#!/usr/bin/env python3
"""
Analysis script for Option C expanded experiments.

Generates:
1. Reliability heatmap (GPU memory vs concurrency)
2. Stability frontier plot
3. Throughput scaling analysis  
4. Latency distributions
5. Paper-ready tables and statistics
6. Corrected paper text suggestions

Usage:
    python analyze_option_c.py --results-dir results/option_c_TIMESTAMP
    python analyze_option_c.py --results-dir results/option_c_TIMESTAMP --plots
"""

import argparse
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import sys


# ============================================================================
# Data Loading
# ============================================================================

def load_results(results_dir: str) -> List[Dict]:
    """Load all experiment results from directory."""
    results = []
    results_path = Path(results_dir)
    
    # Try combined file first
    combined = results_path / "all_results.csv"
    if combined.exists():
        print(f"Loading from combined file: {combined}")
        with open(combined) as f:
            reader = csv.DictReader(f)
            results = list(reader)
    else:
        # Load individual files
        print(f"Loading individual CSV files from {results_dir}")
        for csv_file in sorted(results_path.glob("gpu*.csv")):
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                results.extend(list(reader))
    
    # Convert numeric fields
    numeric_fields = [
        'seed', 'concurrency', 'successful_requests', 'failed_requests', 'total_requests'
    ]
    float_fields = [
        'gpu_memory', 'p50_ms', 'p90_ms', 'p95_ms', 'p99_ms', 'avg_ms', 'min_ms', 'max_ms',
        'throughput_tok_s', 'total_time_s', 'crash_time_s', 'startup_time_s',
        'memory_max_mb', 'memory_mean_mb', 'completion_rate'
    ]
    
    for r in results:
        for key in numeric_fields:
            if key in r and r[key]:
                try:
                    r[key] = int(float(r[key]))
                except:
                    r[key] = None
        for key in float_fields:
            if key in r and r[key]:
                try:
                    r[key] = float(r[key])
                except:
                    r[key] = None
    
    print(f"Loaded {len(results)} experiment records")
    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_reliability_matrix(results: List[Dict]) -> Tuple[Dict, List, List]:
    """
    Compute reliability matrix: success rate for each (gpu_memory, concurrency) pair.
    """
    groups = defaultdict(list)
    
    for r in results:
        gpu_mem = r.get('gpu_memory')
        conc = r.get('concurrency')
        if gpu_mem is not None and conc is not None:
            is_success = r.get('status', '').upper() == 'SUCCESS'
            groups[(gpu_mem, conc)].append({
                'success': is_success,
                'seed': r.get('seed'),
                'crash_reason': r.get('crash_reason'),
                'crash_time_s': r.get('crash_time_s')
            })
    
    reliability = {}
    for key, outcomes in groups.items():
        n = len(outcomes)
        successes = sum(1 for o in outcomes if o['success'])
        
        # Get crash details for failed runs
        crash_times = [o['crash_time_s'] for o in outcomes if not o['success'] and o['crash_time_s']]
        crash_reasons = [o['crash_reason'] for o in outcomes if not o['success'] and o['crash_reason']]
        
        reliability[key] = {
            "success_rate": successes / n if n > 0 else 0,
            "successes": successes,
            "failures": n - successes,
            "n": n,
            "seeds_passed": [o['seed'] for o in outcomes if o['success']],
            "seeds_failed": [o['seed'] for o in outcomes if not o['success']],
            "mean_crash_time": np.mean(crash_times) if crash_times else None,
            "crash_reasons": list(set(crash_reasons))
        }
    
    gpu_memory_levels = sorted(set(k[0] for k in groups.keys()))
    concurrency_levels = sorted(set(k[1] for k in groups.keys()))
    
    return reliability, gpu_memory_levels, concurrency_levels


def find_stability_frontier(reliability: Dict, gpu_memory_levels: List, 
                            concurrency_levels: List) -> Dict[int, Optional[float]]:
    """
    Find minimum GPU memory for 100% reliability at each concurrency level.
    """
    frontier = {}
    
    for c in concurrency_levels:
        min_stable = None
        for gpu_mem in gpu_memory_levels:
            data = reliability.get((gpu_mem, c))
            if data and data["success_rate"] == 1.0:
                min_stable = gpu_mem
                break
        frontier[c] = min_stable
    
    return frontier


def compute_performance_stats(results: List[Dict]) -> Dict:
    """Compute performance statistics for successful runs."""
    successful = [r for r in results if r.get('status', '').upper() == 'SUCCESS']
    
    if not successful:
        return {}
    
    by_config = defaultdict(list)
    for r in successful:
        key = (r.get('gpu_memory'), r.get('concurrency'))
        by_config[key].append(r)
    
    stats = {}
    for (gpu_mem, conc), runs in by_config.items():
        throughputs = [r['throughput_tok_s'] for r in runs if r.get('throughput_tok_s')]
        p50s = [r['p50_ms'] for r in runs if r.get('p50_ms')]
        p99s = [r['p99_ms'] for r in runs if r.get('p99_ms')]
        
        if throughputs:
            stats[(gpu_mem, conc)] = {
                'throughput_mean': np.mean(throughputs),
                'throughput_std': np.std(throughputs),
                'p50_mean': np.mean(p50s) if p50s else None,
                'p50_std': np.std(p50s) if p50s else None,
                'p99_mean': np.mean(p99s) if p99s else None,
                'p99_std': np.std(p99s) if p99s else None,
                'n': len(runs)
            }
    
    return stats


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_reliability_heatmap(reliability: Dict, gpu_memory_levels: List,
                             concurrency_levels: List, output_path: str):
    """Generate heatmap of success rates."""
    matrix = np.zeros((len(gpu_memory_levels), len(concurrency_levels)))
    
    for i, gpu_mem in enumerate(gpu_memory_levels):
        for j, c in enumerate(concurrency_levels):
            data = reliability.get((gpu_mem, c))
            matrix[i, j] = data["success_rate"] * 100 if data else np.nan
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Custom colormap
    colors = ['#d32f2f', '#ff9800', '#ffeb3b', '#8bc34a', '#4caf50']
    cmap = LinearSegmentedColormap.from_list('reliability', colors)
    
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    
    # Labels
    ax.set_xticks(range(len(concurrency_levels)))
    ax.set_xticklabels([f'C={c}' for c in concurrency_levels], fontsize=11)
    ax.set_yticks(range(len(gpu_memory_levels)))
    ax.set_yticklabels([f'{g:.2f}' for g in gpu_memory_levels], fontsize=11)
    
    ax.set_xlabel('Concurrency Level', fontsize=13)
    ax.set_ylabel('GPU Memory Utilization', fontsize=13)
    ax.set_title('vLLM Reliability Matrix: Success Rate by Configuration\n(Llama-3.1-8B on A100 40GB)', 
                 fontsize=14, fontweight='bold')
    
    # Annotations
    for i in range(len(gpu_memory_levels)):
        for j in range(len(concurrency_levels)):
            value = matrix[i, j]
            if not np.isnan(value):
                color = 'white' if value < 50 else 'black'
                data = reliability.get((gpu_memory_levels[i], concurrency_levels[j]))
                n = data["n"] if data else 0
                ax.text(j, i, f'{value:.0f}%\n(n={n})', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Success Rate (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_stability_frontier(frontier: Dict[int, float], concurrency_levels: List,
                           output_path: str):
    """Plot the stability frontier: min GPU memory vs concurrency."""
    valid_points = [(c, frontier[c]) for c in concurrency_levels if frontier.get(c) is not None]
    unstable_points = [c for c in concurrency_levels if frontier.get(c) is None]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if valid_points:
        cs, ts = zip(*valid_points)
        ax.plot(cs, ts, 'o-', color='#1976d2', linewidth=3, markersize=12, 
                label='Minimum stable GPU memory', zorder=3)
        
        # Fill regions
        ax.fill_between(cs, 0.65, ts, alpha=0.25, color='#d32f2f', 
                       label='Unstable region (crashes)')
        ax.fill_between(cs, ts, 1.0, alpha=0.25, color='#4caf50', 
                       label='Stable region (100% reliable)')
        
        # Annotations
        for c, t in valid_points:
            ax.annotate(f'{t:.2f}', (c, t), textcoords="offset points", 
                       xytext=(0, 15), ha='center', fontsize=11, fontweight='bold')
    
    # Mark unstable concurrency levels
    if unstable_points:
        ax.scatter(unstable_points, [0.98] * len(unstable_points), marker='x', s=150, 
                  color='red', linewidths=3, label='No stable configuration', zorder=4)
    
    ax.set_xlabel('Concurrency Level', fontsize=13)
    ax.set_ylabel('GPU Memory Utilization', fontsize=13)
    ax.set_title('Stability Frontier: Minimum GPU Memory for 100% Reliability\n(Llama-3.1-8B on A100 40GB)', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.65, 1.02)
    ax.set_xlim(0, max(concurrency_levels) + 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def plot_throughput_scaling(stats: Dict, concurrency_levels: List, output_path: str):
    """Plot throughput scaling with concurrency."""
    # Group by concurrency (average across GPU memory settings for stable configs)
    by_concurrency = defaultdict(list)
    for (gpu_mem, c), s in stats.items():
        if s.get('throughput_mean'):
            by_concurrency[c].append(s['throughput_mean'])
    
    if not by_concurrency:
        print("  ⚠ No throughput data for scaling plot")
        return
    
    cs = sorted(by_concurrency.keys())
    means = [np.mean(by_concurrency[c]) for c in cs]
    stds = [np.std(by_concurrency[c]) for c in cs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(cs, means, yerr=stds, fmt='o-', capsize=5, capthick=2,
                color='#1976d2', linewidth=2, markersize=10, label='Mean throughput')
    
    # Add scaling reference line
    if len(cs) > 1:
        ideal_scaling = [means[0] * c / cs[0] for c in cs]
        ax.plot(cs, ideal_scaling, '--', color='gray', alpha=0.5, label='Linear scaling (ideal)')
    
    ax.set_xlabel('Concurrency Level', fontsize=13)
    ax.set_ylabel('Throughput (tokens/second)', fontsize=13)
    ax.set_title('Throughput Scaling with Concurrency\n(Stable configurations only)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Annotations
    for c, m, s in zip(cs, means, stds):
        ax.annotate(f'{m:.1f}', (c, m), textcoords="offset points", 
                   xytext=(0, 10), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


# ============================================================================
# Report Generation
# ============================================================================

def generate_paper_table(reliability: Dict, gpu_memory_levels: List,
                        concurrency_levels: List) -> str:
    """Generate LaTeX table for paper."""
    lines = []
    lines.append("% Auto-generated table from Option C experiments")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Reliability by GPU Memory and Concurrency}")
    lines.append("\\label{tab:reliability}")
    
    cols = "l" + "c" * len(concurrency_levels)
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\toprule")
    
    # Header
    header = "GPU Memory & " + " & ".join([f"C={c}" for c in concurrency_levels]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Data rows
    for gpu_mem in gpu_memory_levels:
        row = f"{gpu_mem:.2f}"
        for c in concurrency_levels:
            data = reliability.get((gpu_mem, c))
            if data:
                rate = data["success_rate"] * 100
                n = data["n"]
                if rate == 100:
                    row += f" & \\textbf{{{rate:.0f}\\%}} ({n})"
                elif rate == 0:
                    row += f" & \\textcolor{{red}}{{{rate:.0f}\\%}} ({n})"
                else:
                    row += f" & {rate:.0f}\\% ({n})"
            else:
                row += " & N/A"
        row += " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_text_report(reliability: Dict, frontier: Dict, gpu_memory_levels: List,
                        concurrency_levels: List, stats: Dict, output_path: str):
    """Generate comprehensive text report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("OPTION C EXPERIMENTAL RESULTS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Reliability matrix
    lines.append("RELIABILITY MATRIX")
    lines.append("-" * 80)
    header = f"{'GPU Mem':<10}" + "".join([f"{'C='+str(c):<12}" for c in concurrency_levels])
    lines.append(header)
    lines.append("-" * 80)
    
    for gpu_mem in gpu_memory_levels:
        row = f"{gpu_mem:<10.2f}"
        for c in concurrency_levels:
            data = reliability.get((gpu_mem, c))
            if data:
                rate = data["success_rate"] * 100
                cell = f"{rate:3.0f}% ({data['n']})"
            else:
                cell = "N/A"
            row += f"{cell:<12}"
        lines.append(row)
    
    lines.append("")
    
    # Stability frontier
    lines.append("STABILITY FRONTIER (Minimum GPU memory for 100% reliability)")
    lines.append("-" * 80)
    for c in concurrency_levels:
        threshold = frontier.get(c)
        if threshold:
            lines.append(f"  Concurrency {c:2d}: GPU memory >= {threshold:.2f}")
        else:
            lines.append(f"  Concurrency {c:2d}: No stable configuration found")
    
    lines.append("")
    
    # Key findings
    lines.append("KEY FINDINGS FOR PAPER")
    lines.append("-" * 80)
    
    # Find patterns
    stable_at_all_c = [g for g in gpu_memory_levels 
                      if all(reliability.get((g, c), {}).get('success_rate', 0) == 1.0 
                             for c in concurrency_levels)]
    
    if stable_at_all_c:
        lines.append(f"  • GPU memory >= {min(stable_at_all_c):.2f} is stable at ALL concurrency levels")
    
    # Scaling pattern
    if frontier:
        valid_frontier = [(c, t) for c, t in frontier.items() if t is not None]
        if len(valid_frontier) > 1:
            cs, ts = zip(*sorted(valid_frontier))
            if ts[-1] > ts[0]:
                lines.append(f"  • Higher concurrency requires more GPU memory reservation")
                lines.append(f"    (threshold increases from {ts[0]:.2f} at C={cs[0]} to {ts[-1]:.2f} at C={cs[-1]})")
    
    lines.append("")
    
    # Corrected paper text
    lines.append("SUGGESTED PAPER TEXT (corrected from original)")
    lines.append("-" * 80)
    lines.append("""
The stability frontier shows that minimum safe GPU memory utilization
scales with concurrency level. For our test configuration (Llama-3.1-8B
on A100 40GB), we observe:

[INSERT SPECIFIC VALUES FROM YOUR RESULTS]

Importantly, we observed run-to-run variance in crash behavior at
boundary configurations, with identical settings sometimes succeeding
and sometimes failing. This variance reflects non-deterministic factors
including GPU memory allocator state, request scheduling timing, and
prompt ordering effects. Our use of shuffled prompts (randomized per
seed) contributes to this variance by creating different KV cache growth
patterns across runs.

This finding underscores that single-run validation is insufficient for
production readiness testing—multiple runs with varied conditions are
essential to characterize true reliability.
""")
    
    lines.append("")
    lines.append("=" * 80)
    
    # Write report
    report_text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"  ✓ Saved: {output_path}")
    return report_text


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Option C experimental results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results-dir/analysis)')
    parser.add_argument('--plots', action='store_true',
                        help='Generate plots')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX table')
    
    args = parser.parse_args()
    
    output_dir = args.output_dir or f"{args.results_dir}/analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("")
    print("=" * 70)
    print("  OPTION C RESULTS ANALYSIS")
    print("=" * 70)
    print(f"  Results: {args.results_dir}")
    print(f"  Output:  {output_dir}")
    print("=" * 70)
    print("")
    
    # Load data
    print("Loading results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("ERROR: No results found!")
        return 1
    
    # Compute statistics
    print("\nComputing reliability matrix...")
    reliability, gpu_levels, conc_levels = compute_reliability_matrix(results)
    
    print(f"  GPU memory levels: {gpu_levels}")
    print(f"  Concurrency levels: {conc_levels}")
    
    print("\nFinding stability frontier...")
    frontier = find_stability_frontier(reliability, gpu_levels, conc_levels)
    for c, t in sorted(frontier.items()):
        status = f">= {t:.2f}" if t else "NONE STABLE"
        print(f"  C={c}: {status}")
    
    print("\nComputing performance statistics...")
    stats = compute_performance_stats(results)
    print(f"  Configurations with performance data: {len(stats)}")
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    # Text report (always generated)
    report = generate_text_report(
        reliability, frontier, gpu_levels, conc_levels, stats,
        f"{output_dir}/report.txt"
    )
    
    # Print summary to console
    print("\n" + "=" * 70)
    print(report[:2000] + "..." if len(report) > 2000 else report)
    
    # Plots
    if args.plots:
        print("\nGenerating plots...")
        
        plot_reliability_heatmap(
            reliability, gpu_levels, conc_levels,
            f"{output_dir}/reliability_heatmap.png"
        )
        
        plot_stability_frontier(
            frontier, conc_levels,
            f"{output_dir}/stability_frontier.png"
        )
        
        plot_throughput_scaling(
            stats, conc_levels,
            f"{output_dir}/throughput_scaling.png"
        )
    
    # LaTeX
    if args.latex:
        latex_table = generate_paper_table(reliability, gpu_levels, conc_levels)
        latex_path = f"{output_dir}/table_reliability.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"  ✓ Saved: {latex_path}")
    
    # Save data as JSON
    analysis_data = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": args.results_dir,
        "total_experiments": len(results),
        "gpu_memory_levels": gpu_levels,
        "concurrency_levels": conc_levels,
        "stability_frontier": {str(k): v for k, v in frontier.items()},
        "reliability": {
            f"{k[0]}_{k[1]}": v for k, v in reliability.items()
        }
    }
    
    json_path = f"{output_dir}/analysis.json"
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  ✓ Saved: {json_path}")
    
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  All outputs in: {output_dir}")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
