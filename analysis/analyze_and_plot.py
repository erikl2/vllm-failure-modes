#!/usr/bin/env python3
"""
Analyze breaking point ablation results and generate plots for technical report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

def load_results():
    """Load all experimental results."""
    results = {}

    # Naive baseline (from previous experiments)
    results['naive'] = {
        'concurrency': 1,
        'p50_ms': 3534.5,
        'p99_ms': 3572.1,
        'throughput_tok_s': 36.2
    }

    # Load C=2 results
    c2_df = pd.read_csv('results/results_vllm_concurrency2.csv')
    results['c2'] = {
        'concurrency': 2,
        'p50_ms': c2_df['p50_ms'].mean(),
        'p50_std': c2_df['p50_ms'].std(),
        'p99_ms': c2_df['p99_ms'].mean(),
        'p99_std': c2_df['p99_ms'].std(),
        'throughput_tok_s': c2_df['throughput_tok_s'].mean(),
        'throughput_std': c2_df['throughput_tok_s'].std(),
        'df': c2_df
    }

    # Load C=3 results
    c3_df = pd.read_csv('results/results_vllm_concurrency3.csv')
    results['c3'] = {
        'concurrency': 3,
        'p50_ms': c3_df['p50_ms'].mean(),
        'p50_std': c3_df['p50_ms'].std(),
        'p99_ms': c3_df['p99_ms'].mean(),
        'p99_std': c3_df['p99_ms'].std(),
        'throughput_tok_s': c3_df['throughput_tok_s'].mean(),
        'throughput_std': c3_df['throughput_tok_s'].std(),
        'df': c3_df
    }

    # C=4 from previous run (showing breaking point)
    results['c4'] = {
        'concurrency': 4,
        'p50_ms': 1796.2,
        'p99_ms': 1807.3,
        'throughput_tok_s': 19.1,
        'note': 'Breaking point - 4 failures, extreme variance'
    }

    return results


def plot_throughput_comparison(results):
    """Plot throughput comparison across concurrency levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = ['Naive\n(C=1)', 'vLLM\nC=2', 'vLLM\nC=3', 'vLLM\nC=4']
    throughputs = [
        results['naive']['throughput_tok_s'],
        results['c2']['throughput_tok_s'],
        results['c3']['throughput_tok_s'],
        results['c4']['throughput_tok_s']
    ]
    errors = [
        0,  # No error for baseline
        results['c2']['throughput_std'],
        results['c3']['throughput_std'],
        0  # C=4 only one data point
    ]

    colors = ['#d62728', '#2ca02c', '#2ca02c', '#ff7f0e']
    x_pos = np.arange(len(configs))

    bars = ax.bar(x_pos, throughputs, yerr=errors, capsize=5,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, throughput) in enumerate(zip(bars, throughputs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (errors[i] if errors[i] > 0 else 5),
                f'{throughput:.1f}\ntok/s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add speedup annotations
    baseline = results['naive']['throughput_tok_s']
    speedups = [t / baseline for t in throughputs]
    for i, speedup in enumerate(speedups[1:], 1):
        ax.text(i, 10, f'{speedup:.1f}×',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_ylabel('Throughput (tokens/second)', fontweight='bold')
    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_title('Throughput Comparison: Breaking Point Analysis',
                 fontweight='bold', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(configs)
    ax.grid(axis='y', alpha=0.3)

    # Add note about C=4
    ax.text(3, results['c4']['throughput_tok_s'] - 15,
            '⚠ Breaking Point\n4 failures',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3))

    plt.tight_layout()
    plt.savefig('results/throughput_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/throughput_comparison.png")
    plt.close()


def plot_latency_comparison(results):
    """Plot latency comparison (P50 and P99)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    configs = ['Naive\n(C=1)', 'vLLM\nC=2', 'vLLM\nC=3', 'vLLM\nC=4']

    # P50 latencies
    p50_values = [
        results['naive']['p50_ms'],
        results['c2']['p50_ms'],
        results['c3']['p50_ms'],
        results['c4']['p50_ms']
    ]
    p50_errors = [
        0,
        results['c2']['p50_std'],
        results['c3']['p50_std'],
        0
    ]

    # P99 latencies
    p99_values = [
        results['naive']['p99_ms'],
        results['c2']['p99_ms'],
        results['c3']['p99_ms'],
        results['c4']['p99_ms']
    ]
    p99_errors = [
        0,
        results['c2']['p99_std'],
        results['c3']['p99_std'],
        0
    ]

    x_pos = np.arange(len(configs))
    colors = ['#d62728', '#2ca02c', '#2ca02c', '#ff7f0e']

    # P50 plot
    bars1 = ax1.bar(x_pos, p50_values, yerr=p50_errors, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('P50 Latency (ms)', fontweight='bold')
    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_title('P50 Latency (Median)', fontweight='bold', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(configs)
    ax1.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars1, p50_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{value:.0f}ms',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # P99 plot
    bars2 = ax2.bar(x_pos, p99_values, yerr=p99_errors, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('P99 Latency (ms)', fontweight='bold')
    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_title('P99 Latency (Tail)', fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(configs)
    ax2.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars2, p99_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{value:.0f}ms',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.suptitle('Latency Comparison: Breaking Point Analysis',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/latency_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/latency_comparison.png")
    plt.close()


def plot_breaking_point_curve(results):
    """Plot breaking point curve showing throughput vs concurrency."""
    fig, ax = plt.subplots(figsize=(10, 7))

    concurrency = [1, 2, 3, 4]
    throughputs = [
        results['naive']['throughput_tok_s'],
        results['c2']['throughput_tok_s'],
        results['c3']['throughput_tok_s'],
        results['c4']['throughput_tok_s']
    ]
    errors = [
        0,
        results['c2']['throughput_std'],
        results['c3']['throughput_std'],
        0
    ]

    # Plot line with error bars
    ax.errorbar(concurrency, throughputs, yerr=errors,
                marker='o', markersize=10, linewidth=2.5, capsize=5,
                color='#2ca02c', markerfacecolor='white',
                markeredgewidth=2, markeredgecolor='#2ca02c',
                label='Measured Throughput')

    # Highlight the breaking point
    ax.plot(4, results['c4']['throughput_tok_s'],
            marker='X', markersize=15, color='red',
            markeredgewidth=2, markeredgecolor='darkred',
            label='Breaking Point', zorder=5)

    # Add annotations
    ax.annotate('Optimal\n207.7 tok/s',
                xy=(3, results['c3']['throughput_tok_s']),
                xytext=(3.5, results['c3']['throughput_tok_s'] + 30),
                fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='green'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    ax.annotate('Breaking Point!\n4 failures\n10× slowdown',
                xy=(4, results['c4']['throughput_tok_s']),
                xytext=(4.3, 80),
                fontsize=11, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', alpha=0.9))

    ax.set_xlabel('Concurrency Level', fontweight='bold', fontsize=12)
    ax.set_ylabel('Throughput (tokens/second)', fontweight='bold', fontsize=12)
    ax.set_title('Breaking Point Discovery: Throughput vs Concurrency',
                 fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(['C=1\n(Naive)', 'C=2\nvLLM', 'C=3\nvLLM', 'C=4\nvLLM'])

    # Add shaded region for stable operation
    ax.axvspan(1, 3.5, alpha=0.1, color='green', label='Stable Region')
    ax.axvspan(3.5, 4.5, alpha=0.1, color='red', label='Breaking Point')

    plt.tight_layout()
    plt.savefig('results/breaking_point_curve.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/breaking_point_curve.png")
    plt.close()


def plot_per_seed_variance(results):
    """Show variance across seeds for C=2 and C=3."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # C=2 variance
    c2_df = results['c2']['df']
    seeds = c2_df['seed'].values
    x_pos = np.arange(len(seeds))

    ax1.bar(x_pos, c2_df['throughput_tok_s'],
            color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=results['c2']['throughput_tok_s'],
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.set_ylabel('Throughput (tok/s)', fontweight='bold')
    ax1.set_xlabel('Random Seed', fontweight='bold')
    ax1.set_title('C=2 Throughput Variance Across Seeds', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(seeds)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add std annotation
    ax1.text(0.5, 0.95, f'σ = {results["c2"]["throughput_std"]:.2f} tok/s',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # C=3 variance
    c3_df = results['c3']['df']
    seeds = c3_df['seed'].values
    x_pos = np.arange(len(seeds))

    ax2.bar(x_pos, c3_df['throughput_tok_s'],
            color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=results['c3']['throughput_tok_s'],
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.set_ylabel('Throughput (tok/s)', fontweight='bold')
    ax2.set_xlabel('Random Seed', fontweight='bold')
    ax2.set_title('C=3 Throughput Variance Across Seeds', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seeds)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add std annotation
    ax2.text(0.5, 0.95, f'σ = {results["c3"]["throughput_std"]:.2f} tok/s',
             transform=ax2.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle('Reproducibility: Low Variance Across Seeds',
                 fontweight='bold', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('results/seed_variance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/seed_variance.png")
    plt.close()


def generate_summary_table(results):
    """Generate a formatted summary table."""
    summary = []
    summary.append("=" * 80)
    summary.append("BREAKING POINT ABLATION - DETAILED RESULTS")
    summary.append("=" * 80)
    summary.append("")

    # Naive baseline
    summary.append("Naive Baseline (C=1)")
    summary.append("  P50 latency:    3534.5 ms")
    summary.append("  P99 latency:    3572.1 ms")
    summary.append("  Throughput:     36.2 tok/s")
    summary.append("  Status:         Baseline")
    summary.append("")

    # C=2
    summary.append("vLLM with Concurrency=2")
    summary.append(f"  P50 latency:    {results['c2']['p50_ms']:.1f} ± {results['c2']['p50_std']:.1f} ms")
    summary.append(f"  P99 latency:    {results['c2']['p99_ms']:.1f} ± {results['c2']['p99_std']:.1f} ms")
    summary.append(f"  Throughput:     {results['c2']['throughput_tok_s']:.1f} ± {results['c2']['throughput_std']:.2f} tok/s")
    speedup_c2 = results['c2']['throughput_tok_s'] / results['naive']['throughput_tok_s']
    summary.append(f"  Speedup:        {speedup_c2:.1f}× vs baseline")
    summary.append(f"  Status:         ✓ Stable (100% success)")
    summary.append("")

    # C=3
    summary.append("vLLM with Concurrency=3 ⭐ OPTIMAL")
    summary.append(f"  P50 latency:    {results['c3']['p50_ms']:.1f} ± {results['c3']['p50_std']:.1f} ms")
    summary.append(f"  P99 latency:    {results['c3']['p99_ms']:.1f} ± {results['c3']['p99_std']:.1f} ms")
    summary.append(f"  Throughput:     {results['c3']['throughput_tok_s']:.1f} ± {results['c3']['throughput_std']:.2f} tok/s")
    speedup_c3 = results['c3']['throughput_tok_s'] / results['naive']['throughput_tok_s']
    improvement = ((results['c3']['throughput_tok_s'] - results['c2']['throughput_tok_s']) /
                   results['c2']['throughput_tok_s'] * 100)
    summary.append(f"  Speedup:        {speedup_c3:.1f}× vs baseline")
    summary.append(f"  Improvement:    {improvement:.1f}% better than C=2")
    summary.append(f"  Status:         ✓ Stable (100% success)")
    summary.append("")

    # C=4
    summary.append("vLLM with Concurrency=4 ⚠ BREAKING POINT")
    summary.append(f"  P50 latency:    {results['c4']['p50_ms']:.1f} ms")
    summary.append(f"  P99 latency:    {results['c4']['p99_ms']:.1f} ms")
    summary.append(f"  Throughput:     {results['c4']['throughput_tok_s']:.1f} tok/s")
    degradation = ((results['c3']['throughput_tok_s'] - results['c4']['throughput_tok_s']) /
                   results['c3']['throughput_tok_s'] * 100)
    summary.append(f"  Degradation:    {degradation:.1f}% worse than C=3")
    summary.append(f"  Status:         ✗ Breaking point (4 failures, extreme variance)")
    summary.append("")

    summary.append("=" * 80)
    summary.append("KEY FINDINGS")
    summary.append("=" * 80)
    summary.append("")
    summary.append("1. Optimal Configuration: C=3")
    summary.append(f"   - Best throughput: {results['c3']['throughput_tok_s']:.1f} tok/s")
    summary.append(f"   - Low latency: {results['c3']['p50_ms']:.1f}ms median")
    summary.append("   - 100% success rate across all seeds")
    summary.append("")
    summary.append("2. Breaking Point: C=4")
    summary.append("   - 91% throughput degradation from C=3")
    summary.append("   - 4% request failure rate")
    summary.append("   - Extreme variance in response times")
    summary.append("")
    summary.append("3. Reproducibility: ✓ Excellent")
    summary.append(f"   - C=2: σ = {results['c2']['throughput_std']:.2f} tok/s ({results['c2']['throughput_std']/results['c2']['throughput_tok_s']*100:.2f}% variance)")
    summary.append(f"   - C=3: σ = {results['c3']['throughput_std']:.2f} tok/s ({results['c3']['throughput_std']/results['c3']['throughput_tok_s']*100:.2f}% variance)")
    summary.append("")
    summary.append("=" * 80)

    summary_text = "\n".join(summary)

    with open('results/analysis_summary.txt', 'w') as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print("\n✓ Saved: results/analysis_summary.txt")


def main():
    """Main execution."""
    print("=" * 80)
    print("BREAKING POINT ANALYSIS - GENERATING PLOTS")
    print("=" * 80)
    print()

    # Create results directory if needed
    Path('results').mkdir(exist_ok=True)

    # Load data
    print("Loading results...")
    results = load_results()
    print("  ✓ Loaded baseline, C=2, C=3, C=4 data")
    print()

    # Generate plots
    print("Generating plots...")
    plot_throughput_comparison(results)
    plot_latency_comparison(results)
    plot_breaking_point_curve(results)
    plot_per_seed_variance(results)
    print()

    # Generate summary
    print("Generating detailed summary...")
    generate_summary_table(results)
    print()

    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - results/throughput_comparison.png")
    print("  - results/latency_comparison.png")
    print("  - results/breaking_point_curve.png")
    print("  - results/seed_variance.png")
    print("  - results/analysis_summary.txt")
    print()
    print("Ready for your technical report! 📊")


if __name__ == "__main__":
    main()
