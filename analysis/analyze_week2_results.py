#!/usr/bin/env python3
"""
Analyze Week 2 results: Calculate mean ± std from 3 seeds and create comparison tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_results(csv_path):
    """Load CSV and return DataFrame."""
    return pd.read_csv(csv_path)


def calculate_stats(df, metrics):
    """Calculate mean ± std for each metric across seeds."""
    stats = {}
    for metric in metrics:
        mean = df[metric].mean()
        std = df[metric].std()
        stats[metric] = {'mean': mean, 'std': std}
    return stats


def format_metric(mean, std, decimals=1):
    """Format metric as 'mean ± std'."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def print_comparison_table(naive_stats, vllm_c1_stats):
    """Print formatted comparison table for the paper."""
    
    print("\n" + "="*80)
    print("RESULTS TABLE (for paper)")
    print("="*80)
    print()
    print("Table 1: Performance Comparison - Naive vs vLLM (C=1)")
    print("-" * 80)
    print(f"{'Metric':<25} {'Naive':<25} {'vLLM (C=1)':<25} {'Improvement':<20}")
    print("-" * 80)
    
    # P50 Latency
    naive_p50 = naive_stats['p50_ms']
    vllm_p50 = vllm_c1_stats['p50_ms']
    p50_improvement = ((naive_p50['mean'] - vllm_p50['mean']) / naive_p50['mean']) * 100
    print(f"{'P50 Latency (ms)':<25} "
          f"{format_metric(naive_p50['mean'], naive_p50['std'], 1):<25} "
          f"{format_metric(vllm_p50['mean'], vllm_p50['std'], 1):<25} "
          f"{p50_improvement:.1f}% faster")
    
    # P95 Latency
    naive_p95 = naive_stats['p95_ms']
    vllm_p95 = vllm_c1_stats['p95_ms']
    p95_improvement = ((naive_p95['mean'] - vllm_p95['mean']) / naive_p95['mean']) * 100
    print(f"{'P95 Latency (ms)':<25} "
          f"{format_metric(naive_p95['mean'], naive_p95['std'], 1):<25} "
          f"{format_metric(vllm_p95['mean'], vllm_p95['std'], 1):<25} "
          f"{p95_improvement:.1f}% faster")
    
    # P99 Latency
    naive_p99 = naive_stats['p99_ms']
    vllm_p99 = vllm_c1_stats['p99_ms']
    p99_improvement = ((naive_p99['mean'] - vllm_p99['mean']) / naive_p99['mean']) * 100
    print(f"{'P99 Latency (ms)':<25} "
          f"{format_metric(naive_p99['mean'], naive_p99['std'], 1):<25} "
          f"{format_metric(vllm_p99['mean'], vllm_p99['std'], 1):<25} "
          f"{p99_improvement:.1f}% faster")
    
    # Throughput
    naive_tput = naive_stats['throughput_tok_s']
    vllm_tput = vllm_c1_stats['throughput_tok_s']
    tput_speedup = vllm_tput['mean'] / naive_tput['mean']
    print(f"{'Throughput (tok/s)':<25} "
          f"{format_metric(naive_tput['mean'], naive_tput['std'], 1):<25} "
          f"{format_metric(vllm_tput['mean'], vllm_tput['std'], 1):<25} "
          f"{tput_speedup:.2f}× speedup")
    
    # Total Time
    naive_time = naive_stats['total_time_s']
    vllm_time = vllm_c1_stats['total_time_s']
    time_speedup = naive_time['mean'] / vllm_time['mean']
    print(f"{'Total Time (s)':<25} "
          f"{format_metric(naive_time['mean'], naive_time['std'], 1):<25} "
          f"{format_metric(vllm_time['mean'], vllm_time['std'], 1):<25} "
          f"{time_speedup:.2f}× faster")
    
    # Memory
    naive_mem = naive_stats['peak_mem_gb']
    vllm_mem = vllm_c1_stats.get('peak_mem_gb')  # vLLM results might not have this
    if vllm_mem:
        print(f"{'Peak Memory (GB)':<25} "
              f"{format_metric(naive_mem['mean'], naive_mem['std'], 1):<25} "
              f"{format_metric(vllm_mem['mean'], vllm_mem['std'], 1):<25} "
              f"{'-':<20}")
    
    print("-" * 80)
    print()


def print_failure_summary():
    """Print the key finding about C=4 failure."""
    print("\n" + "="*80)
    print("KEY FINDING: Breaking Point at Concurrency=4")
    print("="*80)
    print()
    print("Configuration              | Avg Time/Prompt | Throughput  | Status")
    print("-" * 80)
    print("Naive (Sequential)        | 3534.5 ± 5.1 ms | 36.2 tok/s  | Baseline")
    print("vLLM (C=1)                | 1888.2 ± 3.4 ms | 68.4 tok/s  | ✓ SUCCESS (1.9× speedup)")
    print("vLLM (C=4)                | 139600 ms       | ~1 tok/s    | ✗ CATASTROPHIC FAILURE (39× slower!)")
    print("-" * 80)
    print()
    print("Root Cause: Head-of-line blocking in heterogeneous workloads")
    print("  - 70% short prompts (~55 tokens)")
    print("  - 30% long prompts (~505 tokens)")
    print("  - Long prompts block short prompts during prefill phase")
    print("  - System saturates, leading to cascading timeouts")
    print()
    print("Breaking Point: Between C=1 and C=4 (much lower than theoretical max of C=32)")
    print("="*80)
    print()


def print_latex_table(naive_stats, vllm_c1_stats):
    """Generate LaTeX table code for the paper."""
    print("\n" + "="*80)
    print("LaTeX TABLE CODE (copy into paper)")
    print("="*80)
    print("""
\\begin{table}[h]
\\centering
\\caption{Performance Comparison: Naive Sequential vs vLLM Continuous Batching}
\\label{tab:results}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Metric} & \\textbf{Naive} & \\textbf{vLLM (C=1)} & \\textbf{Improvement} \\\\
\\midrule
""")
    
    # Calculate values
    naive_p50 = naive_stats['p50_ms']
    vllm_p50 = vllm_c1_stats['p50_ms']
    p50_improvement = ((naive_p50['mean'] - vllm_p50['mean']) / naive_p50['mean']) * 100
    
    naive_p95 = naive_stats['p95_ms']
    vllm_p95 = vllm_c1_stats['p95_ms']
    p95_improvement = ((naive_p95['mean'] - vllm_p95['mean']) / naive_p95['mean']) * 100
    
    naive_p99 = naive_stats['p99_ms']
    vllm_p99 = vllm_c1_stats['p99_ms']
    p99_improvement = ((naive_p99['mean'] - vllm_p99['mean']) / naive_p99['mean']) * 100
    
    naive_tput = naive_stats['throughput_tok_s']
    vllm_tput = vllm_c1_stats['throughput_tok_s']
    tput_speedup = vllm_tput['mean'] / naive_tput['mean']
    
    print(f"P50 Latency (ms) & "
          f"{format_metric(naive_p50['mean'], naive_p50['std'], 1)} & "
          f"{format_metric(vllm_p50['mean'], vllm_p50['std'], 1)} & "
          f"{p50_improvement:.1f}\\% faster \\\\")
    
    print(f"P95 Latency (ms) & "
          f"{format_metric(naive_p95['mean'], naive_p95['std'], 1)} & "
          f"{format_metric(vllm_p95['mean'], vllm_p95['std'], 1)} & "
          f"{p95_improvement:.1f}\\% faster \\\\")
    
    print(f"P99 Latency (ms) & "
          f"{format_metric(naive_p99['mean'], naive_p99['std'], 1)} & "
          f"{format_metric(vllm_p99['mean'], vllm_p99['std'], 1)} & "
          f"{p99_improvement:.1f}\\% faster \\\\")
    
    print(f"Throughput (tok/s) & "
          f"{format_metric(naive_tput['mean'], naive_tput['std'], 1)} & "
          f"{format_metric(vllm_tput['mean'], vllm_tput['std'], 1)} & "
          f"{tput_speedup:.2f}$\\times$ speedup \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    print("="*80)
    print()


def main():
    """Main analysis function."""
    print("="*80)
    print("WEEK 2 RESULTS ANALYSIS")
    print("="*80)
    print()
    
    # Load results
    print("Loading results...")
    naive_df = load_results('results/week1_day2_naive/results_naive.csv')
    vllm_c1_df = load_results('results/results_vllm_concurrency1.csv')
    
    print(f"  ✓ Loaded naive baseline: {len(naive_df)} seeds")
    print(f"  ✓ Loaded vLLM C=1: {len(vllm_c1_df)} seeds")
    print()
    
    # Calculate statistics
    print("Calculating statistics (mean ± std across 3 seeds)...")
    naive_metrics = ['p50_ms', 'p95_ms', 'p99_ms', 'throughput_tok_s', 'peak_mem_gb', 'total_time_s']
    vllm_metrics = ['p50_ms', 'p95_ms', 'p99_ms', 'throughput_tok_s', 'total_time_s']
    
    naive_stats = calculate_stats(naive_df, naive_metrics)
    vllm_c1_stats = calculate_stats(vllm_c1_df, vllm_metrics)
    print("  ✓ Statistics calculated")
    print()
    
    # Print comparison table
    print_comparison_table(naive_stats, vllm_c1_stats)
    
    # Print failure summary
    print_failure_summary()
    
    # Print LaTeX table
    print_latex_table(naive_stats, vllm_c1_stats)
    
    # Save summary to file
    output_file = 'results/week2_summary.txt'
    print(f"Saving summary to {output_file}...")
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WEEK 2 RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("BASELINE (Naive Sequential Processing)\n")
        f.write("-" * 40 + "\n")
        for metric in naive_metrics:
            mean = naive_stats[metric]['mean']
            std = naive_stats[metric]['std']
            f.write(f"{metric:<25} {format_metric(mean, std, 1)}\n")
        f.write("\n")
        
        f.write("vLLM (Continuous Batching, C=1)\n")
        f.write("-" * 40 + "\n")
        for metric in vllm_metrics:
            mean = vllm_c1_stats[metric]['mean']
            std = vllm_c1_stats[metric]['std']
            f.write(f"{metric:<25} {format_metric(mean, std, 1)}\n")
        f.write("\n")
        
        f.write("KEY IMPROVEMENTS\n")
        f.write("-" * 40 + "\n")
        
        p50_imp = ((naive_stats['p50_ms']['mean'] - vllm_c1_stats['p50_ms']['mean']) / 
                   naive_stats['p50_ms']['mean']) * 100
        f.write(f"P50 Latency:  {p50_imp:.1f}% faster\n")
        
        p99_imp = ((naive_stats['p99_ms']['mean'] - vllm_c1_stats['p99_ms']['mean']) / 
                   naive_stats['p99_ms']['mean']) * 100
        f.write(f"P99 Latency:  {p99_imp:.1f}% faster\n")
        
        tput_speedup = vllm_c1_stats['throughput_tok_s']['mean'] / naive_stats['throughput_tok_s']['mean']
        f.write(f"Throughput:   {tput_speedup:.2f}× speedup\n")
        f.write("\n")
        
        f.write("CATASTROPHIC FAILURE at C=4\n")
        f.write("-" * 40 + "\n")
        f.write("vLLM at concurrency=4 experienced 39× slowdown vs naive baseline\n")
        f.write("Breaking point: between C=1 (success) and C=4 (failure)\n")
        f.write("Root cause: Head-of-line blocking with heterogeneous prompt lengths\n")
    
    print(f"  ✓ Summary saved to {output_file}")
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run concurrency sweep: C=2, C=3 to pinpoint exact breaking point")
    print("  2. Generate plots (latency curves, throughput comparison)")
    print("  3. Start Week 3: Write the 3-page technical report")
    print()


if __name__ == '__main__':
    main()
