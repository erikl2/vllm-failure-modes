#!/usr/bin/env python3
"""
Analyze breaking point with complete concurrency sweep: C=1, C=2, C=3, C=4.
Generates summary statistics and identifies the exact breaking point.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_concurrency_results(concurrency):
    """Load results for a specific concurrency level."""
    csv_path = f'results/results_vllm_concurrency{concurrency}.csv'
    if not Path(csv_path).exists():
        return None
    return pd.read_csv(csv_path)


def calculate_stats(df):
    """Calculate mean ± std for key metrics."""
    if df is None:
        return None
    
    return {
        'p50_ms': (df['p50_ms'].mean(), df['p50_ms'].std()),
        'p95_ms': (df['p95_ms'].mean(), df['p95_ms'].std()),
        'p99_ms': (df['p99_ms'].mean(), df['p99_ms'].std()),
        'throughput': (df['throughput_tok_s'].mean(), df['throughput_tok_s'].std()),
        'total_time': (df['total_time_s'].mean(), df['total_time_s'].std()),
    }


def identify_breaking_point(all_stats):
    """
    Identify the exact breaking point based on performance degradation.
    Breaking point = first concurrency level where performance is worse than naive.
    """
    naive_baseline = 3534.5  # P50 latency from naive benchmark
    
    breaking_point = None
    for c in [1, 2, 3, 4]:
        if all_stats[c] is None:
            continue
        
        p50_mean = all_stats[c]['p50_ms'][0]
        if p50_mean > naive_baseline:
            breaking_point = c
            break
    
    return breaking_point


def print_summary_table(all_stats):
    """Print comprehensive summary table."""
    print("\n" + "="*80)
    print("BREAKING POINT ANALYSIS: Complete Concurrency Sweep")
    print("="*80)
    print()
    print(f"{'Concurrency':<12} {'P50 (ms)':<20} {'P99 (ms)':<20} "
          f"{'Throughput':<20} {'Status':<20}")
    print("-"*80)
    
    naive_p50 = 3534.5
    
    for c in [1, 2, 3, 4]:
        stats = all_stats[c]
        if stats is None:
            print(f"{'C=' + str(c):<12} {'NOT RUN':<20} {'NOT RUN':<20} "
                  f"{'NOT RUN':<20} {'---':<20}")
            continue
        
        p50_mean, p50_std = stats['p50_ms']
        p99_mean, p99_std = stats['p99_ms']
        tput_mean, tput_std = stats['throughput']
        
        # Determine status
        if p50_mean < naive_p50:
            status = "✓ SUCCESS"
            improvement = ((naive_p50 - p50_mean) / naive_p50) * 100
            status_detail = f"✓ {improvement:.1f}% faster"
        else:
            status = "✗ FAILURE"
            degradation = p50_mean / naive_p50
            status_detail = f"✗ {degradation:.1f}× slower"
        
        print(f"{'C=' + str(c):<12} "
              f"{p50_mean:.1f} ± {p50_std:.1f}{'':6} "
              f"{p99_mean:.1f} ± {p99_std:.1f}{'':6} "
              f"{tput_mean:.1f} ± {tput_std:.1f}{'':6} "
              f"{status_detail:<20}")
    
    print("-"*80)
    print(f"{'Naive baseline':<12} {naive_p50:.1f}{'':15} ---{'':17} "
          f"36.2{'':16} Baseline")
    print()


def print_breaking_point_summary(all_stats):
    """Print breaking point identification."""
    breaking_point = identify_breaking_point(all_stats)
    
    print("\n" + "="*80)
    print("BREAKING POINT IDENTIFICATION")
    print("="*80)
    print()
    
    if breaking_point is None:
        print("Status: Breaking point not found in tested range (C=1 to C=4)")
        print("All configurations perform better than naive baseline.")
    else:
        print(f"Breaking Point: Concurrency = {breaking_point}")
        print()
        
        if breaking_point == 1:
            print("Performance degrades immediately at the first concurrency level.")
        else:
            prev_c = breaking_point - 1
            prev_stats = all_stats[prev_c]
            curr_stats = all_stats[breaking_point]
            
            if prev_stats and curr_stats:
                prev_p50 = prev_stats['p50_ms'][0]
                curr_p50 = curr_stats['p50_ms'][0]
                degradation = curr_p50 / prev_p50
                
                print(f"Last successful concurrency: C={prev_c}")
                print(f"  P50 latency: {prev_p50:.1f} ms")
                print()
                print(f"First failure: C={breaking_point}")
                print(f"  P50 latency: {curr_p50:.1f} ms")
                print(f"  Degradation: {degradation:.2f}× slower than C={prev_c}")
                print()
                
                naive_p50 = 3534.5
                vs_naive = curr_p50 / naive_p50
                print(f"  vs Naive baseline: {vs_naive:.2f}× slower")
        
        print()
        print("Root Cause: Head-of-line blocking with heterogeneous prompt lengths")
        print("  - 70% short prompts (~55 tokens)")
        print("  - 30% long prompts (~505 tokens)")
        print("  - Long prompts block short prompts during prefill phase")
        print("  - System saturates as queue depth increases")
    
    print("="*80)
    print()


def save_summary_to_file(all_stats, output_file='results/breaking_point_summary.txt'):
    """Save detailed summary to file."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BREAKING POINT ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Complete Concurrency Sweep Results\n")
        f.write("-"*40 + "\n\n")
        
        for c in [1, 2, 3, 4]:
            stats = all_stats[c]
            if stats is None:
                f.write(f"C={c}: NOT RUN\n\n")
                continue
            
            f.write(f"Concurrency = {c}\n")
            p50_mean, p50_std = stats['p50_ms']
            p95_mean, p95_std = stats['p95_ms']
            p99_mean, p99_std = stats['p99_ms']
            tput_mean, tput_std = stats['throughput']
            
            f.write(f"  P50 latency:    {p50_mean:.1f} ± {p50_std:.1f} ms\n")
            f.write(f"  P95 latency:    {p95_mean:.1f} ± {p95_std:.1f} ms\n")
            f.write(f"  P99 latency:    {p99_mean:.1f} ± {p99_std:.1f} ms\n")
            f.write(f"  Throughput:     {tput_mean:.1f} ± {tput_std:.1f} tok/s\n")
            f.write("\n")
        
        breaking_point = identify_breaking_point(all_stats)
        f.write("Breaking Point\n")
        f.write("-"*40 + "\n")
        if breaking_point:
            f.write(f"First failure occurs at: C={breaking_point}\n")
            
            if breaking_point > 1:
                prev_c = breaking_point - 1
                prev_p50 = all_stats[prev_c]['p50_ms'][0]
                curr_p50 = all_stats[breaking_point]['p50_ms'][0]
                degradation = curr_p50 / prev_p50
                f.write(f"Performance cliff: {degradation:.2f}× degradation from C={prev_c} to C={breaking_point}\n")
        else:
            f.write("No breaking point found in tested range\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
    
    print(f"✓ Summary saved to {output_file}")


def main():
    """Main analysis function."""
    print("="*80)
    print("BREAKING POINT ANALYSIS")
    print("="*80)
    print()
    
    # Load all results
    print("Loading results...")
    all_stats = {}
    for c in [1, 2, 3, 4]:
        df = load_concurrency_results(c)
        if df is not None:
            print(f"  ✓ Loaded C={c}: {len(df)} seeds")
            all_stats[c] = calculate_stats(df)
        else:
            print(f"  ⚠ C={c}: NOT FOUND (run experiments first)")
            all_stats[c] = None
    
    print()
    
    # Check if we have enough data
    available_configs = [c for c in [1, 2, 3, 4] if all_stats[c] is not None]
    if len(available_configs) < 2:
        print("ERROR: Need at least 2 concurrency levels to analyze breaking point")
        print("Please run the experiments first using run_breaking_point_ablation.sh")
        return
    
    # Print summary table
    print_summary_table(all_stats)
    
    # Identify and print breaking point
    print_breaking_point_summary(all_stats)
    
    # Save to file
    save_summary_to_file(all_stats)
    
    # Recommendations
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    if all_stats[2] is None or all_stats[3] is None:
        print("Missing data for C=2 or C=3:")
        print("  Run: bash run_breaking_point_ablation.sh")
        print()
    
    print("After all experiments complete:")
    print("  1. Review breaking_point_summary.txt")
    print("  2. Run: python generate_plots.py (to update plots)")
    print("  3. Incorporate findings into technical report")
    print()


if __name__ == '__main__':
    main()
