#!/usr/bin/env python3
"""
Analyze benchmark results and compute statistics across seeds.
"""

import sys
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: pandas not found. Install with: pip install pandas numpy")
    exit(1)


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")
    return df


def compute_statistics(df: pd.DataFrame) -> dict:
    """
    Compute mean and std for each metric.

    Args:
        df: DataFrame with benchmark results

    Returns:
        Dictionary with statistics
    """
    stats = {}

    # Metrics to analyze
    metrics = [
        ('p50_ms', 'p50 latency', 'ms'),
        ('p95_ms', 'p95 latency', 'ms'),
        ('p99_ms', 'p99 latency', 'ms'),
        ('throughput_tok_s', 'throughput', 'tok/s'),
        ('peak_mem_gb', 'peak memory', 'GB'),
        ('total_time_s', 'total time', 's')
    ]

    for col, label, unit in metrics:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            stats[col] = {
                'label': label,
                'mean': mean_val,
                'std': std_val,
                'unit': unit
            }

    return stats


def format_value_with_std(mean: float, std: float, unit: str = '') -> str:
    """
    Format mean ± std with appropriate precision.

    Args:
        mean: Mean value
        std: Standard deviation
        unit: Unit string (e.g., 'ms', 'GB')

    Returns:
        Formatted string
    """
    # Determine decimal places based on magnitude
    if mean >= 100:
        # Large values: 1 decimal place
        mean_str = f"{mean:.1f}"
        std_str = f"{std:.1f}"
    elif mean >= 10:
        # Medium values: 1 decimal place
        mean_str = f"{mean:.1f}"
        std_str = f"{std:.1f}"
    elif mean >= 1:
        # Small values: 2 decimal places
        mean_str = f"{mean:.2f}"
        std_str = f"{std:.2f}"
    else:
        # Very small values: 2 decimal places
        mean_str = f"{mean:.2f}"
        std_str = f"{std:.2f}"

    if unit:
        return f"{mean_str} ± {std_str} {unit}"
    else:
        return f"{mean_str} ± {std_str}"


def generate_report(df: pd.DataFrame, stats: dict, output_name: str = "Naive Baseline") -> str:
    """
    Generate formatted report text.

    Args:
        df: DataFrame with results
        stats: Statistics dictionary
        output_name: Name for the report header

    Returns:
        Formatted report string
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(f"{output_name} Results (N={len(df)} seeds)")
    lines.append("=" * 80)
    lines.append("")

    # Metrics table header
    lines.append(f"{'Metric':<25} {'Mean ± Std':>30}")
    lines.append("-" * 80)

    # Main metrics (exclude total_time for now)
    main_metrics = ['p50_ms', 'p95_ms', 'p99_ms', 'throughput_tok_s', 'peak_mem_gb']

    for metric_key in main_metrics:
        if metric_key in stats:
            stat = stats[metric_key]
            label = stat['label'].title()
            value_str = format_value_with_std(stat['mean'], stat['std'], stat['unit'])
            lines.append(f"{label:<25} {value_str:>30}")

    lines.append("-" * 80)
    lines.append("")

    # Additional info
    if 'total_time_s' in stats:
        stat = stats['total_time_s']
        value_str = format_value_with_std(stat['mean'], stat['std'], stat['unit'])
        lines.append(f"Total time:              {value_str:>30}")

    # Get total prompts from metadata or estimate
    lines.append(f"Seeds:                   {', '.join(map(str, df['seed'].tolist())):>30}")
    lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def save_report(report: str, output_path: str):
    """Save report to text file."""
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to {output_path}")


def print_detailed_stats(df: pd.DataFrame):
    """Print detailed per-seed statistics."""
    print("\nPer-Seed Results:")
    print("-" * 80)
    print(df.to_string(index=False))
    print("-" * 80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze benchmark results and compute statistics across seeds"
    )
    parser.add_argument(
        'csv_file',
        nargs='?',
        default='results_naive.csv',
        help='Path to results CSV file (default: results_naive.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        default='summary_naive.txt',
        help='Output text file for summary (default: summary_naive.txt)'
    )
    parser.add_argument(
        '--name',
        default='Naive Baseline',
        help='Name for the report (default: Naive Baseline)'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed per-seed results'
    )

    args = parser.parse_args()

    try:
        # Load results
        df = load_results(args.csv_file)

        # Show detailed stats if requested
        if args.detailed:
            print_detailed_stats(df)

        print()  # Blank line

        # Compute statistics
        stats = compute_statistics(df)

        # Generate report
        report = generate_report(df, stats, args.name)

        # Print to console
        print(report)

        # Save to file
        save_report(report, args.output)

        # Additional summary
        print(f"\n✓ Analysis complete!")
        print(f"  Input:  {args.csv_file}")
        print(f"  Output: {args.output}")
        print(f"  Seeds:  {len(df)}")

    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you've run the benchmark first:")
        print("  python3 benchmark_naive.py")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
