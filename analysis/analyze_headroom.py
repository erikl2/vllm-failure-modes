#!/usr/bin/env python3
"""
Analyze headroom experiment results.
Determine which GPU memory utilization settings prevent C=4 wedge.
"""

import os
import glob
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_results(base_dir="results/headroom_experiments"):
    """Load all headroom experiment results."""

    results = defaultdict(lambda: defaultdict(list))

    # Parse all experiment directories
    for exp_dir in glob.glob(os.path.join(base_dir, "headroom_*")):
        # Skip if not a directory
        if not os.path.isdir(exp_dir):
            continue

        exp_name = os.path.basename(exp_dir)

        # Parse exp_name: headroom_mem0.80_c3
        parts = exp_name.split('_')
        gpu_mem = float(parts[1].replace('mem', ''))
        concurrency = int(parts[2].replace('c', ''))

        # Load results for each seed
        for result_file in glob.glob(os.path.join(exp_dir, "result_seed*.txt")):
            seed = int(result_file.split('seed')[-1].replace('.txt', ''))

            with open(result_file) as f:
                result_line = f.read().strip()

            if result_line.startswith('SUCCESS'):
                duration = int(result_line.split()[1])
                status = 'success'
            else:
                duration = None
                status = 'failed'

            # Try to load CSV for metrics
            csv_file = result_file.replace('result_', 'results_').replace('.txt', '.csv')
            throughput = None
            avg_latency = None
            if os.path.exists(csv_file):
                try:
                    with open(csv_file) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row.get('throughput_tok_s'):
                                throughput = float(row['throughput_tok_s'])
                            if row.get('avg_ms'):
                                avg_latency = float(row['avg_ms'])
                            break  # Just get first row
                except:
                    pass

            results[gpu_mem][concurrency].append({
                'seed': seed,
                'status': status,
                'duration': duration,
                'throughput': throughput,
                'avg_latency': avg_latency
            })

    return results

def analyze_results(results):
    """Analyze headroom experiment results."""

    print("="*60)
    print("Headroom Experiment Analysis")
    print("="*60)
    print()

    # For each GPU memory setting
    for gpu_mem in sorted(results.keys()):
        print(f"GPU Memory Utilization: {gpu_mem}")
        print("-" * 40)

        for concurrency in sorted(results[gpu_mem].keys()):
            seeds = results[gpu_mem][concurrency]
            total = len(seeds)
            successes = sum(1 for s in seeds if s['status'] == 'success')

            success_rate = (successes / total * 100) if total > 0 else 0

            # Average throughput for successful runs
            throughputs = [s['throughput'] for s in seeds if s['throughput'] is not None]
            avg_throughput = np.mean(throughputs) if throughputs else 0

            status_icon = "✓" if success_rate == 100 else "⚠" if success_rate > 0 else "✗"

            print(f"  C={concurrency}: {status_icon} {successes}/{total} seeds succeeded "
                  f"({success_rate:.0f}%)", end="")

            if avg_throughput > 0:
                print(f" - Avg throughput: {avg_throughput:.1f} tok/s")
            else:
                print()

        print()

    print("="*60)
    print("Key Findings:")
    print("="*60)

    # Check if C=4 works at lower memory settings
    c4_results = {gpu_mem: results[gpu_mem][4] for gpu_mem in results.keys() if 4 in results[gpu_mem]}

    for gpu_mem, seeds in sorted(c4_results.items()):
        successes = sum(1 for s in seeds if s['status'] == 'success')
        total = len(seeds)

        if successes == total:
            print(f"✓ C=4 is STABLE at GPU mem util {gpu_mem} (all {total} seeds succeeded)")
        elif successes > 0:
            print(f"⚠ C=4 is UNSTABLE at GPU mem util {gpu_mem} ({successes}/{total} seeds succeeded)")
        else:
            print(f"✗ C=4 FAILS at GPU mem util {gpu_mem} (0/{total} seeds succeeded)")

    print()

    # Determine safe operating point
    safe_settings = []
    for gpu_mem in sorted(results.keys()):
        for concurrency in sorted(results[gpu_mem].keys()):
            seeds = results[gpu_mem][concurrency]
            if all(s['status'] == 'success' for s in seeds):
                safe_settings.append((gpu_mem, concurrency))

    if safe_settings:
        max_c = max(c for _, c in safe_settings)
        print(f"Maximum safe concurrency: C={max_c}")
        print(f"Safe at these settings:")
        for gpu_mem, c in safe_settings:
            if c == max_c:
                print(f"  - GPU mem {gpu_mem}, C={c}")

def plot_headroom_results(results, output_dir="results/headroom_experiments"):
    """Generate plots for headroom experiments."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Success rate heatmap
    gpu_mems = sorted(results.keys())
    concurrencies = sorted(set(c for gm in results.values() for c in gm.keys()))

    success_matrix = []
    for gpu_mem in gpu_mems:
        row = []
        for c in concurrencies:
            if c in results[gpu_mem]:
                seeds = results[gpu_mem][c]
                success_rate = sum(1 for s in seeds if s['status'] == 'success') / len(seeds)
                row.append(success_rate * 100)
            else:
                row.append(np.nan)
        success_matrix.append(row)

    im1 = ax1.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax1.set_xticks(range(len(concurrencies)))
    ax1.set_xticklabels([f'C={c}' for c in concurrencies])
    ax1.set_yticks(range(len(gpu_mems)))
    ax1.set_yticklabels([f'{gm:.2f}' for gm in gpu_mems])
    ax1.set_xlabel('Concurrency Level')
    ax1.set_ylabel('GPU Memory Utilization')
    ax1.set_title('Success Rate (%)')
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i, gpu_mem in enumerate(gpu_mems):
        for j, c in enumerate(concurrencies):
            if not np.isnan(success_matrix[i][j]):
                text = ax1.text(j, i, f'{success_matrix[i][j]:.0f}%',
                              ha="center", va="center", color="black", fontsize=10)

    # Plot 2: Throughput comparison
    for gpu_mem in gpu_mems:
        throughputs = []
        throughput_stds = []
        concurrency_plot = []

        for c in concurrencies:
            if c in results[gpu_mem]:
                seeds = results[gpu_mem][c]
                tps = [s['throughput'] for s in seeds if s['throughput'] is not None]
                if tps:
                    throughputs.append(np.mean(tps))
                    throughput_stds.append(np.std(tps))
                    concurrency_plot.append(c)

        if throughputs:
            ax2.errorbar(concurrency_plot, throughputs, yerr=throughput_stds,
                        marker='o', label=f'GPU mem {gpu_mem:.2f}', capsize=5)

    ax2.set_xlabel('Concurrency Level')
    ax2.set_ylabel('Throughput (tokens/s)')
    ax2.set_title('Throughput vs Concurrency by GPU Memory Util')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'headroom_analysis.png'), dpi=150)
    print(f"\n✓ Plot saved to {output_dir}/headroom_analysis.png")

if __name__ == "__main__":
    results = load_experiment_results()
    if not results:
        print("No results found! Run headroom experiments first.")
    else:
        analyze_results(results)
        plot_headroom_results(results)
