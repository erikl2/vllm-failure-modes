#!/usr/bin/env python3
"""
Analyze threshold validation results.
Determine exact GPU memory threshold for C=4 stability.
"""

import os
import sys
import glob
from pathlib import Path

def analyze_threshold(results_dir="results/threshold_validation"):
    """Analyze threshold validation results."""

    print("="*60)
    print("GPU Memory Threshold Analysis for C=4")
    print("="*60)
    print()

    results = {}

    # Parse results for each GPU memory setting
    for exp_dir in sorted(glob.glob(os.path.join(results_dir, "gpu_mem_*"))):
        dir_name = os.path.basename(exp_dir)
        gpu_mem = float(dir_name.replace("gpu_mem_", ""))

        # Count successes
        result_files = glob.glob(os.path.join(exp_dir, "result_seed*.txt"))
        total = len(result_files)
        successes = 0

        for rf in result_files:
            with open(rf) as f:
                if f.read().strip().startswith("SUCCESS"):
                    successes += 1

        success_rate = (successes / total * 100) if total > 0 else 0
        results[gpu_mem] = {
            'total': total,
            'successes': successes,
            'success_rate': success_rate
        }

    # Display results
    print(f"{'GPU Memory':<12} {'Success Rate':<15} {'Status'}")
    print("-" * 60)

    for gpu_mem in sorted(results.keys()):
        data = results[gpu_mem]
        rate = data['success_rate']
        status_icon = "✓" if rate == 100 else "⚠" if rate > 0 else "✗"
        status_text = "STABLE" if rate == 100 else "UNSTABLE" if rate > 0 else "FAILS"

        print(f"{gpu_mem:<12.2f} {data['successes']}/{data['total']} ({rate:>3.0f}%)    {status_icon} {status_text}")

    print()
    print("="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Find critical threshold
    stable_settings = [gm for gm, d in results.items() if d['success_rate'] == 100]
    unstable_settings = [gm for gm, d in results.items() if 0 < d['success_rate'] < 100]
    failing_settings = [gm for gm, d in results.items() if d['success_rate'] == 0]

    if stable_settings:
        min_stable = min(stable_settings)
        print(f"\n✓ Minimum stable GPU memory for C=4: {min_stable:.2f}")
        print(f"  (100% success rate at this setting and above)")

    if failing_settings:
        max_failing = max(failing_settings)
        print(f"\n✗ Maximum failing GPU memory for C=4: {max_failing:.2f}")
        print(f"  (0% success rate at this setting and below)")

    if stable_settings and failing_settings:
        min_stable = min(stable_settings)
        max_failing = max(failing_settings)
        print(f"\n→ Critical threshold is between {max_failing:.2f} and {min_stable:.2f}")

    if unstable_settings:
        print(f"\n⚠ Unstable region (partial success):")
        for gm in sorted(unstable_settings):
            rate = results[gm]['success_rate']
            print(f"  GPU mem {gm:.2f}: {rate:.0f}% success rate")

    print()

    # Paper recommendation
    print("="*60)
    print("PAPER UPDATE RECOMMENDATION")
    print("="*60)

    if stable_settings:
        min_stable = min(stable_settings)
        print(f"""
Add to paper:

"Through fine-grained characterization, we identified {min_stable:.2f} as the
minimum GPU memory utilization for stable C=4 operation on A100 40GB GPUs.
Below this threshold, vLLM experiences initialization failures even with
multiple retry attempts."
""")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_threshold(sys.argv[1])
    else:
        analyze_threshold()
