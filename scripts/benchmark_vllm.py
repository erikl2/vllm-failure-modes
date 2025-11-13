#!/usr/bin/env python3
"""
vLLM benchmark for comparing against naive baseline.
Tests multiple concurrency levels with async requests.
"""

import json
import time
import csv
import asyncio
import subprocess
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Tuple
import random
from pathlib import Path

import numpy as np
import aiohttp
from tqdm.asyncio import tqdm_asyncio

# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 128
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
GPU_MEMORY_UTIL = 0.9
DTYPE = "float16"

SEEDS = [42, 43, 44]
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
WORKLOAD_FILE = "workload.json"
RESULTS_DIR = "results_vllm"
RESULTS_META = "results_vllm_meta.json"
COMPARISON_FILE = "comparison.txt"


class vLLMServer:
    """Manages vLLM server lifecycle."""

    def __init__(self, model_name: str, host: str, port: int, dtype: str, gpu_memory_util: float):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.dtype = dtype
        self.gpu_memory_util = gpu_memory_util
        self.process = None
        self.base_url = f"http://{host}:{port}"

    def start(self):
        """Start vLLM server in background."""
        print(f"\nStarting vLLM server...")
        print(f"  Model: {self.model_name}")
        print(f"  Host: {self.host}:{self.port}")
        print(f"  Dtype: {self.dtype}")
        print(f"  GPU Memory Utilization: {self.gpu_memory_util}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_util),
        ]

        # Start server in background
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        print(f"  Server starting (PID: {self.process.pid})...")
        print(f"  Waiting for server to be ready...")

        # Wait for server to be ready
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                # Check if server is responding
                import requests
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  ✓ vLLM server ready!")
                    return
            except:
                pass

            # Check if process died
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                print(f"\n✗ Server process died!")
                print(f"STDOUT:\n{stdout}")
                print(f"STDERR:\n{stderr}")
                raise RuntimeError("vLLM server failed to start")

            time.sleep(2)

        raise TimeoutError("vLLM server did not start within timeout")

    def stop(self):
        """Stop vLLM server."""
        if self.process:
            print(f"\nStopping vLLM server (PID: {self.process.pid})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("  ✓ Server stopped")
            except subprocess.TimeoutExpired:
                print("  Server didn't stop gracefully, killing...")
                self.process.kill()
                self.process.wait()
                print("  ✓ Server killed")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_workload(filepath: str) -> List[Dict]:
    """Load workload from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data["prompts"]


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    request_id: int
) -> Tuple[float, int, bool]:
    """
    Send async request to vLLM server.

    Returns:
        (latency_ms, tokens_generated, success)
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,  # Deterministic
        "top_p": 1.0,
    }

    start_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                print(f"Request {request_id} failed: {response.status} - {error_text}")
                return 0.0, 0, False

            result = await response.json()
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000

            # Extract tokens generated
            if "choices" in result and len(result["choices"]) > 0:
                completion = result["choices"][0].get("text", "")
                # Estimate tokens (rough approximation)
                tokens_generated = result.get("usage", {}).get("completion_tokens", len(completion.split()))
            else:
                tokens_generated = 0

            return latency_ms, tokens_generated, True

    except asyncio.TimeoutError:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        print(f"Request {request_id} TIMEOUT after {latency_ms/1000:.1f}s")
        return latency_ms, 0, False
    except Exception as e:
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        print(f"Request {request_id} error after {latency_ms/1000:.1f}s: {e}")
        return latency_ms, 0, False


async def run_benchmark_concurrent(
    base_url: str,
    prompts: List[Dict],
    concurrency: int,
    max_tokens: int,
    seed: int
) -> Dict:
    """
    Run benchmark with specified concurrency level.

    Args:
        base_url: vLLM server base URL
        prompts: List of prompt dictionaries
        concurrency: Number of concurrent requests
        max_tokens: Max tokens to generate
        seed: Random seed

    Returns:
        Dictionary with metrics
    """
    set_seeds(seed)

    url = f"{base_url}/v1/completions"

    # Prepare results storage
    latencies = []
    tokens_list = []
    failed_requests = 0

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session, prompt_data, request_id):
        async with semaphore:
            return await send_request(
                session=session,
                url=url,
                prompt=prompt_data["prompt"],
                max_tokens=max_tokens,
                request_id=request_id
            )

    total_start = time.perf_counter()

    # Create session and send all requests
    timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout per request
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Create tasks for all prompts
        tasks = [
            bounded_request(session, prompt, i)
            for i, prompt in enumerate(prompts)
        ]

        # Run with progress bar
        results = await tqdm_asyncio.gather(
            *tasks,
            desc=f"Seed {seed} (c={concurrency})"
        )

    total_end = time.perf_counter()
    total_time_s = total_end - total_start

    # Process results
    for latency_ms, tokens_generated, success in results:
        if success:
            latencies.append(latency_ms)
            tokens_list.append(tokens_generated)
        else:
            failed_requests += 1

    # Calculate metrics
    if latencies:
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        total_tokens = sum(tokens_list)
        throughput_tok_s = total_tokens / total_time_s

        avg_latency = np.mean(latencies)
    else:
        p50_latency = 0
        p95_latency = 0
        p99_latency = 0
        throughput_tok_s = 0
        avg_latency = 0

    metrics = {
        "seed": seed,
        "concurrency": concurrency,
        "p50_ms": p50_latency,
        "p95_ms": p95_latency,
        "p99_ms": p99_latency,
        "avg_ms": avg_latency,
        "throughput_tok_s": throughput_tok_s,
        "total_time_s": total_time_s,
        "total_tokens": sum(tokens_list),
        "num_prompts": len(prompts),
        "successful_requests": len(latencies),
        "failed_requests": failed_requests
    }

    return metrics


def save_results_csv(results: List[Dict], concurrency: int, filepath: str):
    """Save results to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'seed',
            'concurrency',
            'p50_ms',
            'p95_ms',
            'p99_ms',
            'avg_ms',
            'throughput_tok_s',
            'total_time_s'
        ])

        # Write data
        for result in results:
            writer.writerow([
                result['seed'],
                result['concurrency'],
                f"{result['p50_ms']:.1f}",
                f"{result['p95_ms']:.1f}",
                f"{result['p99_ms']:.1f}",
                f"{result['avg_ms']:.1f}",
                f"{result['throughput_tok_s']:.1f}",
                f"{result['total_time_s']:.1f}"
            ])


def print_results_summary(metrics: Dict):
    """Print summary of benchmark results."""
    success_rate = (metrics['successful_requests'] / metrics['num_prompts']) * 100 if metrics['num_prompts'] > 0 else 0

    print(f"\nSeed {metrics['seed']} (Concurrency={metrics['concurrency']}) Results:")
    print("-" * 60)
    print(f"p50 latency:        {metrics['p50_ms']:.1f} ms")
    print(f"p95 latency:        {metrics['p95_ms']:.1f} ms")
    print(f"p99 latency:        {metrics['p99_ms']:.1f} ms")
    print(f"Avg latency:        {metrics['avg_ms']:.1f} ms")
    print(f"Throughput:         {metrics['throughput_tok_s']:.1f} tok/s")
    print(f"Total Time:         {metrics['total_time_s']:.1f} s")
    print(f"Successful:         {metrics['successful_requests']}/{metrics['num_prompts']} ({success_rate:.1f}%)")
    if metrics['failed_requests'] > 0:
        print(f"Failed:             {metrics['failed_requests']} ({100-success_rate:.1f}%)")
    print("-" * 60)


def save_metadata(prompts: List[Dict], filepath: str):
    """Save benchmark metadata."""
    metadata = {
        "model_name": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "dtype": DTYPE,
        "gpu_memory_utilization": GPU_MEMORY_UTIL,
        "num_prompts": len(prompts),
        "seeds": SEEDS,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "vllm_server": {
            "host": VLLM_HOST,
            "port": VLLM_PORT
        }
    }

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def generate_comparison_table(results_by_concurrency: Dict[int, List[Dict]]):
    """Generate comparison table with naive baseline."""
    # Load naive baseline results
    try:
        with open("results/week1_day2_naive/results_naive.csv", 'r') as f:
            reader = csv.DictReader(f)
            naive_results = list(reader)

            # Calculate averages
            naive_p50 = np.mean([float(r['p50_ms']) for r in naive_results])
            naive_p99 = np.mean([float(r['p99_ms']) for r in naive_results])
            naive_throughput = np.mean([float(r['throughput_tok_s']) for r in naive_results])
    except FileNotFoundError:
        print("Warning: Naive baseline results not found, using placeholder values")
        naive_p50 = 3534.5
        naive_p99 = 3572.1
        naive_throughput = 36.2

    lines = []
    lines.append("=" * 80)
    lines.append("Naive vs vLLM Performance Comparison")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"{'Configuration':<20} | {'p50 (ms)':>10} | {'p99 (ms)':>10} | {'Throughput':>12} | {'Speedup':>8}")
    lines.append("-" * 80)

    # Naive baseline
    lines.append(f"{'Naive Baseline':<20} | {naive_p50:>10.1f} | {naive_p99:>10.1f} | {naive_throughput:>10.1f} t/s | {'1.0×':>8}")

    # vLLM results
    for concurrency in sorted(results_by_concurrency.keys()):
        results = results_by_concurrency[concurrency]

        # Calculate averages
        avg_p50 = np.mean([r['p50_ms'] for r in results])
        avg_p99 = np.mean([r['p99_ms'] for r in results])
        avg_throughput = np.mean([r['throughput_tok_s'] for r in results])

        speedup = avg_throughput / naive_throughput

        config_name = f"vLLM (c={concurrency})"
        lines.append(f"{config_name:<20} | {avg_p50:>10.1f} | {avg_p99:>10.1f} | {avg_throughput:>10.1f} t/s | {speedup:>7.1f}×")

    lines.append("=" * 80)
    lines.append("")
    lines.append("Notes:")
    lines.append("- p50/p99: Lower is better (latency)")
    lines.append("- Throughput: Higher is better (tokens/second)")
    lines.append("- Speedup: Relative to naive baseline")
    lines.append("=" * 80)

    return "\n".join(lines)


async def main():
    """Main execution function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="vLLM Benchmark")
    parser.add_argument("--concurrency", type=int, default=None, help="Single concurrency level to test")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to use")
    parser.add_argument("--workload", type=str, default=WORKLOAD_FILE, help="Path to workload JSON file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV file")
    args = parser.parse_args()

    # Determine what to run
    if args.concurrency is not None and args.seed is not None:
        # Single run mode
        concurrency_levels = [args.concurrency]
        seeds = [args.seed]
        single_run = True
    else:
        # Full benchmark mode
        concurrency_levels = CONCURRENCY_LEVELS
        seeds = SEEDS
        single_run = False

    print("=" * 80)
    print("vLLM BENCHMARK")
    print("=" * 80)

    if single_run:
        print(f"Mode: Single run (C={args.concurrency}, seed={args.seed})")
    else:
        print(f"Mode: Full benchmark")

    # Create results directory
    if not single_run:
        Path(RESULTS_DIR).mkdir(exist_ok=True)

    # Load workload
    print(f"\nLoading workload from {args.workload}...")
    prompts = load_workload(args.workload)
    print(f"  Loaded {len(prompts)} prompts")

    # Save metadata (only in full benchmark mode)
    if not single_run:
        save_metadata(prompts, RESULTS_META)
        print(f"  Metadata saved to {RESULTS_META}")

    # Start vLLM server
    with vLLMServer(MODEL_NAME, VLLM_HOST, VLLM_PORT, DTYPE, GPU_MEMORY_UTIL) as server:

        # Store all results
        all_results = {}

        # Run benchmarks for each concurrency level
        for concurrency in concurrency_levels:
            print(f"\n{'=' * 80}")
            print(f"Testing Concurrency Level: {concurrency}")
            print(f"{'=' * 80}")

            concurrency_results = []

            for seed in seeds:
                try:
                    metrics = await run_benchmark_concurrent(
                        base_url=server.base_url,
                        prompts=prompts,
                        concurrency=concurrency,
                        max_tokens=MAX_NEW_TOKENS,
                        seed=seed
                    )

                    concurrency_results.append(metrics)
                    print_results_summary(metrics)

                except Exception as e:
                    print(f"\n✗ Error during benchmark with seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()

            # Save results for this concurrency level
            if args.output:
                csv_file = args.output
            else:
                csv_file = f"{RESULTS_DIR}/results_vllm_concurrency{concurrency}.csv"

            # Handle append mode
            if args.append and Path(csv_file).exists():
                # Append to existing file
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    for result in concurrency_results:
                        writer.writerow([
                            result['seed'],
                            result['concurrency'],
                            f"{result['p50_ms']:.1f}",
                            f"{result['p95_ms']:.1f}",
                            f"{result['p99_ms']:.1f}",
                            f"{result['avg_ms']:.1f}",
                            f"{result['throughput_tok_s']:.1f}",
                            f"{result['total_time_s']:.1f}"
                        ])
                print(f"\n✓ Results appended to {csv_file}")
            else:
                save_results_csv(concurrency_results, concurrency, csv_file)
                print(f"\n✓ Results saved to {csv_file}")

            all_results[concurrency] = concurrency_results

    # Generate comparison table (only in full benchmark mode)
    if not single_run:
        print(f"\n{'=' * 80}")
        print("Generating Comparison Table")
        print(f"{'=' * 80}")

        comparison = generate_comparison_table(all_results)
        print("\n" + comparison)

        # Save comparison
        with open(COMPARISON_FILE, 'w') as f:
            f.write(comparison)
        print(f"\n✓ Comparison saved to {COMPARISON_FILE}")

    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    if single_run:
        print(f"\nResults saved to: {args.output if args.output else csv_file}")
    else:
        print(f"\nAll results saved to: {RESULTS_DIR}/")
        print(f"Comparison table: {COMPARISON_FILE}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
