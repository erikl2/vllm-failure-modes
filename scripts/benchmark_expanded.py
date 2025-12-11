#!/usr/bin/env python3
"""
Expanded vLLM benchmark for Option C: Full experimental matrix.

Key improvements over original:
1. Shuffles prompts based on seed (makes seed comparison meaningful)
2. Passes --seed to vLLM server
3. Uses temperature > 0 so sampling is affected by seed
4. Adds GPU memory profiling
5. Better crash detection with timing
6. Supports the full experimental matrix

Usage:
    python benchmark_expanded.py \
        --gpu-memory 0.85 \
        --concurrency 4 \
        --seed 42 \
        --workload workload.json \
        --output results/exp_gpu0.85_c4_s42.csv \
        --duration 300 \
        --profile-memory
"""

import json
import time
import csv
import asyncio
import subprocess
import sys
import argparse
import signal
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import random
from pathlib import Path
import threading

import numpy as np
import aiohttp

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 128
VLLM_HOST = "0.0.0.0"
VLLM_PORT = 8000
DTYPE = "float16"

# Use non-zero temperature so seed affects sampling
TEMPERATURE = 0.7
TOP_P = 0.95


# ============================================================================
# Memory Profiler
# ============================================================================

class MemoryProfiler:
    """Profiles GPU memory usage during benchmark runs."""
    
    def __init__(self, output_file: str, interval_ms: int = 100):
        self.output_file = output_file
        self.interval_ms = interval_ms
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.data: List[Dict] = []
        self.start_timestamp = None
        
    def start(self):
        """Start memory profiling in background thread."""
        self.running = True
        self.data = []
        self.start_timestamp = time.time()
        self.thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.thread.start()
        print(f"  📊 Memory profiler started (interval: {self.interval_ms}ms)")
        
    def stop(self) -> Dict:
        """Stop profiling, save results, and return summary."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        summary = self._compute_summary()
        self._save_results()
        print(f"  📊 Memory profile: {len(self.data)} samples saved to {self.output_file}")
        return summary
        
    def _profile_loop(self):
        """Main profiling loop."""
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", 
                     "--query-gpu=memory.used,memory.total,utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.0
                )
                
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        self.data.append({
                            "timestamp": time.time() - self.start_timestamp,
                            "memory_used_mb": float(parts[0]),
                            "memory_total_mb": float(parts[1]),
                            "gpu_util_pct": float(parts[2])
                        })
            except Exception:
                pass
                
            time.sleep(self.interval_ms / 1000.0)
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics from profiling data."""
        if not self.data:
            return {}
        
        mem_used = [d["memory_used_mb"] for d in self.data]
        return {
            "memory_min_mb": min(mem_used),
            "memory_max_mb": max(mem_used),
            "memory_mean_mb": np.mean(mem_used),
            "memory_std_mb": np.std(mem_used),
            "samples": len(self.data)
        }
            
    def _save_results(self):
        """Save profiling results to CSV."""
        if not self.data:
            return
            
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "memory_used_mb", 
                                                    "memory_total_mb", "gpu_util_pct"])
            writer.writeheader()
            writer.writerows(self.data)


# ============================================================================
# vLLM Server Manager
# ============================================================================

class VLLMServer:
    """Manages vLLM server lifecycle with proper process group management."""

    def __init__(self, model_name: str, host: str, port: int, dtype: str, 
                 gpu_memory_util: float, seed: Optional[int] = None,
                 max_model_len: int = 4096):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.dtype = dtype
        self.gpu_memory_util = gpu_memory_util
        self.seed = seed
        self.max_model_len = max_model_len
        self.process = None
        self.process_group_id = None
        self.base_url = f"http://{host}:{port}"
        
        # Tracking
        self.startup_time = None
        self.crash_time = None
        self.crash_reason = None
        self.stderr_output = ""

    def start(self) -> bool:
        """Start vLLM server in new process group."""
        print(f"\n🚀 Starting vLLM server...")
        print(f"   Model: {self.model_name}")
        print(f"   GPU Memory: {self.gpu_memory_util}")
        print(f"   Seed: {self.seed if self.seed is not None else 'None (default)'}")
        print(f"   Port: {self.port}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--gpu-memory-utilization", str(self.gpu_memory_util),
            "--max-model-len", str(self.max_model_len),
            "--disable-log-requests",
            "--trust-remote-code",
        ]
        
        # Pass seed to vLLM server (this is one of the key fixes!)
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])

        # Start in new process group for clean cleanup
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1
        )

        self.process_group_id = self.process.pid
        print(f"   PID: {self.process_group_id}")

        return self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        print("   Waiting for server to be ready...")
        start_time = time.time()
        
        import requests

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                self.stderr_output = stderr
                self.crash_reason = self._parse_crash_reason(stderr)
                print(f"   ❌ Server died during startup!")
                print(f"   Crash reason: {self.crash_reason}")
                return False

            # Check if server is responding
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    self.startup_time = time.time() - start_time
                    print(f"   ✅ Server ready in {self.startup_time:.1f}s")
                    return True
            except:
                pass

            time.sleep(2)

        print(f"   ❌ Server failed to start within {timeout}s")
        self.crash_reason = "STARTUP_TIMEOUT"
        return False

    def _parse_crash_reason(self, stderr: str) -> str:
        """Parse crash reason from stderr."""
        stderr_lower = stderr.lower()
        if "cuda out of memory" in stderr_lower or "outofmemoryerror" in stderr_lower:
            return "CUDA_OOM"
        elif "assertionerror" in stderr_lower:
            return "ASSERTION_FAILURE"
        elif "runtimeerror" in stderr_lower:
            return "RUNTIME_ERROR"
        elif "torch.cuda.outofmemoryerror" in stderr_lower:
            return "TORCH_OOM"
        else:
            # Return first 100 chars of error
            for line in stderr.split('\n'):
                if 'error' in line.lower():
                    return line[:100]
            return "UNKNOWN"

    def is_healthy(self) -> bool:
        """Check if server is still healthy."""
        if self.process.poll() is not None:
            if self.crash_time is None:
                self.crash_time = time.time()
                _, stderr = self.process.communicate()
                self.stderr_output = stderr
                self.crash_reason = self._parse_crash_reason(stderr)
            return False

        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def stop(self):
        """Stop vLLM server and ALL child processes."""
        if self.process_group_id is None:
            return

        print(f"\n🛑 Stopping vLLM server (PID: {self.process_group_id})...")

        try:
            os.killpg(self.process_group_id, signal.SIGTERM)
            time.sleep(3)

            try:
                os.killpg(self.process_group_id, 0)  # Check if still alive
                print("   Force killing...")
                os.killpg(self.process_group_id, signal.SIGKILL)
                time.sleep(2)
            except ProcessLookupError:
                pass

            print("   ✅ Server stopped")

        except ProcessLookupError:
            print("   ✅ Server already stopped")
        except Exception as e:
            print(f"   ⚠️ Cleanup error: {e}")
            # Fallback: kill all vLLM processes
            subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
            time.sleep(2)
        
        self._verify_gpu_cleared()

    def _verify_gpu_cleared(self) -> bool:
        """Verify GPU memory was released."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            used_mb = float(result.stdout.strip())

            if used_mb < 1000:
                print(f"   ✅ GPU memory clear: {used_mb:.0f}MB")
                return True
            else:
                print(f"   ⚠️ GPU memory leak: {used_mb:.0f}MB still in use")
                return False
        except Exception as e:
            print(f"   ❓ Could not verify GPU: {e}")
            return False

    def __enter__(self):
        if not self.start():
            raise RuntimeError(f"Failed to start vLLM: {self.crash_reason}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ============================================================================
# Benchmark Functions
# ============================================================================

def set_seeds(seed: int):
    """Set all random seeds for reproducibility."""
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
) -> Tuple[float, int, bool, str]:
    """Send async request to vLLM server."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,  # Non-zero so seed affects sampling
        "top_p": TOP_P,
    }

    start_time = time.perf_counter()

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return 0.0, 0, False, f"HTTP_{response.status}"

            result = await response.json()
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return latency_ms, tokens, True, ""

    except asyncio.TimeoutError:
        return 0.0, 0, False, "TIMEOUT"
    except aiohttp.ClientError as e:
        return 0.0, 0, False, "CONNECTION_ERROR"
    except Exception as e:
        return 0.0, 0, False, f"ERROR: {str(e)[:50]}"


async def run_benchmark(
    base_url: str,
    prompts: List[Dict],
    concurrency: int,
    max_tokens: int,
    seed: int,
    duration_seconds: int = 300,
    health_check_fn=None
) -> Dict:
    """
    Run benchmark with specified concurrency level.
    
    Key fix: Shuffles prompts based on seed for meaningful seed comparison.
    """
    set_seeds(seed)
    
    # CRITICAL FIX: Shuffle prompts so seed actually affects request order
    shuffled_prompts = prompts.copy()
    random.shuffle(shuffled_prompts)
    
    url = f"{base_url}/v1/completions"

    latencies = []
    tokens_list = []
    errors = []
    
    semaphore = asyncio.Semaphore(concurrency)
    start_time = time.time()
    server_crashed = False

    async def bounded_request(session, prompt_data, request_id):
        async with semaphore:
            return await send_request(
                session=session,
                url=url,
                prompt=prompt_data["prompt"],
                max_tokens=max_tokens,
                request_id=request_id
            )

    timeout = aiohttp.ClientTimeout(total=120)
    
    print(f"\n📈 Running benchmark: C={concurrency}, seed={seed}")
    print(f"   Prompts: {len(shuffled_prompts)} (shuffled)")
    print(f"   Duration limit: {duration_seconds}s")
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            bounded_request(session, prompt, i)
            for i, prompt in enumerate(shuffled_prompts)
        ]

        completed = 0
        last_progress = 0
        
        for coro in asyncio.as_completed(tasks):
            try:
                latency_ms, tokens, success, error = await coro
                
                if success:
                    latencies.append(latency_ms)
                    tokens_list.append(tokens)
                else:
                    errors.append(error)
                
                completed += 1
                
                # Progress update every 10%
                progress = int(completed * 100 / len(shuffled_prompts))
                if progress >= last_progress + 10:
                    elapsed = time.time() - start_time
                    print(f"   Progress: {progress}% ({completed}/{len(shuffled_prompts)}) - {elapsed:.0f}s elapsed")
                    last_progress = progress
                
                # Check duration limit
                if time.time() - start_time > duration_seconds:
                    print(f"   ⏰ Duration limit reached after {completed} requests")
                    break
                
                # Check server health periodically
                if health_check_fn and completed % 10 == 0:
                    if not health_check_fn():
                        print(f"   💥 Server crashed after {completed} requests!")
                        server_crashed = True
                        break
                    
            except Exception as e:
                errors.append(str(e))

    total_time = time.time() - start_time

    # Calculate metrics
    if latencies:
        metrics = {
            "seed": seed,
            "concurrency": concurrency,
            "p50_ms": np.percentile(latencies, 50),
            "p90_ms": np.percentile(latencies, 90),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "avg_ms": np.mean(latencies),
            "min_ms": np.min(latencies),
            "max_ms": np.max(latencies),
            "throughput_tok_s": sum(tokens_list) / total_time if total_time > 0 else 0,
            "total_tokens": sum(tokens_list),
            "total_time_s": total_time,
            "successful_requests": len(latencies),
            "failed_requests": len(errors),
            "total_requests": len(shuffled_prompts),
            "completion_rate": len(latencies) / len(shuffled_prompts),
            "status": "CRASHED" if server_crashed else "SUCCESS"
        }
    else:
        metrics = {
            "seed": seed,
            "concurrency": concurrency,
            "p50_ms": 0, "p90_ms": 0, "p95_ms": 0, "p99_ms": 0,
            "avg_ms": 0, "min_ms": 0, "max_ms": 0,
            "throughput_tok_s": 0, "total_tokens": 0,
            "total_time_s": total_time,
            "successful_requests": 0,
            "failed_requests": len(errors),
            "total_requests": len(shuffled_prompts),
            "completion_rate": 0,
            "status": "CRASHED" if server_crashed else "FAILED",
            "error_sample": errors[0] if errors else "No requests completed"
        }

    return metrics


def save_result(result: Dict, filepath: str):
    """Save single result to CSV."""
    fieldnames = [
        "timestamp", "seed", "concurrency", "gpu_memory",
        "p50_ms", "p90_ms", "p95_ms", "p99_ms", "avg_ms", "min_ms", "max_ms",
        "throughput_tok_s", "total_tokens", "total_time_s",
        "successful_requests", "failed_requests", "total_requests", "completion_rate",
        "status", "crash_reason", "crash_time_s", "startup_time_s",
        "memory_max_mb", "memory_mean_mb"
    ]
    
    result["timestamp"] = datetime.now().isoformat()
    
    file_exists = Path(filepath).exists()
    
    with open(filepath, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def print_result_summary(result: Dict):
    """Print result summary."""
    print(f"\n{'='*60}")
    print(f"RESULT SUMMARY")
    print(f"{'='*60}")
    print(f"Status:      {result.get('status', 'UNKNOWN')}")
    
    if result.get('crash_reason'):
        print(f"Crash:       {result['crash_reason']} at {result.get('crash_time_s', 'N/A')}s")
    
    if result.get('status') == 'SUCCESS':
        print(f"Throughput:  {result.get('throughput_tok_s', 0):.1f} tok/s")
        print(f"P50 Latency: {result.get('p50_ms', 0):.1f} ms")
        print(f"P95 Latency: {result.get('p95_ms', 0):.1f} ms")
        print(f"P99 Latency: {result.get('p99_ms', 0):.1f} ms")
        print(f"Completed:   {result.get('successful_requests', 0)}/{result.get('total_requests', 0)}")
    
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Expanded vLLM Benchmark for Option C',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--gpu-memory', type=float, required=True,
                        help='GPU memory utilization (0.0-1.0)')
    parser.add_argument('--concurrency', type=int, required=True,
                        help='Concurrency level')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--workload', type=str, required=True,
                        help='Path to workload JSON')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV path')
    parser.add_argument('--duration', type=int, default=300,
                        help='Max duration in seconds')
    parser.add_argument('--profile-memory', action='store_true',
                        help='Enable GPU memory profiling')
    parser.add_argument('--max-model-len', type=int, default=4096,
                        help='Maximum model context length')
    
    args = parser.parse_args()

    print("="*70)
    print("  EXPANDED vLLM BENCHMARK (Option C)")
    print("="*70)
    print(f"  GPU Memory:    {args.gpu_memory}")
    print(f"  Concurrency:   {args.concurrency}")
    print(f"  Seed:          {args.seed}")
    print(f"  Duration:      {args.duration}s")
    print(f"  Memory Prof:   {args.profile_memory}")
    print(f"  Output:        {args.output}")
    print("="*70)

    # Load workload
    prompts = load_workload(args.workload)
    print(f"\n📂 Loaded {len(prompts)} prompts from {args.workload}")

    # Setup output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Initialize result
    result = {
        "seed": args.seed,
        "concurrency": args.concurrency,
        "gpu_memory": args.gpu_memory,
        "status": "FAILED_TO_START",
        "crash_reason": None,
        "crash_time_s": None,
        "startup_time_s": None
    }

    # Memory profiler
    profiler = None
    if args.profile_memory:
        profile_file = args.output.replace('.csv', f'_memory_s{args.seed}.csv')
        profiler = MemoryProfiler(profile_file)

    server = None
    benchmark_start = None

    try:
        # Start vLLM server
        server = VLLMServer(
            MODEL_NAME, VLLM_HOST, VLLM_PORT, DTYPE,
            gpu_memory_util=args.gpu_memory,
            seed=args.seed,
            max_model_len=args.max_model_len
        )

        if not server.start():
            result["crash_reason"] = server.crash_reason
            save_result(result, args.output)
            print_result_summary(result)
            return 1

        result["startup_time_s"] = server.startup_time

        # Start memory profiling
        if profiler:
            profiler.start()

        # Run benchmark
        benchmark_start = time.time()
        
        metrics = await run_benchmark(
            base_url=server.base_url,
            prompts=prompts,
            concurrency=args.concurrency,
            max_tokens=MAX_NEW_TOKENS,
            seed=args.seed,
            duration_seconds=args.duration,
            health_check_fn=server.is_healthy
        )
        
        result.update(metrics)
        result["gpu_memory"] = args.gpu_memory
        result["startup_time_s"] = server.startup_time
        
        # Check final server health
        if not server.is_healthy():
            result["status"] = "CRASHED"
            result["crash_reason"] = server.crash_reason
            result["crash_time_s"] = server.crash_time - benchmark_start if server.crash_time else None

    except Exception as e:
        result["status"] = "ERROR"
        result["crash_reason"] = str(e)
        if benchmark_start:
            result["crash_time_s"] = time.time() - benchmark_start

    finally:
        # Stop profiler and get summary
        if profiler:
            mem_summary = profiler.stop()
            result.update({
                "memory_max_mb": mem_summary.get("memory_max_mb"),
                "memory_mean_mb": mem_summary.get("memory_mean_mb")
            })

        # Stop server
        if server:
            server.stop()

    # Save and display results
    save_result(result, args.output)
    print_result_summary(result)
    
    print(f"\n📁 Results saved to: {args.output}")
    
    return 0 if result.get("status") == "SUCCESS" else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
