#!/usr/bin/env python3
"""
Health check and GPU cleanup system for vLLM experiments.
Runs between seeds to ensure clean state.
"""

import subprocess
import requests
import time
import sys
from typing import Tuple, Optional

class HealthCheck:
    """Health check and cleanup for vLLM experiments."""

    def __init__(self, vllm_url: str = "http://localhost:8000"):
        self.vllm_url = vllm_url

    def check_vllm_responsive(self) -> bool:
        """Check if vLLM server is responsive."""
        try:
            response = requests.get(f"{self.vllm_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get GPU memory usage in MB.

        Returns:
            (used_mb, total_mb)
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            used, total = result.stdout.strip().split(',')
            return float(used.strip()), float(total.strip())
        except:
            return 0.0, 0.0

    def kill_vllm_processes(self):
        """Kill all vLLM processes."""
        print("Killing vLLM processes...")
        try:
            # Kill by process name
            subprocess.run(
                ["pkill", "-9", "-f", "vllm.entrypoints.openai.api_server"],
                check=False
            )
            time.sleep(2)

            # Verify no vLLM processes remain
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True
            )

            if result.returncode == 0:
                print("  Warning: Some vLLM processes still alive")
                return False
            else:
                print("  ✓ All vLLM processes killed")
                return True
        except Exception as e:
            print(f"  Error killing processes: {e}")
            return False

    def wait_for_gpu_memory_release(self, timeout: int = 30) -> bool:
        """Wait for GPU memory to be released."""
        print("Waiting for GPU memory to clear...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            used_mb, total_mb = self.get_gpu_memory_usage()

            # Consider GPU clear if less than 1GB used
            if used_mb < 1000:
                print(f"  ✓ GPU memory clear: {used_mb:.0f}MB used")
                return True

            print(f"  GPU memory: {used_mb:.0f}MB / {total_mb:.0f}MB used (waiting...)")
            time.sleep(5)

        print(f"  ✗ GPU memory not cleared after {timeout}s")
        return False

    def full_cleanup(self) -> bool:
        """Perform full cleanup: kill vLLM, wait for GPU clear."""
        print("\n=== Performing full cleanup ===")

        # Kill processes
        if not self.kill_vllm_processes():
            print("Warning: Process cleanup incomplete")

        # Wait for GPU memory
        if not self.wait_for_gpu_memory_release():
            print("Warning: GPU memory not fully cleared")
            print("This may cause issues for next seed")
            return False

        print("=== Cleanup complete ===\n")
        return True

    def check_system_healthy(self) -> bool:
        """Check if system is healthy for next experiment."""
        print("\n=== Health Check ===")

        # Check vLLM responsiveness
        vllm_ok = self.check_vllm_responsive()
        print(f"vLLM server: {'✓ responsive' if vllm_ok else '✗ not responsive'}")

        # Check GPU memory
        used_mb, total_mb = self.get_gpu_memory_usage()
        gpu_ok = used_mb < total_mb * 0.95  # Less than 95% used
        print(f"GPU memory: {used_mb:.0f}MB / {total_mb:.0f}MB used ({'✓ OK' if gpu_ok else '✗ HIGH'})")

        print("===================\n")
        return vllm_ok and gpu_ok


def health_check_between_seeds(seed_just_completed: int, next_seed: Optional[int] = None):
    """Run health check between seeds and cleanup if needed."""
    print(f"\n{'='*60}")
    print(f"INTER-SEED HEALTH CHECK (completed seed {seed_just_completed})")
    print(f"{'='*60}")

    checker = HealthCheck()

    # Check if system is healthy
    if checker.check_system_healthy():
        print("✓ System healthy, no cleanup needed")
        return True

    # System not healthy, needs cleanup
    print("⚠ System not healthy, performing cleanup...")

    success = checker.full_cleanup()

    if not success:
        print("\n" + "="*60)
        print("✗ CRITICAL: Cleanup failed!")
        print("Manual intervention may be required")
        print("="*60 + "\n")
        return False

    # Verify cleanup worked
    time.sleep(5)
    if checker.check_system_healthy():
        print("✓ System now healthy after cleanup")
        return True
    else:
        print("✗ System still unhealthy after cleanup")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        next_seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
        success = health_check_between_seeds(seed, next_seed)
        sys.exit(0 if success else 1)
    else:
        # Just run health check
        checker = HealthCheck()
        healthy = checker.check_system_healthy()
        sys.exit(0 if healthy else 1)
