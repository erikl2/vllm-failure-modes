#!/usr/bin/env python3
"""
vLLM runner with proper process group management.
Ensures all child processes are killed when parent exits.
"""

import os
import sys
import signal
import subprocess
import time
import requests
from typing import Optional

class VLLMRunner:
    """Manages vLLM server with proper cleanup."""

    def __init__(self,
                 model: str,
                 gpu_memory_util: float = 0.9,
                 max_model_len: int = 4096,
                 port: int = 8000,
                 seed: int = None):
        self.model = model
        self.gpu_memory_util = gpu_memory_util
        self.max_model_len = max_model_len
        self.port = port
        self.seed = seed
        self.process: Optional[subprocess.Popen] = None
        self.process_group_id: Optional[int] = None

        # Register cleanup handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle signals by cleaning up process group."""
        print(f"\nReceived signal {signum}, cleaning up...")
        self.stop()
        sys.exit(0)

    def start(self) -> bool:
        """Start vLLM server in new process group."""
        print(f"Starting vLLM server...")
        print(f"  Model: {self.model}")
        print(f"  GPU Memory: {self.gpu_memory_util}")
        print(f"  Port: {self.port}")
        if self.seed is not None:
            print(f"  Seed: {self.seed}")

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_util),
            "--max-model-len", str(self.max_model_len),
            "--disable-log-requests",  # Reduce noise
            "--trust-remote-code",
        ]

        # Pass seed to vLLM server for reproducible sampling (when temperature > 0)
        if self.seed is not None:
            cmd.extend(["--seed", str(self.seed)])

        # CRITICAL: Start in new process group
        # This allows us to kill the entire process tree
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
            text=True,
            bufsize=1
        )

        # Store process group ID (same as process PID when using setsid)
        self.process_group_id = self.process.pid
        print(f"  Process Group ID: {self.process_group_id}")

        # Wait for server to be ready
        return self._wait_for_ready()

    def _wait_for_ready(self, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        print("  Waiting for server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self.process.poll() is not None:
                print(f"  ✗ Process died with exit code {self.process.returncode}")
                stdout, stderr = self.process.communicate()
                print(f"STDOUT:\n{stdout}")
                print(f"STDERR:\n{stderr}")
                return False

            # Check if server is responding
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"  ✓ Server ready in {elapsed:.1f}s")
                    return True
            except:
                pass

            time.sleep(5)

        print(f"  ✗ Server failed to start within {timeout}s")
        return False

    def stop(self):
        """Stop vLLM server and ALL child processes."""
        if self.process_group_id is None:
            print("No process group to stop")
            return

        print(f"Stopping vLLM process group {self.process_group_id}...")

        try:
            # Kill entire process group (parent + all children)
            # Negative PID means process group
            os.killpg(self.process_group_id, signal.SIGTERM)

            # Wait for graceful shutdown
            time.sleep(3)

            # Check if any processes remain
            try:
                os.killpg(self.process_group_id, 0)  # Check if group exists
                # Still exists, force kill
                print("  Process group still alive, force killing...")
                os.killpg(self.process_group_id, signal.SIGKILL)
                time.sleep(2)
            except ProcessLookupError:
                # Process group gone, good
                pass

            print("  ✓ Process group terminated")

            # Verify GPU memory cleared
            self._verify_gpu_cleared()

        except ProcessLookupError:
            print("  ✓ Process group already terminated")
        except Exception as e:
            print(f"  ✗ Error during cleanup: {e}")
            # Force kill by PID as fallback
            self._force_kill_fallback()

    def _force_kill_fallback(self):
        """Fallback: brute force kill all vLLM processes."""
        print("  Attempting fallback cleanup...")
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], check=False)
        time.sleep(2)

    def _verify_gpu_cleared(self):
        """Verify GPU memory was actually cleared."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            used_mb = float(result.stdout.strip())

            if used_mb < 1000:
                print(f"  ✓ GPU memory clear: {used_mb:.0f}MB")
            else:
                print(f"  ⚠ GPU memory still in use: {used_mb:.0f}MB")
                print(f"  This may indicate lingering processes")
        except Exception as e:
            print(f"  ? Could not verify GPU status: {e}")

    def is_healthy(self) -> bool:
        """Check if server is still healthy."""
        if self.process.poll() is not None:
            return False

        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        if not self.start():
            raise RuntimeError("Failed to start vLLM server")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.stop()


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Run vLLM with proper cleanup')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--gpu-memory-util', type=float, default=0.9)
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible sampling (only affects temperature > 0)')

    args = parser.parse_args()

    # Use as context manager to ensure cleanup
    with VLLMRunner(
        model=args.model,
        gpu_memory_util=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        port=args.port,
        seed=args.seed
    ) as runner:
        print("\nvLLM server running. Press Ctrl+C to stop.\n")

        # Keep running until interrupted
        try:
            while runner.is_healthy():
                time.sleep(10)

            print("\n✗ Server became unhealthy, stopping...")
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")


if __name__ == "__main__":
    main()
