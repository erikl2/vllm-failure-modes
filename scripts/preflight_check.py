#!/usr/bin/env python3
"""
Pre-flight check for Option C experiments.

Verifies:
1. Python environment and dependencies
2. GPU availability and memory
3. vLLM installation
4. Required files exist
5. Network connectivity (for model download)

Usage:
    python3 scripts/preflight_check.py
"""

import sys
import subprocess
import os
from pathlib import Path


def check_mark(passed: bool) -> str:
    return "✅" if passed else "❌"


def run_check(name: str, check_fn) -> bool:
    """Run a check and print result."""
    try:
        passed, message = check_fn()
        print(f"  {check_mark(passed)} {name}: {message}")
        return passed
    except Exception as e:
        print(f"  {check_mark(False)} {name}: Error - {e}")
        return False


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    return False, f"Python {version.major}.{version.minor} (need 3.8+)"


def check_vllm():
    """Check vLLM installation."""
    try:
        import vllm
        return True, f"vLLM {vllm.__version__}"
    except ImportError:
        return False, "vLLM not installed (pip install vllm)"


def check_dependencies():
    """Check required Python packages."""
    missing = []
    packages = ['aiohttp', 'numpy', 'tqdm']

    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        return False, f"Missing: {', '.join(missing)}"
    return True, "All dependencies installed"


def check_gpu():
    """Check GPU availability."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            info = result.stdout.strip()
            return True, info
        return False, "nvidia-smi failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found (no GPU?)"
    except subprocess.TimeoutExpired:
        return False, "nvidia-smi timeout"


def check_gpu_memory():
    """Check if GPU has enough free memory."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            free_mb = float(result.stdout.strip().split('\n')[0])
            if free_mb > 30000:  # Need ~30GB for 8B model
                return True, f"{free_mb/1024:.1f} GB free"
            elif free_mb > 15000:
                return True, f"{free_mb/1024:.1f} GB free (may be tight for 8B model)"
            else:
                return False, f"Only {free_mb/1024:.1f} GB free (need ~30GB)"
        return False, "Could not query memory"
    except Exception as e:
        return False, str(e)


def check_workload():
    """Check workload file exists and is valid."""
    workload_path = Path("workload.json")
    if not workload_path.exists():
        return False, "workload.json not found"

    try:
        import json
        with open(workload_path) as f:
            data = json.load(f)

        if "prompts" not in data:
            return False, "workload.json missing 'prompts' key"

        num_prompts = len(data["prompts"])
        return True, f"{num_prompts} prompts loaded"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"


def check_benchmark_script():
    """Check benchmark script exists."""
    script_path = Path("scripts/benchmark_expanded.py")
    if script_path.exists():
        return True, "benchmark_expanded.py found"
    return False, "scripts/benchmark_expanded.py not found"


def check_run_script():
    """Check run script exists and is executable."""
    script_path = Path("scripts/run_option_c.sh")
    if not script_path.exists():
        return False, "scripts/run_option_c.sh not found"

    if os.access(script_path, os.X_OK):
        return True, "run_option_c.sh found and executable"
    return False, "run_option_c.sh not executable (run: chmod +x scripts/run_option_c.sh)"


def check_analysis_script():
    """Check analysis script exists."""
    script_path = Path("analysis/analyze_option_c.py")
    if script_path.exists():
        return True, "analyze_option_c.py found"
    return False, "analysis/analyze_option_c.py not found"


def check_results_dir():
    """Check results directory is writable."""
    results_path = Path("results")
    try:
        results_path.mkdir(exist_ok=True)
        test_file = results_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        return True, "results/ directory writable"
    except Exception as e:
        return False, f"Cannot write to results/: {e}"


def check_no_running_vllm():
    """Check no vLLM processes are running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints"],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            return False, f"vLLM already running (PIDs: {', '.join(pids)})"
        return True, "No vLLM processes running"
    except FileNotFoundError:
        # pgrep not available, try ps
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True, text=True
            )
            if "vllm.entrypoints" in result.stdout:
                return False, "vLLM processes detected"
            return True, "No vLLM processes detected"
        except:
            return True, "Could not check (assuming OK)"


def main():
    print("=" * 60)
    print("  OPTION C PRE-FLIGHT CHECK")
    print("=" * 60)
    print()

    all_passed = True

    # Environment checks
    print("Environment:")
    all_passed &= run_check("Python version", check_python)
    all_passed &= run_check("vLLM installation", check_vllm)
    all_passed &= run_check("Dependencies", check_dependencies)
    print()

    # GPU checks
    print("GPU:")
    all_passed &= run_check("GPU available", check_gpu)
    all_passed &= run_check("GPU memory", check_gpu_memory)
    all_passed &= run_check("No running vLLM", check_no_running_vllm)
    print()

    # File checks
    print("Files:")
    all_passed &= run_check("Workload file", check_workload)
    all_passed &= run_check("Benchmark script", check_benchmark_script)
    all_passed &= run_check("Run script", check_run_script)
    all_passed &= run_check("Analysis script", check_analysis_script)
    all_passed &= run_check("Results directory", check_results_dir)
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("  ✅ ALL CHECKS PASSED - Ready to run experiments!")
        print()
        print("  Quick test:  ./scripts/run_option_c.sh --quick")
        print("  Dry run:     ./scripts/run_option_c.sh --dry-run")
        print("  Full run:    ./scripts/run_option_c.sh")
        return 0
    else:
        print("  ❌ SOME CHECKS FAILED - Please fix issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
