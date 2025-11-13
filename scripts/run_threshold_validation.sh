#!/bin/bash
# Fine-grained GPU memory threshold validation for C=4

set -e

echo "=========================================="
echo "GPU Memory Threshold Validation"
echo "=========================================="
echo "Testing GPU memory: 0.82, 0.83, 0.84, 0.85, 0.87"
echo "Concurrency: C=4 only"
echo "Seeds: 42, 43, 44"
echo "=========================================="

# GPU memory settings to test
GPU_MEMS=(0.82 0.83 0.84 0.85 0.87)
CONCURRENCY=4
SEEDS=(42 43 44)

# Results directory
RESULTS_BASE="results/threshold_validation"
mkdir -p "$RESULTS_BASE"

# Main log
MAIN_LOG="$RESULTS_BASE/threshold_validation.log"
exec 1> >(tee -a "$MAIN_LOG")
exec 2>&1

echo "[$(date)] Starting threshold validation"

# Track start time
START_TIME=$(date +%s)

# For each GPU memory setting
for GPU_MEM in "${GPU_MEMS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Testing GPU Memory: $GPU_MEM"
    echo "=========================================="

    EXP_DIR="$RESULTS_BASE/gpu_mem_${GPU_MEM}"
    mkdir -p "$EXP_DIR"

    # For each seed
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- Seed $SEED (GPU mem $GPU_MEM, C=$CONCURRENCY) ---"

        # Pre-experiment health check
        echo "Pre-experiment health check..."
        if ! python health_check.py; then
            echo "⚠ System unhealthy, cleaning up..."
            python -c "from health_check import HealthCheck; HealthCheck().full_cleanup()"
            sleep 10
        fi

        # Start vLLM using new runner (with process group management)
        echo "Starting vLLM with process group management..."

        python -c "
from vllm_runner import VLLMRunner
import sys
import time

runner = VLLMRunner(
    model='meta-llama/Llama-3.1-8B-Instruct',
    gpu_memory_util=$GPU_MEM,
    max_model_len=4096,
    port=8000
)

if not runner.start():
    print('✗ Failed to start vLLM')
    sys.exit(1)

# Write PID for later cleanup
with open('$EXP_DIR/vllm_runner_seed${SEED}.pid', 'w') as f:
    f.write(str(runner.process_group_id))

print('✓ vLLM started with process group', runner.process_group_id)

# Keep running (benchmark will connect to this server)
try:
    while runner.is_healthy():
        time.sleep(10)
except KeyboardInterrupt:
    pass
finally:
    runner.stop()
" > "$EXP_DIR/vllm_runner_seed${SEED}.log" 2>&1 &

        RUNNER_PID=$!
        echo "Runner PID: $RUNNER_PID"
        echo $RUNNER_PID > "$EXP_DIR/runner_pid_seed${SEED}.txt"

        # Wait for vLLM to be ready
        echo "Waiting for vLLM to be ready..."
        for i in {1..60}; do
            if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                echo "✓ vLLM ready"
                break
            fi

            if [ $i -eq 60 ]; then
                echo "✗ vLLM failed to start within 5 minutes"
                kill $RUNNER_PID 2>/dev/null || true
                echo "FAILED_TO_START" > "$EXP_DIR/result_seed${SEED}.txt"
                continue 2
            fi

            sleep 5
        done

        # Run benchmark
        echo "Running benchmark..."
        START_BENCHMARK=$(date +%s)

        if python benchmark_vllm.py \
            --seed $SEED \
            --concurrency $CONCURRENCY \
            --output "$EXP_DIR/results_seed${SEED}.csv"; then

            END_BENCHMARK=$(date +%s)
            DURATION=$((END_BENCHMARK - START_BENCHMARK))
            echo "✓ Benchmark completed in ${DURATION}s"
            echo "SUCCESS $DURATION" > "$EXP_DIR/result_seed${SEED}.txt"
        else
            echo "✗ Benchmark failed"
            echo "FAILED" > "$EXP_DIR/result_seed${SEED}.txt"
        fi

        # Stop vLLM runner (will cleanup process group)
        echo "Stopping vLLM runner..."
        kill $RUNNER_PID 2>/dev/null || true
        sleep 5

        # Verify cleanup
        echo "Verifying cleanup..."
        GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        echo "GPU memory after cleanup: ${GPU_MEM_USED}MB"

        if [ "$GPU_MEM_USED" -gt 1000 ]; then
            echo "⚠ GPU memory not fully cleared, running emergency cleanup..."
            python -c "from health_check import HealthCheck; HealthCheck().full_cleanup()"
            sleep 5
        fi

        # Inter-seed delay
        echo "Waiting 30s before next seed..."
        sleep 30
    done

    # Summary for this GPU memory setting
    echo ""
    echo "Configuration complete: GPU mem $GPU_MEM"
    echo "Results:"
    cat "$EXP_DIR"/result_seed*.txt 2>/dev/null || echo "No results"
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "Threshold Validation Complete"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results in: $RESULTS_BASE"
echo "=========================================="

# Generate summary
python analyze_threshold.py "$RESULTS_BASE"
