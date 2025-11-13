#!/bin/bash
# Validate that Phase One's 0.95 setting works with proper cleanup

set -e

echo "=========================================="
echo "Phase One Configuration Validation"
echo "=========================================="
echo "Testing: GPU mem 0.95, C=4"
echo "This was Phase One's config that showed failures"
echo "With proper cleanup, it should succeed"
echo "=========================================="

GPU_MEM=0.95
CONCURRENCY=4
SEEDS=(42 43 44)

RESULTS_DIR="results/phase1_validation"
mkdir -p "$RESULTS_DIR"

LOGFILE="$RESULTS_DIR/phase1_validation.log"
exec 1> >(tee -a "$LOGFILE")
exec 2>&1

echo "[$(date)] Starting Phase One config validation"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Seed $SEED (GPU mem $GPU_MEM, C=$CONCURRENCY)"
    echo "=========================================="

    # Health check
    python -c "from health_check import HealthCheck; HealthCheck().full_cleanup()"
    sleep 10

    # Start vLLM with process group management
    echo "Starting vLLM..."
    python -c "
from vllm_runner import VLLMRunner
import time

runner = VLLMRunner(
    model='meta-llama/Llama-3.1-8B-Instruct',
    gpu_memory_util=$GPU_MEM,
    max_model_len=4096,
    port=8000
)

if not runner.start():
    import sys
    sys.exit(1)

print('✓ vLLM started')

# Keep running
try:
    while runner.is_healthy():
        time.sleep(10)
except KeyboardInterrupt:
    pass
finally:
    runner.stop()
" > "$RESULTS_DIR/vllm_seed${SEED}.log" 2>&1 &

    RUNNER_PID=$!
    echo "Runner PID: $RUNNER_PID"

    # Wait for ready
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ vLLM ready"
            break
        fi
        sleep 5
    done

    # Run benchmark
    echo "Running benchmark..."
    if python benchmark_vllm.py \
        --seed $SEED \
        --concurrency $CONCURRENCY \
        --output "$RESULTS_DIR/results_seed${SEED}.csv"; then

        echo "✓ SUCCESS"
        echo "SUCCESS" > "$RESULTS_DIR/result_seed${SEED}.txt"
    else
        echo "✗ FAILED"
        echo "FAILED" > "$RESULTS_DIR/result_seed${SEED}.txt"
    fi

    # Cleanup
    kill $RUNNER_PID 2>/dev/null || true
    sleep 5

    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    echo "GPU memory after cleanup: ${GPU_MEM_USED}MB"

    sleep 30
done

echo ""
echo "=========================================="
echo "Phase One Validation Results"
echo "=========================================="
cat "$RESULTS_DIR"/result_seed*.txt

SUCCESSES=$(grep -c "SUCCESS" "$RESULTS_DIR"/result_seed*.txt || echo 0)
echo ""
if [ $SUCCESSES -eq 3 ]; then
    echo "✓ ALL 3 SEEDS SUCCEEDED"
    echo ""
    echo "CONCLUSION: Phase One failures at GPU mem 0.95 were due to"
    echo "GPU memory leaks, NOT the 0.95 setting itself."
    echo "With proper cleanup, 0.95 works fine for C=4."
else
    echo "⚠ Only $SUCCESSES/3 seeds succeeded"
    echo "This suggests 0.95 may have inherent instability"
fi
