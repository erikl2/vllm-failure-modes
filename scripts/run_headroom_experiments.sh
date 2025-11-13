#!/bin/bash
# Headroom axis experiments
# Tests if lower GPU memory utilization prevents C=4 wedge

set -e

echo "=========================================="
echo "Headroom Axis Experiments"
echo "=========================================="
echo "Testing GPU memory utilization: 0.80, 0.85, 0.90"
echo "Concurrency levels: 1, 2, 3, 4"
echo "Seeds: 42, 43, 44"
echo "=========================================="

# Experiment parameters
GPU_MEM_UTILS=(0.80 0.85 0.90)
CONCURRENCY_LEVELS=(1 2 3 4)
SEEDS=(42 43 44)

# Results directory
RESULTS_BASE="results/headroom_experiments"
mkdir -p "$RESULTS_BASE"

# Main log
MAIN_LOG="$RESULTS_BASE/headroom_experiments.log"
exec 1> >(tee -a "$MAIN_LOG")
exec 2>&1

echo "[$(date)] Starting headroom experiments"

# Track total time and cost
START_TIME=$(date +%s)

# For each memory utilization setting
for GPU_MEM in "${GPU_MEM_UTILS[@]}"; do
    echo ""
    echo "=========================================="
    echo "GPU Memory Utilization: $GPU_MEM"
    echo "=========================================="

    # For each concurrency level
    for CONCURRENCY in "${CONCURRENCY_LEVELS[@]}"; do
        echo ""
        echo "--- Concurrency: $CONCURRENCY ---"

        EXP_NAME="headroom_mem${GPU_MEM}_c${CONCURRENCY}"
        EXP_DIR="$RESULTS_BASE/$EXP_NAME"
        mkdir -p "$EXP_DIR"

        # Run all seeds for this configuration
        for SEED in "${SEEDS[@]}"; do
            echo ""
            echo "Running: GPU=$GPU_MEM, C=$CONCURRENCY, Seed=$SEED"

            # Health check before starting
            if ! python health_check.py; then
                echo "⚠ System unhealthy before starting, cleaning up..."
                python -c "from health_check import HealthCheck; HealthCheck().full_cleanup()"
                sleep 10
            fi

            # Start vLLM with specific memory utilization
            echo "Starting vLLM (GPU mem util: $GPU_MEM)..."
            python -m vllm.entrypoints.openai.api_server \
                --model meta-llama/Llama-3.1-8B-Instruct \
                --host 0.0.0.0 \
                --port 8000 \
                --gpu-memory-utilization $GPU_MEM \
                --max-model-len 4096 > "$EXP_DIR/vllm_seed${SEED}.log" 2>&1 &

            VLLM_PID=$!
            echo "vLLM PID: $VLLM_PID"

            # Wait for vLLM ready
            echo "Waiting for vLLM to be ready..."
            for i in {1..60}; do
                if curl -s http://localhost:8000/health > /dev/null 2>&1; then
                    echo "✓ vLLM ready"
                    break
                fi

                if [ $i -eq 60 ]; then
                    echo "✗ vLLM failed to start"
                    kill -9 $VLLM_PID 2>/dev/null || true
                    echo "FAILED" > "$EXP_DIR/result_seed${SEED}.txt"
                    continue 2  # Skip to next seed
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

            # Always cleanup after each seed
            echo "Performing cleanup..."
            python -c "from health_check import HealthCheck; HealthCheck().full_cleanup()"

            # Verify cleanup
            GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            echo "GPU memory after cleanup: ${GPU_MEM_USED}MB"

            # Wait before next seed
            echo "Waiting 30s before next seed..."
            sleep 30
        done

        # Summary for this configuration
        echo ""
        echo "Configuration complete: $EXP_NAME"
        echo "Results:"
        cat "$EXP_DIR"/result_seed*.txt 2>/dev/null || echo "No results"
    done
done

# Final summary
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "=========================================="
echo "Headroom Experiments Complete"
echo "=========================================="
echo "Total time: ${HOURS}h ${MINUTES}m"
echo "Results in: $RESULTS_BASE"
echo "=========================================="

# Generate summary
python -c "
import os
import glob

results_dir = '$RESULTS_BASE'
configs = {}

for exp_dir in glob.glob(os.path.join(results_dir, 'headroom_*')):
    exp_name = os.path.basename(exp_dir)

    # Count successes
    result_files = glob.glob(os.path.join(exp_dir, 'result_seed*.txt'))
    successes = 0
    for rf in result_files:
        with open(rf) as f:
            if f.read().strip().startswith('SUCCESS'):
                successes += 1

    configs[exp_name] = {'total': len(result_files), 'success': successes}

print('\\nConfiguration Summary:')
print('-' * 60)
for exp_name in sorted(configs.keys()):
    data = configs[exp_name]
    print(f'{exp_name}: {data[\"success\"]}/{data[\"total\"]} seeds succeeded')
"
