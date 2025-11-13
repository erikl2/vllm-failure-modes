#!/bin/bash
# Monitor C=4 experiments progress on Lambda

# Load environment variables
source "$(dirname "$0")/.env"

echo "=== C=4 Experiments Progress Check ==="
echo "Time: $(date)"
echo ""

# Check if tmux session exists
echo "--- Tmux Session Status ---"
ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux ls | grep c4_ablation || echo 'No c4_ablation session found'"
echo ""

# Show last 20 lines of experiment log
echo "--- Recent Experiment Output ---"
ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tail -20 ~/c4_experiments.log 2>/dev/null || echo 'No c4_experiments.log yet'"
echo ""

# Show last 10 lines of each seed log if they exist
for seed in 42 43 44; do
    echo "--- Seed $seed (last 10 lines) ---"
    ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tail -10 ~/c4_seed${seed}.log 2>/dev/null || echo 'Seed $seed not started yet'"
    echo ""
done

# Check if results file exists and show line count
echo "--- Results File Status ---"
ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls -lh ~/results/results_vllm_concurrency4.csv 2>/dev/null && echo \"Lines: \$(wc -l < ~/results/results_vllm_concurrency4.csv)\" || echo 'No results file yet'"
echo ""

echo "=== End of Progress Check ==="
