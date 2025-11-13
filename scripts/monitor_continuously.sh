#!/bin/bash
# Continuous monitoring script for C=4 experiments

source "$(dirname "$0")/.env"

echo "=== Starting Continuous Monitoring ==="
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=== C=4 Experiments Monitor ==="
    echo "Time: $(date)"
    echo "================================================"
    echo ""

    # Check tmux session
    echo "--- Tmux Session ---"
    ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux ls 2>&1 | grep c4_ablation || echo 'Session ended'"
    echo ""

    # Get live progress from tmux
    echo "--- Current Progress ---"
    ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux capture-pane -t c4_ablation -p 2>/dev/null | tail -5 || echo 'Cannot capture tmux output'"
    echo ""

    # Check results file
    echo "--- Results Status ---"
    ssh -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "if [ -f results/results_vllm_concurrency4.csv ]; then echo 'Results file exists'; wc -l results/results_vllm_concurrency4.csv; tail -3 results/results_vllm_concurrency4.csv; else echo 'No results file yet'; fi"
    echo ""

    echo "================================================"
    echo "Next update in 2 minutes... (Ctrl+C to stop)"

    sleep 120
done
