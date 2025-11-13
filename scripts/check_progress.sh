#!/bin/bash
# Quick progress checker for breaking point ablation experiments

source .env

echo "═══════════════════════════════════════════════════════════"
echo "BREAKING POINT ABLATION - PROGRESS CHECK"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "Time: $(date)"
echo ""

echo "Tmux sessions:"
ssh -i $LAMBDA_SSH_KEY $LAMBDA_HOST "tmux ls" 2>&1
echo ""

echo "CSV files:"
ssh -i $LAMBDA_SSH_KEY $LAMBDA_HOST "ls -lh results/results_vllm_concurrency*.csv 2>/dev/null" || echo "No results yet"
echo ""

echo "Line counts (excluding header):"
ssh -i $LAMBDA_SSH_KEY $LAMBDA_HOST "wc -l results/results_vllm_concurrency*.csv 2>/dev/null" || echo "No results yet"
echo ""

echo "Expected: 3 lines in concurrency2.csv, 3 lines in concurrency3.csv"
echo "Status: $(ssh -i $LAMBDA_SSH_KEY $LAMBDA_HOST "ps aux | grep 'python.*benchmark_vllm' | grep -v grep" >/dev/null 2>&1 && echo '✓ RUNNING' || echo '✗ NOT RUNNING')"
echo ""

echo "To view live output:"
echo "  ssh -i \$LAMBDA_SSH_KEY \$LAMBDA_HOST"
echo "  tmux attach -t ablation"
echo ""
