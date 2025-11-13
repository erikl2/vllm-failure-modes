#!/bin/bash
################################################################################
# Run C=2 and C=3 experiments to complete the breaking point analysis
# 
# This script will:
# 1. Run vLLM at concurrency=2 (3 seeds)
# 2. Run vLLM at concurrency=3 (3 seeds)  
# 3. Update the breaking point analysis
#
# Estimated time: 6-8 hours
# Estimated cost: ~$5 GPU time
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "BREAKING POINT ABLATION: C=2, C=3 EXPERIMENTS"
echo "================================================================================"
echo ""
echo "This will run:"
echo "  - vLLM at concurrency=2 (seeds: 42, 43, 44)"
echo "  - vLLM at concurrency=3 (seeds: 42, 43, 44)"
echo ""
echo "Estimated time: 6-8 hours"
echo "Estimated cost: ~\$5"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "================================================================================"
echo "CONCURRENCY = 2"
echo "================================================================================"
echo ""

for SEED in 42 43 44; do
    echo "Running vLLM C=2, seed=$SEED..."
    python benchmark_vllm.py \
        --concurrency 2 \
        --seed $SEED \
        --workload workload.json \
        --output results/results_vllm_concurrency2.csv \
        --append
    
    echo "  ✓ Seed $SEED complete"
    echo ""
done

echo "✓ All C=2 experiments complete"
echo ""

echo "================================================================================"
echo "CONCURRENCY = 3"
echo "================================================================================"
echo ""

for SEED in 42 43 44; do
    echo "Running vLLM C=3, seed=$SEED..."
    python benchmark_vllm.py \
        --concurrency 3 \
        --seed $SEED \
        --workload workload.json \
        --output results/results_vllm_concurrency3.csv \
        --append
    
    echo "  ✓ Seed $SEED complete"
    echo ""
done

echo "✓ All C=3 experiments complete"
echo ""

echo "================================================================================"
echo "ANALYZING RESULTS"
echo "================================================================================"
echo ""

echo "Calculating statistics..."
python analyze_breaking_point.py

echo ""
echo "Regenerating plots with updated data..."
python generate_plots.py

echo ""
echo "================================================================================"
echo "ABLATION COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/results_vllm_concurrency2.csv"
echo "  - results/results_vllm_concurrency3.csv"
echo "  - results/breaking_point_summary.txt"
echo ""
echo "Updated plots:"
echo "  - results/breaking_point_curve.png (now with real C=2, C=3 data)"
echo "  - results/latency_comparison.png"
echo ""
echo "Next step: Review the breaking point curve and incorporate into paper"
echo ""
