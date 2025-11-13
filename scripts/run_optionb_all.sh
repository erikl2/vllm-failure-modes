#!/bin/bash
# Run all Option B validation experiments in sequence

set -e

echo "============================================"
echo "Option B Validation Experiments - Master Runner"
echo "============================================"
echo ""
echo "This will run:"
echo "  1. Threshold Validation (GPU 0.82-0.87 at C=4)"
echo "  2. Phase One Validation (GPU 0.95 at C=4)"
echo ""
echo "Estimated time: 6-9 hours"
echo "Estimated cost: $8-11"
echo "============================================"
echo ""

# Record start time
START_TIME=$(date +%s)
START_DATE=$(date)

echo "Started: $START_DATE"
echo ""

# Create top-level results directory
mkdir -p results

# Experiment 1: Threshold Validation
echo ""
echo "============================================"
echo "EXPERIMENT 1: Threshold Validation"
echo "============================================"
echo "Testing GPU memory: 0.82, 0.83, 0.84, 0.85, 0.87"
echo "Concurrency: C=4"
echo "Seeds: 3 per setting (42, 43, 44)"
echo "Expected duration: 4-6 hours"
echo ""

if ./run_threshold_validation.sh; then
    echo "✓ Threshold validation completed successfully"
else
    echo "⚠ Threshold validation encountered errors"
fi

# Delay between experiments
echo ""
echo "Waiting 5 minutes before next experiment..."
sleep 300

# Experiment 2: Phase One Validation
echo ""
echo "============================================"
echo "EXPERIMENT 2: Phase One Validation"
echo "============================================"
echo "Testing GPU memory: 0.95 (original Phase One setting)"
echo "Concurrency: C=4"
echo "Seeds: 3 (42, 43, 44)"
echo "Expected duration: 2-3 hours"
echo ""

if ./run_phase1_validation.sh; then
    echo "✓ Phase One validation completed successfully"
else
    echo "⚠ Phase One validation encountered errors"
fi

# Final summary
END_TIME=$(date +%s)
END_DATE=$(date)
TOTAL_DURATION=$((END_TIME - START_TIME))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))

echo ""
echo "============================================"
echo "ALL OPTION B EXPERIMENTS COMPLETE"
echo "============================================"
echo "Started:  $START_DATE"
echo "Finished: $END_DATE"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results directories:"
echo "  - results/threshold_validation/"
echo "  - results/phase1_validation/"
echo ""
echo "Run analyze_threshold.py to see detailed analysis"
echo "============================================"

# Signal completion
touch results/OPTIONB_COMPLETE
echo "$(date)" > results/OPTIONB_COMPLETE
