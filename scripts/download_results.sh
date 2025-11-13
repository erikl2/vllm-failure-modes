#!/bin/bash
################################################################################
# Download Results from Lambda Labs
# Run this in the morning to get your results
################################################################################

source .env

echo "════════════════════════════════════════════════════════════"
echo "  DOWNLOADING RESULTS FROM LAMBDA LABS"
echo "════════════════════════════════════════════════════════════"
echo ""

SCP_OPTS="-i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=no"

echo "▶ Downloading CSV results..."
scp $SCP_OPTS "$LAMBDA_HOST:~/results/results_vllm_concurrency*.csv" results/ 2>/dev/null || echo "  No CSV files found"

echo "▶ Downloading plots..."
scp $SCP_OPTS "$LAMBDA_HOST:~/results/*.png" results/ 2>/dev/null || echo "  No PNG files found"

echo "▶ Downloading summary..."
scp $SCP_OPTS "$LAMBDA_HOST:~/results/breaking_point_summary.txt" results/ 2>/dev/null || echo "  No summary found"

echo ""
echo "✓ Download complete!"
echo ""

echo "Results downloaded to: ./results/"
ls -lh results/results_vllm_concurrency*.csv 2>/dev/null || echo "  No concurrency CSV files"
echo ""

echo "View summary:"
echo "  cat results/breaking_point_summary.txt"
echo ""

echo "View plots:"
echo "  open results/breaking_point_curve.png"
echo "  open results/latency_comparison.png"
echo "  open results/throughput_comparison.png"
echo ""

echo "NEXT: Terminate your Lambda instance to avoid costs!"
echo "  https://cloud.lambdalabs.com/instances"
echo ""
