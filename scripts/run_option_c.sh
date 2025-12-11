#!/bin/bash
#
# run_option_c.sh - Full Experimental Matrix for Paper Expansion
# 
# Runs 90 experiments:
# - 6 GPU memory settings: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
# - 5 concurrency levels: 1, 2, 4, 8, 16
# - 3 seeds per config: 42, 43, 44
#
# Estimated time: ~12-15 GPU hours
# Estimated cost: ~$15-20 on Lambda Labs A100 @ $1.29/hr
#
# Usage:
#   ./run_option_c.sh                    # Run full matrix
#   ./run_option_c.sh --resume           # Resume from checkpoint
#   ./run_option_c.sh --dry-run          # Show what would run
#   ./run_option_c.sh --quick            # Quick test (2 configs only)

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_DIR}/results/option_c_${TIMESTAMP}"
WORKLOAD="${PROJECT_DIR}/workload.json"
BENCHMARK_SCRIPT="${SCRIPT_DIR}/benchmark_expanded.py"

# Experiment duration per run (seconds)
DURATION=300  # 5 minutes

# Experimental matrix
GPU_MEMORY_SETTINGS=(0.70 0.75 0.80 0.85 0.90 0.95)
CONCURRENCY_LEVELS=(1 2 4 8 16)
SEEDS=(42 43 44)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ============================================================================
# Argument Parsing
# ============================================================================

DRY_RUN=false
RESUME=false
QUICK_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --quick)
            QUICK_TEST=true
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Quick test mode: only 2 configurations
if [ "$QUICK_TEST" = true ]; then
    echo -e "${YELLOW}QUICK TEST MODE: Running only 2 configurations${NC}"
    GPU_MEMORY_SETTINGS=(0.85 0.90)
    CONCURRENCY_LEVELS=(4)
    SEEDS=(42)
fi

# ============================================================================
# Setup
# ============================================================================

mkdir -p "${RESULTS_DIR}"

# Combined results file
COMBINED_CSV="${RESULTS_DIR}/all_results.csv"
LOG_FILE="${RESULTS_DIR}/experiment.log"
CHECKPOINT_FILE="${RESULTS_DIR}/.checkpoint"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

# Calculate totals
TOTAL=$((${#GPU_MEMORY_SETTINGS[@]} * ${#CONCURRENCY_LEVELS[@]} * ${#SEEDS[@]}))

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           OPTION C: FULL EXPERIMENTAL MATRIX                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  GPU Memory Settings: ${GPU_MEMORY_SETTINGS[*]}"
echo "  Concurrency Levels:  ${CONCURRENCY_LEVELS[*]}"
echo "  Seeds:               ${SEEDS[*]}"
echo "  Duration per run:    ${DURATION}s"
echo "  Total experiments:   ${TOTAL}"
echo "  Results directory:   ${RESULTS_DIR}"
echo ""
echo "Estimated time: $((TOTAL * (DURATION + 120) / 3600)) hours"
echo "Estimated cost: \$$(echo "scale=2; $TOTAL * ($DURATION + 120) / 3600 * 1.29" | bc)"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN - No experiments will be executed${NC}"
    echo ""
    echo "Would run the following experiments:"
    for gpu_mem in "${GPU_MEMORY_SETTINGS[@]}"; do
        for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "  gpu=${gpu_mem} c=${concurrency} seed=${seed}"
            done
        done
    done
    exit 0
fi

# ============================================================================
# Pre-flight Checks
# ============================================================================

echo "Running pre-flight checks..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

# Check benchmark script
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    echo -e "${RED}ERROR: Benchmark script not found: $BENCHMARK_SCRIPT${NC}"
    exit 1
fi

# Check workload
if [ ! -f "$WORKLOAD" ]; then
    echo -e "${RED}ERROR: Workload file not found: $WORKLOAD${NC}"
    exit 1
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found - no GPU available${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown")
echo -e "  ${GREEN}✓${NC} GPU: $GPU_INFO"

# Check vLLM
if ! python3 -c "import vllm" 2>/dev/null; then
    echo -e "${RED}ERROR: vLLM not installed${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} vLLM installed"

# Check dependencies
python3 -c "import aiohttp, numpy" 2>/dev/null || {
    echo -e "${RED}ERROR: Missing dependencies. Run: pip install aiohttp numpy${NC}"
    exit 1
}
echo -e "  ${GREEN}✓${NC} Dependencies OK"

echo ""

# ============================================================================
# Helper Functions
# ============================================================================

cleanup_gpu() {
    echo "  Cleaning up GPU processes..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    
    # Verify cleanup
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ ! -z "$mem_used" ]] && (( $(echo "$mem_used > 2000" | bc -l) )); then
        echo -e "  ${YELLOW}Warning: GPU memory still in use (${mem_used}MB). Waiting...${NC}"
        sleep 10
        pkill -9 -f "python" 2>/dev/null || true
        sleep 5
    fi
}

is_completed() {
    local exp_name=$1
    if [ -f "$CHECKPOINT_FILE" ] && grep -q "^${exp_name}$" "$CHECKPOINT_FILE"; then
        return 0
    fi
    return 1
}

mark_completed() {
    local exp_name=$1
    echo "$exp_name" >> "$CHECKPOINT_FILE"
}

# ============================================================================
# Main Experiment Loop
# ============================================================================

START_TIME=$(date +%s)
COMPLETED=0
SUCCESSES=0
FAILURES=0
SKIPPED=0

echo -e "${BLUE}Starting experiments at $(date)${NC}"
echo ""

for gpu_mem in "${GPU_MEMORY_SETTINGS[@]}"; do
    for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            
            EXP_NAME="gpu${gpu_mem}_c${concurrency}_s${seed}"
            OUTPUT_FILE="${RESULTS_DIR}/${EXP_NAME}.csv"
            
            # Check if already completed (for resume)
            if [ "$RESUME" = true ] && is_completed "$EXP_NAME"; then
                echo -e "${YELLOW}⏭  Skipping ${EXP_NAME} (already completed)${NC}"
                ((SKIPPED++))
                ((COMPLETED++))
                continue
            fi
            
            echo ""
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${BLUE}  Experiment $((COMPLETED + 1))/${TOTAL}: ${EXP_NAME}${NC}"
            echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo "  GPU Memory:  ${gpu_mem}"
            echo "  Concurrency: ${concurrency}"
            echo "  Seed:        ${seed}"
            echo "  Output:      ${OUTPUT_FILE}"
            echo ""
            
            # Cleanup before experiment
            cleanup_gpu
            
            # Run experiment
            EXP_START=$(date +%s)
            
            if python3 "$BENCHMARK_SCRIPT" \
                --gpu-memory "$gpu_mem" \
                --concurrency "$concurrency" \
                --seed "$seed" \
                --workload "$WORKLOAD" \
                --output "$OUTPUT_FILE" \
                --duration "$DURATION" \
                --profile-memory; then
                
                EXP_END=$(date +%s)
                EXP_DURATION=$((EXP_END - EXP_START))
                
                # Check result
                if grep -q "SUCCESS" "$OUTPUT_FILE" 2>/dev/null; then
                    echo -e "${GREEN}  ✓ PASSED (${EXP_DURATION}s)${NC}"
                    ((SUCCESSES++))
                else
                    echo -e "${RED}  ✗ FAILED (${EXP_DURATION}s)${NC}"
                    ((FAILURES++))
                fi
                
                mark_completed "$EXP_NAME"
                
            else
                echo -e "${RED}  ✗ CRASHED${NC}"
                ((FAILURES++))
                
                # Record crash in results
                echo "CRASHED,${gpu_mem},${concurrency},${seed},PROCESS_CRASH" >> "${RESULTS_DIR}/crashes.txt"
            fi
            
            ((COMPLETED++))
            
            # Progress update
            NOW=$(date +%s)
            ELAPSED=$((NOW - START_TIME))
            AVG_PER_EXP=$((ELAPSED / COMPLETED))
            REMAINING=$(((TOTAL - COMPLETED) * AVG_PER_EXP / 60))
            
            echo ""
            echo "  Progress: ${COMPLETED}/${TOTAL} | ✓ ${SUCCESSES} | ✗ ${FAILURES} | ⏭ ${SKIPPED}"
            echo "  Time elapsed: $((ELAPSED / 60))m | Remaining: ~${REMAINING}m"
            
            # Post-experiment cleanup
            cleanup_gpu
            
        done
    done
done

# ============================================================================
# Generate Combined Results
# ============================================================================

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Combining Results${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Combine all individual results
FIRST_FILE=true
for f in "${RESULTS_DIR}"/gpu*.csv; do
    if [ -f "$f" ]; then
        if [ "$FIRST_FILE" = true ]; then
            cat "$f" > "$COMBINED_CSV"
            FIRST_FILE=false
        else
            tail -n +2 "$f" >> "$COMBINED_CSV"
        fi
    fi
done

echo "  Combined results: $COMBINED_CSV"

# ============================================================================
# Summary
# ============================================================================

END_TIME=$(date +%s)
TOTAL_TIME=$(((END_TIME - START_TIME) / 60))

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    EXPERIMENT SUMMARY                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Total experiments:  ${COMPLETED}"
echo "  Successful:         ${SUCCESSES}"
echo "  Failed:             ${FAILURES}"
echo "  Skipped (resumed):  ${SKIPPED}"
echo "  Total time:         ${TOTAL_TIME} minutes"
echo ""
echo "  Results directory:  ${RESULTS_DIR}"
echo "  Combined results:   ${COMBINED_CSV}"
echo ""

# Quick reliability summary by config
echo "RELIABILITY BY CONFIGURATION:"
echo "─────────────────────────────"
printf "%-12s" "GPU Mem"
for c in "${CONCURRENCY_LEVELS[@]}"; do
    printf "  C=%-4s" "$c"
done
echo ""

for gpu_mem in "${GPU_MEMORY_SETTINGS[@]}"; do
    printf "%-12s" "$gpu_mem"
    for c in "${CONCURRENCY_LEVELS[@]}"; do
        pattern="gpu${gpu_mem}_c${c}_s"
        total=$(ls "${RESULTS_DIR}"/${pattern}*.csv 2>/dev/null | wc -l)
        success=$(grep -l "SUCCESS" "${RESULTS_DIR}"/${pattern}*.csv 2>/dev/null | wc -l || echo 0)
        if [ "$total" -gt 0 ]; then
            rate=$((success * 100 / total))
            printf "  %3d%%  " "$rate"
        else
            printf "   N/A  "
        fi
    done
    echo ""
done

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    EXPERIMENTS COMPLETE                          ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Run analysis:"
echo "     python3 analysis/analyze_option_c.py --results-dir ${RESULTS_DIR}"
echo ""
echo "  2. Generate plots:"
echo "     python3 analysis/analyze_option_c.py --results-dir ${RESULTS_DIR} --plots"
echo ""
echo "  3. Update paper with new results"
echo ""
