#!/bin/bash
# Adaptive auto-cleanup script for C=4 experiments
# This runs in the background, monitors experiments, downloads results, and terminates Lambda when done

LOGFILE="auto_cleanup.log"
CHECK_INTERVAL=600  # Check every 10 minutes

# Load environment variables
source "$(dirname "$0")/.env"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== Auto-cleanup script started ==="
log "Will check every $CHECK_INTERVAL seconds ($(($CHECK_INTERVAL / 60)) minutes)"
log "Lambda Host: $LAMBDA_HOST"

while true; do
    log "Checking experiment status..."

    # Check if tmux session exists
    SESSION_STATUS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux ls 2>&1" | grep "c4_ablation" || echo "")

    if [ -z "$SESSION_STATUS" ]; then
        log "✓ Tmux session ended - experiments appear to be complete!"

        # Give it a moment to finalize any writes
        sleep 30

        # Create results directory locally if it doesn't exist
        mkdir -p results

        log "Downloading results and logs..."

        # Download results file
        if scp -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST:~/results/results_vllm_concurrency4.csv" ./results/ 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Downloaded results_vllm_concurrency4.csv"
        else
            log "✗ Failed to download results CSV"
        fi

        # Download log files
        for seed in 42 43 44; do
            if scp -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST:~/c4_seed${seed}.log" ./results/ 2>&1 | tee -a "$LOGFILE"; then
                log "✓ Downloaded c4_seed${seed}.log"
            else
                log "⚠ Could not download c4_seed${seed}.log (may not exist)"
            fi
        done

        # Download experiment log
        if scp -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST:~/c4_experiments.log" ./results/ 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Downloaded c4_experiments.log"
        else
            log "⚠ Could not download c4_experiments.log"
        fi

        # Check what we got
        log "--- Downloaded Files ---"
        ls -lh results/ | tee -a "$LOGFILE"

        # Show results summary if CSV exists
        if [ -f "results/results_vllm_concurrency4.csv" ]; then
            log "--- Results Preview ---"
            head -5 results/results_vllm_concurrency4.csv | tee -a "$LOGFILE"
            log "Total lines in results: $(wc -l < results/results_vllm_concurrency4.csv)"
        fi

        log "Terminating Lambda instance via API..."

        # Extract IP from LAMBDA_HOST (format: ubuntu@ip.address)
        INSTANCE_IP="${LAMBDA_HOST##*@}"
        log "Instance IP: $INSTANCE_IP"

        # Terminate using Lambda API
        if python3 lambda_api.py "$INSTANCE_IP" 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Instance termination confirmed via API"
            TERMINATION_SUCCESS=true
        else
            log "✗ WARNING: API termination failed or could not be verified"
            log "⚠  MANUAL ACTION REQUIRED: Check Lambda web console"
            log "⚠  Instance may still be running and incurring charges!"
            TERMINATION_SUCCESS=false

            # Send macOS notification
            osascript -e 'display notification "Lambda instance termination failed! Check web console immediately." with title "URGENT: Manual Action Required"' 2>/dev/null || true
        fi

        if [ "$TERMINATION_SUCCESS" = false ]; then
            log "=== AUTO-CLEANUP INCOMPLETE ==="
            log "Results downloaded successfully, but instance termination uncertain"
            log "Please verify instance status at: https://cloud.lambdalabs.com/instances"
            exit 1
        fi

        log "=== AUTO-CLEANUP COMPLETE ==="
        log "All experiments finished, results downloaded, Lambda terminated."
        log "Check the 'results/' directory for your data."

        # Exit successfully
        exit 0
    else
        # Still running - extract progress info
        PROGRESS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux capture-pane -t c4_ablation -p 2>/dev/null | grep -E 'Seed [0-9]+ \(c=' | tail -1" || echo "")

        if [ -n "$PROGRESS" ]; then
            log "Still running: $PROGRESS"
        else
            log "Experiments still running (no progress visible)"
        fi

        log "Next check in $(($CHECK_INTERVAL / 60)) minutes..."
        sleep $CHECK_INTERVAL
    fi
done
