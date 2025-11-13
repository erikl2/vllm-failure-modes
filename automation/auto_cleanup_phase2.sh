#!/bin/bash
# Enhanced auto-cleanup for Phase 2 with API termination

LOGFILE="auto_cleanup_phase2.log"
CHECK_INTERVAL=600  # 10 minutes

source "$(dirname "$0")/.env"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== Phase 2 Auto-Cleanup Started ==="
log "Lambda Host: $LAMBDA_HOST"
log "Check interval: $((CHECK_INTERVAL / 60)) minutes"

# Extract IP for API calls
INSTANCE_IP="${LAMBDA_HOST##*@}"
log "Instance IP: $INSTANCE_IP"

while true; do
    log "Checking experiment status..."

    # Check for active tmux sessions
    SESSIONS=$(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" \
        "tmux ls 2>&1" | grep -E "headroom|version|admission" || echo "")

    if [ -z "$SESSIONS" ]; then
        log "✓ No active experiment sessions - experiments complete!"

        sleep 30

        # Create local results directory
        mkdir -p results_phase2

        log "Downloading results..."

        # Download entire results directory
        if scp -r -o StrictHostKeyChecking=no -i "$LAMBDA_SSH_KEY" \
            "$LAMBDA_HOST:~/results/" \
            results_phase2/ 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Results downloaded"
        else
            log "⚠ Some files may not have downloaded"
        fi

        log "--- Downloaded Files ---"
        ls -lhR results_phase2/ | tee -a "$LOGFILE"

        # Terminate instance via API
        log "Terminating instance via Lambda API..."

        if python3 lambda_api.py "$INSTANCE_IP" 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Instance termination CONFIRMED via API"
            log "=== AUTO-CLEANUP COMPLETE ==="
            exit 0
        else
            log "✗ API termination failed!"
            log "⚠  MANUAL ACTION REQUIRED"
            log "⚠  Check: https://cloud.lambdalabs.com/instances"

            # Urgent notification
            osascript -e 'display notification "Lambda instance termination FAILED! Check console NOW!" with title "URGENT ACTION REQUIRED"' 2>/dev/null || true

            log "=== AUTO-CLEANUP FAILED ==="
            exit 1
        fi
    else
        log "Experiments still running:"
        echo "$SESSIONS" | tee -a "$LOGFILE"
        log "Next check in $((CHECK_INTERVAL / 60)) minutes..."
        sleep $CHECK_INTERVAL
    fi
done
