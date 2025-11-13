#!/bin/bash
# Auto-cleanup for Option B experiments
# Monitors experiments, downloads results, terminates instance

# Get environment variables
source .env

LOGFILE="auto_cleanup_optionb.log"
CHECK_INTERVAL=600  # Check every 10 minutes
COMPLETION_MARKER="results/OPTIONB_COMPLETE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== Option B Auto-Cleanup Started ==="
log "Lambda instance: $LAMBDA_HOST"
log "Check interval: ${CHECK_INTERVAL}s"

while true; do
    log "Checking experiment status..."

    # Check if completion marker exists on remote
    if ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "test -f ~/results/OPTIONB_COMPLETE" 2>/dev/null; then
        log "✓ Experiments complete! Starting cleanup..."

        # Download results
        log "Downloading results..."
        mkdir -p results_optionb
        scp -i "$LAMBDA_SSH_KEY" -r "$LAMBDA_HOST:~/results/*" results_optionb/ 2>&1 | tee -a "$LOGFILE"

        if [ $? -eq 0 ]; then
            log "✓ Results downloaded to results_optionb/"
            ls -lhR results_optionb/ | tee -a "$LOGFILE"
        else
            log "⚠ Failed to download results"
        fi

        # Terminate Lambda instance
        log "Terminating instance via Lambda API..."

        # Extract IP from LAMBDA_HOST
        INSTANCE_IP=$(echo "$LAMBDA_HOST" | sed 's/.*@//')

        # Find instance ID
        INSTANCE_ID=$(python3 lambda_api.py list | grep "$INSTANCE_IP" | awk '{print $1}')

        if [ -z "$INSTANCE_ID" ]; then
            log "⚠ Could not find instance ID for $INSTANCE_IP"
            log "Trying to find instance by IP..."
            INSTANCE_JSON=$(python3 -c "
import json
import requests
import os

api_key = os.environ.get('LAMBDA_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}
response = requests.get('https://cloud.lambdalabs.com/api/v1/instances', headers=headers)
instances = response.json().get('data', [])

for inst in instances:
    if inst.get('ip') == '$INSTANCE_IP':
        print(inst.get('id', ''))
        break
")
            INSTANCE_ID="$INSTANCE_JSON"
        fi

        if [ -n "$INSTANCE_ID" ]; then
            log "Found instance $INSTANCE_ID at $INSTANCE_IP"
            log "Terminating instance $INSTANCE_ID via Lambda Labs API..."

            python3 lambda_api.py "$INSTANCE_ID" 2>&1 | tee -a "$LOGFILE"

            if [ $? -eq 0 ]; then
                log "✓ Termination request sent"

                # Verify termination
                log "Verifying termination of instance $INSTANCE_ID..."
                for i in {1..12}; do
                    sleep 5
                    STATUS=$(python3 -c "
import json
import requests
import os

api_key = os.environ.get('LAMBDA_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}
response = requests.get('https://cloud.lambdalabs.com/api/v1/instances', headers=headers)
instances = response.json().get('data', [])

for inst in instances:
    if inst.get('id') == '$INSTANCE_ID':
        print(inst.get('status', 'unknown'))
        break
else:
    print('gone')
" 2>/dev/null)

                    if [ "$STATUS" = "gone" ] || [ -z "$STATUS" ]; then
                        log "✓ Instance termination CONFIRMED via API"
                        break
                    else
                        log "  Instance is $STATUS... (waiting)"
                    fi
                done

                if [ "$STATUS" != "gone" ] && [ -n "$STATUS" ]; then
                    log "✗ Warning: Could not verify termination after 60s"
                fi
            else
                log "✗ Failed to terminate instance"
                log "⚠ MANUAL TERMINATION REQUIRED!"

                # Try to send macOS notification
                osascript -e 'display notification "Lambda instance termination FAILED - manual cleanup required!" with title "Option B Auto-Cleanup"' 2>/dev/null || true
            fi
        else
            log "✗ Could not find instance ID"
            log "⚠ MANUAL TERMINATION REQUIRED!"
        fi

        log "=== AUTO-CLEANUP COMPLETE ==="
        exit 0
    else
        log "Experiments still running..."

        # Show progress if possible
        ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "ls -1 ~/results/*/result_seed*.txt 2>/dev/null | wc -l" 2>/dev/null | while read count; do
            log "  Results files so far: $count"
        done
    fi

    log "Next check in $((CHECK_INTERVAL / 60)) minutes..."
    sleep $CHECK_INTERVAL
done
