#!/bin/bash
# Auto-retry Lambda instance launch and deploy Option B experiments
# Checks every 2 minutes until capacity is available

LOGFILE="auto_launch_and_deploy.log"
CHECK_INTERVAL=120  # 2 minutes

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== Auto-Launch and Deploy Started ==="
log "Will check Lambda capacity every $((CHECK_INTERVAL / 60)) minutes"
log "Press Ctrl+C to stop"
log ""

ATTEMPT=1

while true; do
    log "Attempt #$ATTEMPT - Checking Lambda GPU availability..."

    # Try to launch instance
    source .env
    LAUNCH_OUTPUT=$(python3 launch_lambda.py 2>&1)
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        log "✓ Instance launched successfully!"
        echo "$LAUNCH_OUTPUT" | tee -a "$LOGFILE"

        # Extract IP from output
        INSTANCE_IP=$(echo "$LAUNCH_OUTPUT" | grep "export LAMBDA_HOST" | sed 's/.*ubuntu@\([0-9.]*\).*/\1/')

        if [ -z "$INSTANCE_IP" ]; then
            log "✗ Could not extract instance IP from output"
            exit 1
        fi

        log "Instance IP: $INSTANCE_IP"

        # Update .env file
        log "Updating .env with new instance..."

        # Backup current .env
        cp .env .env.backup.$(date +%s)

        # Update LAMBDA_HOST in .env
        sed -i.bak "s/^export LAMBDA_HOST=.*/export LAMBDA_HOST=\"ubuntu@$INSTANCE_IP\"/" .env

        log "✓ .env updated: LAMBDA_HOST=ubuntu@$INSTANCE_IP"

        # Reload environment
        source .env

        # Wait for SSH to be ready
        log "Waiting for SSH to be ready..."
        for i in {1..30}; do
            if ssh -i "$LAMBDA_SSH_KEY" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$LAMBDA_HOST" "echo 'SSH ready'" 2>/dev/null; then
                log "✓ SSH connection established"
                break
            fi

            if [ $i -eq 30 ]; then
                log "✗ SSH not ready after 5 minutes"
                exit 1
            fi

            sleep 10
        done

        # Deploy Option B
        log ""
        log "=== Starting Option B Deployment ==="
        log ""

        # Upload deployment package
        log "Uploading deployment package..."
        if scp -i "$LAMBDA_SSH_KEY" -o StrictHostKeyChecking=no optionb_files.tar.gz "$LAMBDA_HOST:~/" 2>&1 | tee -a "$LOGFILE"; then
            log "✓ Deployment package uploaded"
        else
            log "✗ Failed to upload deployment package"
            exit 1
        fi

        # Extract and setup
        log "Setting up files on Lambda..."
        ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "
            cd ~
            tar -xzf optionb_files.tar.gz
            chmod +x *.sh *.py
            mkdir -p results
            echo '✓ Files extracted and permissions set'
        " 2>&1 | tee -a "$LOGFILE"

        # Install dependencies
        log "Installing dependencies on Lambda..."
        ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "
            pip install --user --upgrade pip
            pip install --user vllm==0.6.3 'numpy<2' requests aiohttp matplotlib
            echo '✓ Dependencies installed'
        " 2>&1 | tee -a "$LOGFILE"

        # Start experiments in tmux
        log "Starting Option B experiments in tmux..."
        ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "
            tmux new-session -d -s optionb 'cd ~ && ./run_optionb_all.sh'
            echo '✓ Experiments started in tmux session: optionb'
        " 2>&1 | tee -a "$LOGFILE"

        # Verify tmux session
        sleep 5
        TMUX_CHECK=$(ssh -i "$LAMBDA_SSH_KEY" "$LAMBDA_HOST" "tmux ls 2>/dev/null | grep optionb")
        if [ -n "$TMUX_CHECK" ]; then
            log "✓ Tmux session confirmed running"
        else
            log "⚠ Could not verify tmux session"
        fi

        # Start local auto-cleanup monitor
        log ""
        log "Starting local auto-cleanup monitor..."

        nohup ./auto_cleanup_optionb.sh > auto_cleanup_optionb_console.log 2>&1 &
        CLEANUP_PID=$!
        echo $CLEANUP_PID > auto_cleanup_optionb.pid

        log "✓ Auto-cleanup started (PID: $CLEANUP_PID)"
        log "  Log: auto_cleanup_optionb.log"
        log "  Console: auto_cleanup_optionb_console.log"

        # Success summary
        log ""
        log "=== DEPLOYMENT COMPLETE ==="
        log ""
        log "Instance Details:"
        log "  IP: $INSTANCE_IP"
        log "  SSH: ssh -i \"$LAMBDA_SSH_KEY\" $LAMBDA_HOST"
        log ""
        log "Experiments Running:"
        log "  1. Threshold Validation (GPU 0.82-0.87, C=4)"
        log "  2. Phase One Validation (GPU 0.95, C=4)"
        log "  Expected duration: 6-9 hours"
        log "  Expected cost: \$8-11"
        log ""
        log "Monitoring:"
        log "  Auto-cleanup PID: $CLEANUP_PID"
        log "  Check progress: tail -f auto_cleanup_optionb.log"
        log "  View experiments: ssh -i \"\$LAMBDA_SSH_KEY\" \$LAMBDA_HOST \"tmux attach -t optionb\""
        log ""
        log "Auto-cleanup will:"
        log "  - Monitor experiments every 10 minutes"
        log "  - Download results when complete"
        log "  - Terminate instance via Lambda API"
        log "  - Verify termination"
        log ""
        log "You can safely close this terminal - everything runs in background!"
        log ""

        # Create status file
        cat > OPTION_B_DEPLOYMENT_STATUS.md <<EOF
# Option B Deployment - LIVE STATUS

**Deployment Time**: $(date)
**Status**: ✓ ALL SYSTEMS OPERATIONAL

---

## Instance Details

- **IP**: $INSTANCE_IP
- **Type**: A100 40GB
- **SSH**: \`ssh -i "$LAMBDA_SSH_KEY" $LAMBDA_HOST\`

---

## Experiments Running

1. **Threshold Validation**
   - GPU memory: 0.82, 0.83, 0.84, 0.85, 0.87
   - Concurrency: C=4
   - Seeds: 3 per setting
   - Total: 15 experiments

2. **Phase One Validation**
   - GPU memory: 0.95
   - Concurrency: C=4
   - Seeds: 3
   - Total: 3 experiments

**Expected Duration**: 6-9 hours
**Expected Cost**: \$8-11

---

## Monitoring

### Auto-Cleanup (Local)

\`\`\`bash
# Check status
ps -p $CLEANUP_PID

# View log
tail -f auto_cleanup_optionb.log

# Stop (emergency only)
kill $CLEANUP_PID
\`\`\`

### Experiments (On Lambda)

\`\`\`bash
# View progress
source .env
ssh -i "\$LAMBDA_SSH_KEY" "\$LAMBDA_HOST" "tmux capture-pane -t optionb -p | tail -50"

# Attach to tmux (Ctrl+B then D to detach)
ssh -i "\$LAMBDA_SSH_KEY" "\$LAMBDA_HOST" "tmux attach -t optionb"

# Check results count
ssh -i "\$LAMBDA_SSH_KEY" "\$LAMBDA_HOST" "ls -1 ~/results/*/result_seed*.txt 2>/dev/null | wc -l"
\`\`\`

---

## What Happens Automatically

1. **Experiments run unattended** in tmux on Lambda
2. **Auto-cleanup monitors** every 10 minutes
3. **When complete**:
   - Results downloaded to \`results_optionb/\`
   - Lambda instance terminated via API
   - Termination verified
   - If termination fails → macOS notification

---

## Expected Timeline

- **Now**: Experiments starting
- **+3-4 hours**: Threshold validation ~50% complete
- **+6 hours**: Threshold validation complete, Phase 1 starting
- **+8 hours**: Phase 1 complete, downloading results
- **+8.5 hours**: Instance terminated, all done!

---

**Status**: Fully automated - you can close your laptop! 🚀
EOF

        log "Status document created: OPTION_B_DEPLOYMENT_STATUS.md"
        log ""
        log "=== AUTO-LAUNCH AND DEPLOY COMPLETE ==="

        exit 0

    else
        # No capacity available
        if echo "$LAUNCH_OUTPUT" | grep -q "Not enough capacity"; then
            log "No capacity available in any region"
        else
            log "Launch attempt failed:"
            echo "$LAUNCH_OUTPUT" | tail -5 | tee -a "$LOGFILE"
        fi

        log "Waiting $((CHECK_INTERVAL / 60)) minutes before retry #$((ATTEMPT + 1))..."
        ATTEMPT=$((ATTEMPT + 1))
        sleep $CHECK_INTERVAL
    fi
done
