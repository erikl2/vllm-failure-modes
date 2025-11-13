#!/bin/bash
################################################################################
# Auto-terminate Lambda instance after experiment completes
# Run this AFTER starting the experiment
################################################################################

source .env

echo "Auto-terminate monitor starting..."
echo "Will check every 15 minutes if experiment is complete"
echo "Press Ctrl+C to cancel"
echo ""

MAX_WAIT_HOURS=10
MAX_CHECKS=$((MAX_WAIT_HOURS * 4))  # 4 checks per hour
CHECK_COUNT=0

while [ $CHECK_COUNT -lt $MAX_CHECKS ]; do
    echo "[$(date +%H:%M:%S)] Checking status (check $((CHECK_COUNT + 1))/$MAX_CHECKS)..."
    
    # Check if tmux session still exists
    TMUX_STATUS=$(ssh -i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=no $LAMBDA_HOST 'tmux ls 2>&1 | grep ablation' || echo "")
    
    if [ -z "$TMUX_STATUS" ]; then
        echo ""
        echo "✓ Tmux session ended - experiment appears complete!"
        echo ""
        
        # Verify by checking for "COMPLETE" in log
        COMPLETE_CHECK=$(ssh -i $LAMBDA_SSH_KEY -o StrictHostKeyChecking=no $LAMBDA_HOST 'grep "COMPLETE" ablation.log' || echo "")
        
        if [ -n "$COMPLETE_CHECK" ]; then
            echo "✓ Confirmed: Found COMPLETE marker in log"
            echo ""
            echo "Downloading results..."
            ./download_results.sh
            
            echo ""
            echo "========================================="
            echo "EXPERIMENT COMPLETE!"
            echo "========================================="
            echo ""
            echo "Total runtime: ~$((CHECK_COUNT * 15 / 60)) hours"
            echo "Estimated cost: ~\$$(echo "scale=2; $CHECK_COUNT * 15 / 60 * 1.29" | bc)"
            echo ""
            echo "Results saved to: ./results/"
            echo ""
            echo "⚠️  REMEMBER TO TERMINATE LAMBDA INSTANCE!"
            echo "   https://cloud.lambdalabs.com/instances"
            echo ""
            
            exit 0
        fi
    fi
    
    # Wait 15 minutes
    sleep 900
    CHECK_COUNT=$((CHECK_COUNT + 1))
done

echo ""
echo "⚠️  Reached maximum wait time ($MAX_WAIT_HOURS hours)"
echo "Please check manually: ./check_progress.sh"
