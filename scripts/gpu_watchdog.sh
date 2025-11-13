#!/bin/bash
# GPU Memory Watchdog - Automatically kills stuck processes holding GPU memory
# Run this in background on Lambda to prevent GPU memory leaks

LOGFILE="gpu_watchdog.log"
CHECK_INTERVAL=120  # Check every 2 minutes

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOGFILE"
}

log "=== GPU Watchdog Started ==="
log "Check interval: ${CHECK_INTERVAL}s ($(($CHECK_INTERVAL / 60)) minutes)"

while true; do
    # Get GPU memory usage
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # Check if GPU memory is high (>1GB) but no vLLM server responding
    if [ "$GPU_MEM" -gt 1000 ]; then
        # Check if vLLM server is actually responsive
        if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
            log "⚠ GPU leak detected: ${GPU_MEM}MB used but no responsive vLLM server"

            # Find PIDs using GPU
            GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

            if [ -n "$GPU_PIDS" ]; then
                log "Killing GPU processes: $GPU_PIDS"
                for PID in $GPU_PIDS; do
                    kill -9 $PID 2>/dev/null && log "  Killed PID $PID"
                done

                # Wait for GPU to clear
                sleep 5

                NEW_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

                if [ "$NEW_GPU_MEM" -lt 1000 ]; then
                    log "✓ GPU memory cleared: ${NEW_GPU_MEM}MB"
                else
                    log "⚠ GPU memory still high: ${NEW_GPU_MEM}MB"

                    # Try killing all Python processes as last resort
                    log "Force killing all user Python processes..."
                    pkill -9 -u $(whoami) python 2>/dev/null
                    sleep 5

                    FINAL_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
                    log "Final GPU memory: ${FINAL_GPU_MEM}MB"
                fi
            else
                log "No GPU processes found by nvidia-smi, trying pkill..."
                pkill -9 -u $(whoami) python 2>/dev/null
                sleep 5
            fi
        else
            log "GPU usage ${GPU_MEM}MB but vLLM server is responsive - OK"
        fi
    else
        log "GPU memory OK: ${GPU_MEM}MB used"
    fi

    sleep $CHECK_INTERVAL
done
