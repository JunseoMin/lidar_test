#!/bin/bash

__conda_setup="$('/home/server01/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/server01/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/server01/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/server01/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate lidarus

export PATH=/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
export OMP_NUM_THREADS=12

python3 /home/server01/js_ws/lidar_test/train_encoding.py

PY_CMD="python3 /home/server01/js_ws/lidar_test/train_encoding.py \
--resume-from /home/server01/js_ws/lidar_test/ckpt/latest_encoding_logger.pth"

function run_and_monitor {
    while true; do
        echo "[Monitor Script] Starting training command in background..."
        $PY_CMD &
        TRAIN_PID=$!

        echo "[Monitor Script] Training PID: $TRAIN_PID"
        
        # We'll monitor GPU usage for GPU index 0, checking every 1 second
        # If usage == 0% for 60 consecutive checks, we kill the process and restart
        IDLE_COUNT=0
        CHECK_INTERVAL=1    # seconds
        IDLE_THRESHOLD=6000   # how many consecutive seconds of 0% usage before restart

        # Wait until the process finishes OR we detect GPU idle for too long
        while kill -0 $TRAIN_PID 2>/dev/null; do
            # Query GPU utilization for GPU index 0
            GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)

            if [ "$GPU_USAGE" == "" ]; then
                echo "[Monitor Script] WARNING: Could not read GPU usage. Exiting monitor loop."
                break
            fi

            # Convert usage to integer if needed
            GPU_USAGE_INT=$(echo "$GPU_USAGE" | awk '{print int($1)}')

            if [ $GPU_USAGE_INT -eq 0 ]; then
                IDLE_COUNT=$((IDLE_COUNT+1))
            else
                IDLE_COUNT=0
            fi

            # Check if we've hit the threshold
            if [ $IDLE_COUNT -ge $IDLE_THRESHOLD ]; then
                echo "[Monitor Script] GPU usage is 0% for $IDLE_THRESHOLD seconds. Killing process $TRAIN_PID..."
                kill $TRAIN_PID
                # Wait briefly for kill to take effect
                sleep 2
                # Optionally 'kill -9' if it's stubborn, but usually not needed
                break
            fi

            sleep $CHECK_INTERVAL
        done

        # When we exit the while loop: either
        # 1) The process ended by itself (exit code 0 or non-zero), or
        # 2) We killed it due to 0% usage for 60s

        # Let's see if the process ended on its own
        wait $TRAIN_PID
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "[Monitor Script] Command ended successfully (exit code 0). Exiting the loop."
            break
        else
            echo "[Monitor Script] Command ended or was killed with code $EXIT_CODE. Will restart..."
        fi
    done
}

# --- [5] Execute the run_and_monitor function ---
run_and_monitor
echo "[Monitor Script] Done."
