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

PY_CMD="python3 /home/server01/js_ws/lidar_test/train_encoding.py \
--resume-from /home/server01/js_ws/lidar_test/ckpt/latest_encoding_logger.pth"

while true; do
    echo "Running command..."
    $PY_CMD
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Command ended successfully (exit code 0). Exiting."
        break
    else
        echo "Command ended with code $EXIT_CODE. Retrying..."
    fi
done
