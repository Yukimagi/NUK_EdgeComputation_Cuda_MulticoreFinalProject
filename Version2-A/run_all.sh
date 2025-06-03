#!/bin/bash

# 1. 清掉舊檔
rm -f ncu_report.csv power_log.csv

# 2. 編譯
nvcc test_video.cu -o a.out

# 3. 啟動功耗監測
nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 200 > power_log.csv &
GPU_MONITOR_PID=$!
sleep 1

# 4. 確認 power monitor 是否正常啟動
if ! ps -p $GPU_MONITOR_PID > /dev/null; then
    echo "nvidia-smi failed to start power logging"
    exit 1
fi

echo "Power monitor PID = $GPU_MONITOR_PID"

# 5. 執行影片處理（包含拆影格、CUDA 計算、合影片），不帶 ncu
./a.out

# 6. 結束功耗監控
if ps -p $GPU_MONITOR_PID > /dev/null; then
    kill $GPU_MONITOR_PID
    echo "Power monitor stopped"
else
    echo "Power monitor already exited"
fi
