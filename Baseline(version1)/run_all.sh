#!/bin/bash

# 1. 清掉舊檔
rm -f ncu_report.csv power_log.csv

# 2. 啟動功耗監測
nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > power_log.csv &
GPU_MONITOR_PID=$!

# 3. 執行影片處理（包含拆影格、CUDA 計算、合影片），不帶 ncu
./test_vedio input.mp4

# 4. CUDA + ffmpeg 做完之後，先 kill 掉功耗監測
kill $GPU_MONITOR_PID

# 5. 確認沒有運行中的 nvidia-smi（可略過，如果上面 kill 成功，通常已經停止）
#    pgrep -f "nvidia-smi --query-gpu" && killall nvidia-smi

# 6. 等功耗監測停了，再啟動 ncu 去做 Profiling（把結果存在 ncu_report.csv）
ncu --csv --metrics \
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
    l2__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
    sm__avg_active_warps_per_active_cycle,\
    sm__inst_issued.avg.per_cycle_active \
    ./test_vedio input.mp4 \
    > ncu_report.csv

echo "All done.
- output.mp4 (CUDA 處理過的影片)
- power_log.csv (執行時 GPU 功耗)
- ncu_report.csv (Nsight Compute 收集的 Memory/Occupancy/IPC)"
