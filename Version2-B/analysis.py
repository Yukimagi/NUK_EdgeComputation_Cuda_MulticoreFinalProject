import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 1. 讀取 timing.csv
timing = pd.read_csv('timing.csv')

# 把所有 NaN 換成 0
timing['elapsed_page_h2d_ms'] = timing['elapsed_page_h2d_ms'].fillna(0)
timing['elapsed_page_d2h_ms'] = timing['elapsed_page_d2h_ms'].fillna(0)


#timing['bw_page_h2d_MBps'] = timing['bw_page_h2d_MBps'].fillna(0)
#timing['bw_page_d2h_MBps'] = timing['bw_page_d2h_MBps'].fillna(0)

timing['total_time_ms'] = (
    timing['elapsed_page_h2d_ms'] +
    timing['elapsed_pin_h2d_ms'] +
    timing['elapsed_kernels_ms'] +
    timing['elapsed_page_d2h_ms'] +
    timing['elapsed_pin_d2h_ms']
)

# 計算「總執行時間」和平均值
total_exec_time_ms = timing['total_time_ms'].sum()
avg_total_time_ms = timing['total_time_ms'].mean()

# 四個 PCIe 帶寬欄位的平均值
bandwidth_columns = [
    'bw_page_h2d_MBps', 'bw_pin_h2d_MBps',
    'bw_page_d2h_MBps', 'bw_pin_d2h_MBps'
]
avg_bandwidths = timing[bandwidth_columns].mean()

# 2. 讀取 power_log.csv，並去掉欄位名稱的空白
power = pd.read_csv('power_log.csv')
power.columns = power.columns.str.strip()  # 這行把所有欄位名稱前後的空白去掉

# 確認現在欄位長這樣：
# print(power.columns)   # -> e.g. Index(['timestamp', 'power.draw [W]'], dtype='object')

# 解析 timestamp，將 power.draw 轉成 float（單位：W）
power['timestamp'] = pd.to_datetime(
    power['timestamp'],
    format='%Y/%m/%d %H:%M:%S.%f'
)
power['power_W'] = power['power.draw [W]'].str.replace(' W', '').astype(float)

# 計算相鄰時間差（秒）
power['time_diff'] = power['timestamp'].diff().dt.total_seconds().fillna(0)

# 用梯形法近似算出能量 (Wh)
power['time_diff_h'] = power['time_diff'] / 3600.0
power['energy_Wh'] = (
    (power['power_W'] + power['power_W'].shift(1, fill_value=power['power_W'].iloc[0])) / 2
) * power['time_diff_h']

total_energy_Wh = power['energy_Wh'].sum()
avg_power_W     = power['power_W'].mean()

# 計算功耗效率 (frames/J)
total_frames = len(timing)
total_energy_J = total_energy_Wh * 3600  # 1 Wh = 3600 J
frames_per_J   = total_frames / total_energy_J if total_energy_J > 0 else None

# 3. 嘗試讀取 NCU 報表
try:
    ncu = pd.read_csv('ncu_report.csv', engine='python')
except:
    ncu = pd.DataFrame()

ncu_metrics = {}
if not ncu.empty:
    columns_mapping = {
        'memory_throughput': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
        'occupancy':         'sm__warps_active.avg.pct_of_peak_sustained_elapsed',
        'instruction_util':  'sm__inst_issued.avg.per_cycle_active'
    }
    for metric, col in columns_mapping.items():
        if col in ncu.columns:
            ncu_metrics[metric] = ncu[col].astype(float)

# 4. 繪製各圖表

## 4.1 每張影格的「總執行時間 (ms)」
plt.figure(figsize=(10, 4))
plt.plot(timing['frame_id'], timing['total_time_ms'], marker='o')
plt.title('Execution Time per Frame (ms)')
plt.xlabel('Frame ID')
plt.ylabel('Total Pipeline Time (ms)')
plt.grid(True)
plt.tight_layout()
plt.show()

## 4.2 PCIe 帶寬 (MB/s)
plt.figure(figsize=(10, 4))
for col in bandwidth_columns:
    plt.plot(timing['frame_id'], timing[col], label=col)
plt.title('PCIe Bandwidth per Frame (MB/s)')
plt.xlabel('Frame ID')
plt.ylabel('Bandwidth (MB/s)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

## 4.3 功耗隨時間 (W)
plt.figure(figsize=(10, 4))
plt.plot(power['timestamp'], power['power_W'], color='red')
plt.title('Power Consumption Over Time (W)')
plt.xlabel('Timestamp')
plt.ylabel('Power (W)')
plt.grid(True)
plt.tight_layout()
plt.show()

## 4.4 NCU 報表指標（如果有正確欄位）
if ncu_metrics:
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    axs[0].plot(ncu_metrics['memory_throughput'], color='green')
    axs[0].set_title('Memory Throughput')
    axs[0].set_xlabel('Sample Index')
    axs[0].set_ylabel('Throughput Metric')
    axs[0].grid(True)

    axs[1].plot(ncu_metrics['occupancy'], color='blue')
    axs[1].set_title('Occupancy (%)')
    axs[1].set_xlabel('Sample Index')
    axs[1].set_ylabel('Occupancy (%)')
    axs[1].grid(True)

    axs[2].plot(ncu_metrics['instruction_util'], color='purple')
    axs[2].set_title('Instruction Utilization')
    axs[2].set_xlabel('Sample Index')
    axs[2].set_ylabel('Utilization Metric')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("NCU report 資料有誤或欄位不存在，無法繪製 Memory/Occupancy/Instruction 圖。")

# 5. 列印總結指標
summary = {
    'Total Execution Time (ms)':              total_exec_time_ms,
    'Avg Execution Time per Frame (ms)':      avg_total_time_ms,
    'Avg PCIe Bandwidths (MB/s)':             avg_bandwidths.to_dict(),
    'Average Power (W)':                      avg_power_W,
    'Total Energy (Wh)':                      total_energy_Wh,
    'Power Efficiency (frames per Joule)':    frames_per_J
}

print("===== Summary Metrics =====")
for key, value in summary.items():
    print(f"{key}: {value}")
