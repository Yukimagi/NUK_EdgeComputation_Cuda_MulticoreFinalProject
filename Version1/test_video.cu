// test.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>   // 讀取目錄
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

using namespace std::chrono;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"          // header-only image loader
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"    // header-only image writer


// TODO -------------------------------------------------

# define DIR_NAME "frames"      // 更改為存圖片的資料夾

// ------------------------------------------------------


void getImageList(const std::string& folder, std::vector<std::string>& files) {
    DIR* dir = opendir(folder.c_str());
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        std::string name = entry->d_name;
        if (name.length() > 4 && (name.substr(name.length()-4) == ".jpg" || name.substr(name.length()-4) == ".png")) {
            files.push_back(folder + "/" + name);
        }
    }
    closedir(dir);
}

// cuda 執行紀錄
double computePowerStats(const std::string& filepath, double& avg_power_W, double& total_energy_Wh) {
    std::ifstream file(filepath);
    std::string line;
    int count = 0;
    double power_sum = 0;

    // 跳過 header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string timestamp, power_str;
        std::getline(ss, timestamp, ',');
        std::getline(ss, power_str, ',');

        if (power_str.empty()) continue; // 空欄跳過
        size_t pos = power_str.find(" ");
        if (pos == std::string::npos) continue; // 沒找到空格，跳過
        std::string power_num = power_str.substr(0, pos);
        try {
            double power = std::stod(power_num);
            power_sum += power;
            count++;
        } catch (std::exception& e) {
            continue; // 無法轉成數字的跳過
        }
    }

    if (count == 0) return 0;

    avg_power_W = power_sum / count;

    // 每筆紀錄 200 ms → 總時間（小時）
    double total_time_h = count * 0.2 / 3600.0;
    total_energy_Wh = avg_power_W * total_time_h;

    return total_time_h;
}


// 高斯核（已歸一化）
__constant__ float gaussKernel1D[5] = { 1/16.f, 4/16.f, 6/16.f, 4/16.f, 1/16.f };


// 水平高斯模糊
__global__ void gaussianBlur1D_Horizontal(const unsigned char* in, unsigned char* out, int w, int h) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // 每 row + padding 4
    __shared__ unsigned char tile[16][16 + 4]; // 每行加左右 padding 2
    int lx = tx + 2;

    // 複製中心
    if (x < w && y < h)
        tile[ty][lx] = in[y * w + x];
    else
        tile[ty][lx] = 0;

    // padding 左右邊界
    if (tx < 2) {
        int left  = max(x - 2, 0);
        int right = min(x + blockDim.x, w - 1);
        tile[ty][tx] = in[y * w + left];                    // 左
        tile[ty][lx + blockDim.x] = in[y * w + right];      // 右
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    float sum = 0;
    for (int k = -2; k <= 2; ++k) {
        sum += gaussKernel1D[k + 2] * tile[ty][lx + k];
    }

    out[y * w + x] = (unsigned char)sum;
}

// 垂直高斯模糊
__global__ void gaussianBlur1D_Vertical(const unsigned char* in, unsigned char* out, int w, int h) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    __shared__ unsigned char tile[16 + 4][16]; // 每列加上下 padding 2
    int ly = ty + 2;

    // 複製中心
    if (x < w && y < h)
        tile[ly][tx] = in[y * w + x];
    else
        tile[ly][tx] = 0;

    // padding 上下邊界
    if (ty < 2) {
        int top    = max(y - 2, 0);
        int bottom = min(y + blockDim.y, h - 1);
        tile[ty][tx] = in[top * w + x];                      // 上
        tile[ly + blockDim.y][tx] = in[bottom * w + x];     // 下
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    float sum = 0;
    for (int k = -2; k <= 2; ++k) {
        sum += gaussKernel1D[k + 2] * tile[ly + k][tx];
    }

    out[y * w + x] = (unsigned char)sum;
}


__global__ void sobelShared(const unsigned char* in, unsigned char* out, int w,int h){
  int tx = threadIdx.x, ty = threadIdx.y;
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  int lx = tx + 1, ly = ty + 1;

  __shared__ unsigned char tile[16 + 2][16 + 2];

  if (x < w && y < h)
    tile[ly][lx] = in[y*w+x];
  else
    tile[ly][lx] = 0;

  // padding
  if (tx == 0 && x > 0) tile[ly][0] = in[y * w + x - 1];
  if (tx == blockDim.x - 1 && x < w - 1) tile[ly][lx + 1] = in[y * w + x + 1];
  if (ty == 0 && y > 0) tile[0][lx] = in[(y - 1) * w + x];
  if (ty == blockDim.y - 1 && y < h - 1) tile[ly + 1][lx] = in[(y + 1) * w + x];

  // corners padding
  if (tx == 0 && ty == 0) tile[0][0] = in[(y - 1) * w + x - 1];
  if (tx == blockDim.x - 1 && ty == 0) tile[0][lx + 1] = in[(y - 1) * w + x + 1];
  if (tx == 0 && ty == blockDim.y - 1) tile[ly + 1][0] = in[(y + 1) * w + x - 1];
  if (tx == blockDim.x - 1 && ty == blockDim.y - 1) tile[ly + 1][lx + 1] = in[(y + 1) * w + x + 1];

  __syncthreads();

  if (x>=w || y>=h) return;

  int Gx=0, Gy=0;
  int sx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}},
      sy[3][3]={{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
  for (int r = -1; r <= 1; ++r)
    for (int c = -1; c <= 1; ++c) {
      int p = tile[ly + r][lx + c];
      Gx += sx[r + 1][c + 1] * p;
      Gy += sy[r + 1][c + 1] * p;
    }
  out[y*w + x] = (unsigned char)min(255, abs(Gx)+abs(Gy));
}


int main(){
    std::vector<std::string> imageList;
    getImageList(DIR_NAME, imageList);

    cudaEvent_t start, stop;

    // 總時間量測
    auto cpu_start = high_resolution_clock::now();

    // 平均值統計
    double total_time_h2d = 0, total_time_k1 = 0, total_time_k2 = 0, total_time_k3 = 0, total_time_d2h = 0;
    double total_frame_time_ms = 0;
    int frame_count = 0;

    // 功耗資料紀錄
    system("nvidia-smi --query-gpu=timestamp,power.draw --format=csv -lms 200 > power_log.csv &");

    // CUDA Event 初始化
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (const auto& path : imageList) {
        int IMG_W, IMG_H, channels;
        unsigned char* h_in = stbi_load(path.c_str(), &IMG_W, &IMG_H, &channels, 1);
        if (!h_in) {
            fprintf(stderr, "ERROR: failed to load %s\n", path.c_str());
            continue;
        }

        size_t imgBytes = IMG_W * IMG_H;
        unsigned char *h_out = (unsigned char*)malloc(imgBytes);
        unsigned char *d_in, *d_tmp, *d_out;
        cudaMalloc(&d_in, imgBytes);
        cudaMalloc(&d_tmp,imgBytes);
        cudaMalloc(&d_out,imgBytes);

        dim3 blk(16,16),
             grid((IMG_W+15)/16,(IMG_H+15)/16);

        float t_h2d=0, t_k1=0, t_k2=0, t_k3=0, t_d2h=0;

        // Host → Device
        cudaEventRecord(start);
        cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_h2d, start, stop);

        // Kernel 1: Gaussian Horizontal
        cudaEventRecord(start);
        gaussianBlur1D_Horizontal<<<grid, blk>>>(d_in, d_tmp, IMG_W, IMG_H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_k1, start, stop);

        // Kernel 2: Gaussian Vertical
        cudaEventRecord(start);
        gaussianBlur1D_Vertical<<<grid, blk>>>(d_tmp, d_tmp, IMG_W, IMG_H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_k2, start, stop);

        // Kernel 3: Sobel
        cudaEventRecord(start);
        sobelShared<<<grid, blk>>>(d_tmp, d_out, IMG_W, IMG_H);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_k3, start, stop);

        // Device → Host
        cudaEventRecord(start);
        cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t_d2h, start, stop);

        // 加總每個 frame 的統計值
        double frame_total = t_h2d + t_k1 + t_k2 + t_k3 + t_d2h;
        total_time_h2d += t_h2d;
        total_time_k1 += t_k1;
        total_time_k2 += t_k2;
        total_time_k3 += t_k3;
        total_time_d2h += t_d2h;
        total_frame_time_ms += frame_total;
        frame_count++;

        // 印出時間
        printf("[%-20s] H2D=%.3fms | GaussianH=%.3fms | GaussianV=%.3fms | Sobel=%.3fms | D2H=%.3fms\n",
            path.c_str(), t_h2d, t_k1, t_k2, t_k3, t_d2h);

        // 輸出檔名
        std::string filename = path.substr(path.find_last_of("/") + 1);
        std::string outPath = "output_images/" + filename;
        stbi_write_jpg(outPath.c_str(), IMG_W, IMG_H, 1, h_out, 95);

        // 清除
        stbi_image_free(h_in);
        free(h_out);
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);

        printf("Done: %s -> %s\n", path.c_str(), outPath.c_str());
    }

    // 刪除事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    system("pkill -f \"nvidia-smi --query-gpu=timestamp,power.draw\"");

    auto cpu_end = high_resolution_clock::now();
    double total_cpu_time_ms = duration<double, std::milli>(cpu_end - cpu_start).count();

    // 統計指標輸出
    double avg_exec_time = (frame_count > 0) ? total_frame_time_ms / frame_count : 0.0;

    printf("Total frames time ms: %.6f\n", total_frame_time_ms);

    printf("\n========== Summary ==========\n");
    printf("Total frames processed: %d\n", frame_count);
    printf("Total Execution Time (ms): %.16f\n", total_frame_time_ms);
    printf("Avg Execution Time per Frame (ms): %.16f\n", avg_exec_time);
    printf("Avg PCIe Bandwidths (MB/s):\n");
    printf("  bw_page_h2d_MBps: n/a\n");
    printf("  bw_pin_h2d_MBps: n/a  (未使用 pinned memory)\n");
    printf("  bw_page_d2h_MBps: n/a\n");
    printf("  bw_pin_d2h_MBps: n/a\n");
    printf("Total CPU-side time = %.3f ms\n", total_cpu_time_ms);
    
    double avg_power = 0, total_energy = 0;
    double total_power_duration_h = computePowerStats("power_log.csv", avg_power, total_energy);

    double total_energy_joule = total_energy * 3600.0;
    double efficiency = (total_energy_joule > 0) ? frame_count / total_energy_joule : 0;

    printf("\n========== Power Analysis ==========\n");
    printf("Average Power (W): %.3f\n", avg_power);
    printf("Total Energy (Wh): %.6f\n", total_energy);
    printf("Power Efficiency (frames per Joule): %.6f\n", efficiency);


    return 0;
}

