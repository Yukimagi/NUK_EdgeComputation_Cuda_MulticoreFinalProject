#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>    // 用來檢查路徑是否存在
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <errno.h>       // 用於 errno

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ----------------------------------------------
// 版本二修改：
// 1) Sobel 核放到 constant memory
__constant__ int sobelX[9] = {
    -1,  0,  1,
    -2,  0,  2,
    -1,  0,  1
};
__constant__ int sobelY[9] = {
     1,  2,  1,
     0,  0,  0,
    -1, -2, -1
};

// 5×5 高斯核保持放 constant memory（版本一即已如此）
static __constant__ float gaussKernel[25] = {
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f
};

// ----------------------------------------------
// 高斯模糊 Kernel（同版本一，沒有做 unroll）
__global__ void gaussianBlurGlobal(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int ix = min(max(x + kx, 0), w - 1);
            int iy = min(max(y + ky, 0), h - 1);
            sum += gaussKernel[(ky + 2) * 5 + (kx + 2)] * in[iy * w + ix];
        }
    }
    out[y * w + x] = (unsigned char)sum;
}

// 版本二：Sobel Kernel，使用 constant sobelX/sobelY，並在取 3×3 時做 loop unrolling
__global__ void sobelGlobal(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int Gx = 0, Gy = 0;
    unsigned char ptrVals[9];
#pragma unroll
    for (int r = -1; r <= 1; ++r) {
#pragma unroll
        for (int c = -1; c <= 1; ++c) {
            int ix = min(max(x + c, 0), w - 1);
            int iy = min(max(y + r, 0), h - 1);
            int idx = (r + 1) * 3 + (c + 1);
            ptrVals[idx] = in[iy * w + ix];
        }
    }
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        Gx += sobelX[i] * ptrVals[i];
        Gy += sobelY[i] * ptrVals[i];
    }
    int val = abs(Gx) + abs(Gy);
    out[y * w + x] = (unsigned char)(val > 255 ? 255 : val);
}

// 檢查檔案是否存在
bool file_exists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s input.mp4\n", argv[0]);
        return -1;
    }
    const char* input_video = argv[1];

    // 打開 CSV 檔案
    std::ofstream ofs;
    ofs.open("timing.csv", std::ios::out);
    if (!ofs.is_open()) {
        fprintf(stderr, "ERROR: 無法開啟 timing.csv\n");
        return -1;
    }
    ofs << "frame_id,"
        << "elapsed_page_h2d_ms,bw_page_h2d_MBps,"
        << "elapsed_pin_h2d_ms,bw_pin_h2d_MBps,"
        << "elapsed_kernels_ms,"
        << "elapsed_page_d2h_ms,bw_page_d2h_MBps,"
        << "elapsed_pin_d2h_ms,bw_pin_d2h_MBps\n";
    ofs << std::fixed << std::setprecision(3);
    ofs.flush();  // 先把表頭刷到磁碟

    // ──────────────────────────────────────────────
    // 1. 用 ffmpeg 拆影格
    printf(">>> 正在呼叫 ffmpeg 拆影格 ...\n");
    if (mkdir("frames", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir frames 失敗");
        ofs.close();
        return -1;
    }
    {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "ffmpeg -hide_banner -loglevel error -i %s -vsync 0 frames/frame_%%06d.png",
                 input_video);
        int ret = system(buf);
        if (ret != 0) {
            fprintf(stderr, "ERROR: ffmpeg 拆影格失敗 (cmd: %s)\n", buf);
            ofs.close();
            return -1;
        }
    }
    printf(">>> 拆影格完成，請到 `frames/` 資料夾確認 PNG 檔。\n");

    if (!file_exists("frames/frame_000001.png")) {
        fprintf(stderr, "ERROR: 沒有在 frames/ 找到任何影格。\n");
        ofs.close();
        return -1;
    }

    // 2. 讀第一張影格取得解析度
    int IMG_W = 0, IMG_H = 0, channels = 0;
    unsigned char* dummy = stbi_load("frames/frame_000001.png", &IMG_W, &IMG_H, &channels, 1);
    if (!dummy) {
        fprintf(stderr, "ERROR: stbi_load 讀取 frames/frame_000001.png 失敗\n");
        ofs.close();
        return -1;
    }
    stbi_image_free(dummy);
    printf(">>> 影格解析度 = %dx%d (灰階單通道)\n", IMG_W, IMG_H);

    size_t frame_bytes = IMG_W * IMG_H * sizeof(unsigned char);
    int max_frames = 0;
    for (int i = 1;; ++i) {
        char fname[256];
        snprintf(fname, sizeof(fname), "frames/frame_%06d.png", i);
        if (!file_exists(fname)) break;
        ++max_frames;
    }
    printf(">>> 總共在 frames/ 找到 %d 張影格。\n", max_frames);
    if (max_frames == 0) {
        fprintf(stderr, "ERROR: 沒有任何影格可處理。\n");
        ofs.close();
        return -1;
    }

    // ──────────────────────────────────────────────
    // 3. 配置 Host 端的 pinned buffer（cudaHostAlloc）＋建立 2 個 CUDA Streams
    unsigned char* h_pinned_input[2]  = { nullptr, nullptr };
    unsigned char* h_pinned_output[2] = { nullptr, nullptr };
    for (int i = 0; i < 2; ++i) {
        if (cudaHostAlloc((void**)&h_pinned_input[i],  frame_bytes, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void**)&h_pinned_output[i], frame_bytes, cudaHostAllocDefault) != cudaSuccess)
        {
            fprintf(stderr, "ERROR: cudaHostAlloc 失敗\n");
            return -1;
        }
    }

    // 4. Device Buffer（二組，用來在兩個 stream 交錯）
    unsigned char *d_in[2]  = { nullptr, nullptr };
    unsigned char *d_tmp[2] = { nullptr, nullptr };
    unsigned char *d_out[2] = { nullptr, nullptr };
    for (int i = 0; i < 2; ++i) {
        cudaMalloc((void**)&d_in[i],  frame_bytes);
        cudaMalloc((void**)&d_tmp[i], frame_bytes);
        cudaMalloc((void**)&d_out[i], frame_bytes);
    }

    // 5. 建立 2 個 CUDA Streams
    cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    // 6. 建立 cudaEvent（各 Stream 分別用三個：H2D_start/H2D_end, kernel_start/kernel_end, D2H_start/D2H_end）
    cudaEvent_t ev_h2d_start[2], ev_h2d_end[2];
    cudaEvent_t ev_kernel_start[2], ev_kernel_end[2];
    cudaEvent_t ev_d2h_start[2], ev_d2h_end[2];
    for (int i = 0; i < 2; ++i) {
        cudaEventCreate(&ev_h2d_start[i]);
        cudaEventCreate(&ev_h2d_end[i]);
        cudaEventCreate(&ev_kernel_start[i]);
        cudaEventCreate(&ev_kernel_end[i]);
        cudaEventCreate(&ev_d2h_start[i]);
        cudaEventCreate(&ev_d2h_end[i]);
    }

    // 7. 決定 Block/ Grid（同版本一）
    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x, (IMG_H + block.y - 1) / block.y);

    // 8. 逐張讀取並分配到兩個 Stream（ping-pong）：
    //    - 對 frame fid，用 streamID = (fid % 2) 執行 「非同步 H2D → kernel → 非同步 D2H」
    //    - 不立即同步，而是在 fid >= 3 時，才對 fid-2 那張做同步、量時間、寫檔、寫 CSV
    //
    // 【修改】加上對 processed 資料夾的錯誤檢查
    if (mkdir("processed", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir processed 失敗");
        return -1;
    }

    // 用來暫存每幀各個階段耗時，最後寫進 CSV
    float h2d_time_ms[2]    = {0.0f, 0.0f};
    float kernel_time_ms[2] = {0.0f, 0.0f};
    float d2h_time_ms[2]    = {0.0f, 0.0f};
    float bw_h2d[2]         = {0.0f, 0.0f};
    float bw_d2h[2]         = {0.0f, 0.0f};
    float mb = frame_bytes / (1024.0f * 1024.0f);

    // Helper lambda：在事件對上後，計算該張影格最終階段耗時、寫檔與寫 CSV
    auto finalize_frame = [&](int old_fid) {
        int sid = old_fid % 2;
        // 等待該 stream 上的 D2H 結束
        cudaEventSynchronize(ev_d2h_end[sid]);

        // 計算各階段耗時
        cudaEventElapsedTime(&h2d_time_ms[sid],    ev_h2d_start[sid], ev_h2d_end[sid]);
        cudaEventElapsedTime(&kernel_time_ms[sid], ev_kernel_start[sid], ev_kernel_end[sid]);
        cudaEventElapsedTime(&d2h_time_ms[sid],    ev_d2h_start[sid], ev_d2h_end[sid]);

        // 帶寬 (MB/s)：mb / (ms/1000)
        bw_h2d[sid] = mb / (h2d_time_ms[sid] / 1000.0f);
        bw_d2h[sid] = mb / (d2h_time_ms[sid] / 1000.0f);

        // 寫出 processed PNG：直接用 pinned_output 內的資料
        char out_name[256];
        snprintf(out_name, sizeof(out_name), "processed/processed_%06d.png", old_fid);
        int write_ok = stbi_write_png(out_name, IMG_W, IMG_H, 1,
                                      h_pinned_output[sid], IMG_W);
        if (!write_ok) {
            fprintf(stderr, "ERROR: 無法寫出 %s\n", out_name);
            // CSV 填寫 ERROR
            ofs << old_fid << ","
                << "N/A,N/A,"
                << "N/A,N/A,"
                << "ERROR_WRITE,"
                << "N/A,N/A,"
                << "N/A,N/A\n";
            ofs.flush();
            return;
        }

        // 寫入 CSV (只保留 pinned H2D / kernel / pinned D2H)
        ofs << old_fid << ","
            << "N/A,N/A,"                              // Pageable H2D
            << h2d_time_ms[sid]    << "," << bw_h2d[sid]    << ","  // Pinned H2D
            << kernel_time_ms[sid] << ","                      // kernels
            << "N/A,N/A,"                              // Pageable D2H
            << d2h_time_ms[sid]    << "," << bw_d2h[sid]    << "\n"; // Pinned D2H
        ofs.flush();

        // 印終端日誌
        printf("Frame %06d [stream %d]: pinned H→D=%.3fms (%.2fMB/s) | kernels=%.3fms | pinned D→H=%.3fms (%.2fMB/s)\n",
               old_fid, sid,
               h2d_time_ms[sid], bw_h2d[sid],
               kernel_time_ms[sid],
               d2h_time_ms[sid], bw_d2h[sid]);
    };

    // 主要迴圈：對每一張影格進行「非同步 H2D→kernel→非同步 D2H」，並在合適時機 finalize（fid-2）
    for (int fid = 1; fid <= max_frames; ++fid) {
        int sid = fid % 2;  // 0 或 1

        // (1) 讀入 PNG 到 pageable ptr
        char in_name[256];
        snprintf(in_name, sizeof(in_name), "frames/frame_%06d.png", fid);
        int w, h, c;
        unsigned char* ptr_pageable = stbi_load(in_name, &w, &h, &c, 1);
        if (!ptr_pageable) {
            fprintf(stderr, "WARNING: stbi_load 讀取 %s 失敗，跳過這張\n", in_name);
            // 如果這張跳過，也要保證 pipeline 不會卡
            // In practice，直接將這張的 slot 視為 no-op
            continue;
        }

        // (2) copy pageable → pinned_input
        memcpy(h_pinned_input[sid], ptr_pageable, frame_bytes);
        stbi_image_free(ptr_pageable);

        // (3) 非同步 Pinned H2D → device
        cudaEventRecord(ev_h2d_start[sid], stream[sid]);
        cudaMemcpyAsync(d_in[sid], h_pinned_input[sid], frame_bytes,
                        cudaMemcpyHostToDevice, stream[sid]);
        cudaEventRecord(ev_h2d_end[sid], stream[sid]);

        // (4) 非同步執行 kernel（Gaussian→Sobel）
        cudaEventRecord(ev_kernel_start[sid], stream[sid]);
        gaussianBlurGlobal<<<grid, block, 0, stream[sid]>>>(d_in[sid], d_tmp[sid], IMG_W, IMG_H);
        sobelGlobal<<<grid, block, 0, stream[sid]>>>(d_tmp[sid], d_out[sid], IMG_W, IMG_H);
        cudaEventRecord(ev_kernel_end[sid], stream[sid]);

        // (5) 非同步 Device → Pinned D2H
        cudaEventRecord(ev_d2h_start[sid], stream[sid]);
        cudaMemcpyAsync(h_pinned_output[sid], d_out[sid], frame_bytes,
                        cudaMemcpyDeviceToHost, stream[sid]);
        cudaEventRecord(ev_d2h_end[sid], stream[sid]);

        // 當已排到第 fid >= 3 時，就可以 finalize 框架 fid-2
        if (fid >= 3) {
            finalize_frame(fid - 2);
        }
    }

    // 迴圈結束後，還剩最後 2 張影格的 finalize
    if (max_frames >= 2) {
        finalize_frame(max_frames - 1);
    }
    if (max_frames >= 1) {
        finalize_frame(max_frames);
    }

    // 9. 釋放 CUDA 資源
    for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(ev_h2d_start[i]);
        cudaEventDestroy(ev_h2d_end[i]);
        cudaEventDestroy(ev_kernel_start[i]);
        cudaEventDestroy(ev_kernel_end[i]);
        cudaEventDestroy(ev_d2h_start[i]);
        cudaEventDestroy(ev_d2h_end[i]);
        cudaStreamDestroy(stream[i]);
        cudaFree(d_in[i]);
        cudaFree(d_tmp[i]);
        cudaFree(d_out[i]);
        cudaFreeHost(h_pinned_input[i]);
        cudaFreeHost(h_pinned_output[i]);
    }
    ofs.close();

    // 10. 用 ffmpeg 合併 processed/*.png → output.mp4
    printf(">>> 正在呼叫 ffmpeg 合併影格成 output.mp4 ...\n");
    {
        char buf2[512];
        snprintf(buf2, sizeof(buf2),
                 "ffmpeg -hide_banner -loglevel error -y "
                 "-r 30 -start_number 1 "
                 "-i processed/processed_%%06d.png "
                 "-c:v mpeg4 -q:v 5 output.mp4");
        int ret2 = system(buf2);
        if (ret2 != 0) {
            fprintf(stderr, "ERROR: ffmpeg 合併影格失敗 (cmd: %s)\n", buf2);
            return -1;
        }
    }
    printf(">>> 處理完成，輸出影片：output.mp4\n");
    return 0;
}
