// test_vedio_streams.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <errno.h>    // 【修改】 用於 errno

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
// 高斯模糊 Kernel（版本一保持不動）
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
    // 我們把 3×3 展平成一維陣列 idx = (r+1)*3 + (c+1)
    // ptrVals[0..8] 將存取 3×3 區域的 pixel
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
    // 再展開計算 Gx, Gy
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
    // 【確保檔頭先 flush】
    ofs.flush();

    // ──────────────────────────────────────────────
    // 1. 用 ffmpeg 拆影格
    printf(">>> 正在呼叫 ffmpeg 拆影格 ...\n");
    // 【修改】加上錯誤檢查：若 mkdir 失敗且非已存在，就報錯
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
    // 3. 版本二：配置 Host 端的 pinned buffer（cudaHostAlloc）＋建立 2 個 CUDA Streams
    unsigned char* h_frame_pinned0 = nullptr;
    unsigned char* h_frame_pinned1 = nullptr;
    cudaError_t err0 = cudaHostAlloc((void**)&h_frame_pinned0, frame_bytes, cudaHostAllocDefault);
    cudaError_t err1 = cudaHostAlloc((void**)&h_frame_pinned1, frame_bytes, cudaHostAllocDefault);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaHostAlloc 失敗 (%s) or (%s)\n",
                cudaGetErrorString(err0), cudaGetErrorString(err1));
        ofs.close();
        return -1;
    }

    // 4. Device Buffer（兩倍大小，用來在兩個 stream 交錯）
    unsigned char *d_in0 = nullptr, *d_tmp0 = nullptr, *d_out0 = nullptr;
    unsigned char *d_in1 = nullptr, *d_tmp1 = nullptr, *d_out1 = nullptr;
    cudaMalloc((void**)&d_in0,  frame_bytes);
    cudaMalloc((void**)&d_tmp0, frame_bytes);
    cudaMalloc((void**)&d_out0, frame_bytes);
    cudaMalloc((void**)&d_in1,  frame_bytes);
    cudaMalloc((void**)&d_tmp1, frame_bytes);
    cudaMalloc((void**)&d_out1, frame_bytes);

    // 5. 建立 2 個 CUDA Streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // 6. 建立 cudaEvent（各 Stream 分別用一組）
    cudaEvent_t ev_start0, ev_end0, ev_start1, ev_end1;
    cudaEventCreate(&ev_start0);
    cudaEventCreate(&ev_end0);
    cudaEventCreate(&ev_start1);
    cudaEventCreate(&ev_end1);

    // 7. 決定 Block/ Grid（同版本一）
    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x, (IMG_H + block.y - 1) / block.y);

    // 8. 逐張讀取並分配到兩個 Stream（stream0 處理奇數幀、stream1 處理偶數幀）
    // 【修改】加上對 processed 資料夾的錯誤檢查
    if (mkdir("processed", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir processed 失敗");
        // 釋放先前 allocation
        cudaFreeHost(h_frame_pinned0);
        cudaFreeHost(h_frame_pinned1);
        cudaFree(d_in0); cudaFree(d_tmp0); cudaFree(d_out0);
        cudaFree(d_in1); cudaFree(d_tmp1); cudaFree(d_out1);
        ofs.close();
        return -1;
    }

    for (int fid = 1; fid <= max_frames; ++fid) {
        char in_name[256], out_name[256];
        snprintf(in_name,  sizeof(in_name),  "frames/frame_%06d.png", fid);
        snprintf(out_name, sizeof(out_name), "processed/processed_%06d.png", fid);

        int w, h, c;
        unsigned char* ptr = stbi_load(in_name, &w, &h, &c, 1);
        if (!ptr) {
            fprintf(stderr, "WARNING: stbi_load 讀取 %s 失敗，跳過這張\n", in_name);
            // 把失敗的狀況寫到 CSV 裡，並 continue
            ofs << fid << ",ERROR_LOAD,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n";
            continue;
        }

        // 計算帶寬所需：每次拷貝 bytes
        float elapsed_pin_h2d = 0.0f;
        float elapsed_kernels = 0.0f;
        float elapsed_pin_d2h = 0.0f;
        float bw_pin_h2d = 0.0f, bw_pin_d2h = 0.0f;
        float mb = frame_bytes / (1024.0f * 1024.0f);

        if (fid % 2 == 1) {
            // Stream0 處理奇數幀
            // (a) Copy pageable -> pinned -> d_in0
            memcpy(h_frame_pinned0, ptr, frame_bytes);
            cudaEventRecord(ev_start0, stream0);
            cudaMemcpyAsync(d_in0, h_frame_pinned0, frame_bytes, cudaMemcpyHostToDevice, stream0);
            cudaEventRecord(ev_end0, stream0);
            cudaEventSynchronize(ev_end0);
            cudaEventElapsedTime(&elapsed_pin_h2d, ev_start0, ev_end0);
            bw_pin_h2d = mb / (elapsed_pin_h2d / 1000.0f);

            // (b) Kernel on stream0
            cudaEventRecord(ev_start0, stream0);
            gaussianBlurGlobal<<<grid, block, 0, stream0>>>(d_in0, d_tmp0, IMG_W, IMG_H);
            sobelGlobal     <<<grid, block, 0, stream0>>>(d_tmp0, d_out0, IMG_W, IMG_H);
            cudaEventRecord(ev_end0, stream0);
            cudaEventSynchronize(ev_end0);
            cudaEventElapsedTime(&elapsed_kernels, ev_start0, ev_end0);

            // (c) Copy d_out0 -> pinned -> pageable
            cudaEventRecord(ev_start0, stream0);
            cudaMemcpyAsync(h_frame_pinned0, d_out0, frame_bytes, cudaMemcpyDeviceToHost, stream0);
            cudaEventRecord(ev_end0, stream0);
            cudaEventSynchronize(ev_end0);
            cudaEventElapsedTime(&elapsed_pin_d2h, ev_start0, ev_end0);
            bw_pin_d2h = mb / (elapsed_pin_d2h / 1000.0f);

            // 最後把 pinned 回寫到 ptr
            memcpy(ptr, h_frame_pinned0, frame_bytes);

        } else {
            // Stream1 處理偶數幀
            memcpy(h_frame_pinned1, ptr, frame_bytes);
            cudaEventRecord(ev_start1, stream1);
            cudaMemcpyAsync(d_in1, h_frame_pinned1, frame_bytes, cudaMemcpyHostToDevice, stream1);
            cudaEventRecord(ev_end1, stream1);
            cudaEventSynchronize(ev_end1);
            cudaEventElapsedTime(&elapsed_pin_h2d, ev_start1, ev_end1);
            bw_pin_h2d = mb / (elapsed_pin_h2d / 1000.0f);

            cudaEventRecord(ev_start1, stream1);
            gaussianBlurGlobal<<<grid, block, 0, stream1>>>(d_in1, d_tmp1, IMG_W, IMG_H);
            sobelGlobal     <<<grid, block, 0, stream1>>>(d_tmp1, d_out1, IMG_W, IMG_H);
            cudaEventRecord(ev_end1, stream1);
            cudaEventSynchronize(ev_end1);
            cudaEventElapsedTime(&elapsed_kernels, ev_start1, ev_end1);

            cudaEventRecord(ev_start1, stream1);
            cudaMemcpyAsync(h_frame_pinned1, d_out1, frame_bytes, cudaMemcpyDeviceToHost, stream1);
            cudaEventRecord(ev_end1, stream1);
            cudaEventSynchronize(ev_end1);
            cudaEventElapsedTime(&elapsed_pin_d2h, ev_start1, ev_end1);
            bw_pin_d2h = mb / (elapsed_pin_d2h / 1000.0f);

            memcpy(ptr, h_frame_pinned1, frame_bytes);
        }

        // 【修改】——寫出每張處理過的影格到 processed/
        int write_ok = stbi_write_png(out_name, IMG_W, IMG_H, 1, ptr, IMG_W);
        if (!write_ok) {
            fprintf(stderr, "ERROR: 無法寫出 %s\n", out_name);
            ofs << fid << ",ERROR_WRITE,N/A,N/A,N/A,N/A,N/A,N/A,N/A\n";
            stbi_image_free(ptr);
            continue;
        }

        // 輸出結果到 CSV（只保留 pinned bandwidth / 時間）
        ofs << fid << ","
            << "N/A," << "N/A,"          // 不再測 Pageable H2D
            << elapsed_pin_h2d << "," << bw_pin_h2d << ","
            << elapsed_kernels << ","
            << "N/A," << "N/A,"          // 不再測 Pageable D2H
            << elapsed_pin_d2h << "," << bw_pin_d2h << "\n";
        ofs.flush();

        // terminal log
        printf("Frame %06d [stream %d]: pinned H→D=%.3fms (%.2fMB/s) | kernels=%.3fms | pinned D→H=%.3fms (%.2fMB/s)\n",
               fid,
               (fid % 2 == 1 ? 0 : 1),
               elapsed_pin_h2d, bw_pin_h2d,
               elapsed_kernels,
               elapsed_pin_d2h, bw_pin_d2h);

        stbi_image_free(ptr);
    }
    ofs.close();

    // 9. 釋放 CUDA 資源
    cudaEventDestroy(ev_start0);
    cudaEventDestroy(ev_end0);
    cudaEventDestroy(ev_start1);
    cudaEventDestroy(ev_end1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaFree(d_in0); cudaFree(d_tmp0); cudaFree(d_out0);
    cudaFree(d_in1); cudaFree(d_tmp1); cudaFree(d_out1);
    cudaFreeHost(h_frame_pinned0);
    cudaFreeHost(h_frame_pinned1);
    ofs.close();

    // 10. 合成影格
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
