// test_vedio_simple.cu
//
// 純版本 A 風格：
// - 拆 input.mp4 → frames/frame_XXXXXX.png
// - 逐幀讀入灰階 PNG，用同步 cudaMemcpy + 兩個 kernel (Gaussian → Sobel)
// - 僅測量 kernel 耗時，其它 H2D/D2H 寫入 N/A
// - 寫入 timing.csv：若本版本不測某階段，則填 "N/A"
// - 處理後輸出到 processed/processed_XXXXXX.png
// - 最後合成 processed/*.png → output.mp4
//
// 編譯：
//   nvcc -std=c++11 test_vedio_simple.cu -o test_vedio_simple
//
// 執行：
//   ./test_vedio_simple input.mp4
//
// 產出：
//   - frames/frame_000001.png, frame_000002.png, …
//   - processed/processed_000001.png, processed/processed_000002.png, …
//   - timing.csv
//   - output.mp4
//

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>    // 用來檢查路徑是否存在
#include <sys/types.h>
#include <unistd.h>
#include <fstream>       // 用於輸出 CSV
#include <iomanip>       // 控制浮點格式
#include <errno.h>       // errno

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"        // header-only image loader
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // header-only image writer

// ----------------------------------------------
// 5×5 高斯核（已歸一化），放到 constant memory
static __constant__ float gaussKernel[25] = {
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f
};

// 高斯模糊 Kernel（同版本 A）
__global__
void gaussianBlurGlobal(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    float sum = 0.0f;
    for (int ky = -2; ky <= 2; ++ky) {
        for (int kx = -2; kx <= 2; ++kx) {
            int ix = min(max(x + kx, 0), w - 1);
            int iy = min(max(y + ky, 0), h - 1);
            sum += gaussKernel[(ky + 2)*5 + (kx + 2)] * in[iy * w + ix];
        }
    }
    out[y * w + x] = (unsigned char)sum;
}

// Sobel Kernel（同版本 A）
__global__
void sobelGlobal(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int Gx = 0, Gy = 0;
    int sx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};
    int sy[3][3] = {{ 1,  2,  1},
                    { 0,  0,  0},
                    {-1, -2, -1}};
    for (int r = -1; r <= 1; ++r) {
        for (int c = -1; c <= 1; ++c) {
            int ix = min(max(x + c, 0), w - 1);
            int iy = min(max(y + r, 0), h - 1);
            int p = in[iy * w + ix];
            Gx += sx[r + 1][c + 1] * p;
            Gy += sy[r + 1][c + 1] * p;
        }
    }
    int val = abs(Gx) + abs(Gy);
    out[y * w + x] = (unsigned char)(val > 255 ? 255 : val);
}

// 檢查文件或目錄是否存在
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

    // ------------------------------------------------------
    // 1. 打開 CSV（truncate 模式），寫入表頭
    std::ofstream ofs("timing.csv", std::ios::out);
    if (!ofs.is_open()) {
        fprintf(stderr, "ERROR: 無法打開 timing.csv\n");
        return -1;
    }
    // CSV 欄：frame_id, elapsed_page_h2d_ms, bw_page_h2d_MBps,
    //         elapsed_pin_h2d_ms,  bw_pin_h2d_MBps,
    //         elapsed_kernels_ms,
    //         elapsed_page_d2h_ms, bw_page_d2h_MBps,
    //         elapsed_pin_d2h_ms,  bw_pin_d2h_MBps
    //
    ofs << "frame_id,"
        << "elapsed_page_h2d_ms,bw_page_h2d_MBps,"
        << "elapsed_pin_h2d_ms,bw_pin_h2d_MBps,"
        << "elapsed_kernels_ms,"
        << "elapsed_page_d2h_ms,bw_page_d2h_MBps,"
        << "elapsed_pin_d2h_ms,bw_pin_d2h_MBps\n";
    ofs << std::fixed << std::setprecision(3);
    ofs.flush();  // 先把表頭刷到磁碟

    // ------------------------------------------------------
    // 2. 用 FFmpeg 拆幀到 frames/
    printf(">>> 正在呼叫 ffmpeg 拆影格 ...\n");
    if (mkdir("frames", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir frames 失敗");
        ofs.close();
        return -1;
    }
    {
        char cmd[512];
        snprintf(cmd, sizeof(cmd),
                 "ffmpeg -hide_banner -loglevel error -i %s -vsync 0 frames/frame_%%06d.png",
                 input_video);
        int ret = system(cmd);
        if (ret != 0) {
            fprintf(stderr, "ERROR: ffmpeg 拆影格失敗 (cmd: %s)\n", cmd);
            ofs.close();
            return -1;
        }
    }
    printf(">>> 拆影格完成，請檢查 frames/ 下的 PNG 檔。\n");

    if (!file_exists("frames/frame_000001.png")) {
        fprintf(stderr, "ERROR: frames/ 下沒有找到任何影格。\n");
        ofs.close();
        return -1;
    }

    // ------------------------------------------------------
    // 3. 讀取第一張影格取得寬/高（灰階單通道）
    int IMG_W = 0, IMG_H = 0, channels = 0;
    unsigned char* dummy = stbi_load("frames/frame_000001.png", &IMG_W, &IMG_H, &channels, 1);
    if (!dummy) {
        fprintf(stderr, "ERROR: stbi_load 讀取 frames/frame_000001.png 失敗\n");
        ofs.close();
        return -1;
    }
    stbi_image_free(dummy);
    printf(">>> 影格分辨率 = %dx%d (灰階單通道)\n", IMG_W, IMG_H);

    // 計算每幀像素位元組數 & 總幀數
    size_t frame_bytes = IMG_W * IMG_H * sizeof(unsigned char);
    int max_frames = 0;
    for (int i = 1; ; ++i) {
        char fname[256];
        snprintf(fname, sizeof(fname), "frames/frame_%06d.png", i);
        if (!file_exists(fname)) break;
        ++max_frames;
    }
    printf(">>> frames/ 下共找到 %d 張影格。\n", max_frames);
    if (max_frames == 0) {
        fprintf(stderr, "ERROR: 沒有影格可處理。\n");
        ofs.close();
        return -1;
    }

    // ------------------------------------------------------
    // 4. 在 device 上配置 input/temp/output buffer
    unsigned char *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in,  frame_bytes);
    cudaMalloc((void**)&d_tmp, frame_bytes);
    cudaMalloc((void**)&d_out, frame_bytes);

    // 5. 為 kernel 測時準備兩個 event（只測 kernel 耗時）
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    // 6. block & grid 大小
    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x, (IMG_H + block.y - 1) / block.y);

    // ------------------------------------------------------
    // 7. 逐幀處理
    if (mkdir("processed", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir processed 失敗");
        // 釋放資源
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
        ofs.close();
        return -1;
    }

    for (int fid = 1; fid <= max_frames; ++fid) {
        char in_name[256], out_name[256];
        snprintf(in_name,  sizeof(in_name),  "frames/frame_%06d.png", fid);
        snprintf(out_name, sizeof(out_name), "processed/processed_%06d.png", fid);

        // 7.1 用 stbi_load 讀入灰階 PNG
        int w, h, c;
        unsigned char* ptr = stbi_load(in_name, &w, &h, &c, 1);
        if (!ptr) {
            fprintf(stderr, "WARNING: stbi_load 讀取 %s 失敗，跳過此幀\n", in_name);
            // 這幀無法讀入，寫一行 ERROR_LOAD，然後 continue
            ofs << fid << ","
                << "ERROR_LOAD,N/A,"   // Pageable H2D
                << "ERROR_LOAD,N/A,"   // Pinned  H2D
                << "ERROR_LOAD,"       // kernels
                << "ERROR_LOAD,N/A,"   // Pageable D2H
                << "ERROR_LOAD,N/A\n"; // Pinned  D2H
            ofs.flush();
            continue;
        }

        // 7.2 同步 memcpy H2D（版本 A：只用 cudaMemcpy，不分 pinned/pageable，故這兩欄都寫 N/A）
        cudaMemcpy(d_in, ptr, frame_bytes, cudaMemcpyHostToDevice);

        // 7.3 測量 kernel 耗時
        float elapsed_kernels = 0.0f;
        cudaEventRecord(ev_start, 0);
        gaussianBlurGlobal<<<grid, block>>>(d_in, d_tmp, IMG_W, IMG_H);
        sobelGlobal     <<<grid, block>>>(d_tmp, d_out, IMG_W, IMG_H);
        cudaEventRecord(ev_end, 0);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_kernels, ev_start, ev_end);

        // 7.4 同步 memcpy D2H
        cudaMemcpy(ptr, d_out, frame_bytes, cudaMemcpyDeviceToHost);

        // 7.5 寫出 processed PNG
        int write_ok = stbi_write_png(out_name, IMG_W, IMG_H, 1, ptr, IMG_W);
        if (!write_ok) {
            fprintf(stderr, "ERROR: 無法寫出 %s\n", out_name);
            // 寫失敗也要向 CSV 報 ERROR_WRITE
            ofs << fid << ","
                << "N/A,N/A,"   // Pageable H2D
                << "N/A,N/A,"   // Pinned  H2D
                << "ERROR_WRITE," 
                << "N/A,N/A,"   // Pageable D2H
                << "N/A,N/A\n"; // Pinned  D2H
            ofs.flush();
            stbi_image_free(ptr);
            continue;
        }

        // 7.6 把這一幀各欄寫入 CSV，只有 kernel 測了時間，其它都寫 N/A
        ofs << fid << ","
            << "N/A,N/A,"        // elapsed_page_h2d_ms, bw_page_h2d_MBps
            << "N/A,N/A,"        // elapsed_pin_h2d_ms,  bw_pin_h2d_MBps
            << elapsed_kernels << "," 
            << "N/A,N/A,"        // elapsed_page_d2h_ms, bw_page_d2h_MBps
            << "N/A,N/A\n";      // elapsed_pin_d2h_ms,  bw_pin_d2h_MBps
        ofs.flush();

        // 7.7 打印終端日誌
        printf("Frame %06d: kernels=%.3fms\n", fid, elapsed_kernels);

        stbi_image_free(ptr);
    }

    // ------------------------------------------------------
    // 8. 釋放 CUDA 資源
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    ofs.close();

    // ------------------------------------------------------
    // 9. 用 ffmpeg 合併 processed/*.png → output.mp4
    printf(">>> 正在呼叫 ffmpeg 合併影格成 output.mp4 ...\n");
    {
        char cmd2[512];
        snprintf(cmd2, sizeof(cmd2),
                 "ffmpeg -hide_banner -loglevel error -y "
                 "-r 30 -start_number 1 "
                 "-i processed/processed_%%06d.png "
                 "-c:v mpeg4 -q:v 5 output.mp4");
        int ret2 = system(cmd2);
        if (ret2 != 0) {
            fprintf(stderr, "ERROR: ffmpeg 合併影格失敗 (cmd: %s)\n", cmd2);
            return -1;
        }
    }
    printf(">>> 處理完成，生成 output.mp4\n");
    return 0;
}
