#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>    // 用來檢查檔案是否存在
#include <sys/types.h>
#include <unistd.h>
#include <fstream>       // C++ 檔案輸出
#include <iomanip>       // 控制浮點格式

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"        // header-only image loader
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // header-only image writer

// ----------------------------------------------
// 5×5 高斯核（已歸一化）
static __constant__ float gaussKernel[25] = {
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    6/256.f, 24/256.f, 36/256.f, 24/256.f, 6/256.f,
    4/256.f, 16/256.f, 24/256.f, 16/256.f, 4/256.f,
    1/256.f,  4/256.f,  6/256.f,  4/256.f, 1/256.f
};

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

__global__ void sobelGlobal(const unsigned char* in, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int Gx = 0, Gy = 0;
    int sx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sy[3][3] = {{ 1, 2, 1}, { 0, 0, 0}, {-1,-2,-1}};
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

    // ──────────────────────────────────────────────
    //  ** 在程式最前面打開一個 CSV 檔案 **
    std::ofstream ofs;
    ofs.open("timing.csv", std::ios::out);
    if (!ofs.is_open()) {
        fprintf(stderr, "ERROR: 無法開啟 timing.csv\n");
        return -1;
    }
    // CSV 表頭：新增了各傳輸階段的 bandwidth 欄位 (MB/s)
    ofs << "frame_id,"
        << "elapsed_page_h2d_ms,bw_page_h2d_MBps,"
        << "elapsed_pin_h2d_ms,bw_pin_h2d_MBps,"
        << "elapsed_kernels_ms,"
        << "elapsed_page_d2h_ms,bw_page_d2h_MBps,"
        << "elapsed_pin_d2h_ms,bw_pin_d2h_MBps\n";
    ofs << std::fixed << std::setprecision(3);

    // ──────────────────────────────────────────────
    // 1. 用 ffmpeg 拆影格
    printf(">>> 正在呼叫 ffmpeg 拆影格 ...\n");
    mkdir("frames", 0777);
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

    // 3. 配置 Host 端的 pageable & pinned buffer
    unsigned char* h_frame_pinned = nullptr;
    cudaError_t err = cudaMallocHost((void**)&h_frame_pinned, frame_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMallocHost 失敗 (%s)\n", cudaGetErrorString(err));
        ofs.close();
        return -1;
    }

    // 4. 配置 Device 端 Buffer
    unsigned char *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in,  frame_bytes);
    cudaMalloc((void**)&d_tmp, frame_bytes);
    cudaMalloc((void**)&d_out, frame_bytes);

    // 5. 建立 CUDA Stream & cudaEvent
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    // 6. 決定 Block/ Grid
    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x, (IMG_H + block.y - 1) / block.y);

    // 7. 逐張讀、處理、寫檔並量測
    mkdir("processed", 0777);
    for (int fid = 1; fid <= max_frames; ++fid) {
        char in_name[256], out_name[256];
        snprintf(in_name,  sizeof(in_name),  "frames/frame_%06d.png", fid);
        snprintf(out_name, sizeof(out_name), "processed/processed_%06d.png", fid);

        // 7.1 讀入影格 (pageable)
        int w, h, c;
        unsigned char* ptr = stbi_load(in_name, &w, &h, &c, 1);
        if (!ptr) {
            fprintf(stderr, "WARNING: stbi_load 讀取 %s 失敗，跳過這張\n", in_name);
            continue;
        }
        // 7.2 複製到 pinned buffer
        memcpy(h_frame_pinned, ptr, frame_bytes);

        // 宣告變數存放每個階段的耗時 (毫秒)
        float elapsed_page_h2d = 0.0f, elapsed_pin_h2d = 0.0f;
        float elapsed_kernels   = 0.0f;
        float elapsed_page_d2h = 0.0f, elapsed_pin_d2h = 0.0f;

        // (a) Pageable H→D
        cudaEventRecord(ev_start, stream);
        cudaMemcpyAsync(d_in, ptr, frame_bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_page_h2d, ev_start, ev_end);

        // (b) Pinned H→D
        cudaEventRecord(ev_start, stream);
        cudaMemcpyAsync(d_in, h_frame_pinned, frame_bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_pin_h2d, ev_start, ev_end);

        // (c) 執行 kernel
        cudaEventRecord(ev_start, stream);
        gaussianBlurGlobal<<<grid, block, 0, stream>>>(d_in, d_tmp, IMG_W, IMG_H);
        sobelGlobal     <<<grid, block, 0, stream>>>(d_tmp, d_out, IMG_W, IMG_H);
        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_kernels, ev_start, ev_end);

        // (d) Pageable D→H
        cudaEventRecord(ev_start, stream);
        cudaMemcpyAsync(ptr,    d_out, frame_bytes, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_page_d2h, ev_start, ev_end);

        // (e) Pinned D→H
        cudaEventRecord(ev_start, stream);
        cudaMemcpyAsync(h_frame_pinned, d_out, frame_bytes, cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(ev_end, stream);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_pin_d2h, ev_start, ev_end);

        // 把 pinned buffer 裡的資料複回 ptr，方便寫 PNG
        memcpy(ptr, h_frame_pinned, frame_bytes);

        // 7.4 用 stb_image_write 寫檔
        stbi_write_png(out_name, IMG_W, IMG_H, 1, ptr, IMG_W);

        // 7.5 計算帶寬 (MB/s)： frame_bytes / (elapsed_ms / 1000) ÷ (1024²)
        float mb = frame_bytes / (1024.0f * 1024.0f);
        float bw_page_h2d = mb / (elapsed_page_h2d / 1000.0f);
        float bw_pin_h2d  = mb / (elapsed_pin_h2d  / 1000.0f);
        float bw_page_d2h = mb / (elapsed_page_d2h / 1000.0f);
        float bw_pin_d2h  = mb / (elapsed_pin_d2h  / 1000.0f);

        // 7.6 釋放 stbi_load 的記憶體
        stbi_image_free(ptr);

        // 7.7 把這一幀的數據寫入 CSV
        ofs << fid << ","
            << elapsed_page_h2d << "," << bw_page_h2d << ","
            << elapsed_pin_h2d  << "," << bw_pin_h2d  << ","
            << elapsed_kernels  << ","
            << elapsed_page_d2h << "," << bw_page_d2h << ","
            << elapsed_pin_d2h  << "," << bw_pin_d2h  << "\n";

        // 同步印到終端，方便觀察
        printf("Frame %06d: H→D(page)=%.3fms (%.2fMB/s), H→D(pin)=%.3fms (%.2fMB/s) | "
               "kernels=%.3fms | "
               "D→H(page)=%.3fms (%.2fMB/s), D→H(pin)=%.3fms (%.2fMB/s)\n",
               fid,
               elapsed_page_h2d, bw_page_h2d,
               elapsed_pin_h2d,  bw_pin_h2d,
               elapsed_kernels,
               elapsed_page_d2h, bw_page_d2h,
               elapsed_pin_d2h,  bw_pin_d2h);
    }

    // 8. 釋放 CUDA 資源
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    cudaFreeHost(h_frame_pinned);

    // 9. 用 ffmpeg 把 processed/*.png 組回影片
    printf(">>> 正在呼叫 ffmpeg 合併影格成 output.mp4 ...\n");
    {
        char buf2[512];
        snprintf(buf2, sizeof(buf2),
                 "ffmpeg -hide_banner -loglevel error -y -r 30 -start_number 1 -i processed/processed_%%06d.png -c:v mpeg4 -q:v 5 output.mp4");
        int ret2 = system(buf2);
        if (ret2 != 0) {
            fprintf(stderr, "ERROR: ffmpeg 合併影格失敗 (cmd: %s)\n", buf2);
            ofs.close();
            return -1;
        }
    }
    printf(">>> 處理完成，輸出影片：output.mp4\n");

    // 關閉 CSV 檔案
    ofs.close();
    return 0;
}
