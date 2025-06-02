// test_vedio_simple.cu
//
// 纯版本 A 风格：
// - 拆 input.mp4 → frames/frame_XXXXXX.png
// - 逐帧读入灰度 PNG，用同步 cudaMemcpy + 两个 kernel (Gaussian → Sobel)
// - 仅测量 kernel 耗时，其它 H2D/D2H 写入 N/A
// - 写入 timing.csv：若本版本不测某阶段，则填 "N/A"
// - 处理后输出到 processed/processed_XXXXXX.png
// - 最后合成 processed/*.png → output.mp4
//
// 编译：
//   nvcc -std=c++11 test_vedio_simple.cu -o test_vedio_simple
//
// 运行：
//   ./test_vedio_simple input.mp4
//
// 产出：
//   - frames/frame_000001.png, frame_000002.png, …
//   - processed/processed_000001.png, processed/processed_000002.png, …
//   - timing.csv
//   - output.mp4
//

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>    // 用来检查路径是否存在
#include <sys/types.h>
#include <unistd.h>
#include <fstream>       // 用于输出 CSV
#include <iomanip>       // 控制浮点格式
#include <errno.h>       // errno

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"        // header-only image loader
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // header-only image writer

// ----------------------------------------------
// 5×5 高斯核（已归一化），放到 constant memory
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

// 检查文件或目录是否存在
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
    // 1. 打开 CSV（truncate 模式），写入表头
    std::ofstream ofs("timing.csv", std::ios::out);
    if (!ofs.is_open()) {
        fprintf(stderr, "ERROR: 无法打开 timing.csv\n");
        return -1;
    }
    // CSV 列：frame_id, elapsed_page_h2d_ms, bw_page_h2d_MBps,
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
    ofs.flush();  // 先把表头刷到磁盘

    // ------------------------------------------------------
    // 2. 用 FFmpeg 拆帧到 frames/
    printf(">>> 正在调用 ffmpeg 拆影格 ...\n");
    if (mkdir("frames", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir frames 失败");
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
            fprintf(stderr, "ERROR: ffmpeg 拆影格失败 (cmd: %s)\n", cmd);
            ofs.close();
            return -1;
        }
    }
    printf(">>> 拆影格完成，请检查 frames/ 下的 PNG 文件。\n");

    if (!file_exists("frames/frame_000001.png")) {
        fprintf(stderr, "ERROR: frames/ 下没有找到任何影格。\n");
        ofs.close();
        return -1;
    }

    // ------------------------------------------------------
    // 3. 读取第一张影格获取宽/高（灰度单通道）
    int IMG_W = 0, IMG_H = 0, channels = 0;
    unsigned char* dummy = stbi_load("frames/frame_000001.png", &IMG_W, &IMG_H, &channels, 1);
    if (!dummy) {
        fprintf(stderr, "ERROR: stbi_load 读取 frames/frame_000001.png 失败\n");
        ofs.close();
        return -1;
    }
    stbi_image_free(dummy);
    printf(">>> 影格分辨率 = %dx%d (灰度单通道)\n", IMG_W, IMG_H);

    // 计算每帧像素字节数 & 总帧数
    size_t frame_bytes = IMG_W * IMG_H * sizeof(unsigned char);
    int max_frames = 0;
    for (int i = 1; ; ++i) {
        char fname[256];
        snprintf(fname, sizeof(fname), "frames/frame_%06d.png", i);
        if (!file_exists(fname)) break;
        ++max_frames;
    }
    printf(">>> frames/ 下共找到 %d 张影格。\n", max_frames);
    if (max_frames == 0) {
        fprintf(stderr, "ERROR: 没有影格可处理。\n");
        ofs.close();
        return -1;
    }

    // ------------------------------------------------------
    // 4. 在 device 上分配 input/temp/output buffer
    unsigned char *d_in = nullptr, *d_tmp = nullptr, *d_out = nullptr;
    cudaMalloc((void**)&d_in,  frame_bytes);
    cudaMalloc((void**)&d_tmp, frame_bytes);
    cudaMalloc((void**)&d_out, frame_bytes);

    // 5. 为 kernel 测时准备两个 event（只测 kernel 耗时）
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    // 6. block & grid 大小
    dim3 block(16, 16);
    dim3 grid((IMG_W + block.x - 1) / block.x, (IMG_H + block.y - 1) / block.y);

    // ------------------------------------------------------
    // 7. 逐帧处理
    if (mkdir("processed", 0777) && errno != EEXIST) {
        perror("ERROR: mkdir processed 失败");
        // 释放资源
        cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
        ofs.close();
        return -1;
    }

    for (int fid = 1; fid <= max_frames; ++fid) {
        char in_name[256], out_name[256];
        snprintf(in_name,  sizeof(in_name),  "frames/frame_%06d.png", fid);
        snprintf(out_name, sizeof(out_name), "processed/processed_%06d.png", fid);

        // 7.1 用 stbi_load 读入灰度 PNG
        int w, h, c;
        unsigned char* ptr = stbi_load(in_name, &w, &h, &c, 1);
        if (!ptr) {
            fprintf(stderr, "WARNING: stbi_load 读取 %s 失败，跳过此帧\n", in_name);
            // 这帧无法读入，写一行 ERROR_LOAD，然后 continue
            ofs << fid << ","
                << "ERROR_LOAD,N/A,"   // Pageable H2D
                << "ERROR_LOAD,N/A,"   // Pinned  H2D
                << "ERROR_LOAD,"       // kernels
                << "ERROR_LOAD,N/A,"   // Pageable D2H
                << "ERROR_LOAD,N/A\n"; // Pinned  D2H
            ofs.flush();
            continue;
        }

        // 7.2 同步 memcpy H2D（版本 A：只用 cudaMemcpy，不分 pinned/pageable，故这两列都写 N/A）
        cudaMemcpy(d_in, ptr, frame_bytes, cudaMemcpyHostToDevice);

        // 7.3 测量 kernel 耗时
        float elapsed_kernels = 0.0f;
        cudaEventRecord(ev_start, 0);
        gaussianBlurGlobal<<<grid, block>>>(d_in, d_tmp, IMG_W, IMG_H);
        sobelGlobal     <<<grid, block>>>(d_tmp, d_out, IMG_W, IMG_H);
        cudaEventRecord(ev_end, 0);
        cudaEventSynchronize(ev_end);
        cudaEventElapsedTime(&elapsed_kernels, ev_start, ev_end);

        // 7.4 同步 memcpy D2H
        cudaMemcpy(ptr, d_out, frame_bytes, cudaMemcpyDeviceToHost);

        // 7.5 写出 processed PNG
        int write_ok = stbi_write_png(out_name, IMG_W, IMG_H, 1, ptr, IMG_W);
        if (!write_ok) {
            fprintf(stderr, "ERROR: 无法写出 %s\n", out_name);
            // 写失败也要向 CSV 报 ERROR_WRITE
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

        // 7.6 把这一帧各列写入 CSV，只有 kernel 测了时间，其它都写 N/A
        ofs << fid << ","
            << "N/A,N/A,"        // elapsed_page_h2d_ms, bw_page_h2d_MBps
            << "N/A,N/A,"        // elapsed_pin_h2d_ms,  bw_pin_h2d_MBps
            << elapsed_kernels << "," 
            << "N/A,N/A,"        // elapsed_page_d2h_ms, bw_page_d2h_MBps
            << "N/A,N/A\n";      // elapsed_pin_d2h_ms,  bw_pin_d2h_MBps
        ofs.flush();

        // 7.7 打印终端日志
        printf("Frame %06d: kernels=%.3fms\n", fid, elapsed_kernels);

        stbi_image_free(ptr);
    }

    // ------------------------------------------------------
    // 8. 释放 CUDA 资源
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaFree(d_in);
    cudaFree(d_tmp);
    cudaFree(d_out);
    ofs.close();

    // ------------------------------------------------------
    // 9. 用 ffmpeg 合并 processed/*.png → output.mp4
    printf(">>> 正在调用 ffmpeg 合并影格成 output.mp4 ...\n");
    {
        char cmd2[512];
        snprintf(cmd2, sizeof(cmd2),
                 "ffmpeg -hide_banner -loglevel error -y "
                 "-r 30 -start_number 1 "
                 "-i processed/processed_%%06d.png "
                 "-c:v mpeg4 -q:v 5 output.mp4");
        int ret2 = system(cmd2);
        if (ret2 != 0) {
            fprintf(stderr, "ERROR: ffmpeg 合并影格失败 (cmd: %s)\n", cmd2);
            return -1;
        }
    }
    printf(">>> 处理完成，生成 output.mp4\n");
    return 0;
}
