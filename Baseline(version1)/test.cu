// test.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"          // header-only image loader
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"    // header-only image writer

// 5×5 高斯核（已歸一化）
__constant__ float gaussKernel[25] = {
  1/256.f,4/256.f,6/256.f,4/256.f,1/256.f,
  4/256.f,16/256.f,24/256.f,16/256.f,4/256.f,
  6/256.f,24/256.f,36/256.f,24/256.f,6/256.f,
  4/256.f,16/256.f,24/256.f,16/256.f,4/256.f,
  1/256.f,4/256.f,6/256.f,4/256.f,1/256.f
};

// Gaussian + Sobel kernels (跟原本完全一樣)
__global__ void gaussianBlurGlobal(const unsigned char* in, unsigned char* out, int w,int h){
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=w||y>=h) return;
  float sum=0;
  for(int ky=-2;ky<=2;++ky)for(int kx=-2;kx<=2;++kx){
    int ix=min(max(x+kx,0),w-1),
        iy=min(max(y+ky,0),h-1);
    sum += gaussKernel[(ky+2)*5 + (kx+2)] * in[iy*w+ix];
  }
  out[y*w + x] = (unsigned char)sum;
}

__global__ void sobelGlobal(const unsigned char* in, unsigned char* out, int w,int h){
  int x=blockIdx.x*blockDim.x+threadIdx.x;
  int y=blockIdx.y*blockDim.y+threadIdx.y;
  if(x>=w||y>=h) return;
  int Gx=0, Gy=0;
  int sx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}},
      sy[3][3]={{ 1,2,1},{ 0,0,0},{-1,-2,-1}};
  for(int r=-1;r<=1;++r)for(int c=-1;c<=1;++c){
    int ix=min(max(x+c,0),w-1),
        iy=min(max(y+r,0),h-1),
        p=in[iy*w+ix];
    Gx += sx[r+1][c+1]*p;
    Gy += sy[r+1][c+1]*p;
  }
  out[y*w + x] = (unsigned char)min(255, abs(Gx)+abs(Gy));
}

int main(){
  int IMG_W, IMG_H, channels;
  // 用 stb_image 讀進灰階（最後一個參數 1 強制成單通道）
  unsigned char* h_in = stbi_load("test.jpg", &IMG_W, &IMG_H, &channels, 1);
  if(!h_in){
    fprintf(stderr,"ERROR: stbi_load failed\n");
    return -1;
  }
  size_t imgBytes = IMG_W * IMG_H;
  unsigned char *h_out = (unsigned char*)malloc(imgBytes);
  unsigned char *d_in, *d_tmp, *d_out;
  cudaMalloc(&d_in, imgBytes);
  cudaMalloc(&d_tmp,imgBytes);
  cudaMalloc(&d_out,imgBytes);
  cudaMemcpy(d_in, h_in, imgBytes, cudaMemcpyHostToDevice);

  dim3 blk(16,16),
       grd((IMG_W+15)/16,(IMG_H+15)/16);
  gaussianBlurGlobal<<<grd,blk>>>(d_in,d_tmp,IMG_W,IMG_H);
  sobelGlobal     <<<grd,blk>>>(d_tmp,d_out,IMG_W,IMG_H);
  cudaMemcpy(h_out, d_out, imgBytes, cudaMemcpyDeviceToHost);

  // 用 stb_image_write 寫出 JPG（quality=95）
  stbi_write_jpg("output.jpg", IMG_W, IMG_H, 1, h_out, 95);

  // 釋放
  stbi_image_free(h_in);
  free(h_out);
  cudaFree(d_in); cudaFree(d_tmp); cudaFree(d_out);
  printf("Done. output.jpg generated.\n");
  return 0;
}
