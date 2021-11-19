#include "core/include/convolution.hpp"
#include "core/include/exception.h"
#include "core/include/cuda_helper.h"

namespace pyimagematch{

#ifdef PIM_WITH_CUDA

/**
 * input : h*w or h*w*c1
 * kernel: k1*k2 or k1*k2*c2
 * result:
 *  if input h*w, kernel k1*k2, then result h*w. 
 *  if input h*w, kernel k1*k2*c, then result h*w*c. 
 *  if input h*w*c, kernel k1*k2, then result h*w*c.  
 */

__global__ void naive_filter_kernel_single(
                float* input_data, int height, int width, 
                float* kernel_data, int kh, int kw, float* result, float padding=0.f){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_kh = kh / 2;
    int half_hw = kw / 2;
    if (w<width && h<height){
        float sum = 0;
        for(int r=0; r<kh; r++){
            for(int c=0; c<kw; c++){
                int x = w + c - half_hw;
                int y = h + r - half_kh;
                if (x>=0 && x<width && y>=0 && y<height){
                    sum += input_data[x+y*width] * kernel_data[c+r*kw];
                }else{
                    sum += padding * kernel_data[c+r*kw];
                }
            }
        }
        result[w+h*width] = sum;
    }
}

__global__ void naive_filter_kernel_input_c_channel(
                float* input_data, int channel, int height, int width, 
                float* kernel_data, int kh, int kw, float* result, float padding=0.f){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_kh = kh / 2;
    int half_hw = kw / 2;
    if (w<width && h<height){
        for(int ch=0; ch<channel; ch++){
            float sum = 0;
            for(int r=0; r<kh; r++){
                for(int c=0; c<kw; c++){
                    int x = w + c - half_hw;
                    int y = h + r - half_kh;
                    if (x>=0 && x<width && y>=0 && y<height){
                        sum += input_data[ch*height*width + y*width + x] * kernel_data[c+r*kw];
                    }else{
                        sum += padding * kernel_data[c+r*kw];
                    }
                }
            }
            result[ch*height*width+h*width+w] = sum;
        }
    }
}

__global__ void naive_filter_kernel_c_filter(
                float* input_data, int channel, int height, int width, 
                float* kernel_data, int kc, int kh, int kw, float* result, float padding=0.f){
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    
    int half_kh = kh / 2;
    int half_hw = kw / 2;
    if (w<width && h<height){
        for(int ch=0; ch<kc; ch++){
            float sum = 0;
            for(int r=0; r<kh; r++){
                for(int c=0; c<kw; c++){
                    int x = w + c - half_hw;
                    int y = h + r - half_kh;
                    if (x>=0 && x<width && y>=0 && y<height){
                        sum += input_data[x+y*width] * kernel_data[c+r*kw+ch*kw*kh];
                    }else{
                        sum += padding * kernel_data[c+r*kw+ch*kw*kh];
                    }
                }
            }
            result[ch*height*width+h*width+w] = sum;
        }
    }
}

void chw_naive_filter(GpuTensor<float>& input, GpuTensor<float>& kernel, GpuTensor<float>& result){
    PIM_ASSERT(
        (input.shape().size()==2 && kernel.shape().size()==2) &&
        (input.shape().size()==2 && kernel.shape().size()==3) &&
        (input.shape().size()==3 && kernel.shape().size()==2));

    dim3 threads(32, 32);
    // dim3 block(N / threads.x, N / threads.y);
    // MatAdd<<<numBlocks, threadsPerBlock>>>(A,B,C);

    
    
}
#endif

}