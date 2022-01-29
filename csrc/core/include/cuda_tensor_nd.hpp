#include "core/include/cuda_tensor.hpp"
#include "core/include/cuda_helper.h"


namespace pyimagematch{

// weak ptr for cuda tensor
template <typename T> 
struct CudaTensor2dPtr{
public:
    __CUDA_HOST_DEVICE__ CudaTensor2dPtr(int d0, int d1, T* data, int s0):
                            d0(d0), d1(d1), data(data), s0(s0){}

    __CUDA_HOST_DEVICE__       T& operator ()(int h, int w)       { return data[h*s0+w]; }
    __CUDA_HOST_DEVICE__ const T& operator ()(int h, int w) const { return data[h*s0+w]; }

public:
    int d0;
    int d1;
    int s0;
    T* data;
};

template <typename T> 
struct CudaTensor3dPtr{
public:
    __CUDA_HOST_DEVICE__ CudaTensor3dPtr(int d0, int d1, int d2, T* data, int s0, int s1):
                            d0(d0), d1(d1), d2(d2), data(data), s0(s0), s1(s1){}

    __CUDA_HOST_DEVICE__       T& operator ()(int c, int h, int w)       { return data[c*s0+h*s1+w]; }
    __CUDA_HOST_DEVICE__ const T& operator ()(int c, int h, int w) const { return data[c*s0+h*s1+w]; }

public:
    int d0;
    int d1;
    int d2;
    int s0;
    int s1;
    T* data;
};


template <typename T> 
struct CudaTensor4dPtr{
public:
    __CUDA_HOST_DEVICE__ CudaTensor4dPtr(int d0, int d1, int d2, int d3, T* data, int s0, int s1, int s2):
                            d0(d0), d1(d1), d2(d2), d3(d3), data(data), s0(s0), s1(s1), s2(s2){}

    __CUDA_HOST_DEVICE__       T& operator ()(int n, int c, int h, int w)       { return data[n*s0+c*s1+h*s2+w]; }
    __CUDA_HOST_DEVICE__ const T& operator ()(int n, int c, int h, int w) const { return data[n*s0+c*s1+h*s2+w]; }

public:
    int d0;
    int d1;
    int d2;
    int d3;
    int s0;
    int s1;
    int s2
    T* data;
};

}