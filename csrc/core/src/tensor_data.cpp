#include "core/include/tensor_data.h"
#include "core/include/exception.h"
#ifdef PIM_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <iostream>
#include <cstring>

namespace pyimagematch{

#define  MALLOC_ALIGN    64

template<typename _Tp> static inline _Tp* alignPtr(_Tp* ptr, int n=(int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n-1) & -n);
}

void* fastMalloc(std::size_t size)
{
    unsigned char* udata = (unsigned char*) std::malloc(size + sizeof(void*) + MALLOC_ALIGN);
    // TODO: raise out of memory error
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
}

void fastFree(void* ptr)
{
    if(ptr)
    {
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        std::free(udata);
    }
}

TensorData::TensorData(std::size_t data_size, DataDeviceType data_type):
    _data_size(data_size), _own_data(true){
    switch (data_type)
    {
    case DATA_CPU:
        _data = fastMalloc(data_size);
        _data_device_status = AT_CPU;
        break;
    case DATA_GPU:
#ifdef PIM_WITH_CUDA
        PIM_CHECK_CUDA(cudaMalloc( reinterpret_cast<void **>( &_gpu_data ), _data_size));
        _data_device_status = AT_GPU;
#else
        PIM_WITHOUT_CUDA_ERROR;
#endif
        break;
    case AT_BOTH:
#ifdef PIM_WITH_CUDA
        _data = fastMalloc(data_size);
        _data_device_status = AT_BOTH;
        PIM_CHECK_CUDA(cudaMalloc( reinterpret_cast<void **>( &_gpu_data ), _data_size));
        _data_device_status = AT_GPU;
#else
        PIM_WITHOUT_CUDA_ERROR;
#endif
        break;
    }
}

TensorData::TensorData(std::size_t data_size, void* user_data, DataDeviceType data_type):
    _data_size(data_size), _own_data(false){
    _data = user_data;

    switch (data_type)
    {
    case DATA_CPU:
        _data = user_data;
        _data_device_status = AT_CPU;
        break;
    case DATA_GPU:
#ifdef PIM_WITH_CUDA
        _gpu_data = user_data;
        _data_device_status = AT_GPU;
#else
        PIM_WITHOUT_CUDA_ERROR;
#endif
        break;
    }
}


#ifdef PIM_WITH_CUDA
void TensorData::to_cpu(){
    if(!_data){
        _data = fastMalloc(_data_size);
    }

    if(_data_device_status == AT_CPU){
        // do nothing
    }else{
        if (_gpu_data != _data){
            PIM_CHECK_CUDA(cudaMemcpy(_data, _gpu_data, _data_size, cudaMemcpyDefault));
        }
        _data_device_status = AT_BOTH;
    } 
}

void TensorData::to_cuda(){
    if(!_gpu_data){
        PIM_CHECK_CUDA(cudaMalloc( reinterpret_cast<void **>( &_gpu_data ), _data_size));
    }

    if(_data_device_status == AT_GPU){
        // do nothing
    }else{
        if (_gpu_data != _data){
            PIM_CHECK_CUDA(cudaMemcpy(_gpu_data, _data, _data_size, cudaMemcpyDefault));
        }
        _data_device_status = AT_BOTH;
    } 
}
#endif

TensorData::~TensorData(){
    if(!_own_data){
        return;
    }

    if(_data){
        fastFree(_data);
    }
#ifdef PIM_WITH_CUDA
    if(_gpu_data){
        PIM_CHECK_CUDA(cudaFree(_gpu_data));
    }
#endif
}

TensorData TensorData::copy(){
//     switch (_data_device_status)
//     {
//     case AT_CPU:
//         TensorData copy_tensor(_data_size, DATA_CPU);
//         return copy_tensor;
//     case AT_GPU:
// #ifdef PIM_WITH_CUDA
//         TensorData copy_tensor(_data_size, DATA_GPU);
//         return copy_tensor;
// #else
//         PIM_WITHOUT_CUDA_ERROR;
// #endif
//     case AT_BOTH:
// #ifdef PIM_WITH_CUDA
//         TensorData copy_tensor(_data_size, DATA_CPU);
//         return copy_tensor;
// #else
//         PIM_WITHOUT_CUDA_ERROR;
// #endif
    
//     }

    if (_data_device_status == AT_CPU){
        TensorData copy_tensor(_data_size, DATA_CPU);
        std::memcpy(copy_tensor._data, _data,  _data_size);
        return copy_tensor;
    }else if (_data_device_status == AT_GPU){
#ifdef PIM_WITH_CUDA
        TensorData copy_tensor(_data_size, DATA_GPU);
        cudaMemcpyAsync
#else
        PIM_WITHOUT_CUDA_ERROR;
#endif

    }




}


}