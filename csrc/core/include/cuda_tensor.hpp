#pragma once
#include "core/include/tensor.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include "core/include/exception.h"


namespace pyimagematch{

template<class T>
class GpuTensor: public TensorBase<T>{
public:
    GpuTensor(){}

    GpuTensor(const std::vector<int>& shapes){
        create(shapes);
    }

    void create(const std::vector<int>& shapes){
        _shapes = shapes;
        _strides = std::vector<int>(shapes.size(), -1);
        int data_size = calculateDatasizeWithStrides<T>(shapes, _strides);
        // TODO: handle cases data_size < 0
        _tensor_data_ptr = std::make_shared<TensorData>((std::size_t) data_size, TensorData::DATA_GPU);
        _data_ptr = (T*) _tensor_data_ptr->getGpuDataPtr();
    }

    void download(Tensor<T>& tensor, cudaStream_t* stream=nullptr){
        tensor.createContiguous(_shapes);
        if(stream){
            PIM_CHECK_CUDA(cudaMemcpyAsync(_data_ptr, tensor.mutableData(), 
                        _tensor_data_ptr->getDataSize()-_ori_data_offset, cudaMemcpyHostToDevice, *stream));
        }else{
            PIM_CHECK_CUDA(cudaMemcpy(_data_ptr, tensor.mutableData(), 
                        _tensor_data_ptr->getDataSize()-_ori_data_offset, cudaMemcpyHostToDevice));
        }
    }

    std::vector<int> stride(){ return _strides; }
    std::vector<int> shape(){ return _shapes; }

    int shape(int i){
        PIM_ASSERT(i>=0 && i<_shapes.size());
        return _shapes[i];
    }

public:
    static GpuTensor<T> fromCpuTensor(const Tensor<T>& cpu_tensor){
        
    }

private:
    std::vector<int> _strides;
    std::vector<int> _shapes;
    std::shared_ptr<TensorData> _tensor_data_ptr = nullptr;
    T* _data_ptr;
    std::size_t _ori_data_offset = 0;
};
}
