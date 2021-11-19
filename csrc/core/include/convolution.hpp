#include "core/include/tensor.hpp"
#include "core/include/cuda_tensor.hpp"

namespace pyimagematch{

#ifdef PIM_WITH_CUDA
void naive_convolution(GpuTensor<float>& input, GpuTensor<float>& kernel, GpuTensor<float>& result);
#endif

}