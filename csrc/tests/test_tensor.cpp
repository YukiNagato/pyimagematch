#include "core/include/tensor.hpp"
#include <catch2/catch.hpp>

#ifdef PIM_WITH_CUDA
#include "core/include/cuda_tensor.hpp"
#endif

using namespace pyimagematch;
TEST_CASE("Tensor allocate") {
    Tensor<float> a({100, 100, 3});
#ifdef PIM_WITH_CUDA
    GpuTensor<float> a_gpu({100, 100, 3});
#endif
}




