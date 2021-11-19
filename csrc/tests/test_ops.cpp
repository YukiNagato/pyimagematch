#ifdef PIM_WITH_CUDA
#include "core/include/tensor.hpp"
#include <catch2/catch.hpp>
#include "core/include/convolution.hpp"
#include "core/include/cuda_tensor.hpp"
#include "opencv2/opencv.hpp"

using namespace pyimagematch;

TEST_CASE("convolution") {
//     Tensor<float> a({100, 100, 3});
// #ifdef PIM_WITH_CUDA
//     GpuTensor<float> a_gpu({100, 100, 3});
// #endif

    cv::Mat image = cv::imread("/home/ysy/work/pyimagematch/tests/data/car1.jpg");
    image.convertTo(image, CV_32FC3);
    GpuTensor<float> tensor({3, image.rows, image.cols});

}

#endif
