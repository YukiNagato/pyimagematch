#include <pybind11/pybind11.h>
#include "Image.h"
#include "OpticalFlow.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
class BroxLiu{
public:
    double alpha = 1.0;
    double ratio = 0.5;
    int minWidth = 40;
    int nOuterFPIterations = 3;
    int nInnerFPIterations = 1;
    int nSORIterations = 20;
    BroxLiu(){}

    void copyArrayToImage(py::array_t<double, py::array::c_style | py::array::forcecast> & array, DImage& im){
        auto r = array.unchecked<3>();
        int height = array.shape(0);
        int width = array.shape(1);
        int channel = array.shape(2);
        int wc_stride = width*channel;
        for(int h=0; h<height; h++)
        {
            int hwc_stride = h*wc_stride;
            for(int w=0; w<width; w++){
                im.pData[hwc_stride +w*channel] = r(h, w, 0);
                im.pData[hwc_stride +w*channel+1] = r(h, w, 1);
                im.pData[hwc_stride +w*channel+2] = r(h, w, 2);
            }
        }
    }

    std::vector<py::array_t<double>> matching_double(py::array_t<double, py::array::c_style | py::array::forcecast> im1, 
                  py::array_t<double, py::array::c_style | py::array::forcecast> im2){
        
        if ( im1.ndim() != 3 || im2.ndim() != 3 || im1.shape(2) !=3 || im2.shape(2) !=3)
            throw std::runtime_error("images should be H*W*3 NumPy array");

        std::vector<py::array_t<double>> result;

        int h = im1.shape(0);
        int w = im1.shape(1);
        int c = im1.shape(2);

        DImage ImFormatted1, ImFormatted2;
        DImage vxFormatted, vyFormatted, warpI2Formatted;

        ImFormatted1.allocate(w, h, c);
        ImFormatted2.allocate(w, h, c);
        ImFormatted1.setColorType(0);
        ImFormatted2.setColorType(0);

        copyArrayToImage(im1, ImFormatted1);
        copyArrayToImage(im2, ImFormatted2);

        // call optical flow backend
        OpticalFlow::Coarse2FineFlow(vxFormatted, vyFormatted, warpI2Formatted,
                                        ImFormatted1, ImFormatted2,
                                        alpha, ratio, minWidth,
                                        nOuterFPIterations, nInnerFPIterations,
                                        nSORIterations);

        py::array_t<double> wx_array({h, w}, {w * sizeof(double), sizeof(double)}, vxFormatted.pData);
        py::array_t<double> wy_array({h, w}, {w * sizeof(double), sizeof(double)}, vyFormatted.pData);
        result.push_back(wx_array);
        result.push_back(wy_array);

        // clear c memory
        ImFormatted1.clear();
        ImFormatted2.clear();
        vxFormatted.clear();
        vyFormatted.clear();
        warpI2Formatted.clear();

        return result;
    }
};

PYBIND11_MODULE(py_brox_liu, m) {
    py::class_<BroxLiu>(m, "BroxLiu")
        .def(py::init())
        .def_readwrite("alpha", &BroxLiu::alpha)
        .def_readwrite("ratio", &BroxLiu::ratio)
        .def_readwrite("minWidth", &BroxLiu::minWidth)
        .def_readwrite("nOuterFPIterations", &BroxLiu::nOuterFPIterations)
        .def_readwrite("nInnerFPIterations", &BroxLiu::nInnerFPIterations)
        .def_readwrite("nSORIterations", &BroxLiu::nSORIterations)
        .def("matching_double", &BroxLiu::matching_double, R"mydelimiter(
        Generate flow bewween two double color images
        Args:
            im1 & im2: double hwc numpy array
    )mydelimiter");
}
