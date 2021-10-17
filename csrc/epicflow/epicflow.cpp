#include <pybind11/pybind11.h>
#include "epic.h"
#include "image.h"
#include "variational.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;
using namespace epicflow;
class Epicflow{
public:
    epic_params_t epic_params;
    variational_params_t flow_params;
    Epicflow(){
        epic_params_default(&epic_params);
        variational_params_default(&flow_params);
    }

    void copyArrayToImage(py::array_t<float, py::array::c_style | py::array::forcecast> & array, image_t* im){
        const float *array_data = array.data();
        int s0 = array.strides(0) / sizeof(float);
        int height = array.shape(0);
        int width = array.shape(1);
        int n=0;
        for(int j=0; j<height; j++)
            for(int i=0; i<width; i++, n++)
                im->data[i+j*im->stride] = array_data[i+j*s0];
    }

    void copyArrayToColorImage(py::array_t<float, py::array::c_style | py::array::forcecast> & array, color_image_t* im){
        auto r = array.unchecked<3>();
        int height = array.shape(0);
        int width = array.shape(1);
        int channel = array.shape(2);
        for(int j=0; j<height; j++)
            for(int i=0; i<width; i++){
                im->c1[j*im->stride+i] = r(j, i, 0);
                im->c2[j*im->stride+i] = r(j, i, 1);;
                im->c3[j*im->stride+i] = r(j, i, 2);;
            }
    }

    void copyArrayToFloatImage(py::array_t<float, py::array::c_style | py::array::forcecast> & array, float_image* im){
        auto r = array.unchecked<2>();
        int height = array.shape(0);
        int width = array.shape(1); 
        float* data_ptr = im->pixels;
        for(int j=0; j<height; j++)
            for(int i=0; i<width; i++){
                *data_ptr = r(j, i);
                data_ptr++;
            }
    }

    std::vector<py::array_t<float>> matching_float(py::array_t<float, py::array::c_style | py::array::forcecast> im1, 
                  py::array_t<float, py::array::c_style | py::array::forcecast> im2, 
                  py::array_t<float, py::array::c_style | py::array::forcecast> edges,
                  py::array_t<float, py::array::c_style | py::array::forcecast> matches){
        
        if ( im1.ndim() != 3 || im2.ndim() != 3 || im1.shape(2) !=3 || im2.shape(2) !=3)
            throw std::runtime_error("images should be H*W*3 NumPy array");
        
        if (edges.ndim() != 2){
            throw std::runtime_error("edges should be 2 dim NumPy array");
        }

        if (matches.ndim() != 2){
            throw std::runtime_error("matches should be 2 dim NumPy array");
        }     

        std::vector<py::array_t<float>> result;

        color_image_t* im1_t = color_image_new(im1.shape(1), im1.shape(0));
        color_image_t* im2_t = color_image_new(im2.shape(1), im2.shape(0));
        float_image edges_t = empty_image(float, edges.shape(1), edges.shape(0));
        float_image matches_t = empty_image(float, matches.shape(1), matches.shape(0));
        image_t *wx = image_new(im1_t->width, im1_t->height), *wy = image_new(im1_t->width, im1_t->height);

        copyArrayToColorImage(im1, im1_t);
        copyArrayToColorImage(im2, im2_t);
        copyArrayToFloatImage(edges, &edges_t);
        copyArrayToFloatImage(matches, &matches_t);

        color_image_t *imlab = rgb_to_lab(im1_t);
        epic(wx, wy, imlab, &matches_t, &edges_t, &epic_params, 1);

        // energy minimization
        variational(wx, wy, im1_t, im2_t, &flow_params);
        
        py::array_t<float> wx_array({wx->height, wx->width}, {wx->stride*4, 4}, wx->data);
        py::array_t<float> wy_array({wy->height, wy->width}, {wy->stride*4, 4}, wy->data);
        result.push_back(wx_array);
        result.push_back(wy_array);

        color_image_delete(im1_t);
        color_image_delete(imlab);
        color_image_delete(im2_t);
        free(matches_t.pixels);
        free(edges_t.pixels);
        image_delete(wx);
        image_delete(wy);

        return result;
    }
};

PYBIND11_MODULE(py_epicflow, m) {
    py::class_<epic_params_t>(m, "epic_params_t")
        .def_readwrite("saliency_th", &epic_params_t::saliency_th, "matches coming from pixels with a saliency below this threshold are removed before interpolation")
        .def_readwrite("pref_nn", &epic_params_t::pref_nn, "number of neighbors for consistent checking)")
        .def_readwrite("pref_th", &epic_params_t::pref_th, "threshold for the first prefiltering step")
        .def_readwrite("coef_kernel", &epic_params_t::nn, "coefficient in the sigmoid of the interpolation kernel")
        .def_readwrite("euc", &epic_params_t::euc, "constant added to the edge cost")
        .def_readwrite("verbose", &epic_params_t::verbose, "verbose mode")
        .def("set_method", &epic_params_t::set_method, "method for interpolation: la (locally-weighted affine) or nw (nadaraya-watson)");

    py::class_<variational_params_t>(m, "variational_params_t")
        .def_readwrite("alpha", &variational_params_t::alpha, "smoothness weight")
        .def_readwrite("gamma", &variational_params_t::gamma, "gradient constancy assumption weight")
        .def_readwrite("delta", &variational_params_t::delta, "color constancy assumption weight")
        .def_readwrite("sigma", &variational_params_t::sigma, "presmoothing of the images")
        .def_readwrite("niter_outer", &variational_params_t::niter_outer, "number of outer fixed point iterations")
        .def_readwrite("niter_inner", &variational_params_t::niter_inner, "number of inner fixed point iterations")
        .def_readwrite("niter_solver", &variational_params_t::niter_solver, "number of solver iterations")
        .def_readwrite("sor_omega", &variational_params_t::sor_omega, "omega parameter of sor method");

    py::class_<Epicflow>(m, "Epicflow")
        .def(py::init())
        .def_readwrite("epic_params", &Epicflow::epic_params, "params for Epicflow")
        .def_readwrite("flow_params", &Epicflow::flow_params, "params for variational")
        .def("matching_float", &Epicflow::matching_float, R"mydelimiter(
        Generate flow bewween two float color images using sparse matches and edge image
        Args:
            im1 & im2: float32 numpy array
            edges: single channel float edge image
            matches: n*4 float array
    )mydelimiter");
}
