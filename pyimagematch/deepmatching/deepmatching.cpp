#include <pybind11/pybind11.h>
#include "deep_matching.h"
#include "std.h"
#include "image.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
class DeepMatching{
public:
    dm_params_t dm_params;
    DeepMatching(){
        set_default_dm_params(&dm_params);
    }

    void copyArrayToImage(py::array_t<float, py::array::c_style | py::array::forcecast> & array, image_t* im){
        float *array_data = array.data();
        int s0 = array.strides(0) / std::sizeof(float);
        int height = array.shape(0);
        int width = array.shape(1);
        int n=0;
        for(int j=0; j<height; j++)
            for(int i=0; i<width; i++, n++)
                im->data[i+j*im->stride] = array_data[i+j*s0];
    }

    py::array_t<float> matching(py::array_t<float, py::array::c_style | py::array::forcecast> im1, 
                  py::array_t<float, py::array::c_style | py::array::forcecast> im2){
        if ( im1.ndim() != 2 || im2.ndim() != 2)
            throw std::runtime_error("images should be 2-D NumPy array");
        
        image_t* im1_t = image_new(im1.shape()[1], im1.shape()[0]);
        image_t* im2_t = image_new(im2.shape()[1], im2.shape()[0]);

        copyArrayToImage(im1, im1_t);
        copyArrayToImage(im2, im2_t);

        float_image* corres = deep_matching(im1_t, im2_t, &dm_params, nullptr);

        auto result = py::array_t<float>(
            {corres->ty, corres->tx}, // shape
            {corres->tx, 4}, // C-style contiguous strides for double
            corres->pixels, // the data pointer
            );
        });

        free_image(corres);
        image_delete(im1_t);
        image_delete(im2_t);

        return result;
    }
}

PYBIND11_MODULE(deepmatching, m) {
    py::class_<desc_params_t>(m, "desc_params_t")
        .def_readwrite("presmooth_sigma", &desc_params_t::presmooth_sigma)
        .def_readwrite("mid_smoothing", &desc_params_t::mid_smoothing)
        .def_readwrite("post_smoothing", &desc_params_t::post_smoothing)
        .def_readwrite("hog_sigmoid", &desc_params_t::hog_sigmoid)
        .def_readwrite("ninth_dim", &desc_params_t::ninth_dim)
        .def_readwrite("norm_pixels", &desc_params_t::norm_pixels);

    py::class_<dm_params_t>(m, "dm_params_t")
        .def_readwrite("prior_img_downscale", &dm_params_t::prior_img_downscale)
        .def_readwrite("rot45", &dm_params_t::rot45)
        .def_readwrite("overlap", &dm_params_t::overlap)
        .def_readwrite("subsample_ref", &dm_params_t::subsample_ref)
        .def_readwrite("nlpow", &dm_params_t::nlpow)
        .def_readwrite("ngh_rad", &dm_params_t::ngh_rad)
        .def_readwrite("maxima_mode", &dm_params_t::maxima_mode)
        .def_readwrite("min_level", &dm_params_t::min_level)
        .def_readwrite("max_psize", &dm_params_t::max_psize)
        .def_readwrite("low_mem", &dm_params_t::low_mem)
        .def_readwrite("scoring_mode", &dm_params_t::scoring_mode)
        .def_readwrite("verbose", &dm_params_t::verbose)
        .def_readwrite("n_thread", &dm_params_t::n_thread)
        .def_readwrite("desc_params", &dm_params_t::desc_params);

    py::class_<DeepMatching>(m, "DeepMatching")
        .def(py::init())
        .def_readwrite("dm_params", &DeepMatching::dm_params);
        .def("matching", &DeepMatching::matching);
}



