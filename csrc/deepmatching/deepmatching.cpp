#include <pybind11/pybind11.h>
#include "deep_matching.h"
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
        const float *array_data = array.data();
        int s0 = array.strides(0) / sizeof(float);
        int height = array.shape(0);
        int width = array.shape(1);
        int n=0;
        for(int j=0; j<height; j++)
            for(int i=0; i<width; i++, n++)
                im->data[i+j*im->stride] = array_data[i+j*s0];
    }

    py::array_t<float> matching_float(py::array_t<float, py::array::c_style | py::array::forcecast> im1, 
                  py::array_t<float, py::array::c_style | py::array::forcecast> im2){
        if ( im1.ndim() != 2 || im2.ndim() != 2)
            throw std::runtime_error("images should be 2-D NumPy array");
        
        image_t* im1_t = image_new(im1.shape()[1], im1.shape()[0]);
        image_t* im2_t = image_new(im2.shape()[1], im2.shape()[0]);

        copyArrayToImage(im1, im1_t);
        copyArrayToImage(im2, im2_t);

        float_image* corres = deep_matching(im1_t, im2_t, &dm_params, nullptr);

        py::array_t<float> result(
            {corres->ty, corres->tx}, // shape
            {corres->tx*4, 4}, // C-style contiguous strides for double
            corres->pixels // the data pointer
            );

        free_image(corres);
        image_delete(im1_t);
        image_delete(im2_t);

        return result;
    }
};

PYBIND11_MODULE(py_deepmatching, m) {
    py::class_<desc_params_t>(m, "desc_params_t")
        .def_readwrite("presmooth_sigma", &desc_params_t::presmooth_sigma, "image pre-smoothing")
        .def_readwrite("mid_smoothing", &desc_params_t::mid_smoothing, "smoothing of oriented gradients (before sigmoid)")
        .def_readwrite("post_smoothing", &desc_params_t::post_smoothing, "smoothing of oriented gradients (after  sigmoid)")
        .def_readwrite("hog_sigmoid", &desc_params_t::hog_sigmoid, "sigmoid strength")
        .def_readwrite("ninth_dim", &desc_params_t::ninth_dim, "small constant for gradient-less area")
        .def_readwrite("norm_pixels", &desc_params_t::norm_pixels, "normalize pixels separately / 0: normalize atomic patches");

    py::class_<dm_params_t>(m, "dm_params_t")
        .def_readwrite("prior_img_downscale", &dm_params_t::prior_img_downscale, "downscale the image by 2^(this) prior to matching")
        .def_readwrite("rot45", &dm_params_t::rot45, "rotate second img by (45*rot45) prior to matching")
        .def_readwrite("overlap", &dm_params_t::overlap, "pyramid level at which patches starts to overlap (999 => no overlap at all)")
        .def_readwrite("subsample_ref", &dm_params_t::subsample_ref, "true if larger patches higher in the pyramid are not densely sampled")
        .def_readwrite("nlpow", &dm_params_t::nlpow, "non-linear power rectification")
        .def_readwrite("ngh_rad", &dm_params_t::ngh_rad, "neighborhood size in pixels => crop res_map (0 == infinite)")
        .def_readwrite("maxima_mode", &dm_params_t::maxima_mode, "1: standard / 0: from all top-level patches")
        .def_readwrite("min_level", &dm_params_t::min_level, "minimum pyramid level to retrieve maxima")
        .def_readwrite("max_psize", &dm_params_t::max_psize, "maximum patch size")
        .def_readwrite("low_mem", &dm_params_t::low_mem, "use less memory to retrieve the maxima (but approximate result)")
        .def_readwrite("scoring_mode", &dm_params_t::scoring_mode, "0: like ICCV paper / 1: improved scoring mode")
        .def_readwrite("verbose", &dm_params_t::verbose, "verbosity")
        .def_readwrite("n_thread", &dm_params_t::n_thread, "parallelization on several cores, when possible")
        .def_readwrite("desc_params", &dm_params_t::desc_params, "params for descriptors");

    py::class_<DeepMatching>(m, "DeepMatching")
        .def(py::init())
        .def_readwrite("dm_params", &DeepMatching::dm_params, "params for deepmatching")
        .def("matching_float", &DeepMatching::matching_float, R"mydelimiter(
        Match two float and single channel images
        Args:
        im1 & im2: float32 numpy array
    )mydelimiter");
}



