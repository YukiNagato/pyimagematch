#include <pybind11/pybind11.h>
#include "deep_matching.h"
#include "std.h"
#include "image.h"
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


class DeepMatching{
public:
    dm_params_t dm_params;
    DeepMatching(){
        set_default_dm_params(&dm_params);
    }

    void matching(py::array_t<float, py::array::c_style | py::array::forcecast> im1, 
                  py::array_t<float, py::array::c_style | py::array::forcecast> im2){
        
    }

}



