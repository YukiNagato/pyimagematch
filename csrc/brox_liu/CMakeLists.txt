cmake_minimum_required(VERSION 3.10)
add_library(brox_liu_ SHARED
    src/GaussianPyramid.cpp
    src/OpticalFlow.cpp
    src/Stochastic.cpp)

target_include_directories(brox_liu_ PRIVATE src)
target_compile_options(brox_liu_ PRIVATE -g -O3)

pybind11_add_module(py_brox_liu brox_liu.cpp)
target_include_directories(py_brox_liu PRIVATE src)
target_link_libraries(py_brox_liu PRIVATE brox_liu_)

install(TARGETS py_brox_liu DESTINATION .)
install(TARGETS brox_liu_ DESTINATION .)
