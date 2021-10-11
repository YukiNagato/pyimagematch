cmake_minimum_required(VERSION 3.10)
project(pyoptflow)

if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

list(APPEND CMAKE_MODULE_PATH ${pyoptflow_SOURCE_DIR}/cmake_modules)
find_package(BLAS)
find_package(OpenMP)

add_subdirectory(3rdparty/pybind11)
add_subdirectory(pyimagematch/deepmatching)
