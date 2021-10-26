cmake_minimum_required(VERSION 3.10)
project(pyimagematch)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/pyimagematch/clib)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

list(APPEND CMAKE_MODULE_PATH ${pyimagematch_SOURCE_DIR}/cmake_modules)
find_package(BLAS)
find_package(OpenMP)

add_subdirectory(3rdparty/pybind11)
add_subdirectory(csrc/deepmatching)
add_subdirectory(csrc/epicflow)
add_subdirectory(csrc/brox_liu)

