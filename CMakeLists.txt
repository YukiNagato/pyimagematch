cmake_minimum_required(VERSION 3.10)
project(pyimagematch)
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

if(WITH_CUDA)
  add_compile_definitions(PIM_WITH_CUDA)
  # enable_language(CUDA)
  find_package(CUDA REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS})
endif(WITH_CUDA)

set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/pyimagematch/clib)

set(CMAKE_INSTALL_RPATH "$ORIGIN")
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND CMAKE_MODULE_PATH ${pyimagematch_SOURCE_DIR}/cmake_modules)
find_package(BLAS)
find_package(OpenMP)

include_directories(csrc)

add_subdirectory(3rdparty/pybind11)
add_subdirectory(csrc/core)
# add_subdirectory(csrc/deepmatching)
# add_subdirectory(csrc/epicflow)
# add_subdirectory(csrc/brox_liu)

if(WITH_TEST)
  enable_testing()
  add_subdirectory(3rdparty/Catch2)
  list(APPEND CMAKE_MODULE_PATH ${pyimagematch_SOURCE_DIR}/3rdparty/Catch2/contrib)
  add_subdirectory(csrc/tests)
endif(WITH_TEST)

