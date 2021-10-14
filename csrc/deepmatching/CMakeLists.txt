cmake_minimum_required(VERSION 3.10)
add_library(deepmatching_ SHARED
    src/conv.cpp
    src/deep_matching.cpp
    src/hog.cpp
    src/image.cpp
    src/pixel_desc.cpp
    src/maxfilter.cpp
    src/std.cpp
)
find_package(GFortran REQUIRED)
target_include_directories(deepmatching_ PRIVATE src)
target_link_libraries(deepmatching_ PRIVATE
    ${QUADMATH_LIBRARIES}
    ${GFORTRAN_LIBRARIES}
    ${BLAS_LIBRARIES}
)
set(DEEPMATCHING_OPENMP_FLAG OFF)
if(OpenMP_CXX_FOUND)
    target_link_libraries(deepmatching_ PRIVATE OpenMP::OpenMP_CXX)
    set(DEEPMATCHING_OPENMP_FLAG ON)
endif()
target_compile_options(deepmatching_ PRIVATE -fPIC -Wall -g -O3)
if(DEEPMATCHING_OPENMP_FLAG)
    target_compile_options(deepmatching_ PRIVATE -DUSE_OPENMP)
endif()

pybind11_add_module(py_deepmatching deepmatching.cpp)
target_include_directories(py_deepmatching PRIVATE src)
target_link_libraries(py_deepmatching PRIVATE deepmatching_)

