cmake_minimum_required(VERSION 3.10)
add_library(epicflow_ SHARED
    src/epic.cpp
    src/epic_aux.cpp
    src/solver.cpp
    src/image.cpp
    src/variational_aux.cpp
    src/variational.cpp
)

find_package(LAPACK)
target_include_directories(epicflow_ PRIVATE src)
target_link_libraries(epicflow_ PRIVATE
    ${LAPACK_LIBRARIES}
)
target_compile_options(epicflow_ PRIVATE -DFINTEGER=int -fPIC -Wall -g -O3 -msse4)

pybind11_add_module(py_epicflow epicflow.cpp)
target_include_directories(py_epicflow PRIVATE src)
target_link_libraries(py_epicflow PRIVATE epicflow_)

install(TARGETS py_epicflow DESTINATION .)
install(TARGETS epicflow_ DESTINATION .)
