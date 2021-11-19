add_library(
    pim_core SHARED
    src/tensor_data.cpp
)

# target_include_directories(pim_core PRIVATE 
#     ..
#     ${CUDA_INCLUDE_DIRS}
# )

target_link_libraries(pim_core PRIVATE
    ${CUDA_LIBRARIES}
)
