add_executable(tests 
            test_main.cpp
            test_tensor.cpp
            test_ops.cpp)

find_package(OpenCV REQUIRED)

target_include_directories(tests PRIVATE
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(tests 
    pim_core 
    Catch2::Catch2
    ${OpenCV_LIBS})   
    
include(CTest)
include(Catch)
catch_discover_tests(tests)