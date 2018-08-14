#####################################################################################
#                                                                                   #
#                      CMake Configuration                                          #
#                                                                                   #
#####################################################################################

# Minimum version required
cmake_minimum_required(VERSION 2.8)

# Project Name
project(so_examples_avx)

add_compile_options(-std=c++17)
add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-O3)
add_compile_options(-fno-tree-vectorize)
add_compile_options(-mavx2)
add_compile_options(-mavx512f)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_avx2 src/test_avx2.cpp)
target_link_libraries(test_avx2
    opencv_core
    opencv_imgcodecs
)

add_executable(test_avx2_sqrt src/test_avx2_sqrt.cpp)
target_link_libraries(test_avx2_sqrt
    opencv_core
    opencv_imgcodecs
)

add_executable(test_avx512 src/test_avx512.cpp)
target_link_libraries(test_avx512
    opencv_core
    opencv_imgcodecs
)
