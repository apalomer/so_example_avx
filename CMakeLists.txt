#####################################################################################
#                                                                                   #
#                      CMake Configuration                                          #
#                                                                                   #
#####################################################################################

# Minimum version required
cmake_minimum_required(VERSION 2.8)

# Project Name
project(so_examples_avx)

set(CMAKE_CXX_COMPILER clang++)

add_compile_options(-std=c++17)
add_compile_options(-Wall)
add_compile_options(-Wextra)
add_compile_options(-O3)
add_compile_options(-fno-tree-vectorize)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(test_avx2_sqrt src/test_avx2_sqrt.cpp)
target_link_libraries(test_avx2_sqrt
    opencv_core
    opencv_imgcodecs
)
target_compile_options(test_avx2_sqrt PRIVATE -mavx2 -march=native)
