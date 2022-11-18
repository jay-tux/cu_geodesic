cmake_minimum_required(VERSION 3.23)
project(cudijkstra
        VERSION 0.0.1
        LANGUAGES CUDA CXX
        DESCRIPTION "Dijkstra from multiple points")

set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "-g -G")
set(CMAKE_CXX_FLAGS "-g -Wall -Wextra -pedantic")

add_executable(${PROJECT_NAME} main.cu src/io.cpp src/kernel.cu src/cuda_error.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/inc)
