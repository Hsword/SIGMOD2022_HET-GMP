cmake_minimum_required(VERSION 3.18)

project(hetuCTR C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CMAKE_CUDA_FLAGS "-O3 -w")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

set(CUDAToolkit_ROOT /usr/local/cuda)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

set(NCCL_ROOT $ENV{CONDA_PREFIX})
find_package(NCCL 2.8 REQUIRED)

include(FetchContent)

# FetchContent_Declare(thrust URL https://github.com/NVIDIA/thrust/archive/refs/tags/1.12.0.tar.gz)
# message(STATUS "Preparing thrust ...")
# FetchContent_MakeAvailable(thrust)

# FetchContent_Declare(cub URL https://github.com/NVlabs/cub/archive/refs/tags/1.12.0.tar.gz)
# message(STATUS "Preparing cub ...")
# FetchContent_MakeAvailable(cub)

set(cub_SOURCE_DIR $ENV{CONDA_PREFIX}/include)
set(thrust_SOURCE_DIR $ENV{CONDA_PREFIX}/include)

find_package(ZMQ 4 REQUIRED)
find_package(pybind11 2.6 CONFIG)

add_subdirectory(${PROJECT_SOURCE_DIR}/csrc)

# add_subdirectory(${PROJECT_SOURCE_DIR}/csrc/partition)

enable_testing()
