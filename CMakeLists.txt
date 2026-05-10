cmake_minimum_required(VERSION 3.16)

# Project setup
project(CudaBenchmark LANGUAGES CXX CUDA)

# Editor settings
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_definitions(-DHAS_CUDA)

# C++ compiler settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# NVCC compiler settings
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 61;72;75;86;87;89)

# Build settings
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(OpenCV REQUIRED)

message(STATUS "Found OpenCV ${OpenCV_VERSION} in ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Found CUDA ${CUDA_VERSION} in ${CUDA_INCLUDE_DIRS}")

# Compile source files
file(
    GLOB_RECURSE
    SOURCES
    Main/*
    CudaUtils/*
)
add_executable(${PROJECT_NAME} ${SOURCES})

# Include external header files
target_include_directories(${PROJECT_NAME} PRIVATE CudaUtils ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

# Link external libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES} cufft)

# Enable separable compilation for CUDA translation units
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_CUDA_SEPARABLE_COMPILATION ON)
