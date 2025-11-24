cmake_minimum_required(VERSION 3.18)
project(llama2_q4 LANGUAGES CXX CUDA)

# Set standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 120)
endif()

# Find NCCL
find_path(NCCL_INCLUDE_DIR nccl.h 
    PATHS ${NCCL_ROOT_DIR}/include)

find_library(NCCL_LIBRARY NAMES nccl
    PATHS ${NCCL_ROOT_DIR}/lib)

if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
else()
    message(WARNING "NCCL not found. Multi-GPU support will not be available.")
endif()

# Executables
add_executable(weight_packer weight_packer.cpp)
add_executable(llama2_q4 llama2_q4.cu)

# Link NCCL
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    target_include_directories(llama2_q4 PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(llama2_q4 PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(llama2_q4 PRIVATE NCCL_ENABLED)
endif()
