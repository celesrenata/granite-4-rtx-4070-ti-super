# Granite 4.2 8B on one RTX 4070 Ti Super - build root (scaffold).
#
# The host side (model constants, config validation, C ABI) builds without a
# CUDA toolkit. The CUDA objects target sm_89 explicitly and require nvcc;
# configure with -DGRANITE_ENABLE_CUDA=OFF for a host-only build.
cmake_minimum_required(VERSION 3.28)
project(granite4_rtx VERSION 0.0.1
        DESCRIPTION "Single-purpose Granite 4.2 8B engine for one RTX 4070 Ti Super (scaffold)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(GRANITE_ENABLE_CUDA "Compile the CUDA engine objects (targets sm_89; requires the CUDA toolkit)" ON)

enable_testing()

add_subdirectory(engine)
