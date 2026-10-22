# Engine library: host side (constants, config validation, C ABI) + CUDA kernels.
#
# The host side compiles with any C++20 compiler, no CUDA required. The CUDA
# objects target sm_89 (RTX 4070 Ti Super, compute capability 8.9) and need
# the CUDA toolkit; disable with -DGRANITE_ENABLE_CUDA=OFF for a host-only
# build (used by CI and by the Python tests).
set(GRANITE_ENGINE_SOURCES
  src/model_config.h
  src/config_validation.cc
  src/safetensors.h
  src/safetensors.cc
  src/capi.cc)

if(GRANITE_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Explicit arch: sm_89 only. No "native" / no auto-detect, so a build is
  # never silently retargeted at another GPU.
  set(CMAKE_CUDA_ARCHITECTURES 89)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")
  find_package(CUDAToolkit REQUIRED)
  list(APPEND GRANITE_ENGINE_SOURCES src/kernels.cu)
endif()

add_library(granite4 SHARED ${GRANITE_ENGINE_SOURCES})
target_include_directories(granite4 PUBLIC src)

if(GRANITE_ENABLE_CUDA)
  target_link_libraries(granite4 PRIVATE CUDA::cudart)
endif()

# Host-only tests: no CUDA required.
add_executable(granite4-config-test tests/test_config_validation.cc)
target_link_libraries(granite4-config-test PRIVATE granite4)
add_test(NAME config_validation COMMAND granite4-config-test)

add_executable(granite4-safetensors-test tests/test_safetensors.cc)
target_link_libraries(granite4-safetensors-test PRIVATE granite4)
add_test(NAME safetensors_metadata COMMAND granite4-safetensors-test)
