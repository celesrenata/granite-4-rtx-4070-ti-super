# Engine library: host side (constants, config validation, C ABI) + CUDA kernels.
#
# The host side compiles with any C++20 compiler, no CUDA required. The CUDA
# objects target sm_89 (RTX 4070 Ti Super, compute capability 8.9) and need
# the CUDA toolkit; disable with -DGRANITE_ENABLE_CUDA=OFF for a host-only
# build (used by CI and by the Python tests).
set(GRANITE_ENGINE_SOURCES
  src/model_config.h
  src/json.h
  src/json.cc
  src/config_validation.cc
  src/safetensors.h
  src/safetensors.cc
  src/capi.cc
  src/checkpoint.h
  src/fp8_convert.cc
  src/checkpoint_sha256.cc
  src/checkpoint_main.cc
  src/kv_pool.h
  src/kv_pool.cc
  src/draft_lookup.h
  src/draft_lookup.cc
  src/speculative_verify.h
  src/speculative_verify.cc
  src/weights.h
  src/weights.cc)

if(GRANITE_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Explicit arch: sm_89 only. No "native" / no auto-detect, so a build is
  # never silently retargeted at another GPU.
  set(CMAKE_CUDA_ARCHITECTURES 89)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --expt-relaxed-constexpr")
  find_package(CUDAToolkit REQUIRED)
  list(APPEND GRANITE_ENGINE_SOURCES src/kernels.cu src/kernels_phase2.cu
       src/kernels_phase3.cu src/forward_gpu.cu src/verify_gemv.cu)
endif()

add_library(granite4 SHARED ${GRANITE_ENGINE_SOURCES})
target_include_directories(granite4 PUBLIC src)

# A static library of the same sources. The shared library is the primary
# artifact (used by the C ABI and tests); the static one exists so the
# checkpoint CLI can be linked self-contained - its in-$out RPATH must not
# point at a build tree that no longer exists after install.
add_library(granite4_static STATIC ${GRANITE_ENGINE_SOURCES})
target_include_directories(granite4_static PUBLIC src)

if(GRANITE_ENABLE_CUDA)
  target_link_libraries(granite4 PRIVATE CUDA::cudart)
  target_link_libraries(granite4_static PRIVATE CUDA::cudart)
  # GRANITE_CUDA is the C++-side switch (see capi.cc): it enables the CUDA
  # device probe there and the GPU forward path in infer_main.cc. The .cu
  # files self-define it; .cc files get it here.
  target_compile_definitions(granite4 PRIVATE GRANITE_CUDA=1)
  target_compile_definitions(granite4_static PUBLIC GRANITE_CUDA=1)
endif()

# Host-only tests: no CUDA required.
add_executable(granite4-config-test tests/test_config_validation.cc)
target_link_libraries(granite4-config-test PRIVATE granite4)
add_test(NAME config_validation COMMAND granite4-config-test)
add_executable(granite4-safetensors-test tests/test_safetensors.cc)
target_link_libraries(granite4-safetensors-test PRIVATE granite4)
add_test(NAME safetensors_metadata COMMAND granite4-safetensors-test)

add_executable(granite4-json-test tests/test_json.cc)
target_link_libraries(granite4-json-test PRIVATE granite4)
add_test(NAME json_parser COMMAND granite4-json-test)

add_executable(granite4-fp8-test tests/test_fp8_convert.cc)
target_link_libraries(granite4-fp8-test PRIVATE granite4)
add_test(NAME fp8_e4m3 COMMAND granite4-fp8-test)

add_executable(granite4-checkpoint-test tests/test_checkpoint.cc)
target_link_libraries(granite4-checkpoint-test PRIVATE granite4)
add_test(NAME checkpoint_verify COMMAND granite4-checkpoint-test)

# Phase 2 (Kernels): host reference math - always buildable, no CUDA.
add_executable(granite4-kernel-math-test tests/test_kernel_math.cc)
target_link_libraries(granite4-kernel-math-test PRIVATE granite4_static)
add_test(NAME kernel_math COMMAND granite4-kernel-math-test)

# Phase 3 (Integration): paged KV memory manager + prefix cache. Host-only,
# no CUDA; exercises allocation, prefix reuse (ref counting), and LRU
# eviction against a byte budget derived from docs/memory.md.
add_executable(granite4-kv-pool-test tests/test_kv_pool.cc)
target_link_libraries(granite4-kv-pool-test PRIVATE granite4_static)
add_test(NAME kv_pool COMMAND granite4-kv-pool-test)

# Speculative drafters (host-only): prompt-lookup proposals are verified by
# the target model, so they carry no weights and no GPU dependency.
add_executable(granite4-draft-lookup-test tests/test_draft_lookup.cc)
target_link_libraries(granite4-draft-lookup-test PRIVATE granite4_static)
add_test(NAME draft_lookup COMMAND granite4-draft-lookup-test)

# Greedy speculative accept/reject gate (host-only, no weights): the token
# decision the inline verification loop makes, factored into a unit-tested
# contract so the end-to-end path can delegate to it.
add_executable(granite4-speculative-verify-test tests/test_speculative_verify.cc)
target_link_libraries(granite4-speculative-verify-test PRIVATE granite4_static)
add_test(NAME speculative_verify COMMAND granite4-speculative-verify-test)

if(GRANITE_ENABLE_CUDA)
  add_executable(granite4-kernel-test tests/test_kernels.cu)
  # Link the *shared* CUDA runtime so the installed binary carries the CUDA
  # runpath (the shared libcudart points at the host driver lib directory,
  # e.g. /run/opengl-driver/lib on NixOS). Without it, nvcc's default
  # static cudart leaves the binary without a CUDA/driver runpath, so on the
  # GPU host cudaGetDeviceCount() cannot resolve libcuda.so.1 and the test
  # fails with "no CUDA device" even though the GPU is present.
  target_link_libraries(granite4-kernel-test PRIVATE granite4_static CUDA::cudart)
  add_test(NAME kernels_device COMMAND granite4-kernel-test)

  # Micro-benchmark / roofline baseline (P2 exit criterion): measures the
  # card's real bandwidth rather than assuming the spec.
  add_executable(granite4-kernel-bench src/kernels_bench.cu)
  target_link_libraries(granite4-kernel-bench PRIVATE CUDA::cudart)

  # P4: GEMV micro-benchmark - achieved bandwidth of the decode GEMM at the
  # real model shapes, isolates the per-token weight-streaming cost.
  add_executable(granite4-gemv-bench src/gemv_bench.cu)
  target_link_libraries(granite4-gemv-bench PRIVATE CUDA::cudart)

  # Experimental speculative-decoding verify path. Kept out of granite4 and
  # production decode until its batched traffic wins on the target workload.
  add_executable(granite4-verify-gemv-test
                 src/verify_gemv.cu tests/test_verify_gemv.cu)
  target_link_libraries(granite4-verify-gemv-test PRIVATE CUDA::cudart)
  add_test(NAME verify_gemv_device COMMAND granite4-verify-gemv-test)
  add_executable(granite4-verify-gemv-bench
                 src/verify_gemv.cu tests/verify_gemv_bench.cu)
  target_link_libraries(granite4-verify-gemv-bench PRIVATE CUDA::cudart)
endif()

# Phase 1 checkpoint CLI (host-only: verify + FP8 export). Linked statically
# so the installed binary does not depend on the in-tree build directory.
add_executable(granite4-checkpoint src/checkpoint_cli.cc)
target_link_libraries(granite4-checkpoint PRIVATE granite4_static)

# Phase 3 inference CLI (host-only forward pass + generate). Linked
# statically; loads original BF16 or FP8-export checkpoints.
add_executable(granite4-infer src/infer_main.cc)
target_link_libraries(granite4-infer PRIVATE granite4_static)

# Phase 3 weight loading (BF16 + FP8 dequant) - host-only.
add_executable(granite4-weights-test tests/test_weights.cc)
target_link_libraries(granite4-weights-test PRIVATE granite4_static)
add_test(NAME weights COMMAND granite4-weights-test)
