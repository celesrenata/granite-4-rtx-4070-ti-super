// CUDA kernel scaffold for Granite 4.2 8B on sm_89.
//
// Phase 0: no kernels exist yet. This file exists so the CUDA build path is
// exercised (compile, arch selection, ABI linkage) before any real work.
// Each future kernel lands in its own file; this one stays a probe.
//
// The device probe is a no-op kernel; running it is the minimum sanity
// check that the arch target actually executes on the card.
#include <cstdint>

#define GRANITE_CUDA 1
#include "model_config.h"

namespace granite4 {

// Returns 1 on sm_89 devices, 0 anywhere else. Used by tests that run on a
// 4070 Ti Super to confirm the binary was actually compiled for sm_89 and
// is executing on the right silicon.
__global__ void arch_probe(int* out) {
#if __CUDA_ARCH__ == 890
  *out = 1;
#else
  *out = 0;
#endif
}

}  // namespace granite4

// The C ABI does not expose the probe yet; keep the symbol live so the CUDA
// objects are pulled into the library when GRANITE_ENABLE_CUDA=ON.
extern "C" void granite4_cuda_arch_probe(int* out) {
  granite4::arch_probe<<<1, 1>>>(out);
}
