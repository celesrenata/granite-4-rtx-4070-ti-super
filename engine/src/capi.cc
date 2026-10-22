// C ABI implementation. Phase 0: metadata and status placeholders only.
#include "capi.h"

#include <cstdio>

#include "config_validation.h"
#include "model_config.h"
#include "safetensors.h"

// Optional CUDA device probe. Compiled out entirely in host-only builds,
// so the host library has no CUDA runtime dependency.
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
#include <cuda_runtime.h>
static bool cuda_device_present() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}
static const char* cuda_probe_message() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) return "cudaGetDeviceCount failed";
  if (count == 0) return "no CUDA device";
  int major = 0, minor = 0;
  if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess)
    return "cannot query compute capability";
  if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess)
    return "cannot query compute capability";
  if (major * 10 + minor != granite4::kCudaArchMajor * 10 + granite4::kCudaArchMinor) {
    static char buf[64];
    std::snprintf(buf, sizeof buf, "device is sm_%d%d, engine targets %s", major, minor,
                  granite4::kGpuTarget);
    return buf;
  }
  return "";
}
#endif

namespace {
const char* const kStatusTexts[] = {"ok", "unimplemented", "no-device", "unavailable"};
}  // namespace

extern "C" {

void granite4_get_model_info(granite4_model_info* out) {
  if (out == nullptr) return;
  const auto& c = granite4::kModelConfig;
  out->model_id = granite4::kModelId;
  out->num_layers = static_cast<long>(c.num_layers);
  out->hidden_size = static_cast<long>(c.hidden_size);
  out->intermediate_size = static_cast<long>(c.intermediate_size);
  out->num_attention_heads = static_cast<long>(c.num_attention_heads);
  out->num_key_value_heads = static_cast<long>(c.num_key_value_heads);
  out->head_dim = static_cast<long>(c.head_dim);
  out->vocab_size = static_cast<long>(c.vocab_size);
  out->max_context_length = static_cast<long>(c.max_context_length);
  out->gpu_target = granite4::kGpuTarget;
}

void granite4_status(granite4_status_info* out) {
  if (out == nullptr) return;
  // Phase 0 scaffold: inference is not implemented; report it, don't fake it.
  out->status = granite4_unimplemented;
  out->status_text = kStatusTexts[static_cast<int>(out->status)];
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
  out->cuda_device_present = cuda_device_present() ? 1 : 0;
  out->message = cuda_probe_message();
#else
  out->cuda_device_present = 0;
  out->message = "";
#endif
}

void granite4_validate_safetensors(const char* path,
                                   granite4_safetensors_result* out) {
  if (out == nullptr) return;
  out->ok = 0;
  out->tensor_count = 0;
  out->diagnostic[0] = '\0';
  if (path == nullptr) {
    std::snprintf(out->diagnostic, sizeof(out->diagnostic), "invalid-args: path is NULL");
    return;
  }
  granite4::TensorManifest manifest;
  std::string error;
  bool ok = granite4::parse_safetensors_file(path, &manifest, &error);
  if (ok) {
    // Re-check every invariant as part of the const validation API.
    ok = manifest.verify();
    if (!ok) error = manifest.describe_issues();
  }
  out->ok = ok ? 1 : 0;
  out->tensor_count = ok ? static_cast<long>(manifest.tensors.size()) : 0;
  std::snprintf(out->diagnostic, sizeof(out->diagnostic), "%s",
                ok ? "" : error.c_str());
}

}  // extern "C"
