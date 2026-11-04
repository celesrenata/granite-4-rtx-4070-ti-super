// GPU weight packing: checkpoint -> GpuWeightsLayout (forward_gpu.h).
//
// Host-only. The device forward consumes one packed blob: BF16 embeddings
// and norm weights, FP8 e4m3 GEMM weights with one per-tensor dequant scale
// (max_abs / 448, the P1 storage decision). Two producers with identical
// output:
//
//   * pack_gpu_model_from_weights - re-packs an already loaded ModelWeights
//     (float). Single-threaded; it is the reference the fast path is tested
//     against.
//   * pack_gpu_model_from_checkpoint - streams the safetensors shards
//     straight into the packed blob, tensor by tensor, with the max-abs scan
//     and the FP8 encode split across all hardware threads. It never builds
//     the ~34 GB float copy, so the GPU paths skip load_weights entirely.
//
// Both produce bit-identical blobs and scales: every element goes through
// the same float value (BF16 decode, or fp8 * scale for an FP8 export), the
// same double max-abs, and the same fp8_e4m3_encode call.
#ifndef GRANITE4_GPU_PACK_H
#define GRANITE4_GPU_PACK_H

#include "forward_gpu.h"
#include "model_config.h"
#include "weights.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace granite4 {

struct GpuModel {
  GpuWeightsLayout lay{};
  std::unique_ptr<std::uint8_t[]> weights;  // [bytes] packed blob (not zeroed)
  std::size_t bytes = 0;
  std::vector<float> scales;  // [lay.num_scales()]

  void allocate(const GpuWeightsLayout& l) {
    lay = l;
    bytes = l.total_bytes();
    weights.reset(new std::uint8_t[bytes]);
    scales.clear();
    scales.reserve(l.num_scales());
  }
};

GpuWeightsLayout gpu_layout_for(const ModelConfig& cfg);

bool pack_gpu_model_from_weights(const ModelWeights& w, GpuModel* out,
                                 std::string* error);

// `threads` <= 0 selects std::thread::hardware_concurrency().
bool pack_gpu_model_from_checkpoint(const std::string& dir,
                                    const ModelConfig& cfg, GpuModel* out,
                                    std::string* error, int threads = 0);


}  // namespace granite4

#endif  // GRANITE4_GPU_PACK_H
