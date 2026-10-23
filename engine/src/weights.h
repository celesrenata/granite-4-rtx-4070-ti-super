// Phase 3 (Integration): model weight loading for inference.
//
// Host-only, stdlib only (plus the in-tree safetensors / config / FP8
// helpers). Reads a checkpoint directory - the original BF16 checkpoint or
// an FP8 e4m3 export produced by `granite4-checkpoint export` - and produces
// the weight layout `forward_logits_f32` (kernel_math.h) consumes:
//
//   embed      [vocab][hidden]
//   layers     [num_layers][layer_bytes]   // per-layer block, safetensors order
//   final_norm [hidden]
//   lm_head    [vocab][hidden]
//
// All values are dequantized/decoded to float on load, so the forward pass
// (and the GPU path) consume plain floats. Two layouts are supported and are
// distinguished by the weight tensor's dtype:
//   * BF16 (the original checkpoint): every tensor is BF16, no scales.
//   * F8_E4M3 (the FP8 export): every weight tensor is FP8 and has a
//     "<name>__scale" F32 sibling - per-tensor (scale shape [1]) or
//     per-channel (scale shape [row], applied per output row). Dequantizing
//     an FP8 tensor is fp8_to_float * scale.
//
// The load also runs the strict config check (load_and_check_config), so a
// checkpoint that deviates from the spec is rejected before any weight is
// read, and the model dimensions come from the verified config.

#ifndef GRANITE4_WEIGHTS_H
#define GRANITE4_WEIGHTS_H

#include "model_config.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace granite4 {

struct ModelWeights {
  ModelConfig model;
  // Weight layout consumed by forward_logits_f32 (see kernel_math.h for the
  // exact per-layer block offsets).
  std::vector<float> embed;     // [vocab][hidden]
  std::vector<float> layers;    // [num_layers][layer_bytes]
  std::vector<float> final_norm;  // [hidden]
  std::vector<float> lm_head;  // [vocab][hidden]

  // Per-layer block size in floats (= the safetensors tensor-order block).
  std::size_t layer_bytes = 0;
  // Bytes loaded from disk (sum of weight + scale tensors actually read).
  std::uint64_t bytes_read = 0;

  // The per-layer block layout (offsets from the block head), shared with
  // forward_logits_f32 and the tests so all three agree byte-for-byte.
  std::size_t off_q = 0, off_k = 0, off_v = 0, off_o = 0, off_gate = 0,
      off_up = 0, off_down = 0;
};

// Load all weights from `dir` into `out`. `dir` must contain the checkpoint's
// config.json and safetensors shards (original BF16 or FP8 export). Returns
// false (with `error` set) on any failure: config mismatch, missing tensor,
// dtype not supported, or read error. When `enforce_spec` is true (default,
// the production path) the config must match the Granite 4.2 8B spec exactly;
// with it false, any structurally-consistent config is accepted (used by the
// small-model unit tests).
bool load_weights(const std::string& dir, ModelWeights* out, std::string* error,
                  bool enforce_spec = true);

}  // namespace granite4

#endif  // GRANITE4_WEIGHTS_H
