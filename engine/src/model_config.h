// Granite 4.2 8B - model constants (single source of truth, C++ side).
//
// These are the target model's spec. P1 (weight loading) must re-check them
// against the checkpoint's config.json and fail fast on any mismatch; nothing
// in this scaffold loads weights or runs inference.
//
// Keep in sync with frontend/config.py (checked by tests).
#ifndef GRANITE4_MODEL_CONFIG_H
#define GRANITE4_MODEL_CONFIG_H

#include <cstddef>

namespace granite4 {

struct ModelConfig {
  // Scaffold: all fields are positive, so == is a plain field-wise compare.
  bool operator==(const ModelConfig&) const = default;
  std::size_t num_layers = 40;
  std::size_t hidden_size = 4096;
  std::size_t intermediate_size = 12800;
  std::size_t num_attention_heads = 32;
  std::size_t num_key_value_heads = 8;
  std::size_t head_dim = 128;
  std::size_t vocab_size = 100352;
  std::size_t max_context_length = 131072;
};

inline constexpr ModelConfig kModelConfig{};

inline constexpr char kModelId[] = "granite-4.2-8b";
inline constexpr int kCudaArchMajor = 8;   // compute capability 8.9
inline constexpr int kCudaArchMinor = 9;  // RTX 4070 Ti Super (Ada)
inline constexpr char kGpuTarget[] = "sm_89";

}  // namespace granite4

#endif  // GRANITE4_MODEL_CONFIG_H
