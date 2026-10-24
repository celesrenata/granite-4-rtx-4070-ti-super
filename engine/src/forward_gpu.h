// Phase 3 (GPU forward): packed device-weight layout.
//
// Single source of truth for the buffer granite4_gpu_forward_range consumes
// (see forward_gpu_api.h). The CLI (infer_main.cc) builds it from the
// dequantized ModelWeights: the GEMM weight tensors (Q, K, V, O, gate,
// up, down, lm_head) are quantized to FP8 e4m3 per-tensor (scale =
// max_abs/448, the P1 storage decision - Ada has no block-scaled MMA), and
// the embeddings and norm weights stay BF16 (the original checkpoint dtype).
//
// Packed weight layout (byte offsets; each block is 2-byte aligned):
//   [0]              embed      [vocab][hidden]      BF16
//                    layers (num_layers blocks):
//     attn_norm      [hidden]                     BF16
//     ffn_norm       [hidden]                     BF16
//     Q              [hidden][hidden]             FP8
//     K              [Hk*head_dim][hidden]        FP8
//     V              [Hk*head_dim][hidden]        FP8
//     O              [hidden][hidden]             FP8
//     gate           [intermediate][hidden]       FP8
//     up             [intermediate][hidden]       FP8
//     down           [hidden][intermediate]       FP8
//                    final_norm [hidden]           BF16
//                    lm_head    [vocab][hidden]    FP8
//
// Scales (float): kGpuScalesPerLayer per layer, in the order
// Q, K, V, O, gate, up, down, then the single lm_head scale.
//
// Everything else (KV cache, scratch, RoPE tables) is owned by the context
// - see forward_gpu_api.h.

#ifndef GRANITE4_FORWARD_GPU_H
#define GRANITE4_FORWARD_GPU_H

#include <cstddef>

namespace granite4 {

// Per-layer dequant scales: one per GEMM weight tensor (Q,K,V,O,gate,up,down).
inline constexpr int kGpuScalesPerLayer = 7;

struct GpuWeightsLayout {
  // Element counts (the "rows" dimension of each block).
  std::size_t hidden;
  std::size_t intermediate;
  std::size_t vocab;
  std::size_t num_layers;
  std::size_t num_q_heads;
  std::size_t num_kv_heads;
  std::size_t head_dim;

  // K/V rows per layer block.
  std::size_t kv_rows() const { return num_kv_heads * head_dim; }

  // Per-layer block size in bytes (norms + 7 GEMM tensors).
  std::size_t layer_bytes() const {
    return 2 * hidden * 2 +
           (hidden * hidden +              // Q
            2 * kv_rows() * hidden +       // K, V
            hidden * hidden +             // O
            2 * intermediate * hidden +   // gate, up
            hidden * intermediate) * 1;    // down; FP8 = 1 byte
  }

  std::size_t embed_bytes() const { return vocab * hidden * 2; }
  std::size_t layers_bytes() const { return num_layers * layer_bytes(); }
  std::size_t final_norm_bytes() const { return hidden * 2; }
  std::size_t lm_head_bytes() const { return vocab * hidden; }

  // Offsets within one layer block.
  static constexpr std::size_t off_attn_norm() { return 0; }
  std::size_t off_ffn_norm() const { return hidden * 2; }
  std::size_t off_q() const { return 2 * hidden * 2; }
  std::size_t off_k() const { return off_q() + hidden * hidden; }
  std::size_t off_v() const { return off_k() + kv_rows() * hidden; }
  std::size_t off_o() const { return off_v() + kv_rows() * hidden; }
  std::size_t off_gate() const { return off_o() + hidden * hidden; }
  std::size_t off_up() const { return off_gate() + intermediate * hidden; }
  std::size_t off_down() const { return off_up() + intermediate * hidden; }

  std::size_t final_norm_off() const { return embed_bytes() + layers_bytes(); }
  std::size_t lm_head_off() const { return final_norm_off() + final_norm_bytes(); }
  std::size_t total_bytes() const { return lm_head_off() + lm_head_bytes(); }

  std::size_t num_scales() const {
    return num_layers * kGpuScalesPerLayer + 1;
  }
};

}  // namespace granite4

#endif  // GRANITE4_FORWARD_GPU_H
