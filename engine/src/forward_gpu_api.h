// Phase 3 (GPU forward): public API for the device forward pass.
//
// C++ (no CUDA, no STL containers). Consumed by infer_main.cc, which is
// compiled with GRANITE_ENABLE_CUDA and links the shared granite4 library
// that carries the CUDA objects (forward_gpu.cu).
//
// Model: a persistent per-sequence context (GpuCtx) owns the device
// allocations - the full K/V history cache, the per-position scratch, the
// causal-prefix compact buffers, and the RoPE tables sized for max_positions
// - so a generate loop pays for them once and then decodes in place.
//
// gpu_forward runs only the positions past the KV watermark: a prompt is one
// prefill call (tensor-core GEMMs for 5+ rows), a speculative verify is one
// batched 2-4-row call, and gpu_decode_greedy chains single-position greedy
// steps on the device (one CUDA graph launch per token, no host round trip).

#ifndef GRANITE4_FORWARD_GPU_API_H
#define GRANITE4_FORWARD_GPU_API_H

#include "model_config.h"

#include <cstddef>
#include <cstdint>

namespace granite4 {

// Opaque per-sequence context (KV cache + scratch + compact buffers +
// RoPE tables). Defined in forward_gpu.cu.
struct GpuCtx;

// Allocate a context sized for sequences up to max_positions. Model
// dimensions come from `model`; `spec` is the forward spec (copied in at
// create time - the caller need not keep it alive). The RoPE theta is
// spec.rope_theta. Returns a context, or nullptr on allocation failure.
GpuCtx* gpu_ctx_create(const ModelConfig& model, const ForwardSpec& spec,
                        int max_positions);

// Free a context and all its device memory.
void gpu_ctx_free(GpuCtx* ctx);

// Upload (or replace) the persistent packed weights used by
// gpu_forward_range. h_weights is HOST memory (GpuWeightsLayout-packed); it
// is copied to the current device once and kept there for the lifetime of
// ctx (freed by gpu_weights_free or on gpu_ctx_free). h_scales is HOST
// memory with [num_layers * kGpuScalesPerLayer + 1] dequant scales (copied).
// Must be called before gpu_forward_range. Returns 0 on success, -1 on
// failure with a null-terminated message in `error`.
int gpu_weights_load(GpuCtx* ctx, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error, std::size_t error_size);

// Free the persistent weights uploaded by gpu_weights_load (idempotent).
void gpu_weights_free(GpuCtx* ctx);

// Number of sequence positions already processed by this ctx (the
// KV-written watermark); gpu_forward_range recomputes positions at or
// below this value only when end <= written.
int gpu_ctx_written(GpuCtx* ctx);

// Lower the KV watermark without freeing allocations. Rows at or above the
// new watermark are stale and must be overwritten before they are read.
// Returns 0 on success, -1 for an invalid/null context or forward position.
int gpu_ctx_truncate(GpuCtx* ctx, int written);

// Compatibility helper: discard the entire KV watermark.
void gpu_ctx_reset(GpuCtx* ctx);

// Run the forward pass for positions [start, end) of the token sequence on
// the current device, where start = min(ctx->written, end): already-processed
// positions are skipped because their KV entries persist in the ctx caches.
// The sequence prefix [0, start) must have been processed by this ctx (this
// is automatic for incremental decoding on one sequence per ctx).
// d_tokens is device memory: [end] int32 token ids (the full sequence so
// far). d_logits receives [end][vocab] BF16 logits (device memory) - only
// rows [start, end) are written; read back only those. Logits are scaled by
// spec.logits_scaling (identical to the host oracle output).
//
// Returns 0 on success; -1 on failure with a null-terminated message in
// `error` (error may be nullptr; error_size is its capacity).
int gpu_forward_range(GpuCtx* ctx, const std::int32_t* d_tokens, int end,
                      std::uint16_t* d_logits, char* error,
                      std::size_t error_size);

// Output plan for gpu_forward. Only rows t >= logits_from get the final norm
// and lm_head (rows below it just fill the KV cache, e.g. a prompt prefill
// that needs the last row only); row t is written to
// d_logits + (t - logits_from) * vocab. With d_argmax, d_argmax[t -
// logits_from] also receives each row's greedy token (first maximum, the
// host sample_greedy_f32 rule). logits_from == end requests no logits.
// With d_hidden, row t >= hidden_from (-1: logits_from) also gets its
// final-normed hidden state (the lm_head input, [H] bf16) at
// d_hidden + (t - hidden_from) * H; d_logits may be nullptr, which skips the
// lm_head (hidden-state dumps, draft-head features).
struct GpuForwardOut {
  std::uint16_t* d_logits = nullptr;
  int logits_from = 0;
  std::int32_t* d_argmax = nullptr;
  std::uint16_t* d_hidden = nullptr;
  int hidden_from = -1;
};

// Device views of the target's packed embedding (bf16 [vocab][hidden]) and
// FP8 lm_head ([vocab][hidden], dequant scale incl. logits_scaling), for a
// draft head that shares them. Returns -1 before gpu_weights_load.
int gpu_ctx_shared_weights(GpuCtx* ctx, const std::uint16_t** d_embed,
                           const std::uint8_t** d_lm_head, float* lm_scale);

// gpu_forward_range with an explicit output plan (same watermark rules).
int gpu_forward(GpuCtx* ctx, const std::int32_t* d_tokens, int end,
                const GpuForwardOut& out, char* error, std::size_t error_size);

// Greedy decode chained on the device: `steps` single-position forwards
// starting at p = gpu_ctx_written(ctx). Step i forwards position p + i (its
// token d_tokens[p + i] must already be on the device; d_tokens[p] normally
// comes from a prior gpu_forward d_argmax), writes the logits row into
// d_logits_row ([vocab], overwritten each step) and its greedy token into
// d_tokens[p + i + 1]. Nothing is synchronized or copied back: the caller
// reads the generated ids from d_tokens afterwards. The step is captured
// once into a CUDA graph (G4_GRAPH=0 disables) and replayed per token.
// Requires p + steps < max_positions. Advances the watermark by `steps`.
int gpu_decode_greedy(GpuCtx* ctx, std::int32_t* d_tokens, int steps,
                      std::uint16_t* d_logits_row, char* error,
                      std::size_t error_size);

// Run the attention portion of speculative verification for `rows` consecutive
// positions. Inputs and outputs are device memory. Q/K/V are token-major
// projected tensors; cos/sin contain one [d/2] row per position. The stage
// applies RoPE, appends K/V at base_pos..base_pos+rows, and writes causal GQA
// outputs. The cache uses [Hk][cache_stride][d] layout. `rows` is currently
// bounded to four, matching the verified batched kernels. All device pointers
// must belong to the current CUDA device; cos/sin are [rows][head_dim/2].
int gpu_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k,
    const std::uint16_t* d_v, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int cache_stride, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k,
    std::uint16_t* d_cache_v, std::uint16_t* d_out);

// Project normalized speculative rows and run one transformer's attention block.
// This compatibility name delegates to the explicitly batched helper below.
// Q/K/V weights are row-major FP8 device matrices with one scale per matrix.
int gpu_verify_attention_layer(
    const std::uint16_t* d_normed, const std::uint8_t* d_wq,
    const std::uint8_t* d_wk, const std::uint8_t* d_wv, float q_scale,
    float k_scale, float v_scale, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int hidden_size, int cache_stride, std::uint16_t* d_q,
    std::uint16_t* d_k, std::uint16_t* d_v, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_attn_out);

// Batched speculative attention block for already-normalized rows. This is
// intentionally limited to one transformer attention block; it does not yet
// run the FFN/final norm/lm-head stages or the accept/reject loop. It runs one
// already-normalized transformer attention block over 1..4 rows.
// It projects Q/K/V, applies position-specific RoPE, appends K/V, and runs
// causal GQA. All pointers are device memory; weights are row-major FP8 and
// outputs are token-major [rows][heads * head_dim].
int gpu_verify_attention_layer_batched(
    const std::uint16_t* d_normed, const std::uint8_t* d_wq,
    const std::uint8_t* d_wk, const std::uint8_t* d_wv, float q_scale,
    float k_scale, float v_scale, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int hidden_size, int cache_stride, std::uint16_t* d_q,
    std::uint16_t* d_k, std::uint16_t* d_v, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_attn_out);

}  // namespace granite4

#endif  // GRANITE4_FORWARD_GPU_API_H
