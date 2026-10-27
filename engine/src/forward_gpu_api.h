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
// gpu_forward_range replays [0, end): prefill is a full replay; each decode
// step is [0, end) again, and the KV cache regrows only the new positions.
// Every position's attention is computed from scratch over its causal prefix
// (gathered into a contiguous [Hk][kv_len][d] block, the layout the P2 gqa
// kernels require), so re-running a prefix reproduces the same values.

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

}  // namespace granite4

#endif  // GRANITE4_FORWARD_GPU_API_H
