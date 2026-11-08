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

// Opaque persistent weight upload (packed FP8 blob + pinned dequant
// scales). Owned independently of GpuCtx so that several contexts can
// attach to the same object: the ~9.5 GiB packed blob then lives once per
// device instead of once per sequence, which is what lets N sequences fit
// in VRAM at all.
struct GpuWeights;

// Allocate an empty weight object (no device memory). nullptr on failure.
GpuWeights* gpu_weights_create();

// Free a weight object and its device memory (idempotent on the buffers).
void gpu_weights_free_obj(GpuWeights* w);

// Upload packed weights + scales into a weight object (replacing any
// existing upload; idempotent semantics). Host memory in, device memory
// out. Returns 0 on success, -1 with a message in `error`.
int gpu_weights_load(GpuWeights* w, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error,
                     std::size_t error_size);

// Attach a weight object to a context without taking ownership (the
// caller keeps the object alive for the lifetime of every context that
// uses it). Replaces any previously attached/owned weights.
int gpu_ctx_use_weights(GpuCtx* ctx, GpuWeights* w);

// Allocate a context sized for sequences up to max_positions. Model
// dimensions come from `model`; `spec` is the forward spec (copied in at
// create time - the caller need not keep it alive). The RoPE theta is
// spec.rope_theta. Returns a context, or nullptr on allocation failure.
// (A context has no weights until gpu_weights_load / gpu_ctx_use_weights.)
GpuCtx* gpu_ctx_create(const ModelConfig& model, const ForwardSpec& spec,
                        int max_positions);

// Free a context and all its device memory.
void gpu_ctx_free(GpuCtx* ctx);

// ---------------------------------------------------------------------------
// Shared paged KV arena (P4). One allocation holds the persistent KV cache of
// every resident sequence; each context addresses it through its own
// position -> arena-row table, so N sessions share one pool instead of each
// owning a private [max_positions] cache.
// ---------------------------------------------------------------------------
struct GpuKvArena;
// Allocate an arena of `rows` positions per (layer, kv head), in the format
// G4_KV8 selects (unset/0 bf16, 1 int8, 4 int4). Needs split-KV attention
// (d == 128, Hq == 4 * Hk). nullptr + `error` on failure.
GpuKvArena* gpu_kv_arena_create(const ModelConfig& model, int rows, char* error,
                                std::size_t error_size);
void gpu_kv_arena_free(GpuKvArena* arena);
std::size_t gpu_kv_arena_bytes(const GpuKvArena* arena);
int gpu_kv_arena_rows(const GpuKvArena* arena);
// Like gpu_ctx_create, but the context's KV lives in `arena` (not owned: the
// caller keeps the arena alive for every context on it). max_positions is the
// longest sequence the context may hold, <= the arena's rows. The context has
// NO row mapping until gpu_ctx_set_kv_rows_n; forwards past the mapped
// positions are rejected.
GpuCtx* gpu_ctx_create_paged(const ModelConfig& model, const ForwardSpec& spec,
                             int max_positions, GpuKvArena* arena);
// Map the first n logical positions to arena rows (each in [0, arena rows) and
// distinct within the table). 0 on success; -1 leaves the table untouched.
int gpu_ctx_set_kv_rows_n(GpuCtx* ctx, const int* h_rows, int n);
// Positions [0, mapped) of the context have a valid row.
int gpu_ctx_mapped_rows(const GpuCtx* ctx);
// Rows per (layer, head) of the context's KV cache (the arena's pool size).
int gpu_ctx_kv_stride(const GpuCtx* ctx);
// The context's device position -> row table (stable address for its life),
// for kernels outside the forward pass (the EAGLE draft head) that must
// address the same rows.
const int* gpu_ctx_kv_rows_device(const GpuCtx* ctx);
// Borrow `owner`'s prefill scratch (~0.8 GiB) instead of allocating one. Valid
// only for contexts that never prefill concurrently (the serve loop runs one
// request at a time). Call before the context's first prefill.
int gpu_ctx_share_prefill_scratch(GpuCtx* ctx, GpuCtx* owner);
// The context's tensor-core prefill scratch (allocated on first use), for
// kernels outside the forward pass that run a prefill-shaped batch between
// two forwards - the EAGLE head's prompt catch-up. Valid until the next
// forward on any context that shares it. 0 on success, -1 when it cannot be
// allocated.
struct GpuPrefillScratch {
  std::uint16_t* hidden;  // [rows][H]
  std::uint16_t* normed;  // [rows][H]
  std::uint16_t* qkv;     // [rows][(Hq + 2Hk) d]
  std::uint16_t* q_rot;   // [rows][Hq d]
  std::uint16_t* attn;    // [rows][Hq d]
  std::uint16_t* act;     // [rows][I]
  float* gu;              // [rows][2I] (16-byte aligned; f32 GEMM output)
  float* ws;              // split-K workspace
  std::size_t ws_bytes;
  int rows;               // capacity in rows
};
int gpu_ctx_prefill_scratch(GpuCtx* ctx, GpuPrefillScratch* out);

// (See gpu_weights_load above for the weight-object upload; the
// legacy per-ctx signatures below delegate to it.)

// Upload (or replace) the persistent packed weights on a context.
// Equivalent to gpu_weights_load into a context-owned GpuWeights:
// h_weights is HOST memory (GpuWeightsLayout-packed); it is copied to the
// current device once and kept there for the lifetime of the context
// (freed by gpu_weights_free or on gpu_ctx_free). Must be called before
// gpu_forward_range. Returns 0 on success, -1 on failure with a
// null-terminated message in `error`.
int gpu_weights_load(GpuCtx* ctx, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error, std::size_t error_size);

// Free the persistent weights uploaded by gpu_weights_load (idempotent).
void gpu_weights_free(GpuCtx* ctx);

// Number of sequence positions already processed by this ctx (the
// KV-written watermark); gpu_forward_range recomputes positions at or
// below this value only when end <= written.
// Device descriptor of a multi-session batch: row r is position pos[r] of the
// session whose position -> arena-row table is pt[r] and whose token buffer is
// tok[r] (device pointers).
struct GpuMultiRows {
  int n;
  int pos[kMaxVerifyRows];
  const int* pt[kMaxVerifyRows];
  std::int32_t* tok[kMaxVerifyRows];
};
// One forward over rows of several sessions sharing weights + one KV arena:
// row r is position row_pos[r] of session row_ctx[r] (token from
// row_tokens[r][row_pos[r]]). Each session's rows must be consecutive and
// start at its watermark; its watermark then advances past them. Writes
// logits [n][vocab], greedy ids d_argmax[n] and, if d_hidden is non-null, the
// final-normed hidden states [n][hidden]. Each row is bit-identical to that
// session's own (batched-verify or decode) forward at that position.
// row_ctx[0] leads (its scratch runs the step). n <= kMaxVerifyRows.
int gpu_forward_multi(GpuCtx* const* row_ctx, std::int32_t* const* row_tokens,
                      const int* row_pos, int n, std::uint16_t* d_logits,
                      std::int32_t* d_argmax, std::uint16_t* d_hidden, char* error,
                      std::size_t error_size);
// The multi-row RoPE / KV append kernels over a device descriptor (the EAGLE
// head's batched steps use them on its own KV pool). bits: 16 / 8 / 4.
int gpu_multi_rope(const std::uint16_t* x, const float* cos_tbl, const float* sin_tbl,
                   const GpuMultiRows* d_rows, int rows, int heads, int d, std::uint16_t* out);
int gpu_multi_kv_append(const std::uint16_t* k, const std::uint16_t* v, int bits,
                        void* cache_k, void* cache_v, float* k_scale, float* v_scale,
                        const GpuMultiRows* d_rows, int rows, int stride, int Hk, int d);
// Multi-session greedy decode: `steps` steps, each decoding the next token
// of all n sessions in one batched forward (the weights stream once for all
// rows). Session i continues from its watermark exactly as gpu_decode_greedy
// would (its token at d_tokens[i][written] in, the greedy id out at
// [written + 1], watermark advances by `steps`), and each row is bit-identical
// to that session decoding alone. ctxs[0] leads (its scratch runs the step);
// with n > 1 all sessions must share weights and one KV arena (paged
// contexts). n <= kMaxVerifyRows. d_logits holds [n][vocab] (last step).
int gpu_decode_multi(GpuCtx* const* ctxs, std::int32_t* const* d_tokens, int n, int steps,
                     std::uint16_t* d_logits, char* error, std::size_t error_size);
int gpu_ctx_written(GpuCtx* ctx);

// Lower the KV watermark without freeing allocations. Rows at or above the
// new watermark are stale and must be overwritten before they are read.
// Returns 0 on success, -1 for an invalid/null context or forward position.
int gpu_ctx_truncate(GpuCtx* ctx, int written);

// True when the persistent KV cache is stored as int8 codes + per-row float
// scale (Q8_0). Reports the *resolved* mode - i.e. after the split-attention
// capability check in gpu_ctx_create - so a config that requested KV8 but
// fell back to bf16 reports false.
bool gpu_ctx_is_kv8(const GpuCtx* ctx);

// True when the persistent KV cache is stored as int4 (Q4_0: two 4-bit codes
// per byte + per-row float scale). Same ground-truth semantics as
// gpu_ctx_is_kv8() - reports the resolved mode after the split-attention
// capability check, so a config that requested int4 KV but fell back to bf16
// reports false.
bool gpu_ctx_is_kv4(const GpuCtx* ctx);

// True when the split-KV attention (decode / verify / prefill) is the active
// path. Implied by gpu_ctx_is_kv8() in the current implementation (KV8 can
// only use split-KV), but exposed separately because the bf16 mode may use
// either split-KV or the fused GQA depending on the model shape and
// G4_SPLIT_ATTN.
bool gpu_ctx_is_split_attn(const GpuCtx* ctx);

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
    std::uint16_t* d_cache_v, std::uint16_t* d_out,
    const int* d_pt = nullptr);

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
    std::uint16_t* d_attn_out, const int* d_pt = nullptr);

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
    std::uint16_t* d_attn_out, const int* d_pt = nullptr);

}  // namespace granite4

#endif  // GRANITE4_FORWARD_GPU_API_H
