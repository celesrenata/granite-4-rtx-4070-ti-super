// Phase 3 (GPU forward): device forward pass for Granite 4.2 8B.
//
// Composes the P2 device-verified primitives (kernels_phase2.cu) into the
// full forward, mirroring the host oracle forward_logits_f32
// (kernel_math.h) operation-for-operation:
//   embed -> (per layer: rmsnorm -> QKV -> rope -> kv-compact -> gqa -> o
//             -> +residual -> rmsnorm -> swiglu -> down -> +residual)
//   -> rmsnorm -> lm_head
//
// Conventions (shared with the P2 kernels):
//   * BF16 activations; GEMM weights FP8 e4m3, per-tensor dequant scale
//     (the P1 storage decision - Ada has no block-scaled MMA).
//   * One thread per output element (correctness-first).
//   * Logit scale = attention_multiplier, applied in gqa_logits; the device
//     softmax then runs with scale 1.0 - exactly like the host oracle
//     (gqa_attention_bf16 scales the logits, softmax_f32 runs at 1.0).
//   * KV: the P2 gqa kernels consume a *contiguous* causal prefix
//     [Hk][kv_len][d] per call (they index K with position stride kv_len),
//     so the cache is a full [L][Hk][max_pos][d] history per K and V, and
//     each attention step gathers positions 0..t into a per-layer
//     [Hk][t+1][d] working buffer first - the device twin of the oracle's
//     explicit causal-prefix compaction (kernel_math.h).
//
// Kernels in this file are the thin glue the P2 set does not cover:
//   * embed_kernel          - embedding lookup (token id from device memory)
//   * add_resid_kernel      - residual-branch add (scaled)
//   * kv_append_kernel      - append this position's K/V into the cache
//   * kv_compact_kernel     - gather the causal prefix [0..t] per KV head
//   * rope_at_pos_kernel    - Llama half-split RoPE at one position, all
//                             heads (the P2 rope_kernel indexes its table by
//                             row, so a per-position call would need one
//                             table row per head; this folds the position
//                             into the table pointer. Formula identical to
//                             the P2 kernel.)
//   * row_gemm_dequant_kernel - the M=1 form of the P2 gemm_dequant kernel
//   * logits_to_probs_kernel  - max-shift exp (double accumulation), the
//                             device twin of the host's exp() math
//
// Compiled only with GRANITE_ENABLE_CUDA=ON, targets sm_89.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "forward_gpu.h"      // GpuWeightsLayout
#include "forward_gpu_api.h"  // public C ABI
#include "model_config.h"     // ModelConfig, ForwardSpec

#define GRANITE_CUDA 1
#include "kernel_math.h"

#include <cuda_runtime.h>

namespace granite4 {
namespace {

// Device-side helpers (same arithmetic as kernels_phase2.cu).
__device__ __forceinline__ std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, sizeof u);
  u += 0x7fffu + ((u >> 16) & 1u);  // RNE
  u >>= 16;
  return static_cast<std::uint16_t>(u);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof f);
  return f;
}

// Exact e4m3 decode (matches kernels_phase2.cu).
__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const bool neg = (x & 0x80u) != 0;
  const unsigned e = (x >> 3) & 0xFu;
  const unsigned m = x & 0x7u;
  if (e == 0) {
    // Subnormal: m * 2^-9 (m == 0 is the zero values).
    const float v = static_cast<float>(m) * std::ldexp(1.0f, -9);
    return neg ? -v : v;
  }
  if (e == 15) {
    // Top exponent: (1 + m/8) * 2^8; m == 7 is sNaN (map to 0.0; upstream
    // saturation never produces it for real weights).
    if (m == 7) return 0.0f;
    const float v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, 8);
    return neg ? -v : v;
  }
  const float v = (1.0f + static_cast<float>(m) / 8.0f) *
                  std::ldexp(1.0f, static_cast<int>(e) - 7);
  return neg ? -v : v;
}

// ---------------------------------------------------------------------------
// Glue kernels (one thread per element; bound-checked).
// ---------------------------------------------------------------------------

// out[j] = mult * emb[tokens[t], j]  (token id read from device memory)
__global__ void embed_kernel(const std::int32_t* tokens, const std::uint16_t* emb,
                             int t, float mult, std::uint16_t* out, int H) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const std::int32_t tok = tokens[t];
  out[j] =
      f32_to_bf16(mult * bf16_to_f32(emb[static_cast<std::size_t>(tok) * H + j]));
}

// out[i] = a[i] + mult * b[i]
__global__ void add_resid_kernel(const std::uint16_t* a, const std::uint16_t* b,
                                 float mult, std::uint16_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f32_to_bf16(bf16_to_f32(a[i]) + mult * bf16_to_f32(b[i]));
}

// cache[(hkv * pos + t) * d + i] = src[hkv * d + i]  (append KV at t)
__global__ void kv_append_kernel(const std::uint16_t* src, std::uint16_t* cache,
                                 int t, int pos, int Hk, int d) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * d) return;
  const int hkv = idx / d;
  cache[(static_cast<std::size_t>(hkv) * pos + t) * d + (idx % d)] = src[idx];
}

// out[(hkv * kv_len + kv) * d + i] = cache[(hkv * pos + kv) * d + i],
// kv in [0, kv_len) - the causal-prefix gather the P2 gqa kernels require.
// One thread per (kv, d) slot per KV head.
__global__ void kv_compact_kernel(const std::uint16_t* cache, std::uint16_t* out,
                                  int kv_len, int pos, int Hk, int d) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * d) return;
  const int hkv = idx / d;
  const int i = idx % d;
  for (int kv = 0; kv < kv_len; ++kv)
    out[(static_cast<std::size_t>(hkv) * kv_len + kv) * d + i] =
        cache[(static_cast<std::size_t>(hkv) * pos + kv) * d + i];
}

// Llama half-split RoPE at ONE position, applied to every head. cos_row /
// sin_row are that position's [d/2] table rows (rope_freqs_f32 layout).
// Formula identical to the P2 rope_kernel.
__global__ void rope_at_pos_kernel(const std::uint16_t* x, const float* cos_row,
                                   const float* sin_row, int heads, int d,
                                   std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= heads * d) return;
  const int h = idx / d;
  const int col = idx % d;
  const int half = d / 2;
  const int i = col < half ? col : col - half;
  const float c = cos_row[i];
  const float s = sin_row[i];
  float y;
  if (col < half) {
    const float a = bf16_to_f32(x[h * d + col]);
    const float b = bf16_to_f32(x[h * d + col + half]);
    y = a * c - b * s;
  } else {
    // Back half: out[col] = x[col]*c + x[col-half]*s (Llama convention).
    const float a = bf16_to_f32(x[h * d + col]);
    const float b = bf16_to_f32(x[h * d + col - half]);
    y = a * c + b * s;
  }
  out[idx] = f32_to_bf16(y);
}

// out[i] = scale * sum_k a[k] * fp8_to_f32(B[i, k]).
// M=1 form of the P2 gemm_dequant kernel: identical loop order and float
// accumulation, so results are row-for-row identical.
__global__ void row_gemm_dequant_kernel(const std::uint16_t* a, const std::uint8_t* B,
                                        float scale, int N, int K, std::uint16_t* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc += fp8_to_f32(B[i * K + k]) * bf16_to_f32(a[k]);
  out[i] = f32_to_bf16(scale * acc);
}

// P2 device-verified primitives (kernels_phase2.cu; no header, declared the
// way test_kernels.cu does).
extern "C" {
int granite4_rope_tables_build(double theta, int head_dim, int max_pos,
                               float** d_cos, float** d_sin);
int granite4_kernel_rmsnorm(const std::uint16_t* d_x, const std::uint16_t* d_w,
                            float eps, int rows, int cols, std::uint16_t* d_out);
int granite4_kernel_softmax(const std::uint16_t* d_x, float scale, int rows,
                            int seq_len, std::uint16_t* d_out);
int granite4_kernel_gqa_logits(const std::uint16_t* d_q, const std::uint16_t* d_k,
                               float scale, int Hq, int Hk, int d, int kv_len,
                               std::uint16_t* d_logits);
int granite4_kernel_gqa_value(const std::uint16_t* d_p, const std::uint16_t* d_v,
                              int Hq, int Hk, int d, int kv_len, std::uint16_t* d_out);
int granite4_kernel_silu_gate_up(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                 const std::uint16_t* d_x, float scale_g, float scale_u,
                                 int N, int K, std::uint16_t* d_out);
}  // extern "C"

#define G4_CUDA_CHECK(expr)                                                   \
  do {                                                                        \
    cudaError_t e_ = (expr);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "granite4: CUDA error at %s:%d: %s\n", __FILE__,  \
                   __LINE__, cudaGetErrorString(e_));                        \
      return -1;                                                              \
    }                                                                         \
  } while (0)

}  // namespace

// ---------------------------------------------------------------------------
// Persistent per-sequence context (namespace granite4 - the public API's
// GpuCtx, forward-declared in forward_gpu_api.h).
// ---------------------------------------------------------------------------
struct GpuCtx {
  ForwardSpec spec;  // copied from the caller at create time.

  int L, H, I, Hq, Hk, d, V, max_pos;

  std::uint16_t* kv_k = nullptr;  // [L][Hk][max_pos][d] (full history)
  std::uint16_t* kv_v = nullptr;  // [L][Hk][max_pos][d]
  std::uint16_t* compact_k = nullptr;  // [Hk][max_pos][d] working prefix
  std::uint16_t* compact_v = nullptr;  // [Hk][max_pos][d]
  std::uint16_t* logits_scratch = nullptr;  // [Hq][max_pos]
  std::uint16_t* probs_scratch = nullptr;   // [Hq][max_pos] (softmax output; the
                                            // P2 softmax kernel is not in-place
                                            // safe - one thread reads a whole
                                            // row while its peers write theirs)
  std::uint16_t* scratch = nullptr;
  float* cos_tbl = nullptr;  // [max_pos][d/2]
  float* sin_tbl = nullptr;
  int written = 0;  // KV positions written so far

  // Persistent packed weights + scales (uploaded once via gpu_weights_load;
  // the per-step forward must not re-upload ~9.5 GB).
  std::uint8_t* weights = nullptr;  // [total_bytes] packed GpuWeightsLayout (device)
  float* scales = nullptr;          // [num_layers*kGpuScalesPerLayer+1] dequant scales (HOST, host-read)
  std::size_t weights_bytes = 0;
  std::size_t scales_count = 0;
};

// ---------------------------------------------------------------------------
// Public API (forward_gpu_api.h), namespace granite4.
// ---------------------------------------------------------------------------
GpuCtx* gpu_ctx_create(const ModelConfig& model, const ForwardSpec& spec,
                       int max_positions) {
  auto g = new GpuCtx();
  g->spec = spec;
  g->L = static_cast<int>(model.num_layers);
  g->H = static_cast<int>(model.hidden_size);
  g->I = static_cast<int>(model.intermediate_size);
  g->Hq = static_cast<int>(model.num_attention_heads);
  g->Hk = static_cast<int>(model.num_key_value_heads);
  g->d = static_cast<int>(model.head_dim);
  g->V = static_cast<int>(model.vocab_size);
  g->max_pos = max_positions;

  const std::size_t kv_elems =
      static_cast<std::size_t>(g->L) * g->Hk * max_positions * g->d;
  const std::size_t compact_elems =
      static_cast<std::size_t>(g->Hk) * max_positions * g->d;
  // Scratch: hidden(H) + normed(H) + normed2(H) + qf(Hq*d) + k_new(Hk*d)
  //          + v_new(Hk*d) + attn_out(Hq*d) + o_out(H) + gate_out(I)
  //          + ffn_out(H) + qf_rot(Hq*d) + k_new_rot(Hk*d)
  const std::size_t scratch_elems =
      5 * static_cast<std::size_t>(g->H) +
      2 * static_cast<std::size_t>(g->Hq) * g->d +
      2 * static_cast<std::size_t>(g->Hk) * g->d +
      static_cast<std::size_t>(g->I) +
      (static_cast<std::size_t>(g->Hq) + g->Hk) * g->d;

  cudaError_t e;
  e = cudaMalloc(&g->kv_k, kv_elems * sizeof(std::uint16_t));
  if (e == cudaSuccess) e = cudaMalloc(&g->kv_v, kv_elems * sizeof(std::uint16_t));
  if (e == cudaSuccess)
    e = cudaMalloc(&g->compact_k, compact_elems * sizeof(std::uint16_t));
  if (e == cudaSuccess)
    e = cudaMalloc(&g->compact_v, compact_elems * sizeof(std::uint16_t));
  if (e == cudaSuccess)
    e = cudaMalloc(&g->logits_scratch,
                   static_cast<std::size_t>(g->Hq) * max_positions *
                       sizeof(std::uint16_t));
  if (e == cudaSuccess)
    e = cudaMalloc(&g->probs_scratch,
                   static_cast<std::size_t>(g->Hq) * max_positions *
                       sizeof(std::uint16_t));
  if (e == cudaSuccess)
    e = cudaMalloc(&g->scratch, scratch_elems * sizeof(std::uint16_t));
  if (e == cudaSuccess) {
    float* c = nullptr;
    float* s = nullptr;
    if (granite4_rope_tables_build(spec.rope_theta, g->d, max_positions, &c, &s) != 0)
      e = cudaErrorInvalidValue;
    else {
      g->cos_tbl = c;
      g->sin_tbl = s;
    }
  }
  if (e != cudaSuccess) {
    cudaFree(g->kv_k);
    cudaFree(g->kv_v);
    cudaFree(g->compact_k);
    cudaFree(g->compact_v);
    cudaFree(g->logits_scratch);
    cudaFree(g->probs_scratch);
    cudaFree(g->scratch);
    cudaFree(g->cos_tbl);
    cudaFree(g->sin_tbl);
    delete g;
    return nullptr;
  }
  g->written = 0;
  return g;
}

// Free the persistent packed weights/scales if loaded (idempotent).
void gpu_weights_free(GpuCtx* ctx);

void gpu_ctx_free(GpuCtx* ctx) {
  if (ctx == nullptr) return;
  cudaFree(ctx->kv_k);
  cudaFree(ctx->kv_v);
  cudaFree(ctx->compact_k);
  cudaFree(ctx->compact_v);
  cudaFree(ctx->logits_scratch);
  cudaFree(ctx->probs_scratch);
  cudaFree(ctx->scratch);
  cudaFree(ctx->cos_tbl);
  cudaFree(ctx->sin_tbl);
  gpu_weights_free(ctx);
  delete ctx;
}

namespace {

// The forward for positions [start, end), where start = min(g->written, end):
// positions below start already have their KV entries cached and are skipped
// (one sequence per ctx; the prefix [0, start) must have been processed by
// this ctx). d_tokens holds the full sequence so far (end ints); d_logits is
// [end][V] BF16 (caller-allocated); only rows [start, end) are written.
int forward_range_impl(GpuCtx* g, const std::int32_t* d_tokens, int end,
                       std::uint16_t* d_logits, char* error,
                       std::size_t error_size) {
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;  // optional G4_TIMESTEP timing
  auto seterr = [&](const char* what) {
    if (ev0 != nullptr) {
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "%s", what);
    return -1;
  };

  const int L = g->L, H = g->H, I = g->I, Hq = g->Hq, Hk = g->Hk;
  const int d = g->d, V = g->V, max_pos = g->max_pos;
  const int half = d / 2;

  const float rms_eps = g->spec.rms_norm_eps;
  const float att_scale = g->spec.attention_multiplier;
  const float resid_scale = g->spec.residual_multiplier;
  const float emb_scale = g->spec.embedding_multiplier;
  const float logits_scale = g->spec.logits_scaling;

  const GpuWeightsLayout lay{H, I, V, L, Hq, Hk, d};

  if (end > max_pos) return seterr("sequence exceeds max_positions");
  if (g->weights == nullptr || g->scales == nullptr)
    return seterr("weights not loaded (call gpu_weights_load first)");

  // Incremental decode: positions [0, written) already have their KV
  // entries cached; only the new suffix needs the full stack. (One
  // sequence per ctx.) If end <= written the call recomputes the whole
  // prefix (KV appends are idempotent), so start = 0 then.
  const int start = (end > g->written) ? g->written : 0;

  // Optional per-call GPU timing (G4_TIMESTEP=1): prints the device-side
  // duration of this call to stderr.
  const bool timed = std::getenv("G4_TIMESTEP") != nullptr;
  if (timed) {
    if (cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess)
      return seterr("cudaEventCreate failed");
    G4_CUDA_CHECK(cudaEventRecord(ev0));
  }

  std::uint16_t* const d_hidden = g->scratch;
  std::uint16_t* const d_normed = g->scratch + H;
  std::uint16_t* const d_normed2 = g->scratch + 2 * H;
  std::uint16_t* const d_qf = g->scratch + 3 * H;
  std::uint16_t* const d_k_new = d_qf + Hq * d;
  std::uint16_t* const d_v_new = d_k_new + Hk * d;
  std::uint16_t* const d_attn_out = d_v_new + Hk * d;
  std::uint16_t* const d_o_out = d_attn_out + Hq * d;
  std::uint16_t* const d_gate_out = d_o_out + H;
  std::uint16_t* const d_ffn_out = d_gate_out + I;
  // RoPE output buffers: the rope kernel is not in-place safe (a thread
  // reads its partner element while the partner writes it), so rotate into
  // separate scratch and feed those to attention.
  std::uint16_t* const d_qf_rot = d_ffn_out + H;
  std::uint16_t* const d_k_new_rot = d_qf_rot + Hq * d;

  const std::uint16_t* const d_embed_tbl =
      reinterpret_cast<const std::uint16_t*>(g->weights);

  for (int t = start; t < end; ++t) {
    // ---------- embedding lookup (token id from device memory) ----------
    embed_kernel<<<(H + 255) / 256, 256>>>(d_tokens, d_embed_tbl, t, emb_scale,
                                           d_hidden, H);
    G4_CUDA_CHECK(cudaGetLastError());

    for (int layer = 0; layer < L; ++layer) {
      const std::size_t lay_off =
          lay.embed_bytes() + static_cast<std::size_t>(layer) * lay.layer_bytes();
      const std::uint16_t* const attn_norm =
          reinterpret_cast<const std::uint16_t*>(g->weights + lay_off);
      const std::uint16_t* const ffn_norm = attn_norm + H;
      const std::uint8_t* const wq = g->weights + lay_off + lay.off_q();
      const std::uint8_t* const wk = g->weights + lay_off + lay.off_k();
      const std::uint8_t* const wv = g->weights + lay_off + lay.off_v();
      const std::uint8_t* const wo = g->weights + lay_off + lay.off_o();
      const std::uint8_t* const wg = g->weights + lay_off + lay.off_gate();
      const std::uint8_t* const wu = g->weights + lay_off + lay.off_up();
      const std::uint8_t* const wd = g->weights + lay_off + lay.off_down();
      const float* const sl =
          g->scales + static_cast<std::size_t>(layer) * kGpuScalesPerLayer;

      const std::size_t kv_layer =
          static_cast<std::size_t>(layer) * Hk * max_pos * d;

      // ---- attention block (causal over positions 0..t) ----
      // P2 public launchers (device-verified in test_kernels.cu; same
      // arithmetic as the host oracle).
      if (granite4_kernel_rmsnorm(d_hidden, attn_norm, rms_eps, 1, H, d_normed) != 0)
        return seterr("rmsnorm (attention) failed");
      row_gemm_dequant_kernel<<<(Hq * d + 255) / 256, 256>>>(d_normed, wq, sl[0],
                                                             Hq * d, H, d_qf);
      G4_CUDA_CHECK(cudaGetLastError());
      row_gemm_dequant_kernel<<<(Hk * d + 255) / 256, 256>>>(d_normed, wk, sl[1],
                                                             Hk * d, H, d_k_new);
      G4_CUDA_CHECK(cudaGetLastError());
      row_gemm_dequant_kernel<<<(Hk * d + 255) / 256, 256>>>(d_normed, wv, sl[2],
                                                             Hk * d, H, d_v_new);
      G4_CUDA_CHECK(cudaGetLastError());

      const float* const cos_t = g->cos_tbl + static_cast<std::size_t>(t) * half;
      const float* const sin_t = g->sin_tbl + static_cast<std::size_t>(t) * half;
      rope_at_pos_kernel<<<(Hq * d + 255) / 256, 256>>>(d_qf, cos_t, sin_t, Hq, d,
                                                        d_qf_rot);
      G4_CUDA_CHECK(cudaGetLastError());
      rope_at_pos_kernel<<<(Hk * d + 255) / 256, 256>>>(d_k_new, cos_t, sin_t, Hk, d,
                                                        d_k_new_rot);
      G4_CUDA_CHECK(cudaGetLastError());

      // Append this position's RoPE'd K and raw V to the full history.
      kv_append_kernel<<<(Hk * d + 255) / 256, 256>>>(d_k_new_rot, g->kv_k + kv_layer,
                                                       t, max_pos, Hk, d);
      G4_CUDA_CHECK(cudaGetLastError());
      kv_append_kernel<<<(Hk * d + 255) / 256, 256>>>(d_v_new, g->kv_v + kv_layer,
                                                       t, max_pos, Hk, d);
      G4_CUDA_CHECK(cudaGetLastError());

      // Gather the causal prefix into the contiguous [Hk][t+1][d] layout
      // the gqa kernels consume (the oracle's explicit compaction).
      const int kv_len = t + 1;
      kv_compact_kernel<<<(Hk * d + 255) / 256, 256>>>(g->kv_k + kv_layer,
                                                       g->compact_k, kv_len,
                                                       max_pos, Hk, d);
      G4_CUDA_CHECK(cudaGetLastError());
      kv_compact_kernel<<<(Hk * d + 255) / 256, 256>>>(g->kv_v + kv_layer,
                                                       g->compact_v, kv_len,
                                                       max_pos, Hk, d);
      G4_CUDA_CHECK(cudaGetLastError());

      if (granite4_kernel_gqa_logits(d_qf_rot, g->compact_k, att_scale, Hq, Hk, d,
                                     kv_len, g->logits_scratch) != 0)
        return seterr("gqa_logits failed");
      // softmax over the scaled logits (scale 1.0 - the scale was folded
      // into the logits by gqa_logits, exactly as the oracle does). The P2
      // kernel reads every element of each row while writing its own
      // element, so the output must be a separate buffer (no in-place).
      if (granite4_kernel_softmax(g->logits_scratch, 1.0f, Hq, kv_len,
                                  g->probs_scratch) != 0)
        return seterr("softmax failed");
      if (granite4_kernel_gqa_value(g->probs_scratch, g->compact_v, Hq, Hk, d,
                                    kv_len, d_attn_out) != 0)
        return seterr("gqa_value failed");
      row_gemm_dequant_kernel<<<(H + 255) / 256, 256>>>(d_attn_out, wo, sl[3], H, H,
                                                        d_o_out);
      G4_CUDA_CHECK(cudaGetLastError());
      add_resid_kernel<<<(H + 255) / 256, 256>>>(d_hidden, d_o_out, resid_scale,
                                                 d_hidden, H);
      G4_CUDA_CHECK(cudaGetLastError());

      // ---- FFN block ----
      if (granite4_kernel_rmsnorm(d_hidden, ffn_norm, rms_eps, 1, H, d_normed2) != 0)
        return seterr("rmsnorm (ffn) failed");
      if (granite4_kernel_silu_gate_up(wg, wu, d_normed2, sl[4], sl[5], I, H,
                                       d_gate_out) != 0)
        return seterr("silu_gate_up failed");
      row_gemm_dequant_kernel<<<(H + 255) / 256, 256>>>(d_gate_out, wd, sl[6], H, I,
                                                        d_ffn_out);
      G4_CUDA_CHECK(cudaGetLastError());
      add_resid_kernel<<<(H + 255) / 256, 256>>>(d_hidden, d_ffn_out, resid_scale,
                                                 d_hidden, H);
      G4_CUDA_CHECK(cudaGetLastError());
    }

    // ---------- final RMSNorm + lm_head ----------
    const std::uint16_t* const d_final_norm =
        reinterpret_cast<const std::uint16_t*>(g->weights + lay.final_norm_off());
    const std::uint8_t* const d_lm_head = g->weights + lay.lm_head_off();
    // Fold in the spec's logits_scaling (same single final multiply as the
    // host oracle).
    const float lm_scale = g->scales[L * kGpuScalesPerLayer] * logits_scale;
    if (granite4_kernel_rmsnorm(d_hidden, d_final_norm, rms_eps, 1, H, d_normed) != 0)
      return seterr("final rmsnorm failed");
    row_gemm_dequant_kernel<<<(V + 255) / 256, 256>>>(
        d_normed, d_lm_head, lm_scale, V, H,
        d_logits + static_cast<std::size_t>(t) * V);
    G4_CUDA_CHECK(cudaGetLastError());
  }

  if (timed) {
    G4_CUDA_CHECK(cudaEventRecord(ev1));
    G4_CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    std::fprintf(stderr, "G4_TIMESTEP end=%d start=%d gpu_ms=%.2f\n", end,
                 start, ms);
  }

  if (end > g->written) g->written = end;
  return 0;
}

}  // namespace

// Public API (forward_gpu_api.h), namespace granite4.
int gpu_weights_load(GpuCtx* ctx, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error,
                     std::size_t error_size) {
  auto seterr = [&](const char* what) {
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "%s", what);
    return -1;
  };
  if (ctx == nullptr || h_weights == nullptr || bytes == 0 ||
      h_scales == nullptr || n_scales == 0)
    return seterr("invalid arguments");
  // Replacing existing weights: free first (idempotent semantics).
  gpu_weights_free(ctx);
  std::uint8_t* d_weights = nullptr;
  if (cudaMalloc(&d_weights, bytes) != cudaSuccess)
    return seterr("cudaMalloc weights failed");
  if (cudaMemcpy(d_weights, h_weights, bytes, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(d_weights);
    return seterr("cudaMemcpy weights failed");
  }
  // The dequant scales are read on the host (every kernel launcher takes
  // them by value), so they must stay in HOST memory - pinned for the
  // lifetime of the upload.
  float* h_scales_pin = nullptr;
  if (cudaHostAlloc(&h_scales_pin, n_scales * sizeof(float),
                    cudaHostAllocDefault) != cudaSuccess) {
    cudaFree(d_weights);
    return seterr("cudaHostAlloc scales failed");
  }
  std::memcpy(h_scales_pin, h_scales, n_scales * sizeof(float));
  ctx->weights = d_weights;
  ctx->weights_bytes = bytes;
  ctx->scales = h_scales_pin;
  ctx->scales_count = n_scales;
  return 0;
}

void gpu_weights_free(GpuCtx* ctx) {
  if (ctx == nullptr) return;
  if (ctx->weights != nullptr) cudaFree(ctx->weights);
  ctx->weights = nullptr;
  ctx->weights_bytes = 0;
  if (ctx->scales != nullptr) cudaFreeHost(ctx->scales);
  ctx->scales = nullptr;
  ctx->scales_count = 0;
}

int gpu_forward_range(GpuCtx* ctx, const std::int32_t* d_tokens, int end,
                      std::uint16_t* d_logits, char* error,
                      std::size_t error_size) {
  if (ctx == nullptr || d_tokens == nullptr || d_logits == nullptr ||
      end <= 0) {
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "invalid arguments");
    return -1;
  }
  return forward_range_impl(ctx, d_tokens, end, d_logits, error, error_size);
}

int gpu_ctx_written(GpuCtx* ctx) { return ctx != nullptr ? ctx->written : 0; }

}  // namespace granite4
