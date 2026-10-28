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
#include <chrono>
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

// Exact e4m3 decode, branchless (matches the arithmetic decode bit-for-bit on
// all 256 codepoints - verified by tests/test_kernels.cu against the
// fp8_to_f32 reference table). The branchy ldexp form cost 14-65% of GEMV
// throughput on sm_89 (measured per shape: small kernels ~14%, lm_head
// ~65% - divergent e/m predication + ldexp latency per element), so this maps the 3 mantissa bits and the biased exponent straight
// into f32 bits:
//   sign  : bit 7 -> f32 bit 31 (x & 0x80) << 24.
//   1<=e<=14: value (1+m/8)*2^(e-7) -> f32 exp field 120+e, mantissa m ->
//             ((120+e) << 23) | (m << 20).
//   e == 15: value (1+m/8)*2^8 -> 0x43800000 | (m << 20); m == 7 is sNaN
//            (map to 0.0, exactly as the arithmetic form did - upstream
//            saturation never produces it for real weights).
//   e == 0: subnormal m * 2^-9 - all 8 values are exact in f32 (0 and
//           2^-9, 2^-8, ... 7*2^-9).
__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const std::uint32_t s = static_cast<std::uint32_t>(x & 0x80u) << 24;
  const std::uint32_t y = x & 0x7Fu;
  const std::uint32_t e = y >> 3;
  const std::uint32_t m = y & 7u;
  std::uint32_t bits = ((120u + e) << 23) | (m << 20);
  const std::uint32_t top =
      (m == 7u) ? 0u : (0x43800000u | (m << 20));
  bits = (e == 15u) ? top : bits;
  bits = (e == 0u) ? __float_as_uint(static_cast<float>(m) * 0.001953125f)
                   : bits;  // 0.001953125f = 2^-9
  // sNaN (0x7f/0xff): the arithmetic decode returned +0.0 (sign discarded),
  // so drop the sign here too to stay bit-identical.
  return ((e == 15u && m == 7u)) ? 0.0f : __uint_as_float(bits | s);
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

// Append-only form (direct-cache GQA): store this position's RoPE'd K and raw
// V in the persistent [Hk][max_pos][d] cache; attention reads the cache rows
// directly with max_pos as the stride, so the per-step prefix gather is gone.
__global__ void kv_append_kernel(const std::uint16_t* src_k, const std::uint16_t* src_v,
                                 std::uint16_t* cache_k, std::uint16_t* cache_v,
                                 int t, int pos, int Hk, int d) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * d) return;
  const int hkv = idx / d;
  const int i = idx % d;
  const std::size_t cache_row = static_cast<std::size_t>(hkv) * pos * d;
  cache_k[cache_row + static_cast<std::size_t>(t) * d + i] = src_k[idx];
  cache_v[cache_row + static_cast<std::size_t>(t) * d + i] = src_v[idx];
}

__global__ void kv_append_compact_kernel(const std::uint16_t* src_k,
                                         const std::uint16_t* src_v,
                                         std::uint16_t* cache_k,
                                         std::uint16_t* cache_v,
                                         std::uint16_t* compact_k,
                                         std::uint16_t* compact_v,
                                         int kv_len, int t, int pos, int Hk, int d) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * d) return;
  const int hkv = idx / d;
  const int i = idx % d;
  const std::size_t cache_row = static_cast<std::size_t>(hkv) * pos * d;
  const std::size_t comp_row = static_cast<std::size_t>(hkv) * kv_len * d;
  const std::uint16_t kk = src_k[idx];
  const std::uint16_t vv = src_v[idx];
  cache_k[cache_row + static_cast<std::size_t>(t) * d + i] = kk;
  cache_v[cache_row + static_cast<std::size_t>(t) * d + i] = vv;
  for (int kv = 0; kv < kv_len; ++kv) {
    const std::size_t coff = cache_row + static_cast<std::size_t>(kv) * d + i;
    const std::size_t doff = comp_row + static_cast<std::size_t>(kv) * d + i;
    compact_k[doff] = (kv == t) ? kk : cache_k[coff];
    compact_v[doff] = (kv == t) ? vv : cache_v[coff];
  }
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

// Batched RoPE over `rows` consecutive positions (speculative verify): row r
// rotates every head with cos_rows/sin_rows row r. Same formula and
// per-element arithmetic as rope_at_pos_kernel.
__global__ void rope_at_pos_batched_kernel(const std::uint16_t* x,
                                           const float* cos_rows,
                                           const float* sin_rows, int rows,
                                           int heads, int d,
                                           std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * heads * d) return;
  const int row = idx / (heads * d);
  const int local = idx % (heads * d);
  const int h = local / d;
  const int col = local % d;
  const int half = d / 2;
  const int i = col < half ? col : col - half;
  const float c = cos_rows[row * half + i];
  const float s = sin_rows[row * half + i];
  const std::size_t base =
      static_cast<std::size_t>(row) * heads * d + static_cast<std::size_t>(h) * d;
  float y;
  if (col < half) {
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col + half]);
    y = a * c - b * s;
  } else {
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col - half]);
    y = a * c + b * s;
  }
  out[idx] = f32_to_bf16(y);
}

// Llama half-split RoPE at ONE position, fused over the q and k projections
// (one launch instead of two). Same formula and per-element arithmetic as
// rope_at_pos_kernel.
__global__ void rope_at_pos_qk_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                      const float* cos_row, const float* sin_row,
                                      int Hq, int Hk, int d,
                                      std::uint16_t* q_out, std::uint16_t* k_out) {
  const int nq = Hq * d;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nq + Hk * d) return;
  const bool is_q = idx < nq;
  const int local = is_q ? idx : idx - nq;
  const std::uint16_t* x = is_q ? q : k;
  std::uint16_t* out = is_q ? q_out : k_out;
  const int h = local / d;
  const int col = local % d;
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
    const float a = bf16_to_f32(x[h * d + col]);
    const float b = bf16_to_f32(x[h * d + col - half]);
    y = a * c + b * s;
  }
  out[local] = f32_to_bf16(y);
}

// out[i] = scale * sum_k a[k] * fp8_to_f32(B[i, k])  (M=1 GEMV: a is one row).
//
// Coalesced block-per-row GEMV. The previous form (one thread per output row,
// each reading its whole K-byte weight row) made consecutive threads read
// addresses K bytes apart, so a warp fetched 32 separate 32-byte cache lines
// while using only 32 bytes - roughly 1/32 of the achievable bandwidth. At
// this shape that pinned decode at ~3.9 tps against a 76 tps roofline
// (kernels_bench measured 635 GB/s). Here one block computes one row of out:
// its threads stride over K so consecutive lanes read consecutive bytes
// (fully coalesced), every byte of B is read exactly once, and the per-row
// partials are combined with a warp-shuffle + shared reduce. Same per-k
// fp8_to_f32 * bf16_to_f32 float arithmetic as the P2 reference (only the
// summation order differs, within float rounding); parity vs the host oracle
// is unchanged.
//
// P4: 16-byte (uint4) weight loads. Each lane loads 16 fp8 bytes per
// iteration (four 4-element subgroups) instead of one uint32 of 4, so the
// kernel issues 4x fewer global loads and each lane has 4x the bytes in
// flight. On the real decode shapes this lifted the measured read bandwidth
// from 381 to 409 GB/s per token (q 330->368, k 234->251, gate 400->429,
// lm_head 517->542 GB/s) and the GEMV-only bench from 22.1 to 20.5
// ms/token; the uninstrumented decode step went 18.5 -> 18.3 ms. The bf16
// outputs are bit-identical to the uint32 form on every shape (the changed
// summation order stays inside bf16 rounding) and the 24-token greedy
// generate is token-identical. Requires K 16-aligned, a 16-aligned row
// start, and a 4-aligned activation vector (the layout's offsets are all
// multiples of H=4096, so this always holds for the checkpoint; the
// alignment tests keep the fast path honest if that ever changes). The
// 4-element and scalar paths remain as fallbacks for shapes/pointers that
// fail those tests.
//
// P4: 128 threads per block. With 16-byte loads a lane covers one uint4 per
// iteration at K=4096, so 128 threads (not 256) leave no idle lanes; the
// small-grid Q/K/V shapes gain most from the shorter ramp. Same-run A/B on
// the full decode: 18.06-18.08 ms/step vs 18.29-18.33 ms at 256 threads
// (token-identical).
//
// P4: the Q/K/V projections use row_gemm_dequant_qkv_kernel below, one grid
// over three row ranges. It shares this exact per-row arithmetic and launch
// width while retaining each projection's weight, scale, and output buffer.
// The O-projection and FFN-down calls use row_gemm_dequant_resid_kernel,
// which folds the residual add into the thread-0 epilogue (template
// AddResidual) and writes d_hidden in place.
constexpr int kRowGemvThreads = 128;

template <bool AddResidual>
__device__ __forceinline__ void gemv_row_dequant(const std::uint16_t* a,
                                                 const std::uint8_t* row,
                                                 float scale, int K,
                                                 std::uint16_t* out, int i,
                                                 const std::uint16_t* residual = nullptr,
                                                 float residual_scale = 0.0f) {
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const std::uint32_t* const a4 = reinterpret_cast<const std::uint32_t*>(a);
  float acc = 0.0f;
  // 16-byte vector path: one uint4 of 16 fp8 bytes per lane per iteration,
  // with the matching 16 bf16 (two uint32 pairs per 4-element subgroup).
  if ((K & 15) == 0 &&
      (reinterpret_cast<std::uintptr_t>(row) & 15u) == 0 &&
      (reinterpret_cast<std::uintptr_t>(a) & 3u) == 0) {
    const int K16 = K >> 4;
    const uint4* const row16 = reinterpret_cast<const uint4*>(row);
    for (int g = t; g < K16; g += nt) {
      const uint4 w = row16[g];
      const std::uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {  // each ws[q] = 4 fp8 elements
        const int e = 16 * g + 4 * q;  // element index of this subgroup
        const std::uint32_t wq = ws[q];
        const std::uint32_t y0 = a4[e / 2];      // elements e, e+1
        const std::uint32_t y1 = a4[e / 2 + 1];  // elements e+2, e+3
        const float b0 = fp8_to_f32(static_cast<std::uint8_t>(wq & 0xffu));
        const float b1 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 8) & 0xffu));
        const float b2 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 16) & 0xffu));
        const float b3 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 24) & 0xffu));
        const float fa0 = bf16_to_f32(static_cast<std::uint16_t>(y0 & 0xffffu));
        const float fa1 = bf16_to_f32(static_cast<std::uint16_t>((y0 >> 16) & 0xffffu));
        const float fa2 = bf16_to_f32(static_cast<std::uint16_t>(y1 & 0xffffu));
        const float fa3 = bf16_to_f32(static_cast<std::uint16_t>((y1 >> 16) & 0xffffu));
        acc += b0 * fa0 + b1 * fa1 + b2 * fa2 + b3 * fa3;
      }
    }
  } else if ((K & 3) == 0) {
    const std::uint32_t* const row4 =
        reinterpret_cast<const std::uint32_t*>(row);
    const int K4 = K >> 2;
    for (int k4 = t; k4 < K4; k4 += nt) {
      const std::uint32_t b = row4[k4];
      const std::uint32_t x0 = a4[2 * k4];
      const std::uint32_t x1 = a4[2 * k4 + 1];
      // Little-endian: byte j of b = element 4*k4+j.
      const float b0 = fp8_to_f32(static_cast<std::uint8_t>(b & 0xffu));
      const float b1 = fp8_to_f32(static_cast<std::uint8_t>((b >> 8) & 0xffu));
      const float b2 = fp8_to_f32(static_cast<std::uint8_t>((b >> 16) & 0xffu));
      const float b3 = fp8_to_f32(static_cast<std::uint8_t>((b >> 24) & 0xffu));
      const std::uint16_t a0 = static_cast<std::uint16_t>(x0 & 0xffffu);
      const std::uint16_t a1 = static_cast<std::uint16_t>((x0 >> 16) & 0xffffu);
      const std::uint16_t a2 = static_cast<std::uint16_t>(x1 & 0xffffu);
      const std::uint16_t a3 = static_cast<std::uint16_t>((x1 >> 16) & 0xffffu);
      acc += b0 * bf16_to_f32(a0) + b1 * bf16_to_f32(a1) +
             b2 * bf16_to_f32(a2) + b3 * bf16_to_f32(a3);
    }
  } else {
    for (int k = t; k < K; k += nt)
      acc += fp8_to_f32(row[k]) * bf16_to_f32(a[k]);
  }
  // Reduce the block: warp shuffle, then a small shared tree over warps.
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  __shared__ float red[32];
  if ((t & 31) == 0) red[t >> 5] = acc;
  __syncthreads();
  if (t == 0) {
    float s = red[0];
    for (int j = 1; j < (nt >> 5); ++j) s += red[j];
    const float y = scale * s;
    out[i] = AddResidual
                 ? f32_to_bf16(bf16_to_f32(residual[i]) + residual_scale * y)
                 : f32_to_bf16(y);
  }
}
__global__ void row_gemm_dequant_kernel(const std::uint16_t* a, const std::uint8_t* B,
                                        float scale, int N, int K, std::uint16_t* out) {
  const int i = blockIdx.x;  // one output row per block
  if (i >= N) return;
  gemv_row_dequant<false>(a, B + static_cast<std::size_t>(i) * K, scale, K, out, i);
}

__global__ void row_gemm_dequant_resid_kernel(const std::uint16_t* a,
                                               const std::uint8_t* B, float scale,
                                               const std::uint16_t* residual,
                                               float residual_scale, int N, int K,
                                               std::uint16_t* out) {
  const int i = blockIdx.x;
  if (i >= N) return;
  gemv_row_dequant<true>(a, B + static_cast<std::size_t>(i) * K, scale, K, out, i,
                         residual, residual_scale);
}

// Batched residual projection: one block per (token, output row), preserving
// the production row-GEMV arithmetic while eliminating the per-token launch
// loop used by the single-position path.
__global__ void row_gemm_dequant_resid_batched_kernel(
    const std::uint16_t* a, const std::uint8_t* B, float scale,
    const std::uint16_t* residual, float residual_scale, int tokens, int N, int K,
    std::uint16_t* out) {
  const int block = blockIdx.x;
  const int token = block / N;
  const int row = block % N;
  if (token >= tokens) return;
  gemv_row_dequant<true>(
      a + static_cast<std::size_t>(token) * K,
      B + static_cast<std::size_t>(row) * K, scale, K,
      out + static_cast<std::size_t>(token) * N, row,
      residual + static_cast<std::size_t>(token) * N, residual_scale);
}

// Fused Q/K/V decode GEMV: one launch for the three attention projections, so
// the small k/v grids stop paying a separate ramp/drain and launch gap per
// layer. Blocks are dispatched by range; per-row arithmetic is identical to
// row_gemm_dequant_kernel.
__global__ void row_gemm_dequant_qkv_kernel(const std::uint16_t* a,
                                            const std::uint8_t* wq,
                                            const std::uint8_t* wk,
                                            const std::uint8_t* wv,
                                            float sq, float sk, float sv,
                                            int Nq, int Nk, int Nv, int K,
                                            std::uint16_t* oq,
                                            std::uint16_t* ok,
                                            std::uint16_t* ov) {
  const int b = blockIdx.x;
  if (b < Nq) {
    gemv_row_dequant<false>(a, wq + static_cast<std::size_t>(b) * K, sq, K, oq, b);
  } else if (b < Nq + Nk) {
    const int i = b - Nq;
    gemv_row_dequant<false>(a, wk + static_cast<std::size_t>(i) * K, sk, K, ok, i);
  } else {
    const int i = b - Nq - Nk;
    if (i >= Nv) return;
    gemv_row_dequant<false>(a, wv + static_cast<std::size_t>(i) * K, sv, K, ov, i);
  }
}


__global__ void row_gemm_dequant_qkv_batched_kernel(
    const std::uint16_t* a, const std::uint8_t* wq, const std::uint8_t* wk,
    const std::uint8_t* wv, float sq, float sk, float sv, int tokens,
    int Nq, int Nk, int Nv, int K, std::uint16_t* oq, std::uint16_t* ok,
    std::uint16_t* ov) {
  const int b = blockIdx.x;
  const int total = Nq + Nk + Nv;
  const int token = b / total;
  const int row = b % total;
  if (token >= tokens) return;
  const std::uint16_t* at = a + static_cast<std::size_t>(token) * K;
  if (row < Nq) {
    gemv_row_dequant<false>(at, wq + static_cast<std::size_t>(row) * K, sq, K,
                            oq + static_cast<std::size_t>(token) * Nq, row);
  } else if (row < Nq + Nk) {
    const int i = row - Nq;
    gemv_row_dequant<false>(at, wk + static_cast<std::size_t>(i) * K, sk, K,
                            ok + static_cast<std::size_t>(token) * Nk, i);
  } else {
    const int i = row - Nq - Nk;
    if (i >= Nv) return;
    gemv_row_dequant<false>(at, wv + static_cast<std::size_t>(i) * K, sv, K,
                            ov + static_cast<std::size_t>(token) * Nv, i);
  }
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
int granite4_kernel_gqa_fused(const std::uint16_t* d_q, const std::uint16_t* d_k,
                               const std::uint16_t* d_v, float att_scale, int Hq,
                               int Hk, int d, int kv_len, int k_stride,
                               std::uint16_t* d_out);
int granite4_kernel_kv_append_batched(const std::uint16_t* d_src_k,
                                      const std::uint16_t* d_src_v,
                                      std::uint16_t* d_cache_k,
                                      std::uint16_t* d_cache_v, int base_pos,
                                      int rows, int stride, int Hk, int d);
int granite4_kernel_gqa_fused_batched(const std::uint16_t* d_q,
                                      const std::uint16_t* d_k,
                                      const std::uint16_t* d_v, float att_scale,
                                      int Hq, int Hk, int d, int base_kv,
                                      int rows, int k_stride,
                                      std::uint16_t* d_out);
int granite4_kernel_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k,
    const std::uint16_t* d_v, const float* d_cos, const float* d_sin,
    float att_scale, int base_pos, int rows, int Hq, int Hk, int d,
    int cache_stride, std::uint16_t* d_q_rot, std::uint16_t* d_k_rot,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v, std::uint16_t* d_out);
int granite4_kernel_qkv_dequant_batched(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int tokens,
    int Nq, int Nk, int Nv, int K, std::uint16_t* d_q, std::uint16_t* d_k,
    std::uint16_t* d_v);
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
  std::uint16_t* logits_scratch = nullptr;  // [Hq][max_pos] (fallback path)
  std::uint16_t* probs_scratch = nullptr;   // [Hq][max_pos] (fallback path)
  std::uint16_t* scratch = nullptr;
  bool gqa_fusable = true;  // fused GQA fits in the default smem limit
  bool direct_kv = false;   // A/B: fused GQA indexes the persistent cache

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
  // Fused decode GQA keeps [q row + logits + probs] in shared memory:
  // d + 2*max_positions bf16 elements. The default per-block limit is 48 KiB
  // (no attribute bump); beyond it the 3-kernel scratch path runs.
  g->gqa_fusable =
      static_cast<std::size_t>(g->d) + 2u * max_positions <= 49152u / 2u;
  // Direct-cache GQA (skip the per-step prefix gather) is an A/B switch.
  {
    const char* dkv = std::getenv("G4_DIRECT_KV");
    g->direct_kv = dkv == nullptr || dkv[0] == '1';
  }
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

  // Optional per-call GPU timing: G4_TIMESTEP=1 prints the device-side
  // duration of this call; G4_TIMESTEP=2 additionally prints a per-bucket
  // breakdown (embed / attention / FFN / lm_head, summed over layers).
  const char* const ts_env = std::getenv("G4_TIMESTEP");
  const bool timed = ts_env != nullptr;
  const bool bucket = timed && ts_env[0] == '2';
  if (timed) {
    if (cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess)
      return seterr("cudaEventCreate failed");
    G4_CUDA_CHECK(cudaEventRecord(ev0));
  }

  // Per-bucket events (G4_TIMESTEP=2): [bucket][layer] start/end pairs.
  constexpr int kBuckets = 7;  // 0 embed, 1 attn-gemm, 2 attn-attn, 3 attn-out,
                               // 4 ffn-gemm, 5 ffn-out, 6 lm_head
  constexpr int kMaxLayers = 64;
  std::vector<cudaEvent_t> bevs;
  auto stamp = [&](int b, int i, int which) {
    if (!bucket) return;
    cudaError_t e = cudaEventRecord(
        bevs[(b * kMaxLayers + i) * 2 + which]);
    if (e != cudaSuccess)
      std::fprintf(stderr, "granite4: stamp cudaEventRecord: %s\n",
                   cudaGetErrorString(e));
  };
  if (bucket) {
    bevs.resize(2 * kBuckets * kMaxLayers);
    for (auto& e : bevs)
      if (cudaEventCreate(&e) != cudaSuccess)
        return seterr("cudaEventCreate failed");
  }

  // Per-GEMV in-context timing (G4_TIMESTEP=3): [gemv][layer] start/end.
  // kGemvs: 0 qkv, 1 o, 2 gateup, 3 down, 4 lm_head.
  constexpr int kGemvs = 5;
  std::vector<cudaEvent_t> gevs;
  auto gstamp = [&](int g, int i, int which) {
    if (!bucket) return;
    cudaError_t e = cudaEventRecord(gevs[(g * kMaxLayers + i) * 2 + which]);
    if (e != cudaSuccess)
      std::fprintf(stderr, "granite4: gstamp cudaEventRecord: %s\n",
                   cudaGetErrorString(e));
  };
  if (bucket) {
    gevs.resize(2 * kGemvs * kMaxLayers);
    for (auto& e : gevs)
      if (cudaEventCreate(&e) != cudaSuccess)
        return seterr("cudaEventCreate failed");
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

  // Host wall-clock of the whole launch+sync span (G4_TIMESTEP=2): the
  // decisive launch-bound vs GPU-bound test. If wall >> GPU timeline, the
  // ~680 small kernel launches per step are stalling on the CPU, not the
  // card - the fix is then CUDA graphs, not a faster kernel.
  std::chrono::steady_clock::time_point wall_start;
  if (bucket) wall_start = std::chrono::steady_clock::now();

  for (int t = start; t < end; ++t) {
    // ---------- embedding lookup (token id from device memory) ----------
    stamp(0, 0, 0);
    embed_kernel<<<(H + 255) / 256, 256>>>(d_tokens, d_embed_tbl, t, emb_scale,
                                           d_hidden, H);
    G4_CUDA_CHECK(cudaGetLastError());
    stamp(0, 0, 1);

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
      stamp(1, layer, 0);
      if (granite4_kernel_rmsnorm(d_hidden, attn_norm, rms_eps, 1, H, d_normed) != 0)
        return seterr("rmsnorm (attention) failed");
      gstamp(0, layer, 0);
      row_gemm_dequant_qkv_kernel<<<Hq * d + 2 * Hk * d, kRowGemvThreads>>>(
          d_normed, wq, wk, wv, sl[0], sl[1], sl[2], Hq * d, Hk * d, Hk * d, H,
          d_qf, d_k_new, d_v_new);
      G4_CUDA_CHECK(cudaGetLastError());
      gstamp(0, layer, 1);

      stamp(1, layer, 1);
      stamp(2, layer, 0);
      const float* const cos_t = g->cos_tbl + static_cast<std::size_t>(t) * half;
      const float* const sin_t = g->sin_tbl + static_cast<std::size_t>(t) * half;
      rope_at_pos_qk_kernel<<<(Hq * d + Hk * d + 255) / 256, 256>>>(
          d_qf, d_k_new, cos_t, sin_t, Hq, Hk, d, d_qf_rot, d_k_new_rot);
      G4_CUDA_CHECK(cudaGetLastError());

      // Append this position's RoPE'd K and raw V to the history. In direct
      // mode attention reads the persistent cache rows straight out of
      // [Hk][max_pos][d]; otherwise gather the causal prefix into the
      // contiguous [Hk][t+1][d] buffers the gqa kernels consume.
      const int kv_len = t + 1;
      const bool direct_kv = g->direct_kv && g->gqa_fusable;
      if (direct_kv) {
        kv_append_kernel<<<(Hk * d + 255) / 256, 256>>>(
            d_k_new_rot, d_v_new, g->kv_k + kv_layer, g->kv_v + kv_layer, t, max_pos,
            Hk, d);
      } else {
        kv_append_compact_kernel<<<(Hk * d + 255) / 256, 256>>>(
            d_k_new_rot, d_v_new, g->kv_k + kv_layer, g->kv_v + kv_layer,
            g->compact_k, g->compact_v, kv_len, t, max_pos, Hk, d);
      }
      G4_CUDA_CHECK(cudaGetLastError());

      stamp(2, layer, 1);
      stamp(3, layer, 0);
      if (g->gqa_fusable) {
        const std::uint16_t* const k_attn = direct_kv ? g->kv_k + kv_layer : g->compact_k;
        const std::uint16_t* const v_attn = direct_kv ? g->kv_v + kv_layer : g->compact_v;
        const int k_stride = direct_kv ? max_pos : kv_len;
        if (granite4_kernel_gqa_fused(d_qf_rot, k_attn, v_attn, att_scale, Hq, Hk,
                                      d, kv_len, k_stride, d_attn_out) != 0)
          return seterr("gqa_fused failed");
      } else {
        // Fallback for very long sequences: the classic three-kernel path
        // over global scratch (the fused kernel's shared-memory row exceeds
        // the 48 KiB default limit).
        if (granite4_kernel_gqa_logits(d_qf_rot, g->compact_k, att_scale, Hq, Hk,
                                       d, kv_len, g->logits_scratch) != 0)
          return seterr("gqa_logits failed");
        if (granite4_kernel_softmax(g->logits_scratch, 1.0f, Hq, kv_len,
                                   g->probs_scratch) != 0)
          return seterr("softmax failed");
        if (granite4_kernel_gqa_value(g->probs_scratch, g->compact_v, Hq, Hk, d,
                                     kv_len, d_attn_out) != 0)
          return seterr("gqa_value failed");
      }
      gstamp(1, layer, 0);
      row_gemm_dequant_resid_kernel<<<H, kRowGemvThreads>>>(
          d_attn_out, wo, sl[3], d_hidden, resid_scale, H, H, d_hidden);
      G4_CUDA_CHECK(cudaGetLastError());
      gstamp(1, layer, 1);

      stamp(3, layer, 1);
      // ---- FFN block ----
      stamp(4, layer, 0);
      if (granite4_kernel_rmsnorm(d_hidden, ffn_norm, rms_eps, 1, H, d_normed2) != 0)
        return seterr("rmsnorm (ffn) failed");
      gstamp(2, layer, 0);
      if (granite4_kernel_silu_gate_up(wg, wu, d_normed2, sl[4], sl[5], I, H,
                                       d_gate_out) != 0)
        return seterr("silu_gate_up failed");
      gstamp(2, layer, 1);
      stamp(4, layer, 1);
      stamp(5, layer, 0);
      gstamp(3, layer, 0);
      row_gemm_dequant_resid_kernel<<<H, kRowGemvThreads>>>(
          d_gate_out, wd, sl[6], d_hidden, resid_scale, H, I, d_hidden);
      G4_CUDA_CHECK(cudaGetLastError());
      gstamp(3, layer, 1);
      stamp(5, layer, 1);
    }
    stamp(6, 0, 0);

    // ---------- final RMSNorm + lm_head ----------
    const std::uint16_t* const d_final_norm =
        reinterpret_cast<const std::uint16_t*>(g->weights + lay.final_norm_off());
    const std::uint8_t* const d_lm_head = g->weights + lay.lm_head_off();
    // Fold in the spec's logits_scaling (same single final multiply as the
    // host oracle).
    const float lm_scale = g->scales[L * kGpuScalesPerLayer] * logits_scale;
    if (granite4_kernel_rmsnorm(d_hidden, d_final_norm, rms_eps, 1, H, d_normed) != 0)
      return seterr("final rmsnorm failed");
    gstamp(4, 0, 0);
    row_gemm_dequant_kernel<<<V, kRowGemvThreads>>>(
        d_normed, d_lm_head, lm_scale, V, H,
        d_logits + static_cast<std::size_t>(t) * V);
    G4_CUDA_CHECK(cudaGetLastError());
    gstamp(4, 0, 1);
    stamp(6, 0, 1);
  }

  if (timed) {
    G4_CUDA_CHECK(cudaEventRecord(ev1));
    G4_CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev0, ev1);
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    if (bucket) {
      const double wall_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - wall_start)
              .count();
      std::fprintf(stderr,
                   "G4_BUCKET end=%d start=%d gpu=%.2fms wall=%.2fms ", end,
                   start, ms, wall_ms);
      const char* names[kBuckets] = {"embed", "attn_g", "attn_a", "attn_o",
                                     "ffn_g", "ffn_o", "lm"};
      std::fprintf(stderr, "total=%.2fms ", ms);
      for (int b = 0; b < kBuckets; ++b) {
        float sum = 0.0f;
        const int n = (b == 0 || b == 6) ? 1 : L;
        for (int i = 0; i < n; ++i) {
          float d = 0.0f;
          G4_CUDA_CHECK(cudaEventElapsedTime(&d,
                                             bevs[(b * kMaxLayers + i) * 2],
                                             bevs[(b * kMaxLayers + i) * 2 + 1]));
          sum += d;
        }
        std::fprintf(stderr, "%s=%.2f ", names[b], sum);
      }
      const char* gnames[kGemvs] = {"qkv", "o", "gateup", "down", "lm"};
      for (int gv = 0; gv < kGemvs; ++gv) {
        float sum = 0.0f;
        const int n = (gv == 4) ? 1 : L;
        for (int i = 0; i < n; ++i) {
          float d = 0.0f;
          G4_CUDA_CHECK(cudaEventElapsedTime(&d,
                                             gevs[(gv * kMaxLayers + i) * 2],
                                             gevs[(gv * kMaxLayers + i) * 2 + 1]));
          sum += d;
        }
        std::fprintf(stderr, "g%s=%.2f ", gnames[gv], sum);
      }
      std::fprintf(stderr, "\n");
      for (auto& e : gevs) cudaEventDestroy(e);
    } else {
      std::fprintf(stderr, "G4_TIMESTEP end=%d start=%d gpu_ms=%.2f\n", end,
                   start, ms);
    }
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

int gpu_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k,
    const std::uint16_t* d_v, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int cache_stride, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k,
    std::uint16_t* d_cache_v, std::uint16_t* d_out) {
  if (d_q == nullptr || d_k == nullptr || d_v == nullptr || d_cos == nullptr ||
      d_sin == nullptr || d_q_rot == nullptr || d_k_rot == nullptr ||
      d_cache_k == nullptr || d_cache_v == nullptr || d_out == nullptr)
    return -1;
  return granite4_kernel_verify_attention_stage(
      d_q, d_k, d_v, d_cos, d_sin, attention_scale, base_pos, rows,
      q_heads, kv_heads, head_dim, cache_stride, d_q_rot, d_k_rot,
      d_cache_k, d_cache_v, d_out);
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

int gpu_ctx_truncate(GpuCtx* ctx, int written) {
  if (ctx == nullptr || written < 0 || written > ctx->written ||
      written > ctx->max_pos)
    return -1;
  ctx->written = written;
  return 0;
}

void gpu_ctx_reset(GpuCtx* ctx) {
  (void)gpu_ctx_truncate(ctx, 0);
}

// Test launcher for the coalesced GEMV (consumed by tests/test_kernels.cu and
// the verify-GEMV parity test): one block per output row, at the forward's
// launch width. The width matters for bit-identity - the per-lane strided
// accumulation depends on blockDim - so it must stay kRowGemvThreads.
extern "C" int granite4_kernel_row_gemm_dequant(const std::uint16_t* d_a,
                                                const std::uint8_t* d_B,
                                                float scale, int N, int K,
                                                std::uint16_t* d_out) {
  row_gemm_dequant_kernel<<<N, kRowGemvThreads>>>(d_a, d_B, scale, N, K, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_qkv_dequant_batched(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int tokens, int nq,
    int nk, int nv, int k, std::uint16_t* d_q, std::uint16_t* d_kk,
    std::uint16_t* d_v) {
  if (tokens <= 0 || nq <= 0 || nk <= 0 || nv <= 0 || k <= 0) return -1;
  row_gemm_dequant_qkv_batched_kernel<<<tokens * (nq + nk + nv), kRowGemvThreads>>>(
      d_a, d_wq, d_wk, d_wv, sq, sk, sv, tokens, nq, nk, nv, k, d_q, d_kk, d_v);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_qkv_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int nq, int nk, int k,
    std::uint16_t* d_q, std::uint16_t* d_kk, std::uint16_t* d_v) {
  if (nq <= 0 || nk <= 0 || k <= 0) return -1;
  row_gemm_dequant_qkv_kernel<<<nq + 2 * nk, kRowGemvThreads>>>(
      d_a, d_wq, d_wk, d_wv, sq, sk, sv, nq, nk, nk, k, d_q, d_kk, d_v);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_row_gemm_resid_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_B, float scale,
    const std::uint16_t* d_residual, float residual_scale, int tokens, int N, int K,
    std::uint16_t* d_out) {
  if (tokens <= 0 || N <= 0 || K <= 0) return -1;
  row_gemm_dequant_resid_batched_kernel<<<tokens * N, kRowGemvThreads>>>(
      d_a, d_B, scale, d_residual, residual_scale, tokens, N, K, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

// Test launchers for the production one-position RoPE kernel (the verify
// reference) and its batched verify form.
extern "C" int granite4_kernel_rope_at_pos(const std::uint16_t* d_x,
                                           const float* d_cos,
                                           const float* d_sin, int heads, int d,
                                           std::uint16_t* d_out) {
  rope_at_pos_kernel<<<(heads * d + 255) / 256, 256>>>(d_x, d_cos, d_sin,
                                                      heads, d, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_rope_at_pos_batched(
    const std::uint16_t* d_x, const float* d_cos, const float* d_sin, int rows,
    int heads, int d, std::uint16_t* d_out) {
  if (rows <= 0 || rows > 4 || heads <= 0 || d <= 0 || (d & 1) != 0) return -1;
  const std::size_t n = static_cast<std::size_t>(rows) * heads * d;
  rope_at_pos_batched_kernel<<<(n + 255) / 256, 256>>>(
      d_x, d_cos, d_sin, rows, heads, d, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

// Verified attention staging for speculative rows: rotate projected Q/K using
// position-specific tables, then append the rotated K and raw V atomically as
// one host-visible operation. GQA consumes the resulting buffers afterwards.
extern "C" int granite4_kernel_verify_qkv_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k,
    const std::uint16_t* d_v, const float* d_cos, const float* d_sin,
    int base_pos, int rows, int Hq, int Hk, int d, int cache_stride,
    std::uint16_t* d_q_rot, std::uint16_t* d_k_rot,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v) {
  if (rows <= 0 || rows > 4 || Hq <= 0 || Hk <= 0 || d <= 0 ||
      (d & 1) != 0 || cache_stride <= base_pos + rows - 1) return -1;
  const std::size_t nq = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t nk = static_cast<std::size_t>(rows) * Hk * d;
  rope_at_pos_batched_kernel<<<(nq + 255) / 256, 256>>>(
      d_q, d_cos, d_sin, rows, Hq, d, d_q_rot);
  G4_CUDA_CHECK(cudaGetLastError());
  rope_at_pos_batched_kernel<<<(nk + 255) / 256, 256>>>(
      d_k, d_cos, d_sin, rows, Hk, d, d_k_rot);
  G4_CUDA_CHECK(cudaGetLastError());
  return granite4_kernel_kv_append_batched(
      d_k_rot, d_v, d_cache_k, d_cache_v, base_pos, rows,
      cache_stride, Hk, d);
}

// Complete attention-stage verify primitive. Q/K are already projected; this
// composes position-aware RoPE, persistent KV append, and causal batched GQA.
extern "C" int granite4_kernel_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k,
    const std::uint16_t* d_v, const float* d_cos, const float* d_sin,
    float att_scale, int base_pos, int rows, int Hq, int Hk, int d,
    int cache_stride, std::uint16_t* d_q_rot, std::uint16_t* d_k_rot,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v, std::uint16_t* d_out) {
  const int staged = granite4_kernel_verify_qkv_stage(
      d_q, d_k, d_v, d_cos, d_sin, base_pos, rows, Hq, Hk, d, cache_stride,
      d_q_rot, d_k_rot, d_cache_k, d_cache_v);
  if (staged != 0) return staged;
  return granite4_kernel_gqa_fused_batched(
      d_q_rot, d_cache_k, d_cache_v, att_scale, Hq, Hk, d, base_pos,
      rows, cache_stride, d_out);
}

}  // namespace granite4
