// Phase 3/4 (GPU forward): device forward pass for Granite 4.2 8B.
//
// Mirrors the host oracle forward_logits_f32 (kernel_math.h)
// operation-for-operation:
//   embed -> (per layer: rmsnorm -> QKV -> rope -> kv-append -> gqa -> o
//             -> +residual -> rmsnorm -> swiglu -> down -> +residual)
//   -> rmsnorm -> lm_head
//
// Conventions (shared with the P2 kernels):
//   * BF16 activations; GEMM weights FP8 e4m3, per-tensor dequant scale
//     (the P1 storage decision - Ada has no block-scaled MMA).
//   * Logit scale = attention_multiplier, applied to the logits; the softmax
//     then runs with scale 1.0 - exactly like the host oracle.
//   * KV: a persistent [L][Hk][max_pos][d] history per K and V. Attention
//     reads it in place (direct KV); G4_DIRECT_KV=0 restores the gather of
//     the causal prefix into a contiguous [Hk][t+1][d] buffer first.
//
// Execution paths (forward_impl picks one per call, by suffix length):
//   * 1 row: forward_position - decode GEMVs, split-KV attention, every
//     kernel position-indexed from device memory so gpu_decode_greedy can
//     capture it once as a CUDA graph and chain greedy steps on the device.
//   * 2..4 rows: forward_rows_batched - speculative verify, batched GEMVs
//     that stream each weight row once for all rows.
//   * 5+ rows: forward_prefill - tensor-core GEMMs and attention in chunks
//     (prefill_gemm.cu).
//
// Kernels in this file are the thin glue the P2 set does not cover:
//   * embed_kernel          - embedding lookup (token id from device memory)
//   * kv_append_kernel      - append this position's K/V into the cache
//   * kv_append_compact_kernel - append + gather the causal prefix [0..t]
//                             (G4_DIRECT_KV=0 / long-context fallback)
//   * rope_at_pos_*_kernel  - Llama half-split RoPE at one position, all
//                             heads (the P2 rope_kernel indexes its table by
//                             row, so a per-position call would need one
//                             table row per head; this folds the position
//                             into the table pointer. Formula identical to
//                             the P2 kernel.)
//   * row_gemm_dequant_*    - the M=1 form of the P2 gemm_dequant kernel
//   * argmax_rows_kernel    - device greedy pick over a logits row
//
// Compiled only with GRANITE_ENABLE_CUDA=ON, targets sm_89.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <new>
#include <vector>

#include "forward_gpu.h"      // GpuWeightsLayout
#include "forward_gpu_api.h"  // public C ABI
#include "model_config.h"     // ModelConfig, ForwardSpec
#include "attention_gpu.h"
#include "prefill_gemm.h"

#define GRANITE_CUDA 1
#include "device_math.cuh"
#include "kernel_math.h"

#include <cuda_runtime.h>

namespace granite4 {
namespace {

using dev::bf16_hi;
using dev::bf16_lo;
using dev::bf16_to_f32;
using dev::f32_to_bf16;
using dev::fp8_to_f32;
using dev::fp8x4_to_f32;

// ---------------------------------------------------------------------------
// Glue kernels (one thread per element; bound-checked).
// ---------------------------------------------------------------------------

// The single-position decode kernels read the position from device memory
// (*d_pos), so one captured CUDA graph replays the step at any position.

// out[j] = mult * emb[tokens[*d_pos], j]
__global__ void embed_kernel(const std::int32_t* tokens, const int* d_pos,
                             const std::uint16_t* emb, float mult, std::uint16_t* out,
                             int H) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const std::int32_t tok = tokens[*d_pos];
  out[j] =
      f32_to_bf16(mult * bf16_to_f32(emb[static_cast<std::size_t>(tok) * H + j]));
}

// Store position *d_pos's RoPE'd K and raw V in the persistent
// [Hk][stride][d] cache; direct-cache GQA reads the rows in place.
__global__ void kv_append_kernel(const std::uint16_t* src_k, const std::uint16_t* src_v,
                                 std::uint16_t* cache_k, std::uint16_t* cache_v,
                                 const int* d_pos, int stride, int Hk, int d) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * d) return;
  const int t = *d_pos;
  const int hkv = idx / d;
  const int i = idx % d;
  const std::size_t off = (static_cast<std::size_t>(hkv) * stride + t) * d + i;
  cache_k[off] = src_k[idx];
  cache_v[off] = src_v[idx];
}

__global__ void set_pos_kernel(int* d_pos, int t) { *d_pos = t; }

// Greedy pick per logits row: the first index of the row maximum (the host
// sample_greedy_f32 / argmax_f32 tie rule). One block per row. With d_pos
// (single-row decode) the token goes to out[*d_pos + 1] and *d_pos advances,
// which chains captured decode steps on the device.
constexpr int kArgmaxThreads = 1024;
__global__ void __launch_bounds__(kArgmaxThreads)
    argmax_rows_kernel(const std::uint16_t* logits, int V, std::int32_t* out,
                       int* d_pos) {
  const int row = blockIdx.x;
  const std::uint16_t* const p = logits + static_cast<std::size_t>(row) * V;
  float bv = -INFINITY;
  int bi = INT_MAX;
  for (int i = threadIdx.x; i < V; i += blockDim.x) {
    const float v = bf16_to_f32(p[i]);
    if (v > bv) {  // i ascends per thread: strict > keeps the first max
      bv = v;
      bi = i;
    }
  }
  auto better = [](float v, int i, float bv_, int bi_) {
    return v > bv_ || (v == bv_ && i < bi_);
  };
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float ov = __shfl_down_sync(0xffffffffu, bv, off);
    const int oi = __shfl_down_sync(0xffffffffu, bi, off);
    if (better(ov, oi, bv, bi)) {
      bv = ov;
      bi = oi;
    }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    sv[warp] = bv;
    si[warp] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
      if (better(sv[w], si[w], bv, bi)) {
        bv = sv[w];
        bi = si[w];
      }
    const int best = bi == INT_MAX ? 0 : bi;
    if (d_pos != nullptr) {
      const int pos = *d_pos;
      out[pos + 1] = best;
      *d_pos = pos + 1;
    } else {
      out[row] = best;
    }
  }
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

// Llama half-split RoPE at position *d_pos, fused over the q and k
// projections (one launch instead of two). Same formula and per-element
// arithmetic as rope_at_pos_kernel.
__global__ void rope_at_pos_qk_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                      const float* cos_tbl, const float* sin_tbl,
                                      const int* d_pos, int Hq, int Hk, int d,
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
  const std::size_t row = static_cast<std::size_t>(*d_pos) * half;
  const float c = cos_tbl[row + i];
  const float s = sin_tbl[row + i];
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
        const std::uint32_t y0 = a4[e / 2];      // elements e, e+1
        const std::uint32_t y1 = a4[e / 2 + 1];  // elements e+2, e+3
        float b0, b1, b2, b3;
        fp8x4_to_f32(ws[q], b0, b1, b2, b3);
        acc += b0 * bf16_lo(y0) + b1 * bf16_hi(y0) + b2 * bf16_lo(y1) +
               b3 * bf16_hi(y1);
      }
    }
  } else if ((K & 3) == 0) {
    const std::uint32_t* const row4 =
        reinterpret_cast<const std::uint32_t*>(row);
    const int K4 = K >> 2;
    for (int k4 = t; k4 < K4; k4 += nt) {
      const std::uint32_t x0 = a4[2 * k4];
      const std::uint32_t x1 = a4[2 * k4 + 1];
      // Little-endian: byte j of the word = element 4*k4+j.
      float b0, b1, b2, b3;
      fp8x4_to_f32(row4[k4], b0, b1, b2, b3);
      acc += b0 * bf16_lo(x0) + b1 * bf16_hi(x0) + b2 * bf16_lo(x1) +
             b3 * bf16_hi(x1);
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
int granite4_kernel_gqa_fused_pos(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                  const std::uint16_t* d_v, float att_scale, int Hq,
                                  int Hk, int d, int kv_len, int k_stride,
                                  const int* d_pos, std::uint16_t* d_out);
int granite4_kernel_gqa_fused_rows(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                   const std::uint16_t* d_v, float att_scale, int Hq,
                                   int Hk, int d, int base_kv, int rows, int k_stride,
                                   std::uint16_t* d_out);
std::size_t granite4_gqa_fused_smem_bytes(int d, int kv_len_max);
int granite4_gqa_fused_smem_limit();
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
int granite4_kernel_verify_qkv_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    const float* d_cos, const float* d_sin, int base_pos, int rows, int Hq, int Hk,
    int d, int cache_stride, std::uint16_t* d_q_rot, std::uint16_t* d_k_rot,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v);
int granite4_kernel_silu_gate_up(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                 const std::uint16_t* d_x, float scale_g, float scale_u,
                                 int N, int K, std::uint16_t* d_out);
int granite4_kernel_rmsnorm_batched(const std::uint16_t* d_x,
                                    const std::uint16_t* d_w, float eps, int rows,
                                    int cols, std::uint16_t* d_out);
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
// Persistent weight upload, shared across contexts (namespace granite4 -
// GpuWeights, forward-declared in forward_gpu_api.h).
//
// Owns the packed weight blob (device) and the dequant scales (pinned host)
// independently of any GpuCtx. gpu_weights_load uploads into the object;
// every context that needs the model attaches to it with gpu_ctx_use_weights
// and keeps it alive for its own lifetime. One object can therefore serve
// N contexts, so the ~9.5 GiB blob lives once per device instead of once
// per sequence.
// ---------------------------------------------------------------------------
struct GpuWeights {
  std::uint8_t* weights = nullptr;  // [weights_bytes] packed GpuWeightsLayout (device)
  float* scales = nullptr;          // [scales_count] dequant scales (HOST, pinned, host-read)
  std::size_t weights_bytes = 0;
  std::size_t scales_count = 0;
};

// Free the persistent packed weights/scales if loaded (idempotent).
void gpu_weights_free(GpuCtx* ctx);

GpuWeights* gpu_weights_create() {
  GpuWeights* w = new (std::nothrow) GpuWeights();
  return w;
}

void gpu_weights_free_obj(GpuWeights* w) {
  if (w == nullptr) return;
  if (w->weights != nullptr) cudaFree(w->weights);
  if (w->scales != nullptr) cudaFreeHost(w->scales);
  delete w;
}

// Upload packed weights + scales into a weight object (replaces any
// existing upload).
int gpu_weights_load(GpuWeights* w, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error,
                     std::size_t error_size) {
  auto seterr = [&](const char* what) {
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "%s", what);
    return -1;
  };
  if (w == nullptr || h_weights == nullptr || bytes == 0 ||
      h_scales == nullptr || n_scales == 0 ||
      n_scales > SIZE_MAX / sizeof(float))
    return seterr("invalid arguments");
  for (std::size_t i = 0; i < n_scales; ++i) {
    if (!std::isfinite(h_scales[i])) return seterr("non-finite scale");
  }
  // Replacing existing weights: free first (idempotent semantics).
  if (w->weights != nullptr) cudaFree(w->weights);
  if (w->scales != nullptr) cudaFreeHost(w->scales);
  w->weights = nullptr;
  w->scales = nullptr;
  w->weights_bytes = 0;
  w->scales_count = 0;
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
  w->weights = d_weights;
  w->weights_bytes = bytes;
  w->scales = h_scales_pin;
  w->scales_count = n_scales;
  return 0;
}

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
  std::uint16_t* scratch_batched = nullptr;  // token-major [kMaxVerifyRows][...], batched verify forward
  int* d_pos = nullptr;     // device position of the single-position step
  bool gqa_fusable = true;  // fused GQA fits the per-block shared memory at max_pos
  bool direct_kv = false;   // A/B: attention indexes the persistent cache
  // Split-KV attention (attention_split.cu) for decode and verify rows;
  // G4_SPLIT_ATTN=0 selects the fused one-block-per-head GQA instead.
  bool split_attn = false;
  void* attn_scratch = nullptr;

  // Tensor-core prefill scratch (allocated on first use, prefill_rows rows).
  std::uint8_t* prefill = nullptr;
  int prefill_rows = 0;

  // Captured single-position greedy decode step (see gpu_decode_greedy).
  cudaGraphExec_t decode_graph = nullptr;
  const std::int32_t* graph_tokens = nullptr;  // pointers the graph was
  const std::uint16_t* graph_logits = nullptr;  // captured with

  float* cos_tbl = nullptr;  // [max_pos][d/2]
  float* sin_tbl = nullptr;
  int written = 0;  // KV positions written so far

  // Persistent packed weights + scales (uploaded once via gpu_weights_load;
  // the per-step forward must not re-upload ~9.5 GB). `weights`/`scales` are
  // raw pointers into a GpuWeights object so every view helper keeps
  // reading them unchanged. The object is either context-owned
  // (gpu_weights_load on the ctx) or externally shared
  // (gpu_ctx_use_weights), recorded in owns_weights so gpu_ctx_free frees
  // it only when it owns it.
  std::uint8_t* weights = nullptr;  // [weights_bytes] packed GpuWeightsLayout (device)
  float* scales = nullptr;          // [num_layers*kGpuScalesPerLayer+1] dequant scales (HOST)
  GpuWeights* weights_obj = nullptr;  // owner of the above buffers
  bool owns_weights = false;
};

// ---------------------------------------------------------------------------
// Public API (forward_gpu_api.h), namespace granite4.
// ---------------------------------------------------------------------------
GpuCtx* gpu_ctx_create(const ModelConfig& model, const ForwardSpec& spec,
                       int max_positions) {
  if (max_positions <= 0 || model.num_layers == 0 || model.hidden_size == 0 ||
      model.intermediate_size == 0 || model.num_attention_heads == 0 ||
      model.num_key_value_heads == 0 || model.head_dim == 0 ||
      model.vocab_size == 0 || model.num_attention_heads < model.num_key_value_heads ||
      model.num_attention_heads % model.num_key_value_heads != 0 ||
      model.head_dim > static_cast<std::size_t>(INT_MAX) ||
      model.head_dim > static_cast<std::size_t>(granite4::kMaxHeadDim) ||
      model.hidden_size > static_cast<std::size_t>(INT_MAX) ||
      model.intermediate_size > static_cast<std::size_t>(INT_MAX) ||
      model.num_layers > static_cast<std::size_t>(INT_MAX) ||
      model.num_attention_heads > static_cast<std::size_t>(INT_MAX) ||
      model.num_key_value_heads > static_cast<std::size_t>(INT_MAX) ||
      model.vocab_size > static_cast<std::size_t>(INT_MAX) ||
      static_cast<unsigned long long>(model.num_attention_heads) * model.head_dim >
          static_cast<unsigned long long>(INT_MAX) ||
      static_cast<unsigned long long>(model.num_key_value_heads) * model.head_dim >
          static_cast<unsigned long long>(INT_MAX) ||
      !std::isfinite(spec.rope_theta) || spec.rope_theta <= 0.0)
    return nullptr;
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

  const auto checked_mul = [](std::size_t a, std::size_t b, std::size_t* out) {
    if (b != 0 && a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
  };
  const auto checked_add = [](std::size_t a, std::size_t b, std::size_t* out) {
    if (a > SIZE_MAX - b) return false;
    *out = a + b;
    return true;
  };
  std::size_t kv_elems = 0;
  std::size_t compact_elems = 0;
  std::size_t scratch_elems = 0;
  std::size_t term = 0;
  if (!checked_mul(static_cast<std::size_t>(g->L), g->Hk, &term) ||
      !checked_mul(term, static_cast<std::size_t>(max_positions), &term) ||
      !checked_mul(term, static_cast<std::size_t>(g->d), &kv_elems) ||
      !checked_mul(static_cast<std::size_t>(g->Hk), static_cast<std::size_t>(max_positions), &term) ||
      !checked_mul(term, static_cast<std::size_t>(g->d), &compact_elems)) {
    delete g;
    return nullptr;
  }
  // Scratch: hidden(H) + normed(H) + normed2(H) + qf(Hq*d) + k_new(Hk*d)
  //          + v_new(Hk*d) + attn_out(Hq*d) + o_out(H) + gate_out(I)
  //          + ffn_out(H) + qf_rot(Hq*d) + k_new_rot(Hk*d)
  if (!checked_mul(5, static_cast<std::size_t>(g->H), &scratch_elems) ||
      !checked_mul(static_cast<std::size_t>(g->Hq), static_cast<std::size_t>(g->d), &term) ||
      !checked_mul(term, 2, &term) || !checked_add(scratch_elems, term, &scratch_elems) ||
      !checked_mul(static_cast<std::size_t>(g->Hk), static_cast<std::size_t>(g->d), &term) ||
      !checked_mul(term, 2, &term) || !checked_add(scratch_elems, term, &scratch_elems) ||
      !checked_add(scratch_elems, static_cast<std::size_t>(g->I), &scratch_elems) ||
      !checked_add(static_cast<std::size_t>(g->Hq), static_cast<std::size_t>(g->Hk), &term) ||
      !checked_mul(term, static_cast<std::size_t>(g->d), &term) ||
      !checked_add(scratch_elems, term, &scratch_elems)) {
    delete g;
    return nullptr;
  }

  // Batched verify scratch: token-major [rows][...] for rows <= kMaxVerifyRows.
  // Per row: hidden(H) + normed(H) + normed2(H) + qf(Hq*d) + k_new(Hk*d)
  //          + v_new(Hk*d) + qf_rot(Hq*d) + k_new_rot(Hk*d) + attn_out(Hq*d)
  //          + gate_out(I) = 3*H + 3*(Hq*d) + 3*(Hk*d) + I.
  std::size_t scratch_batched_elems = 0;
  if (!checked_mul(static_cast<std::size_t>(g->H), 3, &term) ||
      !checked_add(term, static_cast<std::size_t>(g->I), &term) ||
      !checked_mul(static_cast<std::size_t>(g->Hq), static_cast<std::size_t>(g->d), &scratch_batched_elems) ||
      !checked_mul(scratch_batched_elems, 3, &scratch_batched_elems) ||
      !checked_add(term, scratch_batched_elems, &term) ||
      !checked_mul(static_cast<std::size_t>(g->Hk), static_cast<std::size_t>(g->d), &scratch_batched_elems) ||
      !checked_mul(scratch_batched_elems, 3, &scratch_batched_elems) ||
      !checked_add(term, scratch_batched_elems, &term) ||
      !checked_mul(term, kMaxVerifyRows, &scratch_batched_elems)) {
    delete g;
    return nullptr;
  }

  std::size_t kv_bytes = 0;
  std::size_t compact_bytes = 0;
  std::size_t token_scratch_elems = 0;
  std::size_t token_scratch_bytes = 0;
  std::size_t scratch_batched_bytes = 0;
  if (!checked_mul(kv_elems, sizeof(std::uint16_t), &kv_bytes) ||
      !checked_mul(compact_elems, sizeof(std::uint16_t), &compact_bytes) ||
      !checked_mul(static_cast<std::size_t>(g->Hq),
                   static_cast<std::size_t>(max_positions), &token_scratch_elems) ||
      !checked_mul(token_scratch_elems, sizeof(std::uint16_t), &token_scratch_bytes) ||
      !checked_mul(scratch_elems, sizeof(std::uint16_t), &term) ||
      !checked_mul(scratch_batched_elems, sizeof(std::uint16_t),
                   &scratch_batched_bytes)) {
    delete g;
    return nullptr;
  }
  cudaError_t e;
  e = cudaMalloc(&g->kv_k, kv_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->kv_v, kv_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->compact_k, compact_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->compact_v, compact_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->logits_scratch, token_scratch_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->probs_scratch, token_scratch_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->scratch, term);
  if (e == cudaSuccess) e = cudaMalloc(&g->scratch_batched, scratch_batched_bytes);
  if (e == cudaSuccess) e = cudaMalloc(&g->d_pos, sizeof(int));
  g->split_attn = g->d == 128 && g->Hq == 4 * g->Hk;
  {
    const char* sa = std::getenv("G4_SPLIT_ATTN");
    if (sa != nullptr && sa[0] == '0') g->split_attn = false;
  }
  if (e == cudaSuccess && g->split_attn)
    e = cudaMalloc(&g->attn_scratch, attn_split_scratch_bytes(g->Hq, kMaxVerifyRows, max_positions));
  // Fused GQA keeps [q row + logits + probs (+ exp terms)] in shared memory;
  // past the per-block opt-in limit the 3-kernel scratch path runs.
  g->gqa_fusable = granite4_gqa_fused_smem_bytes(g->d, max_positions) <=
                   static_cast<std::size_t>(granite4_gqa_fused_smem_limit());
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
    gpu_ctx_free(g);
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
  cudaFree(ctx->scratch_batched);
  cudaFree(ctx->d_pos);
  cudaFree(ctx->attn_scratch);
  cudaFree(ctx->prefill);
  if (ctx->decode_graph != nullptr) cudaGraphExecDestroy(ctx->decode_graph);
  cudaFree(ctx->cos_tbl);
  cudaFree(ctx->sin_tbl);
  if (ctx->owns_weights && ctx->weights_obj != nullptr)
    gpu_weights_free_obj(ctx->weights_obj);
  ctx->weights_obj = nullptr;
  ctx->weights = nullptr;
  ctx->scales = nullptr;
  ctx->owns_weights = false;
  delete ctx;
}

namespace {

// Multi-token verify GEMVs (verify_gemv.cu): stream each FP8 weight row once
// while accumulating up to kMaxVerifyRows activation rows; each row is
// bit-identical to its single-token decode twin. Declared here because it is
// a standalone TU linked into the same library (no shared header).
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a,
                                    const std::uint8_t* d_b, float scale,
                                    int tokens, int rows, int k,
                                    std::uint16_t* d_c);
extern "C" int granite4_verify_gemv_resid(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                          float scale, const std::uint16_t* d_residual,
                                          float residual_scale, int tokens, int rows, int k,
                                          std::uint16_t* d_out);
extern "C" int granite4_verify_qkv(const std::uint16_t* d_a, const std::uint8_t* d_wq,
                                   const std::uint8_t* d_wk, const std::uint8_t* d_wv,
                                   float sq, float sk, float sv, int tokens, int nq, int nk,
                                   int nv, int k, std::uint16_t* d_q, std::uint16_t* d_k,
                                   std::uint16_t* d_v);
extern "C" int granite4_verify_silu(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                    const std::uint16_t* d_x, float sg, float su, int n,
                                    int k, int tokens, std::uint16_t* d_out);

bool env_enabled(const char* name) {
  const char* v = std::getenv(name);
  return v == nullptr || v[0] != '0';
}

// Attention reads the persistent [Hk][max_pos][d] cache in place (split-KV
// attention has no length limit; fused GQA needs its row in shared memory).
bool direct_path(const GpuCtx* g) {
  return g->direct_kv && (g->split_attn || g->gqa_fusable);
}

// Device views of one layer's packed weights and its KV cache slice.
struct LayerView {
  const std::uint16_t* attn_norm;
  const std::uint16_t* ffn_norm;
  const std::uint8_t *wq, *wk, *wv, *wo, *wg, *wu, *wd;
  const float* sc;  // dequant scales: q k v o gate up down
  std::uint16_t* kv_k;
  std::uint16_t* kv_v;
};

LayerView layer_view(const GpuCtx* g, int layer) {
  const GpuWeightsLayout lay{static_cast<std::size_t>(g->H), static_cast<std::size_t>(g->I),
                             static_cast<std::size_t>(g->V), static_cast<std::size_t>(g->L),
                             static_cast<std::size_t>(g->Hq), static_cast<std::size_t>(g->Hk),
                             static_cast<std::size_t>(g->d)};
  const std::uint8_t* const b =
      g->weights + lay.embed_bytes() + static_cast<std::size_t>(layer) * lay.layer_bytes();
  const std::size_t kv = static_cast<std::size_t>(layer) * g->Hk * g->max_pos * g->d;
  LayerView v;
  v.attn_norm = reinterpret_cast<const std::uint16_t*>(b + lay.off_attn_norm());
  v.ffn_norm = reinterpret_cast<const std::uint16_t*>(b + lay.off_ffn_norm());
  v.wq = b + lay.off_q();
  v.wk = b + lay.off_k();
  v.wv = b + lay.off_v();
  v.wo = b + lay.off_o();
  v.wg = b + lay.off_gate();
  v.wu = b + lay.off_up();
  v.wd = b + lay.off_down();
  v.sc = g->scales + static_cast<std::size_t>(layer) * kGpuScalesPerLayer;
  v.kv_k = g->kv_k + kv;
  v.kv_v = g->kv_v + kv;
  return v;
}

struct HeadView {
  const std::uint16_t* embed;
  const std::uint16_t* final_norm;
  const std::uint8_t* lm_head;
  float lm_scale;  // dequant scale * logits_scaling
};

HeadView head_view(const GpuCtx* g) {
  const GpuWeightsLayout lay{static_cast<std::size_t>(g->H), static_cast<std::size_t>(g->I),
                             static_cast<std::size_t>(g->V), static_cast<std::size_t>(g->L),
                             static_cast<std::size_t>(g->Hq), static_cast<std::size_t>(g->Hk),
                             static_cast<std::size_t>(g->d)};
  HeadView h;
  h.embed = reinterpret_cast<const std::uint16_t*>(g->weights);
  h.final_norm = reinterpret_cast<const std::uint16_t*>(g->weights + lay.final_norm_off());
  h.lm_head = g->weights + lay.lm_head_off();
  h.lm_scale = g->scales[static_cast<std::size_t>(g->L) * kGpuScalesPerLayer] *
               g->spec.logits_scaling;
  return h;
}

// G4_TIMESTEP=2: per-bucket and per-GEMV device time of one position,
// summed over layers, plus host wall time of the launch+sync span.
class StepProbe {
 public:
  enum Bucket { kEmbed, kAttnGemm, kAttnAttn, kAttnOut, kFfnGemm, kFfnOut, kLm, kBuckets };
  enum Gemv { kQkv, kO, kGateUp, kDown, kLmHead, kGemvs };

  bool init(int layers) {
    layers_ = layers;
    ev_.resize(static_cast<std::size_t>(int{kBuckets} + int{kGemvs}) * layers * 2);
    for (auto& e : ev_)
      if (cudaEventCreate(&e) != cudaSuccess) return false;
    wall_ = std::chrono::steady_clock::now();
    return true;
  }
  ~StepProbe() {
    for (auto& e : ev_)
      if (e != nullptr) cudaEventDestroy(e);
  }
  void bucket(int b, int layer, int which) { mark(b, layer, which); }
  void gemv(int gv, int layer, int which) { mark(kBuckets + gv, layer, which); }

  // Prints one G4_BUCKET line (synchronizes on the last recorded event).
  void print(int end, int start, float total_ms) {
    cudaEventSynchronize(ev_[index(kLm, 0, 1)]);
    const double wall = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - wall_)
                            .count();
    std::fprintf(stderr, "G4_BUCKET end=%d start=%d gpu=%.2fms wall=%.2fms total=%.2fms ",
                 end, start, total_ms, wall, total_ms);
    static const char* const names[kBuckets] = {"embed", "attn_g", "attn_a", "attn_o",
                                                "ffn_g", "ffn_o",  "lm"};
    for (int b = 0; b < kBuckets; ++b)
      std::fprintf(stderr, "%s=%.2f ", names[b],
                   sum(b, (b == kEmbed || b == kLm) ? 1 : layers_));
    static const char* const gnames[kGemvs] = {"qkv", "o", "gateup", "down", "lm"};
    for (int gv = 0; gv < kGemvs; ++gv)
      std::fprintf(stderr, "g%s=%.2f ", gnames[gv],
                   sum(kBuckets + gv, gv == kLmHead ? 1 : layers_));
    std::fprintf(stderr, "\n");
  }

 private:
  std::size_t index(int slot, int layer, int which) const {
    return (static_cast<std::size_t>(slot) * layers_ + layer) * 2 + which;
  }
  void mark(int slot, int layer, int which) {
    if (ev_.empty()) return;
    cudaEventRecord(ev_[index(slot, layer, which)]);
  }
  float sum(int slot, int n) const {
    float total = 0.0f;
    for (int i = 0; i < n; ++i) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev_[index(slot, i, 0)], ev_[index(slot, i, 1)]);
      total += ms;
    }
    return total;
  }

  int layers_ = 0;
  std::vector<cudaEvent_t> ev_;
  std::chrono::steady_clock::time_point wall_;
};

// G4_TIMESTEP=1: device time of one forward call.
class CallTimer {
 public:
  explicit CallTimer(bool on) {
    if (on && cudaEventCreate(&e0_) == cudaSuccess && cudaEventCreate(&e1_) == cudaSuccess)
      cudaEventRecord(e0_);
  }
  ~CallTimer() {
    if (e0_ != nullptr) cudaEventDestroy(e0_);
    if (e1_ != nullptr) cudaEventDestroy(e1_);
  }
  bool on() const { return e1_ != nullptr; }
  float stop() {
    if (!on()) return 0.0f;
    cudaEventRecord(e1_);
    cudaEventSynchronize(e1_);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0_, e1_);
    return ms;
  }

 private:
  cudaEvent_t e0_ = nullptr, e1_ = nullptr;
};

// Where the logits/argmax of position t go (nullptr: this row is not
// requested, so the final norm and lm_head are skipped for it).
struct RowOut {
  std::uint16_t* logits;
  std::int32_t* argmax;
  std::uint16_t* hidden;  // final-normed hidden state (GpuForwardOut::d_hidden)
};

int hidden_from(const GpuForwardOut& out) {
  return out.hidden_from >= 0 ? out.hidden_from : out.logits_from;
}

// First row that needs the final norm (end: none).
int first_out_row(const GpuForwardOut& out, int end) {
  int first = end;
  if (out.d_logits != nullptr) first = std::min(first, out.logits_from);
  if (out.d_hidden != nullptr) first = std::min(first, hidden_from(out));
  return first;
}

bool wants_rows(const GpuForwardOut& out) {
  return out.d_logits != nullptr || out.d_hidden != nullptr;
}

RowOut row_out(const GpuForwardOut& out, int t, int V, int H) {
  RowOut ro{nullptr, nullptr, nullptr};
  if (out.d_logits != nullptr && t >= out.logits_from) {
    const std::size_t r = static_cast<std::size_t>(t - out.logits_from);
    ro.logits = out.d_logits + r * V;
    ro.argmax = out.d_argmax != nullptr ? out.d_argmax + r : nullptr;
  }
  if (out.d_hidden != nullptr && t >= hidden_from(out))
    ro.hidden = out.d_hidden + static_cast<std::size_t>(t - hidden_from(out)) * H;
  return ro;
}

// After the final norm of rows [first, end) into `normed` ([end - first][H]):
// copy the requested hidden rows out, then the lm_head (+ argmax) over the
// requested logits rows (verify GEMV up to kMaxVerifyRows rows, else the
// tensor-core GEMM).
int emit_rows(const GpuForwardOut& out, int first, int end, const std::uint16_t* normed,
              int H, int V, const std::uint8_t* lm_head, float lm_scale);

// ---------------------------------------------------------------------------
// Single position *g->d_pos (= t_host). Every kernel of the direct-KV path
// reads the position from device memory, so this exact launch sequence is
// what gpu_decode_greedy captures into a CUDA graph. t_host only drives the
// compact-prefix fallback (G4_DIRECT_KV=0 or past the fused-GQA limit),
// which is never captured.
//
// logits == nullptr skips the final norm and lm_head. chain_tokens != nullptr
// (decode chaining) writes the greedy token to chain_tokens[*d_pos + 1] and
// advances *d_pos; otherwise argmax (optional) receives it.
// ---------------------------------------------------------------------------
int forward_position(GpuCtx* g, const std::int32_t* d_tokens, int t_host,
                     std::uint16_t* logits, std::int32_t* argmax,
                     std::int32_t* chain_tokens, StepProbe* probe,
                     std::uint16_t* hidden = nullptr) {
  const int L = g->L, H = g->H, I = g->I, Hq = g->Hq, Hk = g->Hk, d = g->d, V = g->V;
  const int max_pos = g->max_pos;
  const float rms_eps = g->spec.rms_norm_eps;
  const float att_scale = g->spec.attention_multiplier;
  const float resid_scale = g->spec.residual_multiplier;
  const bool direct = direct_path(g);
  const HeadView hv = head_view(g);
  auto bucket = [&](int b, int layer, int which) {
    if (probe != nullptr) probe->bucket(b, layer, which);
  };
  auto gemv = [&](int gv, int layer, int which) {
    if (probe != nullptr) probe->gemv(gv, layer, which);
  };

  std::uint16_t* const d_hidden = g->scratch;
  std::uint16_t* const d_normed = d_hidden + H;
  std::uint16_t* const d_normed2 = d_normed + H;
  std::uint16_t* const d_qf = d_normed2 + H;
  std::uint16_t* const d_k_new = d_qf + Hq * d;
  std::uint16_t* const d_v_new = d_k_new + Hk * d;
  std::uint16_t* const d_attn_out = d_v_new + Hk * d;
  std::uint16_t* const d_gate_out = d_attn_out + Hq * d;
  // RoPE is not in-place safe (a thread reads its partner element while the
  // partner writes it), so rotate into separate scratch.
  std::uint16_t* const d_qf_rot = d_gate_out + I;
  std::uint16_t* const d_k_new_rot = d_qf_rot + Hq * d;

  bucket(StepProbe::kEmbed, 0, 0);
  embed_kernel<<<(H + 255) / 256, 256>>>(d_tokens, g->d_pos, hv.embed,
                                         g->spec.embedding_multiplier, d_hidden, H);
  G4_CUDA_CHECK(cudaGetLastError());
  bucket(StepProbe::kEmbed, 0, 1);

  const int qkv_rows = (Hq + 2 * Hk) * d;
  for (int layer = 0; layer < L; ++layer) {
    const LayerView lw = layer_view(g, layer);

    // ---- attention block (causal over positions 0..t) ----
    bucket(StepProbe::kAttnGemm, layer, 0);
    if (granite4_kernel_rmsnorm(d_hidden, lw.attn_norm, rms_eps, 1, H, d_normed) != 0)
      return -1;
    gemv(StepProbe::kQkv, layer, 0);
    row_gemm_dequant_qkv_kernel<<<qkv_rows, kRowGemvThreads>>>(
        d_normed, lw.wq, lw.wk, lw.wv, lw.sc[0], lw.sc[1], lw.sc[2], Hq * d, Hk * d,
        Hk * d, H, d_qf, d_k_new, d_v_new);
    G4_CUDA_CHECK(cudaGetLastError());
    gemv(StepProbe::kQkv, layer, 1);
    bucket(StepProbe::kAttnGemm, layer, 1);

    bucket(StepProbe::kAttnAttn, layer, 0);
    rope_at_pos_qk_kernel<<<(qkv_rows - Hk * d + 255) / 256, 256>>>(
        d_qf, d_k_new, g->cos_tbl, g->sin_tbl, g->d_pos, Hq, Hk, d, d_qf_rot,
        d_k_new_rot);
    G4_CUDA_CHECK(cudaGetLastError());
    const int kv_len = t_host + 1;
    if (direct) {
      kv_append_kernel<<<(Hk * d + 255) / 256, 256>>>(d_k_new_rot, d_v_new, lw.kv_k,
                                                      lw.kv_v, g->d_pos, max_pos, Hk, d);
    } else {
      kv_append_compact_kernel<<<(Hk * d + 255) / 256, 256>>>(
          d_k_new_rot, d_v_new, lw.kv_k, lw.kv_v, g->compact_k, g->compact_v, kv_len,
          t_host, max_pos, Hk, d);
    }
    G4_CUDA_CHECK(cudaGetLastError());
    bucket(StepProbe::kAttnAttn, layer, 1);

    bucket(StepProbe::kAttnOut, layer, 0);
    if (direct && g->split_attn) {
      if (attn_split(d_qf_rot, lw.kv_k, lw.kv_v, att_scale, Hq, Hk, d, 0, g->d_pos, 1,
                     max_pos, g->attn_scratch, d_attn_out) != 0)
        return -1;
    } else if (direct) {
      // kv_len = *d_pos + 1 on the device; max_pos bounds the shared memory.
      if (granite4_kernel_gqa_fused_pos(d_qf_rot, lw.kv_k, lw.kv_v, att_scale, Hq, Hk, d,
                                        max_pos, max_pos, g->d_pos, d_attn_out) != 0)
        return -1;
    } else if (g->gqa_fusable) {
      if (granite4_kernel_gqa_fused(d_qf_rot, g->compact_k, g->compact_v, att_scale, Hq,
                                    Hk, d, kv_len, kv_len, d_attn_out) != 0)
        return -1;
    } else {
      // Very long contexts: the classic three-kernel path over global
      // scratch (the fused row exceeds the shared-memory limit).
      if (granite4_kernel_gqa_logits(d_qf_rot, g->compact_k, att_scale, Hq, Hk, d, kv_len,
                                     g->logits_scratch) != 0 ||
          granite4_kernel_softmax(g->logits_scratch, 1.0f, Hq, kv_len,
                                  g->probs_scratch) != 0 ||
          granite4_kernel_gqa_value(g->probs_scratch, g->compact_v, Hq, Hk, d, kv_len,
                                    d_attn_out) != 0)
        return -1;
    }
    gemv(StepProbe::kO, layer, 0);
    row_gemm_dequant_resid_kernel<<<H, kRowGemvThreads>>>(d_attn_out, lw.wo, lw.sc[3],
                                                          d_hidden, resid_scale, H, H,
                                                          d_hidden);
    G4_CUDA_CHECK(cudaGetLastError());
    gemv(StepProbe::kO, layer, 1);
    bucket(StepProbe::kAttnOut, layer, 1);

    // ---- FFN block ----
    bucket(StepProbe::kFfnGemm, layer, 0);
    if (granite4_kernel_rmsnorm(d_hidden, lw.ffn_norm, rms_eps, 1, H, d_normed2) != 0)
      return -1;
    gemv(StepProbe::kGateUp, layer, 0);
    if (granite4_kernel_silu_gate_up(lw.wg, lw.wu, d_normed2, lw.sc[4], lw.sc[5], I, H,
                                     d_gate_out) != 0)
      return -1;
    gemv(StepProbe::kGateUp, layer, 1);
    bucket(StepProbe::kFfnGemm, layer, 1);
    bucket(StepProbe::kFfnOut, layer, 0);
    gemv(StepProbe::kDown, layer, 0);
    row_gemm_dequant_resid_kernel<<<H, kRowGemvThreads>>>(d_gate_out, lw.wd, lw.sc[6],
                                                          d_hidden, resid_scale, H, I,
                                                          d_hidden);
    G4_CUDA_CHECK(cudaGetLastError());
    gemv(StepProbe::kDown, layer, 1);
    bucket(StepProbe::kFfnOut, layer, 1);
  }

  // ---------- final RMSNorm + lm_head (+ greedy pick) ----------
  bucket(StepProbe::kLm, 0, 0);
  gemv(StepProbe::kLmHead, 0, 0);
  if (hidden != nullptr && logits == nullptr) {
    if (granite4_kernel_rmsnorm(d_hidden, hv.final_norm, rms_eps, 1, H, hidden) != 0)
      return -1;
  }
  if (logits != nullptr) {
    if (granite4_kernel_rmsnorm(d_hidden, hv.final_norm, rms_eps, 1, H, d_normed) != 0)
      return -1;
    if (hidden != nullptr)
      G4_CUDA_CHECK(cudaMemcpyAsync(hidden, d_normed, sizeof(std::uint16_t) * H,
                                    cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    row_gemm_dequant_kernel<<<V, kRowGemvThreads>>>(d_normed, hv.lm_head, hv.lm_scale, V,
                                                    H, logits);
    G4_CUDA_CHECK(cudaGetLastError());
    if (chain_tokens != nullptr) {
      argmax_rows_kernel<<<1, kArgmaxThreads>>>(logits, V, chain_tokens, g->d_pos);
      G4_CUDA_CHECK(cudaGetLastError());
    } else if (argmax != nullptr) {
      argmax_rows_kernel<<<1, kArgmaxThreads>>>(logits, V, argmax, nullptr);
      G4_CUDA_CHECK(cudaGetLastError());
    }
  }
  gemv(StepProbe::kLmHead, 0, 1);
  bucket(StepProbe::kLm, 0, 1);
  return 0;
}

// Positions [start, end) one at a time (the reference path, and the fallback
// for every shape the batched paths do not cover).
int forward_positions(GpuCtx* g, const std::int32_t* d_tokens, int start, int end,
                      const GpuForwardOut& out, bool bucketed) {
  for (int t = start; t < end; ++t) {
    StepProbe probe;
    if (bucketed && !probe.init(g->L)) return -1;
    CallTimer timer(bucketed);
    set_pos_kernel<<<1, 1>>>(g->d_pos, t);
    G4_CUDA_CHECK(cudaGetLastError());
    const RowOut ro = row_out(out, t, g->V, g->H);
    if (forward_position(g, d_tokens, t, ro.logits, ro.argmax, nullptr,
                         bucketed ? &probe : nullptr, ro.hidden) != 0)
      return -1;
    if (bucketed) probe.print(t + 1, t, timer.stop());
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Batched suffix forward, rows = end - start in [2, kMaxVerifyRows]
// (speculative verify): one token-major batch through the verify GEMVs, each
// weight row streamed once for all rows. It reads/writes the same persistent
// KV rows as the direct per-position path, and every stage (RMSNorm, the
// verify GEMVs, RoPE, split-KV attention, lm_head) is bit-identical per row
// to the decode step at that position (test_kernels.cu), so the greedy
// verify gate accepts exactly the tokens plain greedy decoding emits.
// ---------------------------------------------------------------------------
int forward_rows_batched(GpuCtx* g, const std::int32_t* d_tokens, int start, int end,
                         const GpuForwardOut& out) {
  const int rows = end - start;
  const int L = g->L, H = g->H, I = g->I, Hq = g->Hq, Hk = g->Hk, d = g->d, V = g->V;
  const int half = d / 2;
  const float rms_eps = g->spec.rms_norm_eps;
  const float resid_scale = g->spec.residual_multiplier;
  const HeadView hv = head_view(g);

  const std::size_t nq = static_cast<std::size_t>(Hq) * d;
  const std::size_t nk = static_cast<std::size_t>(Hk) * d;
  std::uint16_t* const d_hidden = g->scratch_batched;  // [rows][H]
  std::uint16_t* const d_normed = d_hidden + rows * H;
  std::uint16_t* const d_normed2 = d_normed + rows * H;
  std::uint16_t* const d_qf = d_normed2 + rows * H;
  std::uint16_t* const d_k_new = d_qf + rows * nq;
  std::uint16_t* const d_v_new = d_k_new + rows * nk;
  std::uint16_t* const d_qf_rot = d_v_new + rows * nk;
  std::uint16_t* const d_k_new_rot = d_qf_rot + rows * nq;
  std::uint16_t* const d_attn_out = d_k_new_rot + rows * nk;
  std::uint16_t* const d_gate_out = d_attn_out + rows * nq;

  if (prefill_embed_rows(d_tokens, start, rows, hv.embed, g->spec.embedding_multiplier,
                         d_hidden, H) != 0)
    return -1;
  for (int layer = 0; layer < L; ++layer) {
    const LayerView lw = layer_view(g, layer);
    if (granite4_kernel_rmsnorm_batched(d_hidden, lw.attn_norm, rms_eps, rows, H,
                                        d_normed) != 0 ||
        granite4_verify_qkv(d_normed, lw.wq, lw.wk, lw.wv, lw.sc[0], lw.sc[1], lw.sc[2],
                            rows, Hq * d, Hk * d, Hk * d, H, d_qf, d_k_new, d_v_new) != 0 ||
        granite4_kernel_verify_qkv_stage(
            d_qf, d_k_new, d_v_new, g->cos_tbl + static_cast<std::size_t>(start) * half,
            g->sin_tbl + static_cast<std::size_t>(start) * half, start, rows, Hq, Hk, d,
            g->max_pos, d_qf_rot, d_k_new_rot, lw.kv_k, lw.kv_v) != 0 ||
        (g->split_attn
             ? attn_split(d_qf_rot, lw.kv_k, lw.kv_v, g->spec.attention_multiplier, Hq, Hk,
                          d, start, nullptr, rows, g->max_pos, g->attn_scratch, d_attn_out)
             : granite4_kernel_gqa_fused_batched(d_qf_rot, lw.kv_k, lw.kv_v,
                                                 g->spec.attention_multiplier, Hq, Hk, d,
                                                 start, rows, g->max_pos, d_attn_out)) != 0 ||
        granite4_verify_gemv_resid(d_attn_out, lw.wo, lw.sc[3], d_hidden, resid_scale, rows,
                                   H, H, d_hidden) != 0 ||
        granite4_kernel_rmsnorm_batched(d_hidden, lw.ffn_norm, rms_eps, rows, H,
                                        d_normed2) != 0 ||
        granite4_verify_silu(lw.wg, lw.wu, d_normed2, lw.sc[4], lw.sc[5], I, H, rows,
                             d_gate_out) != 0 ||
        granite4_verify_gemv_resid(d_gate_out, lw.wd, lw.sc[6], d_hidden, resid_scale, rows,
                                   H, I, d_hidden) != 0)
      return -1;
  }

  // Final norm + fused-K lm_head over the requested rows only.
  const int first = std::max(start, first_out_row(out, end));
  const int n = end - first;
  if (n > 0) {
    std::uint16_t* const normed = d_normed + static_cast<std::size_t>(first - start) * H;
    if (granite4_kernel_rmsnorm_batched(
            d_hidden + static_cast<std::size_t>(first - start) * H, hv.final_norm,
            rms_eps, n, H, normed) != 0 ||
        emit_rows(out, first, end, normed, H, V, hv.lm_head, hv.lm_scale) != 0)
      return -1;
  }
  return 0;
}

int emit_rows(const GpuForwardOut& out, int first, int end, const std::uint16_t* normed,
              int H, int V, const std::uint8_t* lm_head, float lm_scale) {
  const RowOut ro = row_out(out, std::max(first, hidden_from(out)), V, H);
  if (ro.hidden != nullptr) {
    const int h0 = std::max(first, hidden_from(out));
    G4_CUDA_CHECK(cudaMemcpyAsync(ro.hidden, normed + static_cast<std::size_t>(h0 - first) * H,
                                  sizeof(std::uint16_t) * (end - h0) * H,
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread));
  }
  if (out.d_logits == nullptr) return 0;
  const int l0 = std::max(first, out.logits_from);
  const int n = end - l0;
  if (n <= 0) return 0;
  const RowOut lo = row_out(out, l0, V, H);
  const std::uint16_t* const a = normed + static_cast<std::size_t>(l0 - first) * H;
  if (n <= kMaxVerifyRows) {
    // Same weight stream and per-row arithmetic as the decode lm_head.
    if (granite4_verify_gemv(a, lm_head, lm_scale, n, V, H, lo.logits) != 0) return -1;
  } else {
    GemmEpilogue lm;
    lm.scale0 = lm.scale1 = lm.scale2 = lm_scale;
    lm.out_bf16 = lo.logits;
    lm.ldc = V;
    if (prefill_gemm(GemmOut::kBf16, a, H, lm_head, n, V, H, lm) != 0) return -1;
  }
  if (lo.argmax != nullptr) {
    argmax_rows_kernel<<<n, kArgmaxThreads>>>(lo.logits, V, lo.argmax, nullptr);
    G4_CUDA_CHECK(cudaGetLastError());
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Tensor-core prefill (prefill_gemm.cu): positions [start, end) in chunks of
// up to g->prefill_rows. Every projection is one tensor-core GEMM over the
// chunk (q|k|v and gate|up are stored back to back and run as one GEMM each),
// attention is the causal fused GQA over all rows of the chunk, and the
// lm_head runs only for the rows whose logits were requested.
// ---------------------------------------------------------------------------
constexpr int kPrefillChunk = 512;

struct PrefillScratch {
  std::uint16_t* hidden;  // [C][H]
  std::uint16_t* normed;  // [C][H]
  std::uint16_t* qkv;     // [C][(Hq + 2Hk) d]
  std::uint16_t* q_rot;   // [C][Hq d]
  std::uint16_t* attn;    // [C][Hq d]
  std::uint16_t* act;     // [C][I]
  float* gu;              // [C][2I]
  float* ws;              // split-K workspace
  std::size_t ws_bytes;
};

// Split-K pays off for chunks of one M tile (short prompts), where every
// projection has too few output tiles to keep the card streaming weights.
// Sized for the widest (gate|up) GEMM at a quarter of the maximum split;
// prefill_gemm lowers the split to fit.
std::size_t prefill_ws_bytes(const GpuCtx* g) {
  return prefill_gemm_workspace_bytes(64, 2 * g->I) / 4;
}

std::size_t prefill_row_bytes(const GpuCtx* g) {
  const std::size_t bf16_elems = 2u * g->H + static_cast<std::size_t>(g->Hq + 2 * g->Hk) * g->d +
                                 2u * static_cast<std::size_t>(g->Hq) * g->d + g->I;
  return bf16_elems * sizeof(std::uint16_t) + 2u * g->I * sizeof(float);
}


PrefillScratch prefill_scratch(const GpuCtx* g) {
  const std::size_t C = static_cast<std::size_t>(g->prefill_rows);
  // f32 block first (16-byte alignment for the GEMM epilogue), then bf16.
  PrefillScratch s;
  s.gu = reinterpret_cast<float*>(g->prefill);
  std::uint16_t* p = reinterpret_cast<std::uint16_t*>(s.gu + C * 2 * g->I);
  s.hidden = p;
  p += C * g->H;
  s.normed = p;
  p += C * g->H;
  s.qkv = p;
  p += C * (g->Hq + 2 * g->Hk) * g->d;
  s.q_rot = p;
  p += C * g->Hq * g->d;
  s.attn = p;
  p += C * g->Hq * g->d;
  s.act = p;
  p += C * g->I;
  s.ws = reinterpret_cast<float*>(
      (reinterpret_cast<std::uintptr_t>(p) + 255u) & ~std::uintptr_t(255));
  s.ws_bytes = prefill_ws_bytes(g);
  return s;
}

bool ensure_prefill_scratch(GpuCtx* g) {
  if (g->prefill != nullptr) return true;
  const int rows = std::min(kPrefillChunk, g->max_pos);
  if (cudaMalloc(&g->prefill, prefill_row_bytes(g) * rows + 256 + prefill_ws_bytes(g)) !=
      cudaSuccess) {
    cudaGetLastError();  // clear the sticky-free allocation error
    g->prefill = nullptr;
    return false;
  }
  g->prefill_rows = rows;
  return true;
}

int forward_prefill(GpuCtx* g, const std::int32_t* d_tokens, int start, int end,
                    const GpuForwardOut& out) {
  const int L = g->L, H = g->H, I = g->I, Hq = g->Hq, Hk = g->Hk, d = g->d, V = g->V;
  const int nq = Hq * d, nqkv = (Hq + 2 * Hk) * d;
  const float rms_eps = g->spec.rms_norm_eps;
  const float resid_scale = g->spec.residual_multiplier;
  const HeadView hv = head_view(g);
  const PrefillScratch s = prefill_scratch(g);

  for (int c0 = start; c0 < end; c0 += g->prefill_rows) {
    const int M = std::min(g->prefill_rows, end - c0);
    if (prefill_embed_rows(d_tokens, c0, M, hv.embed, g->spec.embedding_multiplier,
                           s.hidden, H) != 0)
      return -1;
    for (int layer = 0; layer < L; ++layer) {
      const LayerView lw = layer_view(g, layer);
      GemmEpilogue qkv;
      qkv.scale0 = lw.sc[0];
      qkv.scale1 = lw.sc[1];
      qkv.scale2 = lw.sc[2];
      qkv.seg1 = nq;
      qkv.seg2 = nq + Hk * d;
      qkv.out_bf16 = s.qkv;
      qkv.ldc = nqkv;
      GemmEpilogue o;
      o.scale0 = o.scale1 = o.scale2 = lw.sc[3];
      o.out_bf16 = s.hidden;
      o.residual = s.hidden;
      o.resid_scale = resid_scale;
      o.ldc = H;
      GemmEpilogue gu;
      gu.scale0 = lw.sc[4];
      gu.scale1 = gu.scale2 = lw.sc[5];
      gu.seg1 = gu.seg2 = I;
      gu.out_f32 = s.gu;
      gu.ldc = 2 * I;
      GemmEpilogue down = o;
      down.scale0 = down.scale1 = down.scale2 = lw.sc[6];
      if (granite4_kernel_rmsnorm(s.hidden, lw.attn_norm, rms_eps, M, H, s.normed) != 0 ||
          prefill_gemm(GemmOut::kBf16, s.normed, H, lw.wq, M, nqkv, H, qkv, s.ws, s.ws_bytes) != 0 ||
          prefill_rope_split_append(s.qkv, g->cos_tbl, g->sin_tbl, c0, M, Hq, Hk, d,
                                    g->max_pos, s.q_rot, lw.kv_k, lw.kv_v) != 0 ||
          (prefill_attention(s.q_rot, lw.kv_k, lw.kv_v, g->spec.attention_multiplier, Hq,
                             Hk, d, c0, M, g->max_pos, s.attn) != 0 &&
           granite4_kernel_gqa_fused_rows(s.q_rot, lw.kv_k, lw.kv_v,
                                          g->spec.attention_multiplier, Hq, Hk, d, c0, M,
                                          g->max_pos, s.attn) != 0) ||
          prefill_gemm(GemmOut::kResidual, s.attn, nq, lw.wo, M, H, nq, o, s.ws, s.ws_bytes) != 0 ||
          granite4_kernel_rmsnorm(s.hidden, lw.ffn_norm, rms_eps, M, H, s.normed) != 0 ||
          prefill_gemm(GemmOut::kF32, s.normed, H, lw.wg, M, 2 * I, H, gu, s.ws, s.ws_bytes) != 0 ||
          prefill_silu_mul(s.gu, I, M, s.act) != 0 ||
          prefill_gemm(GemmOut::kResidual, s.act, I, lw.wd, M, H, I, down, s.ws, s.ws_bytes) != 0)
        return -1;
    }

    // Logits for the requested rows of this chunk only.
    const int first = std::max(c0, first_out_row(out, end));
    const int n = c0 + M - first;
    if (n <= 0) continue;
    const std::size_t r0 = static_cast<std::size_t>(first - c0) * H;
    if (granite4_kernel_rmsnorm(s.hidden + r0, hv.final_norm, rms_eps, n, H,
                                s.normed + r0) != 0 ||
        emit_rows(out, first, c0 + M, s.normed + r0, H, V, hv.lm_head, hv.lm_scale) != 0)
      return -1;
  }
  return 0;
}

// The forward for positions [start, end), start = min(g->written, end):
// positions below start already have their KV entries cached and are skipped
// (one sequence per ctx). If end <= written the whole prefix is recomputed
// (KV appends are idempotent), so start = 0 then.
//
// Path selection (all read and write the same persistent KV rows):
//   1 row            per-position decode step
//   2..16 rows       batched GEMV suffix (speculative verify)
//   17+ rows         tensor-core prefill
// The batched paths need the direct fused-attention branch; any failure
// falls back to the per-position loop, which recomputes from the embedding
// and re-appends the same KV rows idempotently.
// A/B switches: G4_BATCH_VERIFY=0, G4_PREFILL_GEMM=0 force the per-position
// loop for their shapes.
int forward_impl(GpuCtx* g, const std::int32_t* d_tokens, int end,
                 const GpuForwardOut& out, char* error, std::size_t error_size) {
  auto seterr = [&](const char* what) {
    if (error != nullptr && error_size > 0) std::snprintf(error, error_size, "%s", what);
    return -1;
  };
  if (g == nullptr || d_tokens == nullptr) return seterr("invalid forward arguments");
  if (end <= 0 || end > g->max_pos) return seterr("invalid sequence length");
  if (out.logits_from < 0 || out.logits_from > end ||
      (out.logits_from < end && !wants_rows(out)) ||
      (out.d_hidden != nullptr && (hidden_from(out) < 0 || hidden_from(out) > end)))
    return seterr("invalid logits range");
  const ForwardSpec& sp = g->spec;
  if (!std::isfinite(sp.rms_norm_eps) || sp.rms_norm_eps <= 0.0f ||
      !std::isfinite(sp.attention_multiplier) || !std::isfinite(sp.residual_multiplier) ||
      !std::isfinite(sp.embedding_multiplier) || !std::isfinite(sp.logits_scaling))
    return seterr("invalid forward scales");
  if (g->weights == nullptr || g->scales == nullptr)
    return seterr("weights not loaded (call gpu_weights_load first)");

  const int start = (end > g->written) ? g->written : 0;
  const int rows = end - start;
  const bool direct = direct_path(g);
  const char* const ts = std::getenv("G4_TIMESTEP");
  const bool bucketed = ts != nullptr && ts[0] == '2';
  CallTimer timer(ts != nullptr && !bucketed);

  int rc = -1;
  const char* path = "";
  if (!bucketed && direct && rows >= 2 && rows <= kMaxVerifyRows &&
      env_enabled("G4_BATCH_VERIFY")) {
    rc = forward_rows_batched(g, d_tokens, start, end, out);
    path = " (batched)";
  } else if (!bucketed && direct && rows > kMaxVerifyRows && env_enabled("G4_PREFILL_GEMM") &&
             ensure_prefill_scratch(g)) {
    rc = forward_prefill(g, d_tokens, start, end, out);
    path = " (prefill)";
  }
  if (rc != 0) {
    path = "";
    if (forward_positions(g, d_tokens, start, end, out, bucketed) != 0)
      return seterr("forward failed");
  }
  if (timer.on())
    std::fprintf(stderr, "G4_TIMESTEP end=%d start=%d gpu_ms=%.2f%s\n", end, start,
                 timer.stop(), path);
  if (end > g->written) g->written = end;
  return 0;
}

// Capture forward_position (direct path, chained greedy pick) into a graph
// bound to (d_tokens, d_logits_row).
bool capture_decode_graph(GpuCtx* g, std::int32_t* d_tokens, std::uint16_t* d_logits_row) {
  if (g->decode_graph != nullptr) {
    if (g->graph_tokens == d_tokens && g->graph_logits == d_logits_row) return true;
    cudaGraphExecDestroy(g->decode_graph);
    g->decode_graph = nullptr;
  }
  cudaStream_t s = cudaStreamPerThread;
  if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
    return false;
  const int rc = forward_position(g, d_tokens, -1, d_logits_row, nullptr, d_tokens, nullptr);
  cudaGraph_t graph = nullptr;
  const cudaError_t ce = cudaStreamEndCapture(s, &graph);
  if (rc != 0 || ce != cudaSuccess || graph == nullptr) {
    if (graph != nullptr) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  const cudaError_t ie = cudaGraphInstantiate(&g->decode_graph, graph, 0);
  cudaGraphDestroy(graph);
  if (ie != cudaSuccess) {
    g->decode_graph = nullptr;
    cudaGetLastError();
    return false;
  }
  g->graph_tokens = d_tokens;
  g->graph_logits = d_logits_row;
  return true;
}

}  // namespace

// Public API (forward_gpu_api.h), namespace granite4.
//
// Legacy per-ctx upload: creates (or replaces) a context-OWNED GpuWeights
// and attaches it. The context frees the object on gpu_ctx_free. Use
// gpu_weights_create + gpu_ctx_use_weights for weights shared across
// several contexts.
int gpu_weights_load(GpuCtx* ctx, const std::uint8_t* h_weights,
                     std::size_t bytes, const float* h_scales,
                     std::size_t n_scales, char* error,
                     std::size_t error_size) {
  auto seterr = [&](const char* what) {
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "%s", what);
    return -1;
  };
  if (ctx == nullptr) return seterr("null context");
  // Reuse the context-owned object if there is one (reload in place);
  // otherwise allocate a fresh owned object.
  GpuWeights* w = (ctx->owns_weights && ctx->weights_obj != nullptr)
                      ? ctx->weights_obj
                      : gpu_weights_create();
  if (w == nullptr) return seterr("out of memory (weight object)");
  if (gpu_weights_load(w, h_weights, bytes, h_scales, n_scales, error,
                       error_size) != 0) {
    // Release the object either way (we own it, whether fresh or reused)
    // and clear the context: a failed reload leaves the context without
    // weights.
    gpu_weights_free_obj(w);
    ctx->weights_obj = nullptr;
    ctx->weights = nullptr;
    ctx->scales = nullptr;
    ctx->owns_weights = false;
    return -1;
  }
  ctx->weights = w->weights;
  ctx->scales = w->scales;
  ctx->weights_obj = w;
  ctx->owns_weights = true;
  return 0;
}

// Free a context-owned weight object, or detach a shared one (neither
// frees the shared object - the owner does). Idempotent.
void gpu_weights_free(GpuCtx* ctx) {
  if (ctx == nullptr) return;
  if (ctx->owns_weights && ctx->weights_obj != nullptr)
    gpu_weights_free_obj(ctx->weights_obj);
  ctx->weights_obj = nullptr;
  ctx->weights = nullptr;
  ctx->scales = nullptr;
  ctx->owns_weights = false;
}

// Attach an externally-owned weight object (shared across contexts). The
// caller keeps `w` alive for the lifetime of every context that uses it.
// Replaces any previously attached weights. Returns 0, -1 on bad args.
int gpu_ctx_use_weights(GpuCtx* ctx, GpuWeights* w) {
  if (ctx == nullptr || w == nullptr || w->weights == nullptr) return -1;
  // If this context owns a previous object, release it; if it merely used
  // one, just detach (do not free).
  if (ctx->owns_weights && ctx->weights_obj != nullptr)
    gpu_weights_free_obj(ctx->weights_obj);
  ctx->weights_obj = w;
  ctx->weights = w->weights;
  ctx->scales = w->scales;
  ctx->owns_weights = false;
  return 0;
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
      d_cache_k == nullptr || d_cache_v == nullptr || d_out == nullptr ||
      !std::isfinite(attention_scale) || rows < 1 || rows > granite4::kMaxVerifyRows ||
      base_pos < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > granite4::kMaxHeadDim || q_heads < kv_heads || q_heads % kv_heads != 0 ||
      static_cast<long long>(q_heads) * head_dim > INT_MAX ||
      static_cast<long long>(kv_heads) * head_dim > INT_MAX ||
      base_pos > INT_MAX - rows + 1 ||
      cache_stride <= base_pos + rows - 1)
    return -1;
  return granite4_kernel_verify_attention_stage(
      d_q, d_k, d_v, d_cos, d_sin, attention_scale, base_pos, rows,
      q_heads, kv_heads, head_dim, cache_stride, d_q_rot, d_k_rot,
      d_cache_k, d_cache_v, d_out);
}

int gpu_verify_attention_layer_batched(
    const std::uint16_t* d_normed, const std::uint8_t* d_wq,
    const std::uint8_t* d_wk, const std::uint8_t* d_wv, float q_scale,
    float k_scale, float v_scale, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int hidden_size, int cache_stride, std::uint16_t* d_q,
    std::uint16_t* d_k, std::uint16_t* d_v, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_attn_out) {
  if (d_normed == nullptr || d_wq == nullptr || d_wk == nullptr || d_wv == nullptr ||
      d_q == nullptr || d_k == nullptr || d_v == nullptr || d_cos == nullptr ||
      d_sin == nullptr || d_q_rot == nullptr || d_k_rot == nullptr ||
      d_cache_k == nullptr || d_cache_v == nullptr || d_attn_out == nullptr ||
      !std::isfinite(q_scale) || !std::isfinite(k_scale) ||
      !std::isfinite(v_scale) || !std::isfinite(attention_scale) ||
      hidden_size <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rows < 1 || rows > granite4::kMaxVerifyRows || base_pos < 0 ||
      head_dim > granite4::kMaxHeadDim ||
      static_cast<long long>(rows) * hidden_size > INT_MAX ||
      static_cast<long long>(rows) * hidden_size * static_cast<long long>(sizeof(std::uint16_t)) > INT_MAX ||
      static_cast<long long>(q_heads) * head_dim > INT_MAX ||
      static_cast<long long>(kv_heads) * head_dim > INT_MAX ||
      static_cast<long long>(rows) * q_heads * head_dim * static_cast<long long>(sizeof(std::uint16_t)) > INT_MAX ||
      static_cast<long long>(rows) * kv_heads * head_dim * static_cast<long long>(sizeof(std::uint16_t)) > INT_MAX ||
      base_pos > INT_MAX - rows + 1 ||
      cache_stride <= base_pos + rows - 1)
    return -1;
  if (q_heads < kv_heads || q_heads % kv_heads != 0 ||
      static_cast<long long>(q_heads) * head_dim <= 0 ||
      static_cast<long long>(kv_heads) * head_dim <= 0)
    return -1;
  if (granite4_kernel_qkv_dequant_batched(
          d_normed, d_wq, d_wk, d_wv, q_scale, k_scale, v_scale, rows,
          q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim,
          hidden_size, d_q, d_k, d_v) != 0)
    return -1;
  return gpu_verify_attention_stage(
      d_q, d_k, d_v, d_cos, d_sin, attention_scale, base_pos, rows,
      q_heads, kv_heads, head_dim, cache_stride, d_q_rot, d_k_rot,
      d_cache_k, d_cache_v, d_attn_out);
}

int gpu_verify_attention_layer(
    const std::uint16_t* d_normed, const std::uint8_t* d_wq,
    const std::uint8_t* d_wk, const std::uint8_t* d_wv, float q_scale,
    float k_scale, float v_scale, const float* d_cos, const float* d_sin,
    float attention_scale, int base_pos, int rows, int q_heads, int kv_heads,
    int head_dim, int hidden_size, int cache_stride, std::uint16_t* d_q,
    std::uint16_t* d_k, std::uint16_t* d_v, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_attn_out) {
  return gpu_verify_attention_layer_batched(
      d_normed, d_wq, d_wk, d_wv, q_scale, k_scale, v_scale, d_cos, d_sin,
      attention_scale, base_pos, rows, q_heads, kv_heads, head_dim, hidden_size,
      cache_stride, d_q, d_k, d_v, d_q_rot, d_k_rot, d_cache_k, d_cache_v,
      d_attn_out);
}

int gpu_forward(GpuCtx* ctx, const std::int32_t* d_tokens, int end,
                const GpuForwardOut& out, char* error, std::size_t error_size) {
  return forward_impl(ctx, d_tokens, end, out, error, error_size);
}

int gpu_forward_range(GpuCtx* ctx, const std::int32_t* d_tokens, int end,
                      std::uint16_t* d_logits, char* error,
                      std::size_t error_size) {
  if (d_logits == nullptr) {
    if (error != nullptr && error_size > 0)
      std::snprintf(error, error_size, "invalid arguments");
    return -1;
  }
  GpuForwardOut out;
  out.d_logits = d_logits;
  return forward_impl(ctx, d_tokens, end, out, error, error_size);
}

int gpu_decode_greedy(GpuCtx* ctx, std::int32_t* d_tokens, int steps,
                      std::uint16_t* d_logits_row, char* error,
                      std::size_t error_size) {
  auto seterr = [&](const char* what) {
    if (error != nullptr && error_size > 0) std::snprintf(error, error_size, "%s", what);
    return -1;
  };
  if (ctx == nullptr || d_tokens == nullptr || d_logits_row == nullptr || steps <= 0 ||
      ctx->written <= 0 || ctx->written > ctx->max_pos - 1 - steps)
    return seterr("invalid decode arguments");
  if (ctx->weights == nullptr) return seterr("weights not loaded");
  const char* const ts = std::getenv("G4_TIMESTEP");
  const bool direct = direct_path(ctx);
  if (!direct || (ts != nullptr && ts[0] == '2')) {
    // Compact-prefix fallback / bucket profiling: host-driven steps.
    for (int i = 0; i < steps; ++i) {
      GpuForwardOut out;
      out.d_logits = d_logits_row;
      out.logits_from = ctx->written;
      out.d_argmax = d_tokens + ctx->written + 1;
      if (forward_impl(ctx, d_tokens, ctx->written + 1, out, error, error_size) != 0)
        return -1;
    }
    return 0;
  }

  set_pos_kernel<<<1, 1>>>(ctx->d_pos, ctx->written);
  if (cudaGetLastError() != cudaSuccess) return seterr("set_pos launch failed");
  const bool graph = env_enabled("G4_GRAPH") && capture_decode_graph(ctx, d_tokens, d_logits_row);
  std::vector<cudaEvent_t> ev;
  if (ts != nullptr) {
    ev.resize(static_cast<std::size_t>(steps) + 1);
    for (auto& e : ev)
      if (cudaEventCreate(&e) != cudaSuccess) return seterr("cudaEventCreate failed");
  }
  int rc = 0;
  for (int i = 0; i < steps && rc == 0; ++i) {
    if (!ev.empty()) cudaEventRecord(ev[static_cast<std::size_t>(i)]);
    if (graph) {
      rc = cudaGraphLaunch(ctx->decode_graph, cudaStreamPerThread) == cudaSuccess ? 0 : -1;
    } else {
      rc = forward_position(ctx, d_tokens, ctx->written + i, d_logits_row, nullptr, d_tokens,
                            nullptr);
    }
  }
  if (!ev.empty()) {
    cudaEventRecord(ev.back());
    cudaEventSynchronize(ev.back());
    for (int i = 0; i < steps; ++i) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev[static_cast<std::size_t>(i)], ev[static_cast<std::size_t>(i) + 1]);
      std::fprintf(stderr, "G4_TIMESTEP end=%d start=%d gpu_ms=%.2f%s\n",
                   ctx->written + i + 1, ctx->written + i, ms, graph ? " (graph)" : "");
    }
    for (auto& e : ev) cudaEventDestroy(e);
  }
  if (rc != 0) return seterr("decode step failed");
  ctx->written += steps;
  return 0;
}

int gpu_ctx_written(GpuCtx* ctx) { return ctx != nullptr ? ctx->written : -1; }

int gpu_ctx_shared_weights(GpuCtx* ctx, const std::uint16_t** d_embed,
                           const std::uint8_t** d_lm_head, float* lm_scale) {
  if (ctx == nullptr || ctx->weights == nullptr || ctx->scales == nullptr) return -1;
  const HeadView hv = head_view(ctx);
  *d_embed = hv.embed;
  *d_lm_head = hv.lm_head;
  *lm_scale = hv.lm_scale;
  return 0;
}

int gpu_ctx_truncate(GpuCtx* ctx, int written) {
  if (ctx == nullptr || written < 0 || written > ctx->written ||
      written > ctx->max_pos)
    return -1;
  ctx->written = written;
  return 0;
}

void gpu_ctx_reset(GpuCtx* ctx) {
  if (ctx != nullptr) (void)gpu_ctx_truncate(ctx, 0);
}

// Test launcher for the coalesced GEMV (consumed by tests/test_kernels.cu and
// the verify-GEMV parity test): one block per output row, at the forward's
// launch width. The width matters for bit-identity - the per-lane strided
// accumulation depends on blockDim - so it must stay kRowGemvThreads.
extern "C" int granite4_kernel_row_gemm_dequant(const std::uint16_t* d_a,
                                                const std::uint8_t* d_B,
                                                float scale, int N, int K,
                                                std::uint16_t* d_out) {
  if (d_a == nullptr || d_B == nullptr || d_out == nullptr ||
      !std::isfinite(scale) || N <= 0 || K <= 0 || N > INT_MAX ||
      static_cast<long long>(N) * K > INT_MAX)
    return -1;
  row_gemm_dequant_kernel<<<N, kRowGemvThreads>>>(d_a, d_B, scale, N, K, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_qkv_dequant_batched(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int tokens, int nq,
    int nk, int nv, int k, std::uint16_t* d_q, std::uint16_t* d_kk,
    std::uint16_t* d_v) {
  const long long total = static_cast<long long>(nq) + nk + nv;
  if (d_a == nullptr || d_wq == nullptr || d_wk == nullptr || d_wv == nullptr ||
      d_q == nullptr || d_kk == nullptr || d_v == nullptr ||
      !std::isfinite(sq) || !std::isfinite(sk) || !std::isfinite(sv) ||
      tokens <= 0 || nq <= 0 || nk <= 0 || nv <= 0 || k <= 0 ||
      total > INT_MAX || static_cast<long long>(tokens) * total > INT_MAX ||
      static_cast<long long>(tokens) * k > INT_MAX)
    return -1;
  row_gemm_dequant_qkv_batched_kernel<<<tokens * static_cast<int>(total), kRowGemvThreads>>>(
      d_a, d_wq, d_wk, d_wv, sq, sk, sv, tokens, nq, nk, nv, k, d_q, d_kk, d_v);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_qkv_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int nq, int nk, int k,
    std::uint16_t* d_q, std::uint16_t* d_kk, std::uint16_t* d_v) {
  const long long total = static_cast<long long>(nq) + 2 * nk;
  if (d_a == nullptr || d_wq == nullptr || d_wk == nullptr || d_wv == nullptr ||
      d_q == nullptr || d_kk == nullptr || d_v == nullptr ||
      !std::isfinite(sq) || !std::isfinite(sk) || !std::isfinite(sv) ||
      nq <= 0 || nk <= 0 || k <= 0 || total > INT_MAX ||
      static_cast<long long>(nq) * k > INT_MAX ||
      static_cast<long long>(nk) * k > INT_MAX)
    return -1;
  row_gemm_dequant_qkv_kernel<<<static_cast<int>(total), kRowGemvThreads>>>(
      d_a, d_wq, d_wk, d_wv, sq, sk, sv, nq, nk, nk, k, d_q, d_kk, d_v);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_row_gemm_resid_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_B, float scale,
    const std::uint16_t* d_residual, float residual_scale, int tokens, int N, int K,
    std::uint16_t* d_out) {
  if (d_a == nullptr || d_B == nullptr || d_residual == nullptr || d_out == nullptr ||
      !std::isfinite(scale) || !std::isfinite(residual_scale) || tokens <= 0 ||
      N <= 0 || K <= 0 || static_cast<long long>(tokens) * N > INT_MAX ||
      static_cast<long long>(tokens) * K > INT_MAX ||
      static_cast<long long>(N) * K > INT_MAX)
    return -1;
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
  if (d_x == nullptr || d_cos == nullptr || d_sin == nullptr || d_out == nullptr ||
      heads <= 0 || d <= 0 || (d & 1) != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(heads) * d > INT_MAX)
    return -1;
  rope_at_pos_kernel<<<(heads * d + 255) / 256, 256>>>(d_x, d_cos, d_sin,
                                                      heads, d, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

extern "C" int granite4_kernel_rope_at_pos_batched(
    const std::uint16_t* d_x, const float* d_cos, const float* d_sin, int rows,
    int heads, int d, std::uint16_t* d_out) {
  if (d_x == nullptr || d_cos == nullptr || d_sin == nullptr || d_out == nullptr ||
      rows <= 0 || rows > granite4::kMaxVerifyRows || heads <= 0 || d <= 0 || (d & 1) != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(rows) * heads * d > INT_MAX)
    return -1;
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
  if (d_q == nullptr || d_k == nullptr || d_v == nullptr || d_cos == nullptr ||
      d_sin == nullptr || d_q_rot == nullptr || d_k_rot == nullptr ||
      d_cache_k == nullptr || d_cache_v == nullptr || base_pos < 0 ||
      rows <= 0 || rows > granite4::kMaxVerifyRows || Hq <= 0 || Hk <= 0 || Hq < Hk ||
      Hq % Hk != 0 || d <= 0 || (d & 1) != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(rows) * Hq * d > INT_MAX ||
      static_cast<long long>(rows) * Hk * d > INT_MAX ||
      base_pos > INT_MAX - rows + 1 ||
      cache_stride <= base_pos + rows - 1)
    return -1;
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
