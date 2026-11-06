// Phase 2 (Kernels): device kernels for Granite 4.2 8B on sm_89.
//
// Conventions (shared with the host reference in kernel_math.h):
//   * BF16 in / BF16 out, float (double for reductions) accumulation.
//   * RMSNorm:  y = x * rsqrt(mean(x^2) + eps) * w
//   * RoPE:     Llama-style half split (pairs i, i+d/2), precomputed
//               [max_pos, d/2] cos/sin tables, theta = 1e7
//   * Softmax:  logit scale = attention_multiplier (1/128, replaces
//               1/sqrt(d_head)); max-subtracted 2-pass, exp in double
//   * GEMM:     FP8 e4m3 weight rows x BF16 activation rows, per-tensor
//               dequant scale (Ada has no block-scaled MMA; per-tensor is
//               the P1 storage decision)
//
// Correctness-first tiling (one thread per output element); the
// micro-benchmark pass (P2 exit criterion) lands in the next milestone.
#include <algorithm>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>

#define GRANITE_CUDA 1
#include "device_math.cuh"
#include "kernel_math.h"
#include "model_config.h"

namespace granite4 {

namespace {

using dev::bf16_hi;
using dev::bf16_lo;
using dev::bf16_to_f32;
using dev::f32_to_bf16;
using dev::fp8_to_f32;
using dev::fp8x4_to_f32;

}  // namespace

// ---------------------------------------------------------------------------
// RMSNorm: one thread per element; the row reduction (mean of squares) is
// re-read per thread (correct; the fast variant tiles over rows).
// ---------------------------------------------------------------------------

// Threads per RMSNorm block. G4 decode rows are 4096 wide: 1024 threads give
// every thread 4 elements, so the strided load/normalize passes and the
// per-thread double accumulation chain are 4x shorter than at 256.
constexpr int kRmsNormThreads = 1024;

__global__ void rmsnorm_kernel(const std::uint16_t* x, const std::uint16_t* w,
                               float eps, int rows, int cols,
                               std::uint16_t* out) {
  // One block per row: the row is read exactly once (O(cols)), not once per
  // element. The previous one-thread-per-element form had every thread re-read
  // the whole row for the sum-of-squares (O(cols^2)); for the decode row
  // (cols=4096) that is ~33 MB of redundant traffic per call, and it is the
  // dominant cost of the forward.
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const std::size_t base = static_cast<std::size_t>(row) * cols;
  const std::uint16_t* const xr = x + base;
  // Each thread owns a strided slice. The sum of squares is accumulated as an
  // unevaluated float pair (hi + lo, Knuth TwoSum): ~48 significant bits, so
  // like the host oracle's double sum it rounds to the same float mean square,
  // at FP32 instead of the 1/64-rate FP64 throughput of consumer Ada. Each
  // square of a bf16 value is exact in float. Rows of up to
  // kRmsCache * blockDim elements (all of Granite's) keep the slice and its
  // norm weights in registers, so the normalize pass re-reads nothing.
  constexpr int kRmsCache = 8;
  const bool cached = cols <= kRmsCache * nt;
  float xv[kRmsCache], wv[kRmsCache];
  float hi = 0.0f, lo = 0.0f;
  auto add = [](float& h, float& l, float x) {
    const float s = h + x;
    const float bb = s - h;
    l += (h - (s - bb)) + (x - bb);
    h = s;
  };
  if (cached) {
#pragma unroll
    for (int j = 0; j < kRmsCache; ++j) {
      const int c = t + j * nt;
      xv[j] = c < cols ? bf16_to_f32(xr[c]) : 0.0f;
      wv[j] = c < cols ? bf16_to_f32(w[c]) : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < kRmsCache; ++j) add(hi, lo, xv[j] * xv[j]);
  } else {
    for (int c = t; c < cols; c += nt) {
      const float v = bf16_to_f32(xr[c]);
      add(hi, lo, v * v);
    }
  }
  // Deterministic tree: lanes, then warps in warp order, merging pairs.
  __shared__ float red_hi[32], red_lo[32];
  __shared__ float total;
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float oh = __shfl_down_sync(0xffffffffu, hi, offset);
    const float ol = __shfl_down_sync(0xffffffffu, lo, offset);
    lo += ol;
    add(hi, lo, oh);
  }
  if ((t & 31) == 0) {
    red_hi[t >> 5] = hi;
    red_lo[t >> 5] = lo;
  }
  __syncthreads();
  if (t == 0) {
    float h = 0.0f, l = 0.0f;
    const int warps = (nt + 31) >> 5;
    for (int i = 0; i < warps; ++i) {
      l += red_lo[i];
      add(h, l, red_hi[i]);
    }
    total = static_cast<float>((static_cast<double>(h) + static_cast<double>(l)) / cols);
  }
  __syncthreads();
  const float r = 1.0f / std::sqrt(total + eps);
  if (cached) {
#pragma unroll
    for (int j = 0; j < kRmsCache; ++j) {
      const int c = t + j * nt;
      if (c < cols) out[base + c] = f32_to_bf16(xv[j] * r * wv[j]);
    }
    return;
  }
  for (int c = t; c < cols; c += nt)
    out[base + c] = f32_to_bf16(bf16_to_f32(xr[c]) * r * bf16_to_f32(w[c]));
}

// ---------------------------------------------------------------------------
// RoPE: one thread per (row, col) of the head vector.
// ---------------------------------------------------------------------------

__global__ void rope_kernel(const std::uint16_t* x, const float* cos_table,
                            const float* sin_table, int rows, int head_dim,
                            std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * head_dim) return;
  const int row = idx / head_dim;
  const int col = idx % head_dim;
  const int half = head_dim / 2;
  const int base = row * head_dim;
  const int i = col < half ? col : col - half;
  const float c = cos_table[row * half + i];
  const float s = sin_table[row * half + i];
  float y;
  if (col < half) {
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col + half]);
    y = a * c - b * s;
  } else {
    // Back half: out[col] = x[col]*c + x[col-half]*s (Llama convention).
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col - half]);
    y = a * c + b * s;
  }
  out[idx] = f32_to_bf16(y);
}

// ---------------------------------------------------------------------------
// Scaled softmax: one thread per element, two passes per row
// (max, then sum - same arithmetic as the host reference).
// ---------------------------------------------------------------------------

__global__ void softmax_kernel(const std::uint16_t* x, float scale, int rows,
                               int seq_len, std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * seq_len) return;
  const int row = idx / seq_len;
  const int col = idx % seq_len;
  const int base = row * seq_len;
  float m = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < seq_len; ++i)
    m = fmaxf(m, scale * bf16_to_f32(x[base + i]));
  double sum = 0.0;
  for (int i = 0; i < seq_len; ++i)
    sum += std::exp(static_cast<double>(scale * bf16_to_f32(x[base + i]) - m));
  out[idx] = f32_to_bf16(
      static_cast<float>(std::exp(static_cast<double>(scale * bf16_to_f32(x[base + col]) - m)) / sum));
}

// ---------------------------------------------------------------------------
// GEMM: C[i, j] = scale * sum_k A[i, k] * fp8_to_f32(B[j, k]).
// One thread per output element.
// ---------------------------------------------------------------------------

__global__ void gemm_dequant_kernel(const std::uint16_t* A, const std::uint8_t* B,
                                    float scale, int M, int N, int K,
                                    std::uint16_t* C) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const int i = idx / N;
  const int j = idx % N;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc += fp8_to_f32(B[j * K + k]) * bf16_to_f32(A[i * K + k]);
  C[idx] = f32_to_bf16(scale * acc);
}

// ---------------------------------------------------------------------------
// GQA attention (decode, batch 1): logits = scale * Q K^T with the K-head
// for Q-head hq being hq/gh (gh = Hq/Hk); value accumulation over p * V.
// Logits are stored BF16 (the softmax kernel consumes them as-is) and the
// softmax probabilities are consumed BF16 by the value kernel - matching
// the reference pipeline exactly.
// ---------------------------------------------------------------------------

__global__ void gqa_logits_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                  float scale, int Hq, int Hk, int d, int kv_len,
                                  std::uint16_t* logits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hq * kv_len) return;
  const int hq = idx / kv_len;
  const int kv = idx % kv_len;
  const int hkv = hq / (Hq / Hk);
  float acc = 0.0f;
  const int kb = (hkv * kv_len + kv) * d;
  for (int i = 0; i < d; ++i)
    acc += bf16_to_f32(k[kb + i]) * bf16_to_f32(q[hq * d + i]);
  logits[idx] = f32_to_bf16(scale * acc);
}

__global__ void gqa_value_kernel(const std::uint16_t* p, const std::uint16_t* v,
                                 int Hq, int Hk, int d, int kv_len,
                                 std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hq * d) return;
  const int hq = idx / d;
  const int i = idx % d;
  const int hkv = hq / (Hq / Hk);
  float acc = 0.0f;
  for (int kv = 0; kv < kv_len; ++kv)
    acc += bf16_to_f32(p[hq * kv_len + kv]) * bf16_to_f32(v[(hkv * kv_len + kv) * d + i]);
  out[idx] = f32_to_bf16(acc);
}

// ---------------------------------------------------------------------------
// Fused decode GQA: logits -> softmax -> value in ONE block per Q head.
// Bit-identical to the three-kernel pipeline (gqa_logits_kernel +
// softmax_kernel + gqa_value_kernel): the bf16 logits and bf16 probabilities
// are materialized in shared memory exactly as the pipeline's global scratch
// buffers would hold them, and each stage's arithmetic (summation order,
// bf16 rounding, the double-precision softmax denominator) is reproduced
// verbatim. Removes two launches per layer and the two global scratch
// round-trips. Needs d + 2*kv_len bf16 of shared memory (48 KiB default
// limit covers kv_len up to ~24 K).
//
// Phase B runs the exp() terms in parallel but accumulates them in ascending
// kv order by one thread: the sequence of double additions is identical to
// the serial reference, so the denominator and probabilities stay
// bit-identical. The parallel form needs kv_len doubles of extra shared
// memory; beyond the shared parallel-softmax limit the exact serial loop runs.
// ---------------------------------------------------------------------------

// One (query head, row) of fused GQA over the causal prefix [0, kv_len).
// q_row is that row's [Hq][d] query; out_row its [Hq][d] output.
//
// The max is a block tree (max is exact in any order). The denominator is
// the reference's ascending-order double sum, run by thread 0 with the
// shared-memory loads batched ahead of the dependent adds; everything else
// is parallel across the block.
__device__ __forceinline__ void gqa_fused_row(const std::uint16_t* q_row,
                                              const std::uint16_t* k,
                                              const std::uint16_t* v, float att_scale,
                                              int hq, int hkv, int d, int kv_len,
                                              int k_stride, std::uint16_t* out_row) {
  const int t = threadIdx.x;
  const int nt = blockDim.x;  // blockDim.x == d

  extern __shared__ std::uint16_t smem[];
  std::uint16_t* const sq = smem;               // [d]
  std::uint16_t* const slog = smem + d;         // [kv_len] bf16 logits
  std::uint16_t* const sp = smem + d + kv_len;  // [kv_len] bf16 probs
  const std::size_t base_elems =
      static_cast<std::size_t>(d) + 2u * static_cast<std::size_t>(kv_len);
  const std::size_t exp_off = (base_elems + 3u) & ~std::size_t(3);  // 8B align
  double* const dexp = (kv_len <= granite4::kGqaSoftmaxParallelMax)
                           ? reinterpret_cast<double*>(smem + exp_off)
                           : nullptr;
  __shared__ float wmax[32];
  __shared__ float smax;     // row max (Phase B)
  __shared__ double sdenom;  // softmax denominator (Phase B)

  // Preload the q head row once (identical values to re-reading global).
  for (int i = t; i < d; i += nt) sq[i] = q_row[hq * d + i];
  __syncthreads();

  // Phase A: bf16 logits[kv] = att_scale * dot(q, k[hkv, kv]), plus the
  // per-thread running max of what this thread wrote.
  float m = -std::numeric_limits<float>::infinity();
  for (int kv = t; kv < kv_len; kv += nt) {
    const int kbase = (hkv * k_stride + kv) * d;
    float acc = 0.0f;
    for (int i = 0; i < d; ++i)
      acc += bf16_to_f32(k[kbase + i]) * bf16_to_f32(sq[i]);
    const std::uint16_t lb = f32_to_bf16(att_scale * acc);
    slog[kv] = lb;
    m = fmaxf(m, bf16_to_f32(lb));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  if ((t & 31) == 0) wmax[t >> 5] = m;
  __syncthreads();
  if (t == 0) {
    float mm = wmax[0];
    for (int w = 1; w < ((nt + 31) >> 5); ++w) mm = fmaxf(mm, wmax[w]);
    smax = mm;
  }
  __syncthreads();

  // Phase B: exp terms in parallel, ordered double denominator.
  if (dexp != nullptr) {
    for (int i = t; i < kv_len; i += nt)
      dexp[i] = std::exp(static_cast<double>(1.0f * bf16_to_f32(slog[i]) - smax));
    __syncthreads();
    if (t == 0) {
      double sum = 0.0;
      int i = 0;
      for (; i + 8 <= kv_len; i += 8) {
        double e[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = dexp[i + j];
#pragma unroll
        for (int j = 0; j < 8; ++j) sum += e[j];
      }
      for (; i < kv_len; ++i) sum += dexp[i];
      sdenom = sum;
    }
    __syncthreads();
    for (int i = t; i < kv_len; i += nt)
      sp[i] = f32_to_bf16(static_cast<float>(dexp[i] / sdenom));
  } else if (t == 0) {
    double sum = 0.0;
    for (int i = 0; i < kv_len; ++i)
      sum += std::exp(static_cast<double>(1.0f * bf16_to_f32(slog[i]) - smax));
    for (int i = 0; i < kv_len; ++i)
      sp[i] = f32_to_bf16(
          static_cast<float>(std::exp(static_cast<double>(1.0f * bf16_to_f32(slog[i]) - smax)) /
                             sum));
  }
  __syncthreads();

  // Phase C: out[hq, i] = sum_kv p[kv] * v[hkv, kv, i] (ascending kv).
  const int i = t;
  float acc = 0.0f;
  for (int kv = 0; kv < kv_len; ++kv)
    acc += bf16_to_f32(sp[kv]) * bf16_to_f32(v[(hkv * k_stride + kv) * d + i]);
  out_row[hq * d + i] = f32_to_bf16(acc);
}

// Single row; d_pos (optional): when non-null the causal length is
// *d_pos + 1, read on the device, so a captured decode graph replays at
// every position.
__global__ void gqa_fused_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                 const std::uint16_t* v, float att_scale, int Hq,
                                 int Hk, int d, int kv_len, int k_stride,
                                 const int* d_pos, std::uint16_t* out) {
  if (d_pos != nullptr) kv_len = *d_pos + 1;
  const int hq = blockIdx.x;
  gqa_fused_row(q, k, v, att_scale, hq, hq / (Hq / Hk), d, kv_len, k_stride, out);
}

// ---------------------------------------------------------------------------
// Batched speculative verify: append `rows` consecutive KV positions in one
// grid, then run one fused-GQA block per (query head, row) with that row's
// causal prefix [0, base_kv + row + 1). The per-row arithmetic is the same
// order as the single-row kernels, so each row stays bit-identical.
// ---------------------------------------------------------------------------
__global__ void kv_append_batched_kernel(const std::uint16_t* src_k,
                                         const std::uint16_t* src_v,
                                         std::uint16_t* cache_k,
                                         std::uint16_t* cache_v,
                                         int base_pos, int rows, int stride,
                                         int Hk, int d) {
  const int r = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows || idx >= Hk * d) return;
  const int hkv = idx / d;
  const int i = idx % d;
  const int pos = base_pos + r;
  const std::size_t cache_row = static_cast<std::size_t>(hkv) * stride * d;
  const std::size_t src_off =
      static_cast<std::size_t>(r) * Hk * d + static_cast<std::size_t>(idx);
  cache_k[cache_row + static_cast<std::size_t>(pos) * d + i] = src_k[src_off];
  cache_v[cache_row + static_cast<std::size_t>(pos) * d + i] = src_v[src_off];
}

// KV8 (Q8_0): int8 form of the batched verify append - `rows` consecutive KV
// positions, one block per (KV head, row), blockDim.x == d so the row amax is
// a block reduce. Quantized per row with dev::kv8_* (the committed math).
__global__ void kv_append_int8_batched_kernel(
    const std::uint16_t* __restrict__ src_k, const std::uint16_t* __restrict__ src_v,
    std::int8_t* __restrict__ cache_k, std::int8_t* __restrict__ cache_v,
    float* __restrict__ cache_k_scale, float* __restrict__ cache_v_scale,
    int base_pos, int rows, int stride, int Hk, int d) {
  const int hkv = blockIdx.x;
  const int r = blockIdx.y;
  const int i = threadIdx.x;  // blockDim.x == d
  const int pos = base_pos + r;
  const std::size_t src = (static_cast<std::size_t>(r) * Hk + hkv) * d + i;
  const std::size_t cache = (static_cast<std::size_t>(hkv) * stride + pos) * d + i;
  const std::size_t sc = static_cast<std::size_t>(hkv) * stride + pos;
  const float x = bf16_to_f32(src_k[src]);
  const float y = bf16_to_f32(src_v[src]);
  __shared__ float smax[2];
  const int lane = i & 31;
  float ax = fabsf(x), ay = fabsf(y);
  for (int off = 16; off > 0; off >>= 1) {
    ax = fmaxf(ax, __shfl_down_sync(0xffffffffu, ax, off));
    ay = fmaxf(ay, __shfl_down_sync(0xffffffffu, ay, off));
  }
  __shared__ float wmax[2][32];
  if (lane == 0) {
    wmax[0][i >> 5] = ax;
    wmax[1][i >> 5] = ay;
  }
  __syncthreads();
  float mk = 0.0f, mv = 0.0f;
  for (int w = 0; w < (d + 31) >> 5; ++w) {
    mk = fmaxf(mk, wmax[0][w]);
    mv = fmaxf(mv, wmax[1][w]);
  }
  if (i == 0) {
    const float sk = dev::kv8_scale(mk);
    const float sv = dev::kv8_scale(mv);
    smax[0] = sk;
    smax[1] = sv;
    cache_k_scale[sc] = sk;
    cache_v_scale[sc] = sv;
  }
  __syncthreads();
  cache_k[cache] = dev::kv8_quant(x, smax[0]);
  cache_v[cache] = dev::kv8_quant(y, smax[1]);
}

// KV4 (Q4_0): int4 form of the batched verify append - `rows` consecutive KV
// positions, one block per (KV head, row). Each thread owns a dim-pair (two
// adjacent dims packed into one byte); the row amax is a block reduce over
// blockDim.x threads, matching the int8 layout. d must be even (128).
__global__ void kv_append_int4_batched_kernel(
    const std::uint16_t* __restrict__ src_k, const std::uint16_t* __restrict__ src_v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    float* __restrict__ cache_k_scale, float* __restrict__ cache_v_scale,
    int base_pos, int rows, int stride, int Hk, int d) {
  const int hkv = blockIdx.x;
  const int r = blockIdx.y;
  const int w = threadIdx.x;  // blockDim.x == d/2
  const int pos = base_pos + r;
  const int d0 = 2 * w;       // even dim (lower nibble)
  const int d1 = d0 + 1;      // odd dim (upper nibble)
  const std::size_t src0 = (static_cast<std::size_t>(r) * Hk + hkv) * d + d0;
  const std::size_t src1 = (static_cast<std::size_t>(r) * Hk + hkv) * d + d1;
  const std::size_t cache = (static_cast<std::size_t>(hkv) * stride + pos) * (d / 2) + w;
  const std::size_t sc = static_cast<std::size_t>(hkv) * stride + pos;
  const float x0 = bf16_to_f32(src_k[src0]);
  const float x1 = bf16_to_f32(src_k[src1]);
  const float y0 = bf16_to_f32(src_v[src0]);
  const float y1 = bf16_to_f32(src_v[src1]);
  __shared__ float smax[2];
  const int lane = w & 31;
  float ax = fmaxf(fabsf(x0), fabsf(x1));
  float ay = fmaxf(fabsf(y0), fabsf(y1));
  for (int off = 16; off > 0; off >>= 1) {
    ax = fmaxf(ax, __shfl_down_sync(0xffffffffu, ax, off));
    ay = fmaxf(ay, __shfl_down_sync(0xffffffffu, ay, off));
  }
  __shared__ float wmax[2][32];
  if (lane == 0) {
    wmax[0][w >> 5] = ax;
    wmax[1][w >> 5] = ay;
  }
  __syncthreads();
  float mk = 0.0f, mv = 0.0f;
  for (int wp = 0; wp < ((d / 2) + 31) >> 5; ++wp) {
    mk = fmaxf(mk, wmax[0][wp]);
    mv = fmaxf(mv, wmax[1][wp]);
  }
  if (w == 0) {
    const float sk = dev::q4_scale(mk);
    const float sv = dev::q4_scale(mv);
    smax[0] = sk;
    smax[1] = sv;
    cache_k_scale[sc] = sk;
    cache_v_scale[sc] = sv;
  }
  __syncthreads();
  const unsigned char ck =
      static_cast<unsigned char>((dev::q4_quant(x0, smax[0]) & 0x0fu) |
                                 (dev::q4_quant(x1, smax[0]) << 4));
  const unsigned char cv =
      static_cast<unsigned char>((dev::q4_quant(y0, smax[1]) & 0x0fu) |
                                 (dev::q4_quant(y1, smax[1]) << 4));
  cache_k[cache] = ck;
  cache_v[cache] = cv;
}

// Row blockIdx.y attends to its causal prefix [0, base_kv + row + 1).
__global__ void gqa_fused_batched_kernel(const std::uint16_t* q,
                                         const std::uint16_t* k,
                                         const std::uint16_t* v,
                                         float att_scale, int Hq, int Hk, int d,
                                         int base_kv, int rows, int k_stride,
                                         std::uint16_t* out) {
  const int hq = blockIdx.x;
  const int row = blockIdx.y;
  const std::size_t off = static_cast<std::size_t>(row) * Hq * d;
  gqa_fused_row(q + off, k, v, att_scale, hq, hq / (Hq / Hk), d, base_kv + row + 1,
                k_stride, out + off);
}

// ---------------------------------------------------------------------------
// FFN gate/up (SwiGLU): y[r] = silu(sg * Wg[r].x) * (su * Wu[r].x).
// ---------------------------------------------------------------------------

__global__ void silu_gate_up_kernel(const std::uint8_t* Wg, const std::uint8_t* Wu,
                                    const std::uint16_t* x, float sg, float su,
                                    int N, int K, std::uint16_t* out) {
  const int r = blockIdx.x;  // one output row per block
  if (r >= N) return;
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  const std::uint8_t* const rg = Wg + static_cast<std::size_t>(r) * K;
  const std::uint8_t* const ru = Wu + static_cast<std::size_t>(r) * K;
  float g = 0.0f, u = 0.0f;
  if ((K & 3) == 0) {
    // 4 k-elements per iteration (2x bytes in flight vs the scalar form):
    // 4 fp8 per row per matrix + 4 bf16 of the shared activation row.
    const std::uint32_t* rg4 = reinterpret_cast<const std::uint32_t*>(rg);
    const std::uint32_t* ru4 = reinterpret_cast<const std::uint32_t*>(ru);
    const std::uint32_t* x4 = reinterpret_cast<const std::uint32_t*>(x);
    const int K4 = K >> 2;
    for (int k4 = t; k4 < K4; k4 += nt) {
      const std::uint32_t xa = x4[2 * k4];
      const std::uint32_t xb = x4[2 * k4 + 1];
      const float x0 = bf16_lo(xa), x1 = bf16_hi(xa);
      const float x2 = bf16_lo(xb), x3 = bf16_hi(xb);
      float g0, g1, g2, g3, u0, u1, u2, u3;
      fp8x4_to_f32(rg4[k4], g0, g1, g2, g3);
      fp8x4_to_f32(ru4[k4], u0, u1, u2, u3);
      g += g0 * x0 + g1 * x1 + g2 * x2 + g3 * x3;
      u += u0 * x0 + u1 * x1 + u2 * x2 + u3 * x3;
    }
  } else {
    for (int i = t; i < K; i += nt) {
      const float xv = bf16_to_f32(x[i]);
      g += fp8_to_f32(rg[i]) * xv;
      u += fp8_to_f32(ru[i]) * xv;
    }
  }
  // Block reduce of both partials (warp shuffles + shared tree).
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    g += __shfl_down_sync(0xffffffffu, g, off);
    u += __shfl_down_sync(0xffffffffu, u, off);
  }
  __shared__ float rg_[32], ru_[32];
  if ((t & 31) == 0) {
    rg_[t >> 5] = g;
    ru_[t >> 5] = u;
  }
  __syncthreads();
  if (t == 0) {
    float s = rg_[0], v = ru_[0];
    for (int j = 1; j < (nt >> 5); ++j) {
      s += rg_[j];
      v += ru_[j];
    }
    const float sig = 1.0f / (1.0f + std::exp(-(sg * s)));
    out[r] = f32_to_bf16(sg * s * sig * (su * v));
  }
}

// Batched SwiGLU for speculative verification. One block owns one output row
// and accumulates all activation rows while streaming each gate/up weight row
// once. The final nonlinear operation and BF16 conversion use the same order
// as silu_gate_up_kernel.
__global__ void silu_gate_up_batched_kernel(const std::uint8_t* Wg,
                                            const std::uint8_t* Wu,
                                            const std::uint16_t* x, float sg,
                                            float su, int N, int K, int rows,
                                            std::uint16_t* out) {
  const int r = blockIdx.x;
  if (r >= N) return;
  const int t = threadIdx.x;
  const int nt = blockDim.x;
  float g[4] = {0.f, 0.f, 0.f, 0.f};
  float u[4] = {0.f, 0.f, 0.f, 0.f};
  const std::uint8_t* const rg = Wg + static_cast<std::size_t>(r) * K;
  const std::uint8_t* const ru = Wu + static_cast<std::size_t>(r) * K;
  if ((K & 3) == 0) {
    const std::uint32_t* rg4 = reinterpret_cast<const std::uint32_t*>(rg);
    const std::uint32_t* ru4 = reinterpret_cast<const std::uint32_t*>(ru);
    const int K4 = K >> 2;
    for (int k4 = t; k4 < K4; k4 += nt) {
      float g0, g1, g2, g3, u0, u1, u2, u3;
      fp8x4_to_f32(rg4[k4], g0, g1, g2, g3);
      fp8x4_to_f32(ru4[k4], u0, u1, u2, u3);
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        if (row >= rows) break;
        const std::uint32_t* x4 = reinterpret_cast<const std::uint32_t*>(
            x + static_cast<std::size_t>(row) * K);
        const std::uint32_t xa = x4[2 * k4], xb = x4[2 * k4 + 1];
        const float x0 = bf16_lo(xa), x1 = bf16_hi(xa);
        const float x2 = bf16_lo(xb), x3 = bf16_hi(xb);
        g[row] += g0 * x0 + g1 * x1 + g2 * x2 + g3 * x3;
        u[row] += u0 * x0 + u1 * x1 + u2 * x2 + u3 * x3;
      }
    }
  } else {
    for (int i = t; i < K; i += nt) {
      const float wg = fp8_to_f32(rg[i]);
      const float wu = fp8_to_f32(ru[i]);
#pragma unroll
      for (int row = 0; row < 4; ++row) {
        if (row >= rows) break;
        const float xv = bf16_to_f32(x[static_cast<std::size_t>(row) * K + i]);
        g[row] += wg * xv;
        u[row] += wu * xv;
      }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      g[row] += __shfl_down_sync(0xffffffffu, g[row], off);
      u[row] += __shfl_down_sync(0xffffffffu, u[row], off);
    }
  }
  __shared__ float rg_[4][32], ru_[4][32];
  if ((t & 31) == 0) {
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      rg_[row][t >> 5] = g[row];
      ru_[row][t >> 5] = u[row];
    }
  }
  __syncthreads();
  if (t == 0) {
    const int warps = nt >> 5;
#pragma unroll
    for (int row = 0; row < 4; ++row) {
      if (row >= rows) break;
      float gs = rg_[row][0], us = ru_[row][0];
      for (int j = 1; j < warps; ++j) {
        gs += rg_[row][j];
        us += ru_[row][j];
      }
      const float sig = 1.0f / (1.0f + std::exp(-(sg * gs)));
      out[static_cast<std::size_t>(row) * N + r] =
          f32_to_bf16(sg * gs * sig * (su * us));
    }
  }
}

// ---------------------------------------------------------------------------
// Sampling. Greedy: one thread per row (any vocab size). Top-p: one block
// per row with a shared-memory stable sort; vocab <= 4096 (the production
// 100352 path uses the same per-row loop with a chunked argsort - a P3
// concern; the math is identical).
// ---------------------------------------------------------------------------

__global__ void sample_greedy_kernel(const std::uint16_t* p, int rows, int n,
                                     int* out) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int best = 0;
  float bv = bf16_to_f32(p[row * n]);
  for (int i = 1; i < n; ++i) {
    const float v = bf16_to_f32(p[row * n + i]);
    if (v > bv) {
      bv = v;
      best = i;
    }
  }
  out[row] = best;
}

constexpr int kTopPVocabMax = 4096;

__global__ void sample_top_p_kernel(const std::uint16_t* p, float nucleus,
                                    const float* u_in, int n, int* out) {
  // One block per row: the shared arrays hold the row being sorted.
  const int row = blockIdx.x;
  if (n > kTopPVocabMax) return;
  __shared__ float sp[kTopPVocabMax];
  __shared__ int st[kTopPVocabMax];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    sp[i] = bf16_to_f32(p[row * n + i]);
    st[i] = i;
  }
  __syncthreads();
  // One thread: stable insertion sort (probability descending, token id
  // ascending on ties), then nucleus truncation and prefix-sum pick.
  // Single-threaded because the sort is in-place over the shared arrays;
  // O(n^2) is fine at the tested sizes (a faster argsort is a P3 concern).
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) {
      float v = sp[i];
      int id = st[i];
      int j = i - 1;
      while (j >= 0 && (sp[j] < v || (sp[j] == v && st[j] > id))) {
        sp[j + 1] = sp[j];
        st[j + 1] = st[j];
        --j;
      }
      sp[j + 1] = v;
      st[j + 1] = id;
    }
    double acc = 0.0;
    int cut = n;
    for (int j = 0; j < n; ++j) {
      acc += sp[j];
      cut = j + 1;
      if (acc >= nucleus) break;
    }
    const double target = u_in[row] * acc;
    double cum = 0.0;
    int pick = st[cut - 1];
    for (int j = 0; j < cut; ++j) {
      cum += sp[j];
      if (cum >= target) {
        pick = st[j];
        break;
      }
    }
    out[row] = pick;
  }
}

// Verification helper: decode all 256 e4m3 codepoints to f32 (one per
// thread). The exhaustive decode test compares these against the arithmetic
// reference (checkpoint.h fp8_e4m3_to_bf16) to prove the branchless
// fp8_to_f32 is bit-exact.
__global__ void fp8_decode_all_kernel(float* out) {
  out[threadIdx.x] = fp8_to_f32(static_cast<std::uint8_t>(threadIdx.x));
}

}  // namespace granite4

// ---------------------------------------------------------------------------
// Host launchers (C ABI surface for the P3 scheduler). 0 on success, -1 on
// CUDA error (diagnostic on stderr).
// ---------------------------------------------------------------------------

#define G4_CUDA_CHECK(expr)                                                 \
  do {                                                                      \
    cudaError_t e_ = (expr);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "granite4: CUDA error at %s:%d: %s\n", __FILE__, \
                   __LINE__, cudaGetErrorString(e_));                      \
      return -1;                                                            \
    }                                                                       \
  } while (0)

extern "C" {

int granite4_kernel_gqa_fused_pos(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                  const std::uint16_t* d_v, float att_scale, int Hq,
                                  int Hk, int d, int kv_len, int k_stride,
                                  const int* d_pos, std::uint16_t* d_out);
int granite4_kernel_gqa_fused_rows(const std::uint16_t* d_q,
                                   const std::uint16_t* d_k,
                                   const std::uint16_t* d_v, float att_scale,
                                   int Hq, int Hk, int d, int base_kv, int rows,
                                   int k_stride, std::uint16_t* d_out);

int granite4_fp8_decode_all(float** d_out) {
  if (d_out == nullptr) return -1;
  *d_out = nullptr;
  // d_out must hold 256 floats.
  float* d_out_p = nullptr;
  G4_CUDA_CHECK(cudaMalloc(&d_out_p, 256 * sizeof(float)));
  if (d_out_p == nullptr) return -1;
  granite4::fp8_decode_all_kernel<<<1, 256>>>(d_out_p);
  if (cudaGetLastError() != cudaSuccess) {
    cudaFree(d_out_p);
    return -1;
  }
  *d_out = d_out_p;
  return 0;
}

int granite4_rope_tables_build(double theta, int head_dim, int max_pos,
                               float** d_cos, float** d_sin) {
  if (d_cos == nullptr || d_sin == nullptr || !std::isfinite(theta) || theta <= 0.0 ||
      head_dim <= 0 || (head_dim & 1) != 0 ||
      head_dim > granite4::kMaxHeadDim || max_pos <= 0 ||
      static_cast<std::size_t>(max_pos) >
          SIZE_MAX / (static_cast<std::size_t>(head_dim) / 2u) ||
      static_cast<std::size_t>(max_pos) * (head_dim / 2u) > SIZE_MAX / sizeof(float))
    return -1;
  *d_cos = nullptr;
  *d_sin = nullptr;
  std::vector<float> h_cos, h_sin;
  granite4::rope_freqs_f32(theta, head_dim, max_pos, &h_cos, &h_sin);
  const std::size_t bytes = h_cos.size() * sizeof(float);
  if (bytes == 0 || h_sin.size() != h_cos.size()) return -1;
  cudaError_t e = cudaMalloc(d_cos, bytes);
  if (e != cudaSuccess) return -1;
  e = cudaMalloc(d_sin, bytes);
  if (e != cudaSuccess) {
    cudaFree(*d_cos);
    *d_cos = nullptr;
    return -1;
  }
  e = cudaMemcpy(*d_cos, h_cos.data(), bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    cudaFree(*d_cos);
    cudaFree(*d_sin);
    *d_cos = nullptr;
    *d_sin = nullptr;
    return -1;
  }
  e = cudaMemcpy(*d_sin, h_sin.data(), bytes, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    cudaFree(*d_cos);
    cudaFree(*d_sin);
    *d_cos = nullptr;
    *d_sin = nullptr;
    return -1;
  }
  return 0;
}

int granite4_kernel_rmsnorm(const std::uint16_t* d_x, const std::uint16_t* d_w,
                            float eps, int rows, int cols, std::uint16_t* d_out) {
  if (d_x == nullptr || d_w == nullptr || d_out == nullptr ||
      !std::isfinite(eps) || eps <= 0.0f || rows <= 0 || cols <= 0 ||
      static_cast<long long>(rows) * cols > INT_MAX)
    return -1;
  granite4::rmsnorm_kernel<<<rows, granite4::kRmsNormThreads>>>(d_x, d_w, eps, rows, cols,
                                                                d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_rmsnorm_batched(const std::uint16_t* d_x,
                                    const std::uint16_t* d_w, float eps,
                                    int rows, int cols, std::uint16_t* d_out) {
  // rmsnorm_kernel is already row-batched (one block per row); this wrapper
  // just pins the verify-path bounds. Reusing it keeps a single
  // implementation, so a batched call cannot drift from the decode path.
  if (d_x == nullptr || d_w == nullptr || d_out == nullptr ||
      !std::isfinite(eps) || eps <= 0.0f || rows <= 0 || rows > granite4::kMaxVerifyRows || cols <= 0 ||
      static_cast<long long>(rows) * cols > INT_MAX)
    return -1;
  granite4::rmsnorm_kernel<<<rows, granite4::kRmsNormThreads>>>(d_x, d_w, eps, rows, cols,
                                                                d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_rope(const std::uint16_t* d_x, const float* d_cos,
                         const float* d_sin, int rows, int head_dim,
                         std::uint16_t* d_out) {
  if (d_x == nullptr || d_cos == nullptr || d_sin == nullptr || d_out == nullptr ||
      rows <= 0 || head_dim <= 0 || (head_dim & 1) != 0 ||
      static_cast<long long>(rows) * head_dim > INT_MAX ||
      head_dim > granite4::kMaxHeadDim)
    return -1;
  granite4::rope_kernel<<<(rows * head_dim + 255) / 256, 256>>>(
      d_x, d_cos, d_sin, rows, head_dim, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_softmax(const std::uint16_t* d_x, float scale, int rows,
                           int seq_len, std::uint16_t* d_out) {
  if (d_x == nullptr || d_out == nullptr || !std::isfinite(scale) ||
      rows <= 0 || seq_len <= 0 ||
      seq_len > granite4::kGqaSoftmaxParallelMaxSeqLen ||
      static_cast<long long>(rows) * seq_len > INT_MAX)
    return -1;
  granite4::softmax_kernel<<<(rows * seq_len + 255) / 256, 256>>>(
      d_x, scale, rows, seq_len, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gemm_dequant(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                 float scale, int M, int N, int K, std::uint16_t* d_c) {
  if (d_a == nullptr || d_b == nullptr || d_c == nullptr ||
      !std::isfinite(scale) || M <= 0 || N <= 0 || K <= 0 ||
      static_cast<long long>(M) * N > INT_MAX ||
      static_cast<long long>(N) * K > INT_MAX)
    return -1;
  granite4::gemm_dequant_kernel<<<(M * N + 255) / 256, 256>>>(
      d_a, d_b, scale, M, N, K, d_c);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_logits(const std::uint16_t* d_q, const std::uint16_t* d_k,
                               float scale, int Hq, int Hk, int d, int kv_len,
                               std::uint16_t* d_logits) {
  if (d_q == nullptr || d_k == nullptr || d_logits == nullptr ||
      !std::isfinite(scale) || Hq <= 0 || Hk <= 0 || d <= 0 || kv_len <= 0 ||
      Hq < Hk || Hq % Hk != 0 ||
      static_cast<long long>(Hq) * kv_len > INT_MAX ||
      static_cast<long long>(Hq) * d > INT_MAX ||
      static_cast<long long>(Hk) * d > INT_MAX)
    return -1;
  granite4::gqa_logits_kernel<<<(Hq * kv_len + 255) / 256, 256>>>(
      d_q, d_k, scale, Hq, Hk, d, kv_len, d_logits);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_value(const std::uint16_t* d_p, const std::uint16_t* d_v,
                              int Hq, int Hk, int d, int kv_len, std::uint16_t* d_out) {
  if (d_p == nullptr || d_v == nullptr || d_out == nullptr || Hq <= 0 ||
      Hk <= 0 || d <= 0 || kv_len <= 0 || Hq < Hk || Hq % Hk != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(Hq) * d > INT_MAX ||
      static_cast<long long>(Hk) * d > INT_MAX)
    return -1;
  granite4::gqa_value_kernel<<<(Hq * d + 255) / 256, 256>>>(
      d_p, d_v, Hq, Hk, d, kv_len, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_fused(const std::uint16_t* d_q, const std::uint16_t* d_k,
                              const std::uint16_t* d_v, float att_scale, int Hq,
                              int Hk, int d, int kv_len, int k_stride,
                              std::uint16_t* d_out) {
  return granite4_kernel_gqa_fused_pos(d_q, d_k, d_v, att_scale, Hq, Hk, d, kv_len,
                                       k_stride, nullptr, d_out);
}

std::size_t granite4_gqa_fused_smem_bytes(int d, int kv_len_max) {
  if (d <= 0 || kv_len_max <= 0) return 0;
  // [q row d][logits kv][probs kv] bf16, 8-byte aligned, then kv doubles for
  // the parallel softmax. Rows shorter than kv_len_max place their double
  // array earlier, and at most min(kv_len_max, parallel max) doubles are
  // ever used, so this bounds every row of a batched launch.
  const std::size_t base =
      (static_cast<std::size_t>(d) + 2u * static_cast<std::size_t>(kv_len_max) + 3u) &
      ~std::size_t(3);
  const std::size_t exp_elems = static_cast<std::size_t>(
      std::min(kv_len_max, granite4::kGqaSoftmaxParallelMax));
  return base * sizeof(std::uint16_t) + exp_elems * sizeof(double);
}

int granite4_gqa_fused_smem_limit() {
  static const int limit = [] {
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
            cudaSuccess) {
      cudaGetLastError();
      return 48 * 1024;
    }
    // Opt both fused kernels into the full per-block shared memory once; the
    // dynamic budget is the opt-in total minus each kernel's static part.
    int dyn = optin;
    for (const void* fn : {reinterpret_cast<const void*>(granite4::gqa_fused_kernel),
                           reinterpret_cast<const void*>(granite4::gqa_fused_batched_kernel)}) {
      cudaFuncAttributes attr{};
      if (cudaFuncGetAttributes(&attr, fn) != cudaSuccess) {
        cudaGetLastError();
        return 48 * 1024;
      }
      dyn = std::min(dyn, optin - static_cast<int>(attr.sharedSizeBytes));
    }
    for (const void* fn : {reinterpret_cast<const void*>(granite4::gqa_fused_kernel),
                           reinterpret_cast<const void*>(granite4::gqa_fused_batched_kernel)}) {
      if (cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn) !=
          cudaSuccess) {
        cudaGetLastError();
        return 48 * 1024;
      }
    }
    return dyn;
  }();
  return limit;
}

int granite4_kernel_gqa_fused_pos(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                  const std::uint16_t* d_v, float att_scale, int Hq,
                                  int Hk, int d, int kv_len, int k_stride,
                                  const int* d_pos, std::uint16_t* d_out) {
  // With d_pos, kv_len is the upper bound the shared memory is sized for.
  if (d_q == nullptr || d_k == nullptr || d_v == nullptr || d_out == nullptr ||
      !std::isfinite(att_scale) || Hq <= 0 || Hk <= 0 || d <= 0 || kv_len <= 0 ||
      k_stride < kv_len || Hq < Hk || Hq % Hk != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(Hq) * d > INT_MAX ||
      static_cast<long long>(Hk) * d > INT_MAX ||
      static_cast<long long>(kv_len) + d > INT_MAX)
    return -1;
  const std::size_t smem_bytes = granite4_gqa_fused_smem_bytes(d, kv_len);
  if (smem_bytes > static_cast<std::size_t>(granite4_gqa_fused_smem_limit())) return -1;
  granite4::gqa_fused_kernel<<<Hq, d, smem_bytes>>>(d_q, d_k, d_v, att_scale, Hq, Hk,
                                                     d, kv_len, k_stride, d_pos, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_kv_append_batched(const std::uint16_t* d_src_k,
                                      const std::uint16_t* d_src_v,
                                      std::uint16_t* d_cache_k,
                                      std::uint16_t* d_cache_v, int base_pos,
                                      int rows, int stride, int Hk, int d) {
  if (d_src_k == nullptr || d_src_v == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || base_pos < 0 || rows <= 0 || rows > granite4::kMaxVerifyRows ||
      Hk <= 0 || d <= 0 || base_pos > INT_MAX - rows ||
      stride < base_pos + rows || static_cast<long long>(Hk) * d > INT_MAX ||
      stride > INT_MAX / std::max(Hk, 1))
    return -1;
  granite4::kv_append_batched_kernel<<<
      dim3(static_cast<unsigned int>((static_cast<std::size_t>(Hk) * d + 255u) / 256u),
           static_cast<unsigned int>(rows)),
      256>>>(d_src_k, d_src_v, d_cache_k, d_cache_v, base_pos, rows, stride, Hk, d);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_kv_append_int8_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::int8_t* d_cache_k,
                                           std::int8_t* d_cache_v,
                                           float* d_cache_k_scale,
                                           float* d_cache_v_scale, int base_pos,
                                           int rows, int stride, int Hk, int d) {
  if (d_src_k == nullptr || d_src_v == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || d_cache_k_scale == nullptr ||
      d_cache_v_scale == nullptr || base_pos < 0 || rows <= 0 ||
      rows > granite4::kMaxVerifyRows || Hk <= 0 || d <= 0 || d > 1024 ||
      base_pos > INT_MAX - rows || stride < base_pos + rows ||
      static_cast<long long>(Hk) * d > INT_MAX || stride > INT_MAX / std::max(Hk, 1))
    return -1;
  // One block per (KV head, row); blockDim.x == d so the row amax is a block
  // reduce (the bf16 form uses a flat per-element grid).
  granite4::kv_append_int8_batched_kernel<<<dim3(static_cast<unsigned int>(Hk),
                                                 static_cast<unsigned int>(rows)),
                                             static_cast<unsigned int>(d)>>>
      (d_src_k, d_src_v, d_cache_k, d_cache_v, d_cache_k_scale, d_cache_v_scale,
       base_pos, rows, stride, Hk, d);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_kv_append_int4_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::uint8_t* d_cache_k,
                                           std::uint8_t* d_cache_v,
                                           float* d_cache_k_scale,
                                           float* d_cache_v_scale, int base_pos,
                                           int rows, int stride, int Hk, int d) {
  if (d_src_k == nullptr || d_src_v == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || d_cache_k_scale == nullptr ||
      d_cache_v_scale == nullptr || base_pos < 0 || rows <= 0 ||
      rows > granite4::kMaxVerifyRows || Hk <= 0 || d <= 0 || d % 2 != 0 || d > 1024 ||
      base_pos > INT_MAX - rows || stride < base_pos + rows ||
      static_cast<long long>(Hk) * d > INT_MAX || stride > INT_MAX / std::max(Hk, 1))
    return -1;
  // One block per (KV head, row); blockDim.x == d/2 (one thread per code
  // byte / dim-pair), so the row amax is a block reduce.
  granite4::kv_append_int4_batched_kernel<<<dim3(static_cast<unsigned int>(Hk),
                                                static_cast<unsigned int>(rows)),
                                            static_cast<unsigned int>(d / 2)>>>
      (d_src_k, d_src_v, d_cache_k, d_cache_v, d_cache_k_scale, d_cache_v_scale,
       base_pos, rows, stride, Hk, d);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_fused_batched(const std::uint16_t* d_q,
                                      const std::uint16_t* d_k,
                                      const std::uint16_t* d_v, float att_scale,
                                      int Hq, int Hk, int d, int base_kv,
                                      int rows, int k_stride,
                                      std::uint16_t* d_out) {
  if (rows > granite4::kMaxVerifyRows) return -1;
  return granite4_kernel_gqa_fused_rows(d_q, d_k, d_v, att_scale, Hq, Hk, d, base_kv,
                                        rows, k_stride, d_out);
}

int granite4_kernel_gqa_fused_rows(const std::uint16_t* d_q,
                                   const std::uint16_t* d_k,
                                   const std::uint16_t* d_v, float att_scale,
                                   int Hq, int Hk, int d, int base_kv, int rows,
                                   int k_stride, std::uint16_t* d_out) {
  if (d_q == nullptr || d_k == nullptr || d_v == nullptr || d_out == nullptr ||
      !std::isfinite(att_scale) || rows <= 0 || rows > 65535 || Hq <= 0 || Hk <= 0 ||
      d <= 0 || base_kv < 0 || Hq < Hk || Hq % Hk != 0 ||
      d > granite4::kMaxHeadDim ||
      static_cast<long long>(Hq) * d > INT_MAX ||
      static_cast<long long>(Hk) * d > INT_MAX ||
      base_kv > INT_MAX - rows || k_stride < base_kv + rows)
    return -1;
  // Shared memory is sized for the longest row (base_kv + rows); each block
  // row uses only its own causal prefix within that allocation.
  const std::size_t smem_bytes = granite4_gqa_fused_smem_bytes(d, base_kv + rows);
  if (smem_bytes > static_cast<std::size_t>(granite4_gqa_fused_smem_limit())) return -1;
  granite4::gqa_fused_batched_kernel<<<dim3(Hq, rows), d, smem_bytes>>>(
      d_q, d_k, d_v, att_scale, Hq, Hk, d, base_kv, rows, k_stride, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_silu_gate_up(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                 const std::uint16_t* d_x, float scale_g, float scale_u,
                                 int N, int K, std::uint16_t* d_out) {
  if (d_wg == nullptr || d_wu == nullptr || d_x == nullptr || d_out == nullptr ||
      !std::isfinite(scale_g) || !std::isfinite(scale_u) || N <= 0 || K <= 0 ||
      static_cast<long long>(N) * K > INT_MAX)
    return -1;
  granite4::silu_gate_up_kernel<<<N, 128>>>(d_wg, d_wu, d_x, scale_g, scale_u, N, K,
                                            d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_silu_gate_up_batched(const std::uint8_t* d_wg,
                                         const std::uint8_t* d_wu,
                                         const std::uint16_t* d_x, float scale_g,
                                         float scale_u, int N, int K, int rows,
                                         std::uint16_t* d_out) {
  if (d_wg == nullptr || d_wu == nullptr || d_x == nullptr || d_out == nullptr ||
      !std::isfinite(scale_g) || !std::isfinite(scale_u) || rows <= 0 || rows > 4 ||
      N <= 0 || K <= 0 || static_cast<long long>(N) * K > INT_MAX ||
      static_cast<long long>(rows) * K > INT_MAX ||
      static_cast<long long>(rows) * N > INT_MAX)
    return -1;
  granite4::silu_gate_up_batched_kernel<<<N, 128>>>(
      d_wg, d_wu, d_x, scale_g, scale_u, N, K, rows, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_sample_greedy(const std::uint16_t* d_p, int rows, int n, int* d_out) {
  if (d_p == nullptr || d_out == nullptr || rows <= 0 || n <= 0 ||
      static_cast<long long>(rows) * n > INT_MAX)
    return -1;
  granite4::sample_greedy_kernel<<<(rows + 255) / 256, 256>>>(d_p, rows, n, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_sample_top_p(const std::uint16_t* d_p, float nucleus,
                                 const float* d_u, int rows, int n, int* d_out) {
  if (d_p == nullptr || d_u == nullptr || d_out == nullptr ||
      !std::isfinite(nucleus) || nucleus <= 0.0f || nucleus > 1.0f ||
      rows <= 0 || n <= 0 || n > granite4::kTopPVocabMax ||
      static_cast<long long>(rows) * n > INT_MAX)
    return -1;
  // One block per row: the shared-memory arrays hold the row being sorted.
  granite4::sample_top_p_kernel<<<rows, 256>>>(d_p, nucleus, d_u, n, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

}  // extern "C"
