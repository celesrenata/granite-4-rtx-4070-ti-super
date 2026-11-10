// Tensor-core prefill: the multi-row forward for prompts and long suffixes.
//
// Decode is weight-bandwidth bound and runs the GEMV kernels in
// forward_gpu.cu. A prefill chunk of M positions instead reuses every weight
// byte M times, so it is compute bound and runs on the sm_89 tensor cores:
//
//   C[M][N] = A[M][K] (bf16, token-major) x W[N][K]^T (fp8 e4m3, row-major)
//
// One 256-thread block owns a BM x 128 output tile (BM = 64, or 16 for short
// chunks) and walks K in steps of 64 (grid.x = M tiles, grid.y = N tiles,
// grid.z = K splits). A 4-stage cp.async pipeline keeps three tiles of
// activations (bf16) and weights (raw fp8) in flight; each weight fragment is
// decoded fp8 -> bf16 right before its MMA (exact: every finite e4m3 value is
// a bf16 value), so the inner loop is plain mma.sync.m16n8k16.bf16 with f32
// accumulation.
//
// Products of bf16 values are exact in f32, so the result differs from the
// decode GEMV only in f32 summation order (within bf16 rounding of the
// output, the same tolerance class as the host-oracle parity gate).
//
// Attention runs prefill_attention_kernel (below), the tensor-core causal
// attention over the chunk.
//
// The epilogue applies the per-tensor dequant scale (up to three row
// segments, so the contiguous q|k|v and gate|up weights run as one GEMM) and
// writes bf16, f32, or bf16(residual + resid_scale * y) in place.
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <cstdio>

#include <cuda_runtime.h>

#include "device_math.cuh"
#include "prefill_gemm.h"

namespace granite4 {
namespace {

using dev::bf16_to_f32;
using dev::f32_to_bf16;

constexpr int kBN = 128;
constexpr int kBK = 64;
constexpr int kThreads = 256;

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const std::uint32_t (&a)[4],
                                               std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two e4m3 bytes -> packed bf16x2 (exact).
__device__ __forceinline__ std::uint32_t fp8x2_to_bf16x2(std::uint16_t pair) {
  const float2 f = dev::fp8x2_to_f32(pair);
  return (__float_as_uint(f.x) >> 16) | (__float_as_uint(f.y) & 0xffff0000u);
}

__device__ __forceinline__ float epi_scale(const GemmEpilogue& ep, int n) {
  return n < ep.seg1 ? ep.scale0 : (n < ep.seg2 ? ep.scale1 : ep.scale2);
}

template <GemmOut Out>
__device__ __forceinline__ void epi_store(const GemmEpilogue& ep, int m, int n,
                                          float v0, float v1) {
  const std::size_t off = static_cast<std::size_t>(m) * ep.ldc + n;
  const float y0 = epi_scale(ep, n) * v0;
  const float y1 = epi_scale(ep, n + 1) * v1;
  if constexpr (Out == GemmOut::kF32) {
    *reinterpret_cast<float2*>(ep.out_f32 + off) = make_float2(y0, y1);
  } else if constexpr (Out == GemmOut::kBf16) {
    *reinterpret_cast<std::uint32_t*>(ep.out_bf16 + off) =
        static_cast<std::uint32_t>(f32_to_bf16(y0)) |
        (static_cast<std::uint32_t>(f32_to_bf16(y1)) << 16);
  } else {  // kResidual: the decode epilogue, bf16(residual + rs * y)
    const std::uint32_t r = *reinterpret_cast<const std::uint32_t*>(ep.residual + off);
    const float o0 = dev::bf16_lo(r) + ep.resid_scale * y0;
    const float o1 = dev::bf16_hi(r) + ep.resid_scale * y1;
    *reinterpret_cast<std::uint32_t*>(ep.out_bf16 + off) =
        static_cast<std::uint32_t>(f32_to_bf16(o0)) |
        (static_cast<std::uint32_t>(f32_to_bf16(o1)) << 16);
  }
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
  const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill (out-of-range rows)
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(gmem),
               "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Shared-memory geometry of one pipeline stage (bytes). A rows are bf16 with a
// 72-element pitch, W rows raw fp8 with an 80-byte pitch: both make the
// fragment loads below bank-conflict free and keep 16-byte cp.async targets.
// Short tiles are weight-latency bound (deep pipeline); 64-row tiles are MMA
// bound and prefer two resident blocks per SM over pipeline depth (measured:
// 3 or 4 stages at BM = 64 cost 13-15% on 500- and 2000-token prompts).
template <int BM>
constexpr int stages() {
  return BM == 16 ? 4 : 2;
}
constexpr int kAPitch = kBK + 8;      // bf16 elements
constexpr int kWPitch = kBK + 16;     // bytes
template <int BM>
constexpr int stage_bytes() {
  return BM * kAPitch * 2 + kBN * kWPitch;
}
template <int BM>
constexpr int smem_bytes() {
  return stages<BM>() * stage_bytes<BM>();
}

// Split-K (gridDim.z > 1): block z covers k-tiles [z * kt_per, (z + 1) *
// kt_per) and stores its raw f32 accumulators to partial[z][M][N]; a second
// kernel sums the splits in z order and applies the epilogue, so the result
// does not depend on scheduling.
//
// BM is the M tile: 64 (8 warps as 2 x 4, 32 x 32 each) for real prefill
// chunks, 16 (8 warps as 1 x 8, 16 x 16 each) for short chunks, where a 64-row
// tile would spend 4x the tensor-core work per weight byte on zero padding.
// Global loads run kStages - 1 tiles ahead through cp.async; the weight tile
// stays fp8 in shared memory and is decoded to bf16 per MMA fragment.
template <GemmOut Out, int BM>
__global__ void __launch_bounds__(kThreads)
    gemm_fp8w_kernel(const std::uint16_t* __restrict__ A, int lda,
                     const std::uint8_t* __restrict__ W, int M, int N, int K,
                     int kt_per, float* __restrict__ partial, GemmEpilogue ep) {
  constexpr int kWarpsM = BM == 64 ? 2 : 1;
  constexpr int kWarpsN = 8 / kWarpsM;
  constexpr int MI = BM / kWarpsM / 16;   // m16 fragments per warp
  constexpr int NI = kBN / kWarpsN / 8;   // n8 fragments per warp
  constexpr int kWarpM = BM / kWarpsM;    // rows per warp
  constexpr int kWarpN = kBN / kWarpsN;   // cols per warp
  constexpr int kAChunks = BM * kBK / 8;  // 16-byte chunks per A tile
  constexpr int kWChunks = kBN * kBK / 16;
  constexpr int kStages = stages<BM>();
  extern __shared__ __align__(16) std::uint8_t smem[];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / kWarpsN;
  const int wn = warp % kWarpsN;
  const int g = lane >> 2;   // fragment group row
  const int t4 = lane & 3;   // fragment thread-in-group
  // M tiles are the fast grid axis: the blocks sharing one weight tile run
  // back to back, so each weight byte comes from DRAM once and the (small)
  // activation matrix stays resident in L2.
  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * kBN;
  const int kt0 = blockIdx.z * kt_per;
  const int ktiles = min(kt_per, K / kBK - kt0);

  auto a_tile = [&](int st) {
    return reinterpret_cast<std::uint16_t*>(smem + st * stage_bytes<BM>());
  };
  auto w_tile = [&](int st) { return smem + st * stage_bytes<BM>() + BM * kAPitch * 2; };
  auto issue = [&](int kt) {
    const int st = kt % kStages;
    const int k0 = (kt0 + kt) * kBK;
    std::uint16_t* const as = a_tile(st);
    std::uint8_t* const wsm = w_tile(st);
    for (int c = tid; c < kAChunks; c += kThreads) {
      const int r = c / (kBK / 8), col = (c % (kBK / 8)) * 8;
      const bool ok = m0 + r < M;
      cp_async16(as + r * kAPitch + col,
                 A + static_cast<std::size_t>(ok ? m0 + r : 0) * lda + k0 + col, ok);
    }
    for (int c = tid; c < kWChunks; c += kThreads) {
      const int r = c / (kBK / 16), col = (c % (kBK / 16)) * 16;
      const bool ok = n0 + r < N;
      cp_async16(wsm + r * kWPitch + col,
                 W + static_cast<std::size_t>(ok ? n0 + r : 0) * K + k0 + col, ok);
    }
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < ktiles) issue(s);
    cp_async_commit();
  }

  for (int kt = 0; kt < ktiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();  // tile kt landed for everyone; tile kt-1's readers done
    if (kt + kStages - 1 < ktiles) issue(kt + kStages - 1);
    cp_async_commit();

    const std::uint16_t* const as = a_tile(kt % kStages);
    const std::uint8_t* const wsm = w_tile(kt % kStages);
#pragma unroll
    for (int ks = 0; ks < kBK; ks += 16) {
      const int kk = ks + 2 * t4;
      std::uint32_t af[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const std::uint16_t* const r0 = as + (wm * kWarpM + mi * 16 + g) * kAPitch + kk;
        af[mi][0] = *reinterpret_cast<const std::uint32_t*>(r0);
        af[mi][1] = *reinterpret_cast<const std::uint32_t*>(r0 + 8 * kAPitch);
        af[mi][2] = *reinterpret_cast<const std::uint32_t*>(r0 + 8);
        af[mi][3] = *reinterpret_cast<const std::uint32_t*>(r0 + 8 * kAPitch + 8);
      }
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) {
        const std::uint8_t* const wr = wsm + (wn * kWarpN + ni * 8 + g) * kWPitch + kk;
        const std::uint32_t b0 = fp8x2_to_bf16x2(*reinterpret_cast<const std::uint16_t*>(wr));
        const std::uint32_t b1 =
            fp8x2_to_bf16x2(*reinterpret_cast<const std::uint16_t*>(wr + 8));
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) mma_bf16_16816(acc[mi][ni], af[mi], b0, b1);
      }
    }
  }
  cp_async_wait<0>();

  float* const pz = partial != nullptr
                       ? partial + static_cast<std::size_t>(blockIdx.z) * M * N
                       : nullptr;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) {
    const int r0 = m0 + wm * kWarpM + mi * 16 + g;
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int c = n0 + wn * kWarpN + ni * 8 + 2 * t4;
      if (c >= N) continue;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = r0 + 8 * h;
        if (r >= M) continue;
        if (pz != nullptr) {
          *reinterpret_cast<float2*>(pz + static_cast<std::size_t>(r) * N + c) =
              make_float2(acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);
        } else {
          epi_store<Out>(ep, r, c, acc[mi][ni][2 * h], acc[mi][ni][2 * h + 1]);
        }
      }
    }
  }
}

template <GemmOut Out>
__global__ void splitk_reduce_kernel(const float* __restrict__ partial, int splits, int M,
                                     int N, GemmEpilogue ep) {
  const std::size_t i =
      2 * (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  const std::size_t total = static_cast<std::size_t>(M) * N;
  if (i >= total) return;
  float v0 = 0.0f, v1 = 0.0f;
  for (int z = 0; z < splits; ++z) {
    const float2 p = *reinterpret_cast<const float2*>(partial + z * total + i);
    v0 += p.x;
    v1 += p.y;
  }
  epi_store<Out>(ep, static_cast<int>(i / N), static_cast<int>(i % N), v0, v1);
}

// ---------------------------------------------------------------------------
// Prefill glue (one launch per layer step, all rows at once).
// ---------------------------------------------------------------------------

// out[r][j] = mult * emb[tokens[start + r]][j]
__global__ void embed_rows_kernel(const std::int32_t* tokens, int start,
                                  const std::uint16_t* emb, float mult,
                                  std::uint16_t* out, int H) {
  const int r = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const std::int32_t tok = tokens[start + r];
  out[static_cast<std::size_t>(r) * H + j] =
      f32_to_bf16(mult * bf16_to_f32(emb[static_cast<std::size_t>(tok) * H + j]));
}

// qkv is token-major [rows][Hq*d | Hk*d | Hk*d]. Row r sits at position
// base_pos + r: q is rotated into q_rot [rows][Hq*d]; k is rotated and v
// copied straight into the persistent [Hk][stride][d] caches. RoPE is the
// Llama half split, the same formula as the decode kernels.
__global__ void rope_split_append_kernel(const std::uint16_t* qkv, const float* cos_tbl,
                                         const float* sin_tbl, int base_pos, int Hq,
                                         int Hk, int d, int stride,
                                         std::uint16_t* q_rot, std::uint16_t* cache_k,
                                         std::uint16_t* cache_v, const int* d_pt) {
  const int r = blockIdx.y;
  const int nq = Hq * d, nk = Hk * d;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nq + 2 * nk) return;
  const int pos = base_pos + r;
  const int row_pos = (d_pt != nullptr) ? d_pt[pos] : pos;  // P4 arena row
  const std::uint16_t* const row = qkv + static_cast<std::size_t>(r) * (nq + 2 * nk);
  if (idx >= nq + nk) {  // v: plain copy into the cache
    const int local = idx - nq - nk;
    const int hkv = local / d, i = local % d;
    cache_v[(static_cast<std::size_t>(hkv) * stride + row_pos) * d + i] = row[idx];
    return;
  }
  const bool is_q = idx < nq;
  const int local = is_q ? idx : idx - nq;
  const std::uint16_t* const x = row + (is_q ? 0 : nq);
  const int h = local / d, col = local % d;
  const int half = d / 2;
  const int i = col < half ? col : col - half;
  const float c = cos_tbl[static_cast<std::size_t>(pos) * half + i];
  const float s = sin_tbl[static_cast<std::size_t>(pos) * half + i];
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
  if (is_q) {
    q_rot[static_cast<std::size_t>(r) * nq + local] = f32_to_bf16(y);
  } else {
    cache_k[(static_cast<std::size_t>(h) * stride + row_pos) * d + col] = f32_to_bf16(y);
  }
}

// KV8 (Q8_0): the int8 mirror of rope_split_append_kernel. One block per
// (chunk row, head) over the q|k|v heads. Q is rotated to bf16 exactly as in
// the bf16 path; K (rotated) and V are quantized per (head, position) row to
// int8 codes + a row scale with dev::kv8_* (the block reduces the row amax
// first, thread 0 publishes the scale). Requires d == 128 (== the prefill
// attention head dim).
constexpr int kPrefillInt8D = 128;
__global__ void __launch_bounds__(kPrefillInt8D)
    rope_split_append_int8_kernel(const std::uint16_t* qkv, const float* cos_tbl,
                                  const float* sin_tbl, int base_pos, int Hq, int Hk,
                                  int d, int stride, std::uint16_t* q_rot,
                                  std::int8_t* cache_k, std::int8_t* cache_v,
                                  float* cache_k_scale, float* cache_v_scale, const int* d_pt) {
  const int r = blockIdx.y;
  const int head = blockIdx.x;  // 0..Hq-1: q, Hq..Hq+Hk-1: k, else v
  const int pos = base_pos + r;
  const int row_pos = (d_pt != nullptr) ? d_pt[pos] : pos;  // P4 arena row
  const int nq = Hq * d, nk = Hk * d;
  const int i = threadIdx.x;
  const int half = d / 2;

  __shared__ float s_val[kPrefillInt8D];
  if (head < Hq) {  // Q: rotate to bf16 (byte-identical to the bf16 path)
    const int h = head, col = i;
    const std::uint16_t* const x = qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + h * d;
    const int j = col < half ? col : col - half;
    const float c = cos_tbl[static_cast<std::size_t>(pos) * half + j];
    const float s = sin_tbl[static_cast<std::size_t>(pos) * half + j];
    float y;
    if (col < half) {
      y = bf16_to_f32(x[col]) * c - bf16_to_f32(x[col + half]) * s;
    } else {
      y = bf16_to_f32(x[col]) * c + bf16_to_f32(x[col - half]) * s;
    }
    q_rot[static_cast<std::size_t>(r) * nq + h * d + col] = f32_to_bf16(y);
    return;
  }

  const bool is_k = head < Hq + Hk;
  const int h = is_k ? head - Hq : head - Hq - Hk;
  const int col = i;
  float val;
  if (is_k) {  // K: rotate, then quantize
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + h * d;
    const int j = col < half ? col : col - half;
    const float c = cos_tbl[static_cast<std::size_t>(pos) * half + j];
    const float s = sin_tbl[static_cast<std::size_t>(pos) * half + j];
    if (col < half)
      val = bf16_to_f32(x[col]) * c - bf16_to_f32(x[col + half]) * s;
    else
      val = bf16_to_f32(x[col]) * c + bf16_to_f32(x[col - half]) * s;
  } else {  // V: quantize the raw bf16 row
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + nk + h * d;
    val = bf16_to_f32(x[col]);
  }
  s_val[col] = val;
  __syncthreads();  // all row values are in s_val before the amax reads them
  __shared__ float s_scale;
  if (i == 0) {
    float a = 0.0f;
    for (int j = 0; j < d; ++j) a = fmaxf(a, fabsf(s_val[j]));
    s_scale = dev::kv8_scale(a);
  }
  __syncthreads();  // s_scale is published before any thread quantizes
  const float sc = s_scale;
  if (i == 0) {
    if (is_k)
      cache_k_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
    else
      cache_v_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
  }
  const std::int8_t code = dev::kv8_quant(s_val[i], sc);
  const std::size_t cbase = (static_cast<std::size_t>(h) * stride + row_pos) * d + col;
  if (is_k)
    cache_k[cbase] = code;
  else
    cache_v[cbase] = code;
}

// KV4 (Q4_0): the int4 mirror of rope_split_append_int8_kernel. One block per
// (chunk row, head) over the q|k|v heads, but blockDim.x == d/2: each thread
// owns a dim-pair (even dim = lower nibble, odd dim = upper nibble) of one
// K/V row. Q is rotated to bf16 exactly as in the bf16 path; K (rotated) and
// V are quantized per (head, position) row to two-per-byte int4 codes + a row
// scale with dev::q4_*. Requires d == 128 (== the prefill attention head dim).
constexpr int kPrefillInt4D = 64;
__global__ void __launch_bounds__(kPrefillInt4D)
    rope_split_append_int4_kernel(const std::uint16_t* qkv, const float* cos_tbl,
                                  const float* sin_tbl, int base_pos, int Hq, int Hk,
                                  int d, int stride, std::uint16_t* q_rot,
                                  std::uint8_t* cache_k, std::uint8_t* cache_v,
                                  float* cache_k_scale, float* cache_v_scale, const int* d_pt) {
  const int r = blockIdx.y;
  const int head = blockIdx.x;  // 0..Hq-1: q, Hq..Hq+Hk-1: k, else v
  const int pos = base_pos + r;
  const int row_pos = (d_pt != nullptr) ? d_pt[pos] : pos;  // P4 arena row
  const int nq = Hq * d, nk = Hk * d;
  const int w = threadIdx.x;
  const int half = d / 2;

  if (head < Hq) {  // Q: rotate to bf16 (byte-identical to the bf16 path)
    const int h = head, col = 2 * w;
    const int h1 = col + 1;
    const std::uint16_t* const x = qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + h * d;
    // Each dim uses its own RoPE frequency: j = col < half ? col : col - half.
    const int j0 = col < half ? col : col - half;
    const int j1 = h1 < half ? h1 : h1 - half;
    const std::size_t p0 = static_cast<std::size_t>(pos) * half + j0;
    const std::size_t p1 = static_cast<std::size_t>(pos) * half + j1;
    const std::size_t qo = static_cast<std::size_t>(r) * nq + h * d;
    q_rot[qo + col] = col < half
        ? f32_to_bf16(bf16_to_f32(x[col]) * cos_tbl[p0] - bf16_to_f32(x[col + half]) * sin_tbl[p0])
        : f32_to_bf16(bf16_to_f32(x[col]) * cos_tbl[p0] + bf16_to_f32(x[col - half]) * sin_tbl[p0]);
    q_rot[qo + h1] = h1 < half
        ? f32_to_bf16(bf16_to_f32(x[h1]) * cos_tbl[p1] - bf16_to_f32(x[h1 + half]) * sin_tbl[p1])
        : f32_to_bf16(bf16_to_f32(x[h1]) * cos_tbl[p1] + bf16_to_f32(x[h1 - half]) * sin_tbl[p1]);
    return;
  }

  const bool is_k = head < Hq + Hk;
  const int h = is_k ? head - Hq : head - Hq - Hk;
  const int col = 2 * w;
  const int h1 = col + 1;
  __shared__ float s_val[kPrefillInt4D][2];  // [thread][dim-pair]
  if (is_k) {  // K: rotate the dim-pair (per-dim frequency), then quantize
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + h * d;
    const int j0 = col < half ? col : col - half;
    const int j1 = h1 < half ? h1 : h1 - half;
    const std::size_t p0 = static_cast<std::size_t>(pos) * half + j0;
    const std::size_t p1 = static_cast<std::size_t>(pos) * half + j1;
    s_val[w][0] = col < half
        ? bf16_to_f32(x[col]) * cos_tbl[p0] - bf16_to_f32(x[col + half]) * sin_tbl[p0]
        : bf16_to_f32(x[col]) * cos_tbl[p0] + bf16_to_f32(x[col - half]) * sin_tbl[p0];
    s_val[w][1] = h1 < half
        ? bf16_to_f32(x[h1]) * cos_tbl[p1] - bf16_to_f32(x[h1 + half]) * sin_tbl[p1]
        : bf16_to_f32(x[h1]) * cos_tbl[p1] + bf16_to_f32(x[h1 - half]) * sin_tbl[p1];
  } else {  // V: raw dim-pair
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + nk + h * d;
    s_val[w][0] = bf16_to_f32(x[col]);
    s_val[w][1] = bf16_to_f32(x[h1]);
  }
  __shared__ float s_scale;
  __syncthreads();  // all threads' s_val writes are visible before the amax
  if (w == 0) {
    float a = 0.0f;
    for (int j = 0; j < d / 2; ++j) {
      a = fmaxf(a, fabsf(s_val[j][0]));
      a = fmaxf(a, fabsf(s_val[j][1]));
    }
    s_scale = dev::q4_scale(a);
  }
  __syncthreads();  // s_scale is published before any thread quantizes
  const float sc = s_scale;
  if (w == 0) {
    if (is_k)
      cache_k_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
    else
      cache_v_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
  }
  __syncthreads();
  const unsigned char code =
      static_cast<unsigned char>((dev::q4_quant(s_val[w][0], sc) & 0x0fu) |
                                 (dev::q4_quant(s_val[w][1], sc) << 4));
  const std::size_t cbase = (static_cast<std::size_t>(h) * stride + row_pos) * (d / 2) + w;
  if (is_k)
    cache_k[cbase] = code;
  else
    cache_v[cbase] = code;
}

// ---- K5:V4 prefill append (a separate kernel so the live int4 kernel above is untouched) ----
__global__ void __launch_bounds__(kPrefillInt4D)
    rope_split_append_k5v4_kernel(const std::uint16_t* qkv, const float* cos_tbl,
                                  const float* sin_tbl, int base_pos, int Hq, int Hk,
                                  int d, int stride, std::uint16_t* q_rot,
                                  std::uint8_t* cache_k, std::uint8_t* cache_v,
                                  float* cache_k_scale, float* cache_v_scale, const int* d_pt) {
  const int r = blockIdx.y;
  const int head = blockIdx.x;  // 0..Hq-1: q, Hq..Hq+Hk-1: k, else v
  const int pos = base_pos + r;
  const int row_pos = (d_pt != nullptr) ? d_pt[pos] : pos;  // P4 arena row
  const int nq = Hq * d, nk = Hk * d;
  const int w = threadIdx.x;
  const int half = d / 2;

  if (head < Hq) {  // Q: rotate to bf16 (byte-identical to the bf16 path)
    const int h = head, col = 2 * w;
    const int h1 = col + 1;
    const std::uint16_t* const x = qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + h * d;
    // Each dim uses its own RoPE frequency: j = col < half ? col : col - half.
    const int j0 = col < half ? col : col - half;
    const int j1 = h1 < half ? h1 : h1 - half;
    const std::size_t p0 = static_cast<std::size_t>(pos) * half + j0;
    const std::size_t p1 = static_cast<std::size_t>(pos) * half + j1;
    const std::size_t qo = static_cast<std::size_t>(r) * nq + h * d;
    q_rot[qo + col] = col < half
        ? f32_to_bf16(bf16_to_f32(x[col]) * cos_tbl[p0] - bf16_to_f32(x[col + half]) * sin_tbl[p0])
        : f32_to_bf16(bf16_to_f32(x[col]) * cos_tbl[p0] + bf16_to_f32(x[col - half]) * sin_tbl[p0]);
    q_rot[qo + h1] = h1 < half
        ? f32_to_bf16(bf16_to_f32(x[h1]) * cos_tbl[p1] - bf16_to_f32(x[h1 + half]) * sin_tbl[p1])
        : f32_to_bf16(bf16_to_f32(x[h1]) * cos_tbl[p1] + bf16_to_f32(x[h1 - half]) * sin_tbl[p1]);
    return;
  }

  const bool is_k = head < Hq + Hk;
  const int h = is_k ? head - Hq : head - Hq - Hk;
  const int col = 2 * w;
  const int h1 = col + 1;
  __shared__ float s_val[kPrefillInt4D][2];  // [thread][dim-pair]
  if (is_k) {  // K: rotate the dim-pair (per-dim frequency), then quantize
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + h * d;
    const int j0 = col < half ? col : col - half;
    const int j1 = h1 < half ? h1 : h1 - half;
    const std::size_t p0 = static_cast<std::size_t>(pos) * half + j0;
    const std::size_t p1 = static_cast<std::size_t>(pos) * half + j1;
    s_val[w][0] = col < half
        ? bf16_to_f32(x[col]) * cos_tbl[p0] - bf16_to_f32(x[col + half]) * sin_tbl[p0]
        : bf16_to_f32(x[col]) * cos_tbl[p0] + bf16_to_f32(x[col - half]) * sin_tbl[p0];
    s_val[w][1] = h1 < half
        ? bf16_to_f32(x[h1]) * cos_tbl[p1] - bf16_to_f32(x[h1 + half]) * sin_tbl[p1]
        : bf16_to_f32(x[h1]) * cos_tbl[p1] + bf16_to_f32(x[h1 - half]) * sin_tbl[p1];
  } else {  // V: raw dim-pair
    const std::uint16_t* const x =
        qkv + static_cast<std::size_t>(r) * (nq + 2 * nk) + nq + nk + h * d;
    s_val[w][0] = bf16_to_f32(x[col]);
    s_val[w][1] = bf16_to_f32(x[h1]);
  }
  __shared__ float s_scale;
  __shared__ unsigned char s_hi[kPrefillInt4D * 2];  // K5: high bit of each dim
  __syncthreads();  // all threads' s_val writes are visible before the amax
  if (w == 0) {
    float a = 0.0f;
    for (int j = 0; j < d / 2; ++j) {
      a = fmaxf(a, fabsf(s_val[j][0]));
      a = fmaxf(a, fabsf(s_val[j][1]));
    }
    s_scale = is_k ? dev::k5_scale(a) : dev::q4_scale(a);
  }
  __syncthreads();  // s_scale is published before any thread quantizes
  const float sc = s_scale;
  if (w == 0) {
    if (is_k)
      cache_k_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
    else
      cache_v_scale[static_cast<std::size_t>(h) * stride + row_pos] = sc;
  }
  __syncthreads();
  if (is_k) {
    // K5: the low 4 bits of each code pack exactly like int4 into bytes [0, 64) of the 80-byte
    // row; the 5th bits go to the 16-byte plane at [64, 80) (bit d % 8 of byte 64 + d / 8).
    const unsigned char c0 = dev::k5_quant(s_val[w][0], sc);
    const unsigned char c1 = dev::k5_quant(s_val[w][1], sc);
    const std::size_t krow = (static_cast<std::size_t>(h) * stride + row_pos) * dev::kK5RowBytes;
    cache_k[krow + w] = static_cast<unsigned char>((c0 & 0x0fu) | ((c1 & 0x0fu) << 4));
    s_hi[col] = static_cast<unsigned char>(c0 >> 4);
    s_hi[h1] = static_cast<unsigned char>(c1 >> 4);
    __syncthreads();
    if (w < dev::kK5RowBytes - dev::kK5NibbleBytes) {  // 16 plane bytes
      unsigned plane = 0;
#pragma unroll
      for (int bit = 0; bit < 8; ++bit) plane |= static_cast<unsigned>(s_hi[8 * w + bit] & 1u) << bit;
      cache_k[krow + dev::kK5NibbleBytes + w] = static_cast<unsigned char>(plane);
    }
    return;
  }
  const unsigned char code =
      static_cast<unsigned char>((dev::q4_quant(s_val[w][0], sc) & 0x0fu) |
                                 (dev::q4_quant(s_val[w][1], sc) << 4));
  const std::size_t cbase = (static_cast<std::size_t>(h) * stride + row_pos) * (d / 2) + w;
  cache_v[cbase] = code;
}

// ---------------------------------------------------------------------------
// Causal prefill attention on the tensor cores (flash-attention style).
//
// One block = one KV head x 16 consecutive query positions; the G = Hq / Hk
// query heads of the group share every K/V tile, so the block's 4 warps each
// own 16 "query rows" = 16 / G positions x G heads (one m16 MMA tile). For
// each 64-position KV tile: S = Q K^T (mma), logits rounded to bf16 as in the
// reference pipeline, causal mask, online softmax in f32 (running max and
// row sum), then O += P V with P as bf16. O is normalized once at the end.
// The decode / verify fused-GQA kernel keeps the reference's double
// denominator; this kernel's f32 online softmax agrees within bf16 rounding
// of the output, the same tolerance as the rest of the tensor-core prefill.
// ---------------------------------------------------------------------------
constexpr int kAttnKvTile = 64;

__device__ __forceinline__ float bf16_round(float x) {
  return __uint_as_float(static_cast<std::uint32_t>(f32_to_bf16(x)) << 16);
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  return static_cast<std::uint32_t>(f32_to_bf16(lo)) |
         (static_cast<std::uint32_t>(f32_to_bf16(hi)) << 16);
}

// Quantized KV: when Int8 or Int4 the persistent cache is int8/int4 codes +
// a per-row float scale; the tile fill dequantizes each row into the same
// bf16 Ks/Vt tiles the bf16 path writes, so everything after the tile fill
// (QK, online softmax, PV) is byte-identical between the three
// instantiations.
template <int D, int G, bool Int8, bool Int4, int W, bool K5 = false>
__global__ void __launch_bounds__(W * 32)
    prefill_attention_kernel(const std::uint16_t* __restrict__ q_rot,
                             const std::uint16_t* __restrict__ cache_k,
                             const std::uint16_t* __restrict__ cache_v, float scale,
                             int Hq, int base_pos, int rows, int stride,
                             std::uint16_t* __restrict__ out,
                             const std::int8_t* __restrict__ cache_k8,
                             const std::int8_t* __restrict__ cache_v8,
                             const float* __restrict__ k_scale,
                             const float* __restrict__ v_scale,
                             const std::uint8_t* __restrict__ cache_k4,
                             const std::uint8_t* __restrict__ cache_v4,
                             const int* __restrict__ d_pt) {
  constexpr int kPosPerWarp = 16 / G;
  // W warps share each K/V tile: the tile fill (load + dequant) is paid once
  // per W * kPosPerWarp query positions, so wider blocks amortize it.
  constexpr int kPosPerBlock = W * kPosPerWarp;
  constexpr int KT = kAttnKvTile;
  constexpr int KP = D + 8;   // K tile pitch (bank-conflict-free B loads)
  constexpr int VP = KT + 8;  // V^T tile pitch
  __shared__ __align__(16) std::uint16_t Ks[KT][KP];
  __shared__ __align__(16) std::uint16_t Vt[D][VP];

  const int hkv = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int p0 = blockIdx.x * kPosPerBlock;  // first local row of the block
  const int wp0 = p0 + warp * kPosPerWarp;

  // This thread's two fragment rows (g and g + 8): local position and head.
  int rl[2], head[2], pabs[2];
  bool ok[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    rl[h] = wp0 + r / G;
    head[h] = hkv * G + r % G;
    ok[h] = rl[h] < rows;
    pabs[h] = base_pos + rl[h];
  }
  const std::size_t qpitch = static_cast<std::size_t>(Hq) * D;

  std::uint32_t qf[D / 16][4];
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
    const int k = ks * 16 + 2 * t4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const std::uint16_t* const q = q_rot + rl[h] * qpitch + head[h] * D;
      qf[ks][h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(q + k) : 0u;
      qf[ks][2 + h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(q + k + 8) : 0u;
    }
  }

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY};
  float l[2] = {0.0f, 0.0f};

  const int last = min(p0 + kPosPerBlock, rows) - 1;
  const int kv_end = base_pos + last + 1;
  const bool quantized = Int8 || Int4;
  const std::uint16_t* const kb =
      quantized ? nullptr : cache_k + static_cast<std::size_t>(hkv) * stride * D;
  const std::uint16_t* const vb =
      quantized ? nullptr : cache_v + static_cast<std::size_t>(hkv) * stride * D;
  const std::int8_t* const k8 =
      Int8 ? cache_k8 + static_cast<std::size_t>(hkv) * stride * D : nullptr;
  const std::int8_t* const v8 =
      Int8 ? cache_v8 + static_cast<std::size_t>(hkv) * stride * D : nullptr;
  const std::uint8_t* const k4 =
      Int4 ? cache_k4 + static_cast<std::size_t>(hkv) * stride * (K5 ? dev::kK5RowBytes : D / 2)
           : nullptr;
  const std::uint8_t* const v4 =
      Int4 ? cache_v4 + static_cast<std::size_t>(hkv) * stride * (D / 2) : nullptr;
  const float* const ksc = quantized ? k_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const float* const vsc = quantized ? v_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);

  for (int j0 = 0; j0 < kv_end; j0 += KT) {
    __syncthreads();  // the previous tile's readers are done
    // K tile row-major (coalesced); V tile transposed, consecutive threads on
    // consecutive positions so the scalar transposed stores do not conflict.
    if (!quantized) {
      for (int idx = threadIdx.x; idx < KT * D / 8; idx += W * 32) {
        const int kr = idx / (D / 8), kc = (idx % (D / 8)) * 8;
        const int kj = j0 + kr;
        // Logical position kj lives in arena row d_pt[kj] (identity when
        // d_pt is null); out-of-tile positions keep the zero fill below.
        const int krow = (kj < kv_end) ? ((d_pt != nullptr) ? d_pt[kj] : kj) : kj;
        *reinterpret_cast<uint4*>(&Ks[kr][kc]) =
            kj < kv_end ? *reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(krow) * D + kc)
                        : zero;
        const int vr = idx % KT, vc = (idx / KT) * 8;
        const int vj = j0 + vr;
        const int vrow = (vj < kv_end) ? ((d_pt != nullptr) ? d_pt[vj] : vj) : vj;
        const uint4 v4 =
            vj < kv_end
                ? *reinterpret_cast<const uint4*>(vb + static_cast<std::size_t>(vrow) * D + vc)
                : zero;
        const std::uint32_t vw[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          Vt[vc + 2 * e][vr] = static_cast<std::uint16_t>(vw[e] & 0xffffu);
          Vt[vc + 2 * e + 1][vr] = static_cast<std::uint16_t>(vw[e] >> 16);
        }
      }
    } else if (Int4) {
      // KV4: one thread per (tile-row, code-byte) - read the two packed int4
      // codes, dequantize with the row's scale, and store the packed bf16
      // values into the same Ks (row-major) / Vt (transposed) tiles the
      // bf16 path writes; out-of-tile rows store 0 (the bf16 path's zero
      // fill) so masked logits and PV stay 0.
      // K: consecutive threads on consecutive code bytes of one row
      // (row-major Ks, conflict-free word stores).
      for (int idx = threadIdx.x; idx < KT * (D / 2); idx += W * 32) {
        const int kr = idx / (D / 2), cw = idx % (D / 2);
        const int kj = j0 + kr;
        std::uint32_t packed = 0u;
        if (kj < kv_end) {
          const int row = (d_pt != nullptr) ? d_pt[kj] : kj;
          float k0, k1;
          if (K5) {
            // 80-byte K row: nibble byte cw, plus the high bits of dims 2cw and 2cw+1 from the plane.
            const std::size_t kr0 = static_cast<std::size_t>(row) * dev::kK5RowBytes;
            const unsigned plane = k4[kr0 + dev::kK5NibbleBytes + cw / 4];
            dev::k5_dequant_pair(k4[kr0 + cw], (plane >> ((2 * cw) & 7)) & 1u,
                                 (plane >> ((2 * cw + 1) & 7)) & 1u, ksc[row], k0, k1);
          } else {
            dev::q4_dequant_pair(k4[static_cast<std::size_t>(row) * (D / 2) + cw], ksc[row], k0, k1);
          }
          packed = static_cast<std::uint32_t>(f32_to_bf16(k0)) |
                   (static_cast<std::uint32_t>(f32_to_bf16(k1)) << 16);
        }
        *reinterpret_cast<std::uint32_t*>(&Ks[kr][2 * cw]) = packed;
      }
      // V (transposed): consecutive threads on consecutive positions, so the
      // Vt[dim][pos] stores hit consecutive halfwords (no bank conflicts).
      for (int idx = threadIdx.x; idx < KT * (D / 2); idx += W * 32) {
        const int vr = idx % KT, cw = idx / KT;
        const int vj = j0 + vr;
        float v0 = 0.0f, v1 = 0.0f;
        if (vj < kv_end) {
          const int row = (d_pt != nullptr) ? d_pt[vj] : vj;
          dev::q4_dequant_pair(v4[static_cast<std::size_t>(row) * (D / 2) + cw], vsc[row], v0, v1);
        }
        Vt[2 * cw][vr] = vj < kv_end ? f32_to_bf16(v0) : std::uint16_t{0};
        Vt[2 * cw + 1][vr] = vj < kv_end ? f32_to_bf16(v1) : std::uint16_t{0};
      }
    } else {
      // KV8: dequant each K row (row-major, one thread per 8-byte chunk) and V
      // element into the same bf16 Ks/Vt tiles the bf16 path writes. Codes are
      // read as aligned uint64 (8 per lane) instead of scalar int8, and the
      // dequant runs over the register codes; out-of-tile positions store 0
      // (the bf16 path's zero fill), so the masked logits (and the Vt columns
      // the PV mma still reads) stay 0.
      for (int idx = threadIdx.x; idx < KT * D / 8; idx += W * 32) {
        const int r = idx / (D / 8), w = (idx % (D / 8)) * 8;
        const int j = j0 + r;
        if (j < kv_end) {
          const int row = (d_pt != nullptr) ? d_pt[j] : j;
          const std::size_t off = static_cast<std::size_t>(row) * D + w;
          const float sk = ksc[row];
          const uint64_t kraw = *reinterpret_cast<const uint64_t*>(k8 + off);
          const std::int8_t* const kp =
              reinterpret_cast<const std::int8_t*>(&kraw);
          std::uint16_t kc[8];
          #pragma unroll
          for (int e = 0; e < 8; ++e)
            kc[e] = f32_to_bf16(dev::kv8_dequant(kp[e], sk));
          *reinterpret_cast<uint4*>(&Ks[r][w]) =
              *reinterpret_cast<const uint4*>(kc);
        } else {
          *reinterpret_cast<uint4*>(&Ks[r][w]) = zero;
        }
      }
      // V (transposed): consecutive threads on consecutive positions (as the
      // bf16 path), so the scalar Vt[dim][pos] stores hit consecutive
      // halfwords. Mapping threads along the dims instead put every thread
      // of a warp on one bank (the Vt row pitch is a multiple of 128 B).
      for (int idx = threadIdx.x; idx < KT * D / 8; idx += W * 32) {
        const int r = idx % KT, w = (idx / KT) * 8;
        const int j = j0 + r;
        if (j < kv_end) {
          const int row = (d_pt != nullptr) ? d_pt[j] : j;
          const std::size_t off = static_cast<std::size_t>(row) * D + w;
          const float sv = vsc[row];
          const uint64_t vraw = *reinterpret_cast<const uint64_t*>(v8 + off);
          const std::int8_t* const vp =
              reinterpret_cast<const std::int8_t*>(&vraw);
          #pragma unroll
          for (int e = 0; e < 8; ++e)
            Vt[w + e][r] = f32_to_bf16(dev::kv8_dequant(vp[e], sv));
        } else {
          #pragma unroll
          for (int e = 0; e < 8; ++e) Vt[w + e][r] = std::uint16_t{0};
        }
      }
    }
    __syncthreads();

    float s[KT / 8][4];
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks) {
#pragma unroll
      for (int ni = 0; ni < KT / 8; ++ni) {
        const std::uint16_t* const kr = &Ks[ni * 8 + g][ks * 16 + 2 * t4];
        mma_bf16_16816(s[ni], qf[ks], *reinterpret_cast<const std::uint32_t*>(kr),
                       *reinterpret_cast<const std::uint32_t*>(kr + 8));
      }
    }

    // bf16 logits (the reference rounding), causal mask, tile row max.
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        const int j = j0 + ni * 8 + 2 * t4 + (e & 1);
        const float v = j <= pabs[h] ? bf16_round(scale * s[ni][e]) : -INFINITY;
        s[ni][e] = v;
        mx[h] = fmaxf(mx[h], v);
      }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      alpha[h] = expf(m[h] - mx[h]);  // 0 on the first tile (m = -inf)
      l[h] *= alpha[h];
      m[h] = mx[h];
    }
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }

    // P = exp(S - m) as bf16 A fragments (two n8 tiles = one k16 step).
    std::uint32_t pf[KT / 16][4];
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni) {
      const float e0 = expf(s[ni][0] - m[0]), e1 = expf(s[ni][1] - m[0]);
      const float e2 = expf(s[ni][2] - m[1]), e3 = expf(s[ni][3] - m[1]);
      l[0] += e0 + e1;
      l[1] += e2 + e3;
      pf[ni / 2][(ni & 1) * 2] = pack_bf16x2(e0, e1);
      pf[ni / 2][(ni & 1) * 2 + 1] = pack_bf16x2(e2, e3);
    }
#pragma unroll
    for (int kk = 0; kk < KT / 16; ++kk) {
#pragma unroll
      for (int nd = 0; nd < D / 8; ++nd) {
        const std::uint16_t* const vr = &Vt[nd * 8 + g][kk * 16 + 2 * t4];
        mma_bf16_16816(o[nd], pf[kk], *reinterpret_cast<const std::uint32_t*>(vr),
                       *reinterpret_cast<const std::uint32_t*>(vr + 8));
      }
    }
  }

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!ok[h]) continue;
    const float inv = 1.0f / l[h];
    std::uint16_t* const dst = out + rl[h] * qpitch + head[h] * D + 2 * t4;
#pragma unroll
    for (int nd = 0; nd < D / 8; ++nd)
      *reinterpret_cast<std::uint32_t*>(dst + nd * 8) =
          pack_bf16x2(o[nd][2 * h] * inv, o[nd][2 * h + 1] * inv);
  }
}

// Quantized-cache prefill attention (KV8 / KV4 / K5:V4; G4_PREFILL_ATTN_CODES=0 restores the
// dequantizing fill above). Same block layout and online softmax as prefill_attention_kernel, but
// the K / V tiles hold the integer codes (exact in bf16) and the per-position scales go to the
// logits (K) and probabilities (V), so a tile fill is vector loads plus bit operations instead of
// a scalar load, multiply and bf16 rounding per element. V is stored row-major and read with
// ldmatrix.trans; the next tile's cache loads are issued before the current tile is attended.
// Numerics: K enters exactly (not rounded to bf16 after scaling); P carries the V scale in bf16.
constexpr int kPqP = 128 + 8;  // K / V tile pitch (bf16): conflict-free fragment loads

struct PqRaw {
  uint4 k, v;     // KV8: 16 codes each; KV4 / K5: .x / .y = 8 nibble bytes (16 dims)
  unsigned hib;   // K5: the 16 high bits of the item's dims
  float sk, sv;
};

template <bool Int8, bool K5>
__device__ __forceinline__ void pq_load(PqRaw& r, int row, bool valid, int part,
                                        const std::int8_t* k8, const std::int8_t* v8,
                                        const std::uint8_t* k4, const std::uint8_t* v4,
                                        const float* ksc, const float* vsc) {
  if (!valid) return;
  const int c0 = 16 * part;
  r.sk = ksc[row];
  r.sv = vsc[row];
  if (Int8) {
    r.k = *reinterpret_cast<const uint4*>(k8 + static_cast<std::size_t>(row) * 128 + c0);
    r.v = *reinterpret_cast<const uint4*>(v8 + static_cast<std::size_t>(row) * 128 + c0);
  } else {
    const std::size_t kb = K5 ? static_cast<std::size_t>(row) * dev::kK5RowBytes + c0 / 2
                              : static_cast<std::size_t>(row) * 64 + c0 / 2;
    const uint2 kr = *reinterpret_cast<const uint2*>(k4 + kb);
    const uint2 vr = *reinterpret_cast<const uint2*>(v4 + static_cast<std::size_t>(row) * 64 + c0 / 2);
    r.k.x = kr.x;
    r.k.y = kr.y;
    r.v.x = vr.x;
    r.v.y = vr.y;
    if (K5)
      r.hib = *reinterpret_cast<const std::uint16_t*>(
          k4 + static_cast<std::size_t>(row) * dev::kK5RowBytes + dev::kK5NibbleBytes + c0 / 8);
  }
}

template <bool Int8, bool K5>
__device__ __forceinline__ void pq_store(std::uint16_t (*Ks)[kPqP], std::uint16_t (*Vs)[kPqP],
                                         float* ksc_s, float* vsc_s, int kr, int part,
                                         const PqRaw& r, bool valid) {
  const int c0 = 16 * part;
  std::uint32_t kw[8], vw[8];
  if (!valid) {
#pragma unroll
    for (int e = 0; e < 8; ++e) kw[e] = vw[e] = 0u;
  } else if (Int8) {
    dev::int8x16_to_bf16(r.k, kw);
    dev::int8x16_to_bf16(r.v, vw);
  } else {
    constexpr std::uint32_t kOff = K5 ? dev::kK5CodeOff : dev::kQ4CodeOff;
    dev::codes8_to_bf16<K5>(r.k.x, r.hib, kOff, &kw[0]);
    dev::codes8_to_bf16<K5>(r.k.y, r.hib >> 8, kOff, &kw[4]);
    dev::codes8_to_bf16<false>(r.v.x, 0u, dev::kQ4CodeOff, &vw[0]);
    dev::codes8_to_bf16<false>(r.v.y, 0u, dev::kQ4CodeOff, &vw[4]);
  }
  *reinterpret_cast<uint4*>(&Ks[kr][c0]) = make_uint4(kw[0], kw[1], kw[2], kw[3]);
  *reinterpret_cast<uint4*>(&Ks[kr][c0 + 8]) = make_uint4(kw[4], kw[5], kw[6], kw[7]);
  *reinterpret_cast<uint4*>(&Vs[kr][c0]) = make_uint4(vw[0], vw[1], vw[2], vw[3]);
  *reinterpret_cast<uint4*>(&Vs[kr][c0 + 8]) = make_uint4(vw[4], vw[5], vw[6], vw[7]);
  if (part == 0) {
    ksc_s[kr] = valid ? r.sk : 0.0f;
    vsc_s[kr] = valid ? r.sv : 0.0f;
  }
}

template <bool Int8, bool K5, int W>
__global__ void __launch_bounds__(W * 32)
    prefill_attention_q_kernel(const std::uint16_t* __restrict__ q_rot, float scale, int Hq,
                               int base_pos, int rows, int stride, std::uint16_t* __restrict__ out,
                               const std::int8_t* __restrict__ cache_k8,
                               const std::int8_t* __restrict__ cache_v8,
                               const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                               const std::uint8_t* __restrict__ cache_k4,
                               const std::uint8_t* __restrict__ cache_v4,
                               const int* __restrict__ d_pt) {
  constexpr int D = 128, G = 4;
  constexpr int kPosPerWarp = 16 / G;
  constexpr int kPosPerBlock = W * kPosPerWarp;
  constexpr int KT = kAttnKvTile;
  constexpr int kItems = KT * (D / 16) / (W * 32);  // fill items (position, 16 dims) per thread
  static_assert(kItems * W * 32 == KT * (D / 16), "fill items");
  __shared__ __align__(16) std::uint16_t Ks[KT][kPqP];
  __shared__ __align__(16) std::uint16_t Vs[KT][kPqP];
  __shared__ __align__(8) float ksc_s[KT], vsc_s[KT];

  const int hkv = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int p0 = blockIdx.x * kPosPerBlock;
  const int wp0 = p0 + warp * kPosPerWarp;

  int rl[2], head[2], pabs[2];
  bool ok[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = g + 8 * h;
    rl[h] = wp0 + r / G;
    head[h] = hkv * G + r % G;
    ok[h] = rl[h] < rows;
    pabs[h] = base_pos + rl[h];
  }
  const std::size_t qpitch = static_cast<std::size_t>(Hq) * D;
  std::uint32_t qf[D / 16][4];
#pragma unroll
  for (int ks = 0; ks < D / 16; ++ks) {
    const int k = ks * 16 + 2 * t4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const std::uint16_t* const q = q_rot + rl[h] * qpitch + head[h] * D;
      qf[ks][h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(q + k) : 0u;
      qf[ks][2 + h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(q + k + 8) : 0u;
    }
  }
  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY};
  float l[2] = {0.0f, 0.0f};

  const int last = min(p0 + kPosPerBlock, rows) - 1;
  const int kv_end = base_pos + last + 1;
  const std::int8_t* const k8 = Int8 ? cache_k8 + static_cast<std::size_t>(hkv) * stride * D : nullptr;
  const std::int8_t* const v8 = Int8 ? cache_v8 + static_cast<std::size_t>(hkv) * stride * D : nullptr;
  const std::uint8_t* const k4 =
      Int8 ? nullptr : cache_k4 + static_cast<std::size_t>(hkv) * stride * (K5 ? dev::kK5RowBytes : D / 2);
  const std::uint8_t* const v4 = Int8 ? nullptr : cache_v4 + static_cast<std::size_t>(hkv) * stride * (D / 2);
  const float* const ksc = k_scale + static_cast<std::size_t>(hkv) * stride;
  const float* const vsc = v_scale + static_cast<std::size_t>(hkv) * stride;

  PqRaw raw[kItems];
  bool rv[kItems];
  auto load_tile = [&](int j0) {
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int it = threadIdx.x + i * W * 32;
      const int j = j0 + (it >> 3);
      rv[i] = j < kv_end;
      const int row = rv[i] ? (d_pt != nullptr ? d_pt[j] : j) : 0;
      pq_load<Int8, K5>(raw[i], row, rv[i], it & 7, k8, v8, k4, v4, ksc, vsc);
    }
  };
  load_tile(0);
  const int mj = lane >> 3, mr = lane & 7;
  for (int j0 = 0; j0 < kv_end; j0 += KT) {
    __syncthreads();  // the previous tile's readers are done
#pragma unroll
    for (int i = 0; i < kItems; ++i) {
      const int it = threadIdx.x + i * W * 32;
      pq_store<Int8, K5>(Ks, Vs, ksc_s, vsc_s, it >> 3, it & 7, raw[i], rv[i]);
    }
    __syncthreads();
    if (j0 + KT < kv_end) load_tile(j0 + KT);

    float s[KT / 8][4];
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < D / 16; ++ks)
#pragma unroll
      for (int ni = 0; ni < KT / 8; ++ni) {
        const std::uint16_t* const kr = &Ks[ni * 8 + g][ks * 16 + 2 * t4];
        mma_bf16_16816(s[ni], qf[ks], *reinterpret_cast<const std::uint32_t*>(kr),
                       *reinterpret_cast<const std::uint32_t*>(kr + 8));
      }
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni) {
      const float2 skp = *reinterpret_cast<const float2*>(&ksc_s[ni * 8 + 2 * t4]);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        const int j = j0 + ni * 8 + 2 * t4 + (e & 1);
        const float dot = s[ni][e] * ((e & 1) ? skp.y : skp.x);
        const float v = j <= pabs[h] ? bf16_round(scale * dot) : -INFINITY;
        s[ni][e] = v;
        mx[h] = fmaxf(mx[h], v);
      }
    }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      alpha[h] = expf(m[h] - mx[h]);  // 0 on the first tile (m = -inf)
      l[h] *= alpha[h];
      m[h] = mx[h];
    }
#pragma unroll
    for (int i = 0; i < D / 8; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }
    std::uint32_t pf[KT / 16][4];
#pragma unroll
    for (int ni = 0; ni < KT / 8; ++ni) {
      const float e0 = expf(s[ni][0] - m[0]), e1 = expf(s[ni][1] - m[0]);
      const float e2 = expf(s[ni][2] - m[1]), e3 = expf(s[ni][3] - m[1]);
      l[0] += e0 + e1;
      l[1] += e2 + e3;
      const float2 svp = *reinterpret_cast<const float2*>(&vsc_s[ni * 8 + 2 * t4]);
      pf[ni / 2][(ni & 1) * 2] = pack_bf16x2(e0 * svp.x, e1 * svp.y);
      pf[ni / 2][(ni & 1) * 2 + 1] = pack_bf16x2(e2 * svp.x, e3 * svp.y);
    }
#pragma unroll
    for (int kk = 0; kk < KT / 16; ++kk)
#pragma unroll
      for (int nd = 0; nd < D / 8; nd += 2) {
        std::uint32_t b0, b1, b2, b3;
        dev::ldsm_x4_trans(b0, b1, b2, b3, &Vs[kk * 16 + 8 * (mj & 1) + mr][8 * (nd + (mj >> 1))]);
        mma_bf16_16816(o[nd], pf[kk], b0, b1);
        mma_bf16_16816(o[nd + 1], pf[kk], b2, b3);
      }
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!ok[h]) continue;
    const float inv = 1.0f / l[h];
    std::uint16_t* const dst = out + rl[h] * qpitch + head[h] * D + 2 * t4;
#pragma unroll
    for (int nd = 0; nd < D / 8; ++nd)
      *reinterpret_cast<std::uint32_t*>(dst + nd * 8) =
          pack_bf16x2(o[nd][2 * h] * inv, o[nd][2 * h + 1] * inv);
  }
}

// act[r][i] = bf16(g * sigmoid(g) * u), g = gu[r][i], u = gu[r][I + i]
// (both already dequant-scaled by the GEMM epilogue) - the decode
// silu_gate_up epilogue arithmetic.
__global__ void silu_mul_rows_kernel(const float* gu, int I, int rows,
                                     std::uint16_t* act) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::size_t>(rows) * I) return;
  const std::size_t r = idx / I, i = idx % I;
  const float gs = gu[r * 2 * I + i];
  const float us = gu[r * 2 * I + I + i];
  const float sig = 1.0f / (1.0f + std::exp(-gs));
  act[idx] = f32_to_bf16(gs * sig * us);
}

#define G4_PF_CHECK()                                                        \
  do {                                                                       \
    const cudaError_t e_ = cudaGetLastError();                               \
    if (e_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "granite4: CUDA error at %s:%d: %s\n", __FILE__,  \
                   __LINE__, cudaGetErrorString(e_));                        \
      return -1;                                                             \
    }                                                                        \
  } while (0)

}  // namespace

namespace {

template <GemmOut Out>
void launch_gemm(int bm, dim3 grid, int splits, int kt_per, const std::uint16_t* A,
                 int lda, const std::uint8_t* W, int M, int N, int K, float* ws,
                 const GemmEpilogue& ep) {
  float* const partial = splits > 1 ? ws : nullptr;
  // Dynamic shared memory beyond the 48 KiB default needs a one-time opt-in.
  static const bool opted = [] {
    return cudaFuncSetAttribute(gemm_fp8w_kernel<Out, 16>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes<16>()) == cudaSuccess &&
           cudaFuncSetAttribute(gemm_fp8w_kernel<Out, 64>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_bytes<64>()) == cudaSuccess;
  }();
  (void)opted;
  if (bm == 16)
    gemm_fp8w_kernel<Out, 16><<<grid, kThreads, smem_bytes<16>()>>>(A, lda, W, M, N, K,
                                                                    kt_per, partial, ep);
  else
    gemm_fp8w_kernel<Out, 64><<<grid, kThreads, smem_bytes<64>()>>>(A, lda, W, M, N, K,
                                                                    kt_per, partial, ep);
  if (splits > 1) {
    const std::size_t pairs = static_cast<std::size_t>(M) * N / 2;
    splitk_reduce_kernel<Out><<<static_cast<unsigned>((pairs + 255) / 256), 256>>>(
        ws, splits, M, N, ep);
  }
}

// Enough blocks to keep every SM streaming weights: small-M GEMMs (short
// prompts) have few output tiles, so their K range is split instead.
constexpr int kTargetBlocks = 6 * 66;
constexpr int kMaxSplits = 8;
constexpr int kMinKTilesPerSplit = 16;
constexpr int kSmallM = 48;  // up to this many rows the 16-row tile wins

}  // namespace

std::size_t prefill_gemm_workspace_bytes(int M, int N) {
  return static_cast<std::size_t>(kMaxSplits) * M * N * sizeof(float);
}

int prefill_gemm(GemmOut out, const std::uint16_t* A, int lda, const std::uint8_t* W,
                 int M, int N, int K, const GemmEpilogue& ep, float* ws,
                 std::size_t ws_bytes) {
  if (A == nullptr || W == nullptr || M <= 0 || N <= 0 || K <= 0 || (K % kBK) != 0 ||
      (lda % 8) != 0 || lda < K || (N & 1) != 0 || (ep.ldc & 1) != 0 || ep.ldc < N ||
      (reinterpret_cast<std::uintptr_t>(A) & 15u) != 0 ||
      (reinterpret_cast<std::uintptr_t>(W) & 15u) != 0 || (K & 15) != 0 ||
      (out == GemmOut::kF32 ? ep.out_f32 == nullptr : ep.out_bf16 == nullptr) ||
      (out == GemmOut::kResidual && ep.residual == nullptr))
    return -1;
  const int bm = M <= kSmallM ? 16 : 64;
  const int gm = (M + bm - 1) / bm, gn = (N + kBN - 1) / kBN;
  const int ktiles = K / kBK;
  int splits = 1;
  if (ws != nullptr && gm * gn < kTargetBlocks) {
    splits = std::min({kMaxSplits, (kTargetBlocks + gm * gn - 1) / (gm * gn),
                       std::max(1, ktiles / kMinKTilesPerSplit)});
    while (splits > 1 &&
           static_cast<std::size_t>(splits) * M * N * sizeof(float) > ws_bytes)
      --splits;
  }
  const int kt_per = (ktiles + splits - 1) / splits;
  splits = (ktiles + kt_per - 1) / kt_per;
  const dim3 grid(static_cast<unsigned>(gm), static_cast<unsigned>(gn),
                  static_cast<unsigned>(splits));
  switch (out) {
    case GemmOut::kBf16:
      launch_gemm<GemmOut::kBf16>(bm, grid, splits, kt_per, A, lda, W, M, N, K, ws, ep);
      break;
    case GemmOut::kF32:
      launch_gemm<GemmOut::kF32>(bm, grid, splits, kt_per, A, lda, W, M, N, K, ws, ep);
      break;
    case GemmOut::kResidual:
      launch_gemm<GemmOut::kResidual>(bm, grid, splits, kt_per, A, lda, W, M, N, K, ws, ep);
      break;
  }
  G4_PF_CHECK();
  return 0;
}

int prefill_embed_rows(const std::int32_t* d_tokens, int start, int rows,
                       const std::uint16_t* emb, float mult, std::uint16_t* out, int H) {
  if (rows <= 0 || rows > 65535 || H <= 0) return -1;
  embed_rows_kernel<<<dim3((H + 255) / 256, rows), 256>>>(d_tokens, start, emb, mult, out, H);
  G4_PF_CHECK();
  return 0;
}

int prefill_rope_split_append(const std::uint16_t* qkv, const float* cos_tbl,
                              const float* sin_tbl, int base_pos, int rows, int Hq,
                              int Hk, int d, int stride, std::uint16_t* q_rot,
                              std::uint16_t* cache_k, std::uint16_t* cache_v, const int* d_pt) {
  if (rows <= 0 || rows > 65535 || (d & 1) != 0 || base_pos < 0 ||
      stride < base_pos + rows)
    return -1;
  const int n = (Hq + 2 * Hk) * d;
  rope_split_append_kernel<<<dim3((n + 255) / 256, rows), 256>>>(
      qkv, cos_tbl, sin_tbl, base_pos, Hq, Hk, d, stride, q_rot, cache_k, cache_v,
      d_pt);
  G4_PF_CHECK();
  return 0;
}
// KV8: append the prefill chunk's K/V to the int8 persistent cache (rotating
// K, quantizing K and V per row). Q is rotated to bf16 as in the bf16 path.
// Requires d == 128 (the prefill attention head dim).
int prefill_rope_split_append_int8(const std::uint16_t* qkv, const float* cos_tbl,
                                   const float* sin_tbl, int base_pos, int rows,
                                   int Hq, int Hk, int d, int stride,
                                   std::uint16_t* q_rot, std::int8_t* cache_k,
                                   std::int8_t* cache_v, float* cache_k_scale,
                                   float* cache_v_scale, const int* d_pt) {
  if (rows <= 0 || rows > 65535 || base_pos < 0 || stride < base_pos + rows ||
      d != kPrefillInt8D || Hq <= 0 || Hk <= 0 || q_rot == nullptr ||
      cache_k == nullptr || cache_v == nullptr || cache_k_scale == nullptr ||
      cache_v_scale == nullptr)
    return -1;
  const int heads = Hq + 2 * Hk;
  rope_split_append_int8_kernel<<<dim3(heads, rows), kPrefillInt8D>>>(
      qkv, cos_tbl, sin_tbl, base_pos, Hq, Hk, d, stride, q_rot, cache_k, cache_v,
      cache_k_scale, cache_v_scale,
      d_pt);
  G4_PF_CHECK();
  return 0;
}

// KV4: append the prefill chunk's K/V to the int4 persistent cache (rotating
// K, quantizing K and V per row as two-per-byte int4 codes). Q is rotated to
// bf16 as in the bf16 path. Requires d == 128 (the prefill attention head
// dim).
int prefill_rope_split_append_int4(const std::uint16_t* qkv, const float* cos_tbl,
                                   const float* sin_tbl, int base_pos, int rows,
                                   int Hq, int Hk, int d, int stride,
                                   std::uint16_t* q_rot, std::uint8_t* cache_k,
                                   std::uint8_t* cache_v, float* cache_k_scale,
                                   float* cache_v_scale, const int* d_pt) {
  if (rows <= 0 || rows > 65535 || base_pos < 0 || stride < base_pos + rows ||
      d != kPrefillInt4D * 2 || (d % 2) != 0 || Hq <= 0 || Hk <= 0 ||
      q_rot == nullptr || cache_k == nullptr || cache_v == nullptr ||
      cache_k_scale == nullptr || cache_v_scale == nullptr)
    return -1;
  const int heads = Hq + 2 * Hk;
  rope_split_append_int4_kernel<<<dim3(heads, rows), kPrefillInt4D>>>(
      qkv, cos_tbl, sin_tbl, base_pos, Hq, Hk, d, stride, q_rot, cache_k, cache_v,
      cache_k_scale, cache_v_scale,
      d_pt);
  G4_PF_CHECK();
  return 0;
}

// K5:V4: append the prefill chunk's K/V (K 5-bit in 80-byte rows, V int4). Same contract as
// prefill_rope_split_append_int4.
int prefill_rope_split_append_k5v4(const std::uint16_t* qkv, const float* cos_tbl,
                                   const float* sin_tbl, int base_pos, int rows,
                                   int Hq, int Hk, int d, int stride,
                                   std::uint16_t* q_rot, std::uint8_t* cache_k,
                                   std::uint8_t* cache_v, float* cache_k_scale,
                                   float* cache_v_scale, const int* d_pt) {
  if (rows <= 0 || rows > 65535 || base_pos < 0 || stride < base_pos + rows ||
      d != kPrefillInt4D * 2 || Hq <= 0 || Hk <= 0 ||
      q_rot == nullptr || cache_k == nullptr || cache_v == nullptr ||
      cache_k_scale == nullptr || cache_v_scale == nullptr)
    return -1;
  const int heads = Hq + 2 * Hk;
  rope_split_append_k5v4_kernel<<<dim3(heads, rows), kPrefillInt4D>>>(
      qkv, cos_tbl, sin_tbl, base_pos, Hq, Hk, d, stride, q_rot, cache_k, cache_v,
      cache_k_scale, cache_v_scale, d_pt);
  G4_PF_CHECK();
  return 0;
}
int prefill_attention(const std::uint16_t* q_rot, const std::uint16_t* cache_k,
                      const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
                      int base_pos, int rows, int stride, std::uint16_t* out,
                      const std::int8_t* cache_k8, const std::int8_t* cache_v8,
                      const float* k_scale, const float* v_scale, bool int8,
                      const std::uint8_t* cache_k4, const std::uint8_t* cache_v4,
                      bool int4, const int* d_pt, bool k5) {
  if (q_rot == nullptr || out == nullptr || !std::isfinite(scale) || Hk <= 0 ||
      Hq % Hk != 0 || rows <= 0 || base_pos < 0 || stride < base_pos + rows ||
      d != 128 || Hq / Hk != 4 ||
      (int8 && int4) ||
      (k5 && !int4) ||
      (int8 ? (cache_k8 == nullptr || cache_v8 == nullptr || k_scale == nullptr ||
               v_scale == nullptr)
            : int4 ? (cache_k4 == nullptr || cache_v4 == nullptr ||
                      k_scale == nullptr || v_scale == nullptr)
                  : (cache_k == nullptr || cache_v == nullptr ||
                     (reinterpret_cast<std::uintptr_t>(cache_k) & 15u) != 0 ||
                     (reinterpret_cast<std::uintptr_t>(cache_v) & 15u) != 0)))
    return -1;
  // Warps per block (each owns 4 positions x 4 heads; all share the K/V
  // tiles). Quantized caches pay a dequant per tile element, so 8 warps
  // amortize it over twice the queries (KV8 30k prefill: -7% vs 4); the bf16
  // fill is a plain copy and 4 warps keep more blocks resident (+3% vs 8).
  // G4_PREFILL_ATTN_WARPS=4|8 overrides (A/B).
  static const int forced = [] {
    const char* e = std::getenv("G4_PREFILL_ATTN_WARPS");
    return e == nullptr ? 0 : (e[0] == '4' ? 4 : (e[0] == '8' ? 8 : 0));
  }();
  const int warps = forced != 0 ? forced : ((int8 || int4) ? 8 : 4);
  static const bool codes = [] {
    const char* e = std::getenv("G4_PREFILL_ATTN_CODES");
    return e == nullptr || e[0] != '0';
  }();
  auto launch = [&](auto wtag) {
    constexpr int Wn = decltype(wtag)::value;
    constexpr int kPosPerBlock = Wn * (16 / 4);
    const dim3 grid(static_cast<unsigned>((rows + kPosPerBlock - 1) / kPosPerBlock),
                    static_cast<unsigned>(Hk));
    if (codes && int8)
      prefill_attention_q_kernel<true, false, Wn><<<grid, Wn * 32>>>(
          q_rot, scale, Hq, base_pos, rows, stride, out, cache_k8, cache_v8, k_scale, v_scale,
          nullptr, nullptr, d_pt);
    else if (codes && int4 && k5)
      prefill_attention_q_kernel<false, true, Wn><<<grid, Wn * 32>>>(
          q_rot, scale, Hq, base_pos, rows, stride, out, nullptr, nullptr, k_scale, v_scale,
          cache_k4, cache_v4, d_pt);
    else if (codes && int4)
      prefill_attention_q_kernel<false, false, Wn><<<grid, Wn * 32>>>(
          q_rot, scale, Hq, base_pos, rows, stride, out, nullptr, nullptr, k_scale, v_scale,
          cache_k4, cache_v4, d_pt);
    else if (int8)
      prefill_attention_kernel<128, 4, true, false, Wn><<<grid, Wn * 32>>>(
          q_rot, cache_k, cache_v, scale, Hq, base_pos, rows, stride, out, cache_k8,
          cache_v8, k_scale, v_scale, nullptr, nullptr, d_pt);
    else if (int4 && k5)
      prefill_attention_kernel<128, 4, false, true, Wn, true><<<grid, Wn * 32>>>(
          q_rot, cache_k, cache_v, scale, Hq, base_pos, rows, stride, out, nullptr,
          nullptr, k_scale, v_scale, cache_k4, cache_v4, d_pt);
    else if (int4)
      prefill_attention_kernel<128, 4, false, true, Wn><<<grid, Wn * 32>>>(
          q_rot, cache_k, cache_v, scale, Hq, base_pos, rows, stride, out, nullptr,
          nullptr, k_scale, v_scale, cache_k4, cache_v4, d_pt);
    else
      prefill_attention_kernel<128, 4, false, false, Wn><<<grid, Wn * 32>>>(
          q_rot, cache_k, cache_v, scale, Hq, base_pos, rows, stride, out, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr, d_pt);
  };
  if (warps == 4)
    launch(std::integral_constant<int, 4>{});
  else
    launch(std::integral_constant<int, 8>{});
  G4_PF_CHECK();
  return 0;
}

int prefill_silu_mul(const float* gu, int I, int rows, std::uint16_t* act) {
  if (rows <= 0 || I <= 0) return -1;
  const std::size_t n = static_cast<std::size_t>(rows) * I;
  silu_mul_rows_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(gu, I, rows, act);
  G4_PF_CHECK();
  return 0;
}

}  // namespace granite4

// C ABI for the device tests.
extern "C" int granite4_prefill_gemm_bf16(const std::uint16_t* d_a, const std::uint8_t* d_w,
                                          float scale, int M, int N, int K,
                                          std::uint16_t* d_c) {
  granite4::GemmEpilogue ep;
  ep.scale0 = ep.scale1 = ep.scale2 = scale;
  ep.seg1 = ep.seg2 = N;
  ep.out_bf16 = d_c;
  ep.ldc = N;
  return granite4::prefill_gemm(granite4::GemmOut::kBf16, d_a, K, d_w, M, N, K, ep);
}
