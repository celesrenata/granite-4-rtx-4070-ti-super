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
                                         std::uint16_t* cache_v) {
  const int r = blockIdx.y;
  const int nq = Hq * d, nk = Hk * d;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nq + 2 * nk) return;
  const int pos = base_pos + r;
  const std::uint16_t* const row = qkv + static_cast<std::size_t>(r) * (nq + 2 * nk);
  if (idx >= nq + nk) {  // v: plain copy into the cache
    const int local = idx - nq - nk;
    const int hkv = local / d, i = local % d;
    cache_v[(static_cast<std::size_t>(hkv) * stride + pos) * d + i] = row[idx];
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
    cache_k[(static_cast<std::size_t>(h) * stride + pos) * d + col] = f32_to_bf16(y);
  }
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
constexpr int kAttnWarps = 4;
constexpr int kAttnKvTile = 64;

__device__ __forceinline__ float bf16_round(float x) {
  return __uint_as_float(static_cast<std::uint32_t>(f32_to_bf16(x)) << 16);
}

__device__ __forceinline__ std::uint32_t pack_bf16x2(float lo, float hi) {
  return static_cast<std::uint32_t>(f32_to_bf16(lo)) |
         (static_cast<std::uint32_t>(f32_to_bf16(hi)) << 16);
}

template <int D, int G>
__global__ void __launch_bounds__(kAttnWarps * 32)
    prefill_attention_kernel(const std::uint16_t* __restrict__ q_rot,
                             const std::uint16_t* __restrict__ cache_k,
                             const std::uint16_t* __restrict__ cache_v, float scale,
                             int Hq, int base_pos, int rows, int stride,
                             std::uint16_t* __restrict__ out) {
  constexpr int kPosPerWarp = 16 / G;
  constexpr int kPosPerBlock = kAttnWarps * kPosPerWarp;
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
  const std::uint16_t* const kb = cache_k + static_cast<std::size_t>(hkv) * stride * D;
  const std::uint16_t* const vb = cache_v + static_cast<std::size_t>(hkv) * stride * D;
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);

  for (int j0 = 0; j0 < kv_end; j0 += KT) {
    __syncthreads();  // the previous tile's readers are done
    // K tile row-major (coalesced); V tile transposed, consecutive threads on
    // consecutive positions so the scalar transposed stores do not conflict.
    for (int idx = threadIdx.x; idx < KT * D / 8; idx += kAttnWarps * 32) {
      const int kr = idx / (D / 8), kc = (idx % (D / 8)) * 8;
      const int kj = j0 + kr;
      *reinterpret_cast<uint4*>(&Ks[kr][kc]) =
          kj < kv_end ? *reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(kj) * D + kc)
                      : zero;
      const int vr = idx % KT, vc = (idx / KT) * 8;
      const int vj = j0 + vr;
      const uint4 v4 =
          vj < kv_end ? *reinterpret_cast<const uint4*>(vb + static_cast<std::size_t>(vj) * D + vc)
                      : zero;
      const std::uint32_t vw[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        Vt[vc + 2 * e][vr] = static_cast<std::uint16_t>(vw[e] & 0xffffu);
        Vt[vc + 2 * e + 1][vr] = static_cast<std::uint16_t>(vw[e] >> 16);
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
                              std::uint16_t* cache_k, std::uint16_t* cache_v) {
  if (rows <= 0 || rows > 65535 || (d & 1) != 0 || base_pos < 0 ||
      stride < base_pos + rows)
    return -1;
  const int n = (Hq + 2 * Hk) * d;
  rope_split_append_kernel<<<dim3((n + 255) / 256, rows), 256>>>(
      qkv, cos_tbl, sin_tbl, base_pos, Hq, Hk, d, stride, q_rot, cache_k, cache_v);
  G4_PF_CHECK();
  return 0;
}

int prefill_attention(const std::uint16_t* q_rot, const std::uint16_t* cache_k,
                      const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
                      int base_pos, int rows, int stride, std::uint16_t* out) {
  if (q_rot == nullptr || cache_k == nullptr || cache_v == nullptr || out == nullptr ||
      !std::isfinite(scale) || Hk <= 0 || Hq % Hk != 0 || rows <= 0 || base_pos < 0 ||
      stride < base_pos + rows || d != 128 || Hq / Hk != 4 ||
      (reinterpret_cast<std::uintptr_t>(cache_k) & 15u) != 0 ||
      (reinterpret_cast<std::uintptr_t>(cache_v) & 15u) != 0)
    return -1;
  constexpr int kPosPerBlock = kAttnWarps * (16 / 4);
  const dim3 grid(static_cast<unsigned>((rows + kPosPerBlock - 1) / kPosPerBlock),
                  static_cast<unsigned>(Hk));
  prefill_attention_kernel<128, 4><<<grid, kAttnWarps * 32>>>(
      q_rot, cache_k, cache_v, scale, Hq, base_pos, rows, stride, out);
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
