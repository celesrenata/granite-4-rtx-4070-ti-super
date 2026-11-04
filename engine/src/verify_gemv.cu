// Multi-token GEMVs for the speculative verify forward (2..16 rows).
//
// Every kernel here streams each FP8 weight row exactly once while
// accumulating up to kMaxVerifyTokens activation rows. Per token, the lane
// arithmetic, the k-chunk assignment (blockDim = 128) and the block
// reduction (warp shuffle tree, then warp totals in warp order by thread 0)
// reproduce the single-token decode kernels exactly:
//
//   granite4_verify_gemv        row_gemm_dequant_kernel       (lm_head)
//   granite4_verify_gemv_resid  row_gemm_dequant_resid_kernel (O, down)
//   granite4_verify_qkv         row_gemm_dequant_qkv_kernel   (Q|K|V)
//   granite4_verify_silu        silu_gate_up_kernel           (gate/up)
//
// so each token's output is bit-identical to the decode step at that
// position, and the greedy verify gate accepts exactly what plain greedy
// decoding would emit. Only the activation loads are wider (one 16-byte load
// per token per 16 weights instead of eight 4-byte loads); the values and the
// order they are combined in are unchanged.
//
// The 16-byte paths need K % 16 == 0 and 16-byte aligned weight rows and
// activation rows (the model's shapes always satisfy this); other shapes
// return -1 and the caller falls back to per-position forwards.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

#include "device_math.cuh"
#include "model_config.h"

namespace {
constexpr int kMaxVerifyTokens = granite4::kMaxVerifyRows;
static_assert(kMaxVerifyTokens == 16, "dispatch() below instantiates 1..16");
constexpr int kThreads = 128;  // == kRowGemvThreads: part of the bit-identity contract

using granite4::dev::bf16_hi;
using granite4::dev::bf16_lo;
using granite4::dev::bf16_to_f32;
using granite4::dev::f32_to_bf16;
using granite4::dev::fp8x4_to_f32;

// Lane partials of one weight row for T tokens (gemv_row_dequant's 16-byte
// path: lane t owns 16-element chunks t, t + 128, ...).
template <int T>
__device__ __forceinline__ void row_partials(const std::uint16_t* __restrict__ a, int lda,
                                             const std::uint8_t* __restrict__ w, int k,
                                             float (&acc)[T]) {
#pragma unroll
  for (int t = 0; t < T; ++t) acc[t] = 0.0f;
  const int k16 = k >> 4;
  const uint4* const w16 = reinterpret_cast<const uint4*>(w);
  for (int g = threadIdx.x; g < k16; g += kThreads) {
    const uint4 wv = w16[g];
    const std::uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
    float b[16];
#pragma unroll
    for (int q = 0; q < 4; ++q) fp8x4_to_f32(ws[q], b[4 * q], b[4 * q + 1], b[4 * q + 2], b[4 * q + 3]);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const uint4* const at = reinterpret_cast<const uint4*>(a + static_cast<std::size_t>(t) * lda);
      const uint4 lo = at[2 * g], hi = at[2 * g + 1];
      const std::uint32_t y[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc[t] += b[4 * q] * bf16_lo(y[2 * q]) + b[4 * q + 1] * bf16_hi(y[2 * q]) +
                  b[4 * q + 2] * bf16_lo(y[2 * q + 1]) + b[4 * q + 3] * bf16_hi(y[2 * q + 1]);
      }
    }
  }
}

// Block totals per token, in the decode kernels' order; valid in thread 0.
template <int T>
__device__ __forceinline__ void block_reduce(float (&acc)[T], float (*red)[kThreads / 32]) {
  const int lane = threadIdx.x;
#pragma unroll
  for (int t = 0; t < T; ++t) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc[t] += __shfl_down_sync(0xffffffffu, acc[t], off);
    if ((lane & 31) == 0) red[t][lane >> 5] = acc[t];
  }
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float s = red[t][0];
      for (int j = 1; j < kThreads / 32; ++j) s += red[t][j];
      acc[t] = s;
    }
  }
}

// out[t][row] = bf16(scale * a[t] . B[row])   (ldo = output row pitch)
template <int T>
__global__ void __launch_bounds__(kThreads)
    gemv_kernel(const std::uint16_t* __restrict__ a, const std::uint8_t* __restrict__ b,
                float scale, int n, int k, std::uint16_t* __restrict__ c) {
  const int row = blockIdx.x;
  if (row >= n) return;
  __shared__ float red[T][kThreads / 32];
  float acc[T];
  row_partials<T>(a, k, b + static_cast<std::size_t>(row) * k, k, acc);
  block_reduce<T>(acc, red);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) c[static_cast<std::size_t>(t) * n + row] = f32_to_bf16(scale * acc[t]);
  }
}

// out[t][row] = bf16(residual[t][row] + rs * scale * a[t] . B[row]); out may
// alias residual (each element is read, then written, by the same thread).
template <int T>
__global__ void __launch_bounds__(kThreads)
    gemv_resid_kernel(const std::uint16_t* __restrict__ a, const std::uint8_t* __restrict__ b,
                      float scale, const std::uint16_t* residual, float rs, int n, int k,
                      std::uint16_t* out) {
  const int row = blockIdx.x;
  if (row >= n) return;
  __shared__ float red[T][kThreads / 32];
  float acc[T];
  row_partials<T>(a, k, b + static_cast<std::size_t>(row) * k, k, acc);
  block_reduce<T>(acc, red);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const std::size_t i = static_cast<std::size_t>(t) * n + row;
      const float y = scale * acc[t];
      out[i] = f32_to_bf16(bf16_to_f32(residual[i]) + rs * y);
    }
  }
}

// Q|K|V in one grid: block b < nq -> q row b, then k rows, then v rows.
template <int T>
__global__ void __launch_bounds__(kThreads)
    qkv_kernel(const std::uint16_t* __restrict__ a, const std::uint8_t* __restrict__ wq,
               const std::uint8_t* __restrict__ wk, const std::uint8_t* __restrict__ wv,
               float sq, float sk, float sv, int nq, int nk, int nv, int k,
               std::uint16_t* __restrict__ oq, std::uint16_t* __restrict__ ok,
               std::uint16_t* __restrict__ ov) {
  int row = blockIdx.x;
  const std::uint8_t* w = wq;
  float scale = sq;
  std::uint16_t* out = oq;
  int n = nq;
  if (row >= nq + nk) {
    row -= nq + nk;
    if (row >= nv) return;
    w = wv, scale = sv, out = ov, n = nv;
  } else if (row >= nq) {
    row -= nq;
    w = wk, scale = sk, out = ok, n = nk;
  }
  __shared__ float red[T][kThreads / 32];
  float acc[T];
  row_partials<T>(a, k, w + static_cast<std::size_t>(row) * k, k, acc);
  block_reduce<T>(acc, red);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) out[static_cast<std::size_t>(t) * n + row] = f32_to_bf16(scale * acc[t]);
  }
}

// SwiGLU: out[t][r] = silu(sg * Wg[r] . x[t]) * (su * Wu[r] . x[t]), with
// silu_gate_up_kernel's 4-element chunking (lane t owns k4 = t, t + 128, ...).
template <int T>
__global__ void __launch_bounds__(kThreads)
    silu_kernel(const std::uint8_t* __restrict__ wg, const std::uint8_t* __restrict__ wu,
                const std::uint16_t* __restrict__ x, float sg, float su, int n, int k,
                std::uint16_t* __restrict__ out) {
  const int r = blockIdx.x;
  if (r >= n) return;
  const std::uint32_t* const rg4 = reinterpret_cast<const std::uint32_t*>(wg + static_cast<std::size_t>(r) * k);
  const std::uint32_t* const ru4 = reinterpret_cast<const std::uint32_t*>(wu + static_cast<std::size_t>(r) * k);
  float g[T], u[T];
#pragma unroll
  for (int t = 0; t < T; ++t) g[t] = u[t] = 0.0f;
  const int k4n = k >> 2;
  for (int k4 = threadIdx.x; k4 < k4n; k4 += kThreads) {
    float g0, g1, g2, g3, u0, u1, u2, u3;
    fp8x4_to_f32(rg4[k4], g0, g1, g2, g3);
    fp8x4_to_f32(ru4[k4], u0, u1, u2, u3);
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const uint2 xw = reinterpret_cast<const uint2*>(x + static_cast<std::size_t>(t) * k)[k4];
      const float x0 = bf16_lo(xw.x), x1 = bf16_hi(xw.x);
      const float x2 = bf16_lo(xw.y), x3 = bf16_hi(xw.y);
      g[t] += g0 * x0 + g1 * x1 + g2 * x2 + g3 * x3;
      u[t] += u0 * x0 + u1 * x1 + u2 * x2 + u3 * x3;
    }
  }
  __shared__ float rg[T][kThreads / 32], ru[T][kThreads / 32];
  const int lane = threadIdx.x;
#pragma unroll
  for (int t = 0; t < T; ++t) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      g[t] += __shfl_down_sync(0xffffffffu, g[t], off);
      u[t] += __shfl_down_sync(0xffffffffu, u[t], off);
    }
    if ((lane & 31) == 0) {
      rg[t][lane >> 5] = g[t];
      ru[t][lane >> 5] = u[t];
    }
  }
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      float s = rg[t][0], v = ru[t][0];
      for (int j = 1; j < kThreads / 32; ++j) {
        s += rg[t][j];
        v += ru[t][j];
      }
      const float sig = 1.0f / (1.0f + std::exp(-(sg * s)));
      out[static_cast<std::size_t>(t) * n + r] = f32_to_bf16(sg * s * sig * (su * v));
    }
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; }

// Calls f(std::integral_constant<int, T>) for T = tokens in [1, 16].
template <class F>
bool dispatch(int tokens, F&& f) {
  switch (tokens) {
#define G4_T(N) \
  case N:       \
    f.template operator()<N>(); \
    return true;
    G4_T(1) G4_T(2) G4_T(3) G4_T(4) G4_T(5) G4_T(6) G4_T(7) G4_T(8)
    G4_T(9) G4_T(10) G4_T(11) G4_T(12) G4_T(13) G4_T(14) G4_T(15) G4_T(16)
#undef G4_T
    default:
      return false;
  }
}

int launched() { return cudaGetLastError() == cudaSuccess ? 0 : -1; }
}  // namespace

// A is [tokens][k] bf16, B is [rows][k] fp8, C is [tokens][rows] bf16.
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                    float scale, int tokens, int rows, int k,
                                    std::uint16_t* d_c) {
  if (rows < 1 || k < 1 || (k & 15) != 0 || !aligned16(d_a) || !aligned16(d_b)) return -1;
  if (!dispatch(tokens, [&]<int T>() { gemv_kernel<T><<<rows, kThreads>>>(d_a, d_b, scale, rows, k, d_c); }))
    return -1;
  return launched();
}

// O / down projection with the residual add (out may alias residual).
extern "C" int granite4_verify_gemv_resid(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                          float scale, const std::uint16_t* d_residual,
                                          float residual_scale, int tokens, int rows, int k,
                                          std::uint16_t* d_out) {
  if (rows < 1 || k < 1 || (k & 15) != 0 || !aligned16(d_a) || !aligned16(d_b)) return -1;
  if (!dispatch(tokens, [&]<int T>() {
        gemv_resid_kernel<T><<<rows, kThreads>>>(d_a, d_b, scale, d_residual, residual_scale,
                                                 rows, k, d_out);
      }))
    return -1;
  return launched();
}

extern "C" int granite4_verify_qkv(const std::uint16_t* d_a, const std::uint8_t* d_wq,
                                   const std::uint8_t* d_wk, const std::uint8_t* d_wv,
                                   float sq, float sk, float sv, int tokens, int nq, int nk,
                                   int nv, int k, std::uint16_t* d_q, std::uint16_t* d_k,
                                   std::uint16_t* d_v) {
  if (nq < 1 || nk < 1 || nv < 1 || k < 1 || (k & 15) != 0 || !aligned16(d_a) ||
      !aligned16(d_wq) || !aligned16(d_wk) || !aligned16(d_wv))
    return -1;
  if (!dispatch(tokens, [&]<int T>() {
        qkv_kernel<T><<<nq + nk + nv, kThreads>>>(d_a, d_wq, d_wk, d_wv, sq, sk, sv, nq, nk, nv,
                                                  k, d_q, d_k, d_v);
      }))
    return -1;
  return launched();
}

extern "C" int granite4_verify_silu(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                    const std::uint16_t* d_x, float sg, float su, int n,
                                    int k, int tokens, std::uint16_t* d_out) {
  if (n < 1 || k < 1 || (k & 3) != 0 || (reinterpret_cast<std::uintptr_t>(d_x) & 7u) != 0 ||
      (reinterpret_cast<std::uintptr_t>(d_wg) & 3u) != 0 ||
      (reinterpret_cast<std::uintptr_t>(d_wu) & 3u) != 0)
    return -1;
  if (!dispatch(tokens, [&]<int T>() {
        silu_kernel<T><<<n, kThreads>>>(d_wg, d_wu, d_x, sg, su, n, k, d_out);
      }))
    return -1;
  return launched();
}
