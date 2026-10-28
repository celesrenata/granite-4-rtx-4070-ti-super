// Experimental fused-k verification GEMV for speculative decoding.
//
// A is [tokens,K], B is [rows,K] FP8 row-major, C is [tokens,rows]. One
// block owns an output row and streams that row of B exactly once while
// accumulating up to kMaxVerifyTokens activation rows. The per-lane
// arithmetic and the block reduction are reproduced from the production
// row_gemm_dequant_kernel (16-byte chunk grouping, warp shuffle, warp
// totals in warp order), so each token's output is bit-identical to the
// single-token path and the greedy token gate stays exact.
//
// Deliberately standalone until the k-token forward/attention path is
// complete.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kMaxVerifyTokens = 4;

__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const std::uint32_t s = static_cast<std::uint32_t>(x & 0x80u) << 24;
  const std::uint32_t y = x & 0x7fu;
  const std::uint32_t e = y >> 3;
  const std::uint32_t m = y & 7u;
  std::uint32_t bits = ((120u + e) << 23) | (m << 20);
  const std::uint32_t top = (m == 7u) ? 0u : (0x43800000u | (m << 20));
  bits = (e == 15u) ? top : bits;
  bits = (e == 0u) ? __float_as_uint(static_cast<float>(m) * 0.001953125f) : bits;
  return ((e == 15u && m == 7u)) ? 0.0f : __uint_as_float(bits | s);
}
__device__ __forceinline__ float bf16_to_f32(std::uint16_t x) {
  return __uint_as_float(static_cast<std::uint32_t>(x) << 16);
}
__device__ __forceinline__ std::uint16_t f32_to_bf16(float x) {
  const std::uint32_t u = __float_as_uint(x);
  return static_cast<std::uint16_t>((u + 0x7ffu + ((u >> 16) & 1u)) >> 16);
}

// Per-token dot products for one output row, from one weight stream.
__global__ void verify_gemv_kernel(const std::uint16_t* __restrict__ a,
                                   const std::uint8_t* __restrict__ b,
                                   float scale, int tokens, int rows, int k,
                                   std::uint16_t* __restrict__ c) {
  const int row = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x;
  const int nt = blockDim.x;
  if (row >= rows) return;
  const std::uint8_t* const w = b + static_cast<std::size_t>(row) * k;
  float acc[kMaxVerifyTokens] = {0.f, 0.f, 0.f, 0.f};

  if ((k & 15) == 0 && (reinterpret_cast<std::uintptr_t>(w) & 15u) == 0 &&
      (reinterpret_cast<std::uintptr_t>(a) & 3u) == 0) {
    // Same 16-byte vector path and subgroup grouping as gemv_row_dequant.
    const int k16 = k >> 4;
    const uint4* const w16 = reinterpret_cast<const uint4*>(w);
    for (int g = lane; g < k16; g += nt) {
      const uint4 wv = w16[g];
      const std::uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {  // ws[q] = 4 fp8 elements
        const int e = 16 * g + 4 * q;
        const std::uint32_t wq = ws[q];
        const float b0 = fp8_to_f32(static_cast<std::uint8_t>(wq & 0xffu));
        const float b1 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 8) & 0xffu));
        const float b2 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 16) & 0xffu));
        const float b3 = fp8_to_f32(static_cast<std::uint8_t>((wq >> 24) & 0xffu));
#pragma unroll
        for (int token = 0; token < kMaxVerifyTokens; ++token) {
          if (token >= tokens) break;
          const std::uint32_t* const a4 =
              reinterpret_cast<const std::uint32_t*>(a + static_cast<std::size_t>(token) * k);
          const std::uint32_t y0 = a4[e / 2];      // elements e, e+1
          const std::uint32_t y1 = a4[e / 2 + 1];  // elements e+2, e+3
          const float fa0 = bf16_to_f32(static_cast<std::uint16_t>(y0 & 0xffffu));
          const float fa1 = bf16_to_f32(static_cast<std::uint16_t>((y0 >> 16) & 0xffffu));
          const float fa2 = bf16_to_f32(static_cast<std::uint16_t>(y1 & 0xffffu));
          const float fa3 = bf16_to_f32(static_cast<std::uint16_t>((y1 >> 16) & 0xffffu));
          acc[token] += b0 * fa0 + b1 * fa1 + b2 * fa2 + b3 * fa3;
        }
      }
    }
  } else {
    for (int col = lane; col < k; col += nt) {
      const float ww = fp8_to_f32(w[col]);
#pragma unroll
      for (int token = 0; token < kMaxVerifyTokens; ++token) {
        if (token >= tokens) break;
        acc[token] += ww * bf16_to_f32(a[static_cast<std::size_t>(token) * k + col]);
      }
    }
  }

  // Same warp tree, then warp totals in warp order by thread 0.
  __shared__ float red[kMaxVerifyTokens][32];
#pragma unroll
  for (int token = 0; token < kMaxVerifyTokens; ++token) {
    for (int off = 16; off > 0; off >>= 1)
      acc[token] += __shfl_down_sync(0xffffffffu, acc[token], off);
    if ((lane & 31) == 0) red[token][lane >> 5] = acc[token];
  }
  __syncthreads();
  if (lane == 0) {
    const int warps = nt >> 5;
#pragma unroll
    for (int token = 0; token < kMaxVerifyTokens; ++token) {
      if (token >= tokens) break;
      float sum = red[token][0];
      for (int warp = 1; warp < warps; ++warp) sum += red[token][warp];
      c[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(scale * sum);
    }
  }
}
}  // namespace

// Experimental API. `tokens` must be in [1, 4]. Returns -1 for unsupported
// widths so callers cannot silently lose the one-weight-stream guarantee.
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                     float scale, int tokens, int rows, int k,
                                     std::uint16_t* d_c) {
  if (tokens < 1 || tokens > kMaxVerifyTokens || rows < 1 || k < 1) return -1;
  verify_gemv_kernel<<<rows, 128>>>(d_a, d_b, scale, tokens, rows, k, d_c);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int granite4_verify_qkv(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int tokens, int nq,
    int nk, int k, std::uint16_t* d_q, std::uint16_t* d_k, std::uint16_t* d_v) {
  if (granite4_verify_gemv(d_a, d_wq, sq, tokens, nq, k, d_q) != 0) return -1;
  if (granite4_verify_gemv(d_a, d_wk, sk, tokens, nk, k, d_k) != 0) return -1;
  if (granite4_verify_gemv(d_a, d_wv, sv, tokens, nk, k, d_v) != 0) return -1;
  return 0;
}
