// Fused-k verification GEMV for speculative decoding.
//
// A is [tokens,K], B is [rows,K] FP8 row-major, C is [tokens,rows]. One
// block owns an output row and streams that row of B exactly once while
// accumulating up to kMaxVerifyTokens activation rows. The per-lane
// arithmetic and the block reduction are reproduced from the production
// row_gemm_dequant_kernel (16-byte chunk grouping, warp shuffle, warp
// totals in warp order), so each token's output is bit-identical to the
// single-token path and the greedy token gate stays exact.
//
// Production: forward_range_batched_impl runs the lm_head through it.
#include <cuda_runtime.h>
#include <cstdint>

#include "device_math.cuh"

namespace {
constexpr int kMaxVerifyTokens = 4;

using granite4::dev::bf16_hi;
using granite4::dev::bf16_lo;
using granite4::dev::bf16_to_f32;
using granite4::dev::f32_to_bf16;
using granite4::dev::fp8_to_f32;
using granite4::dev::fp8x4_to_f32;

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
        float b0, b1, b2, b3;
        fp8x4_to_f32(ws[q], b0, b1, b2, b3);
#pragma unroll
        for (int token = 0; token < kMaxVerifyTokens; ++token) {
          if (token >= tokens) break;
          const std::uint32_t* const a4 =
              reinterpret_cast<const std::uint32_t*>(a + static_cast<std::size_t>(token) * k);
          const std::uint32_t y0 = a4[e / 2];      // elements e, e+1
          const std::uint32_t y1 = a4[e / 2 + 1];  // elements e+2, e+3
          acc[token] += b0 * bf16_lo(y0) + b1 * bf16_hi(y0) + b2 * bf16_lo(y1) +
                        b3 * bf16_hi(y1);
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

// `tokens` must be in [1, 4]. Returns -1 for unsupported
// widths so callers cannot silently lose the one-weight-stream guarantee.
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                     float scale, int tokens, int rows, int k,
                                     std::uint16_t* d_c) {
  if (tokens < 1 || tokens > kMaxVerifyTokens || rows < 1 || k < 1) return -1;
  verify_gemv_kernel<<<rows, 128>>>(d_a, d_b, scale, tokens, rows, k, d_c);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
