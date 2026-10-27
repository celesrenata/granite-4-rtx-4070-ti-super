// Experimental k-token speculative-decoding verify GEMV.
// One block computes one (token, output-row) dot product. This is deliberately
// a separate API: it is useful for measuring verify traffic without changing
// the production M=1 decode path.
#include <cuda_runtime.h>
#include <cstdint>

namespace {
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
  const std::uint32_t lsb = (u >> 16) & 1u;
  return static_cast<std::uint16_t>((u + 0x7ffu + lsb) >> 16);
}

// A is [tokens,K], B is [rows,K] row-major FP8, C is [tokens,rows] BF16.
__global__ void verify_gemv_kernel(const std::uint16_t* __restrict__ a,
                                   const std::uint8_t* __restrict__ b,
                                   float scale, int tokens, int rows, int k,
                                   std::uint16_t* __restrict__ c) {
  const int row = static_cast<int>(blockIdx.x);
  const int token = static_cast<int>(blockIdx.y);
  if (row >= rows || token >= tokens) return;
  const std::uint16_t* avec = a + static_cast<std::size_t>(token) * k;
  const std::uint8_t* wrow = b + static_cast<std::size_t>(row) * k;
  float sum = 0.0f;
  for (int j = threadIdx.x; j < k; j += blockDim.x)
    sum = fmaf(bf16_to_f32(avec[j]), fp8_to_f32(wrow[j]), sum);
  __shared__ float partial[128];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    c[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(partial[0] * scale);
}
}  // namespace

extern "C" int granite4_verify_gemv(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                     float scale, int tokens, int rows, int k,
                                     std::uint16_t* d_c) {
  if (!d_a || !d_b || !d_c || tokens <= 0 || rows <= 0 || k <= 0) return 1;
  verify_gemv_kernel<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(tokens)), 128>>>(
      d_a, d_b, scale, tokens, rows, k, d_c);
  return static_cast<int>(cudaGetLastError());
}
