// Phase 3 (Integration): paged KV compaction device kernel.
//
// The paged KV pool (kv_pool.h) stores a sequence's KV as pages in a
// token-major FP8 e4m3 layout:
//   page = [token_local][layer][k|v][kv_head][head_dim]
//   page_bytes = tokens_per_page * num_layers * 2 * Hk * head_dim
// A token's page is block_table[token / tokens_per_page]. The P2 attention
// kernels consume *contiguous* [Hk, kv_len, d] BF16 K/V, so this kernel
// gathers one layer's K (or V) out of the paged pool into that layout, with
// the FP8->BF16 dequant folded in. One thread per output element (correct-
// first tiling, matching the P2 convention); FP8->BF16 is exact, so the
// result is bit-identical to the host reference paged_kv_compact_bf16.
#include <cmath>
#include <cstdint>
#include <cstdio>

#define GRANITE_CUDA 1
#include "kernel_math.h"
#include "model_config.h"

namespace granite4 {

namespace {

__device__ __forceinline__ std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, sizeof u);
  u += 0x7fffu + ((u >> 16) & 1u);  // RNE
  u >>= 16;
  return static_cast<std::uint16_t>(u);
}

// Exact e4m3 decode to float (matches the P2 device helper); f32_to_bf16 is
// exact on it, so the composed decode equals fp8->bf16 bit-for-bit.
__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const bool neg = (x & 0x80u) != 0;
  const unsigned e = (x >> 3) & 0xFu;
  const unsigned m = x & 0x7u;
  if (e == 0) return neg ? -static_cast<float>(m) * std::ldexp(1.0f, -9)
                        : static_cast<float>(m) * std::ldexp(1.0f, -9);
  if (e == 15) {
    if (m == 7) return 0.0f;  // sNaN -> 0
    const float v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, 8);
    return neg ? -v : v;
  }
  const float v = (1.0f + static_cast<float>(m) / 8.0f) *
                  std::ldexp(1.0f, static_cast<int>(e) - 7);
  return neg ? -v : v;
}

// out[head, token, dim] <- decode(paged[page][token_local][layer][k|v][head][dim])
__global__ void paged_kv_compact_kernel(const std::uint8_t* paged,
                                        const std::size_t* block_table,
                                        int kv_len, int tokens_per_page,
                                        int num_layers, int Hk, int d, int layer,
                                        int which, std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hk * kv_len * d) return;
  const int dim = idx % d;
  const int token = (idx / d) % kv_len;
  const int head = idx / (kv_len * d);
  const std::size_t per_token =
      static_cast<std::size_t>(num_layers) * 2ull * Hk * d;
  const std::size_t page_bytes =
      static_cast<std::size_t>(tokens_per_page) * per_token;
  const std::size_t page = block_table[token / tokens_per_page];
  const std::size_t token_local = token % tokens_per_page;
  const std::size_t src =
      page * page_bytes + token_local * per_token +
      (static_cast<std::size_t>(layer) * 2ull + static_cast<std::size_t>(which)) *
          Hk * d +
      static_cast<std::size_t>(head) * d + dim;
  out[idx] = f32_to_bf16(fp8_to_f32(paged[src]));
}

}  // namespace

}  // namespace granite4

// C ABI: gather one layer's K (which=0) or V (which=1) from the paged FP8
// pool into contiguous [Hk, kv_len, d] BF16. `block_table` must have
// ceil(kv_len / tokens_per_page) entries (device pointers throughout).
extern "C" int granite4_kernel_paged_kv_compact(
    const std::uint8_t* d_paged, const std::size_t* d_block_table, int kv_len,
    int tokens_per_page, int num_layers, int Hk, int d, int layer, int which,
    std::uint16_t* d_out) {
  if (kv_len <= 0 || tokens_per_page <= 0 || num_layers <= 0 || Hk <= 0 ||
      d <= 0 || layer < 0 || layer >= num_layers || (which != 0 && which != 1)) {
    return 1;
  }
  const int total = Hk * kv_len * d;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  granite4::paged_kv_compact_kernel<<<blocks, threads>>>(
      d_paged, d_block_table, kv_len, tokens_per_page, num_layers, Hk, d, layer,
      which, d_out);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
