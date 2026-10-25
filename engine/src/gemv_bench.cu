// Phase 4 (performance): GEMV micro-benchmark. Measures the *achieved* read
// bandwidth of the coalesced block-per-row GEMV (the decode GEMM) at the real
// model shapes, against the card's measured roofline (kernels_bench: ~631.9
// GB/s read on the RTX 4070 Ti Super). This isolates the per-token
// weight-streaming cost that decode is bound by: each step reads every FP8
// weight once.
//
//   granite4-gemv-bench [--iters n]   (default 200)
//
// Prints one JSON object to stdout: per-shape achieved GB/s + the per-layer /
// per-token totals. The measured numbers are kept out of the repo's prose per
// the cross-phase rule; they are reproducible from this binary.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernel_math.h"  // (host-only helpers; device twins below)

namespace granite4 {
namespace {

// Exact e4m3 decode, branchless - same bit-for-bit values as the forward
// kernel's fp8_to_f32 (kept in sync by hand; both are trivial).
__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const std::uint32_t s = static_cast<std::uint32_t>(x & 0x80u) << 24;
  const std::uint32_t y = x & 0x7Fu;
  const std::uint32_t e = y >> 3;
  const std::uint32_t m = y & 7u;
  std::uint32_t bits = ((120u + e) << 23) | (m << 20);
  const std::uint32_t top = (m == 7u) ? 0u : (0x43800000u | (m << 20));
  bits = (e == 15u) ? top : bits;
  bits = (e == 0u) ? __float_as_uint(static_cast<float>(m) * 0.001953125f)
                   : bits;  // 0.001953125f = 2^-9
  // sNaN (0x7f/0xff): the arithmetic decode returned +0.0 (sign discarded),
  // so drop the sign here too to stay bit-identical.
  return ((e == 15u && m == 7u)) ? 0.0f : __uint_as_float(bits | s);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof f);
  return f;
}

__device__ __forceinline__ std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, sizeof u);
  u += 0x7fffu + ((u >> 16) & 1u);
  u >>= 16;
  return static_cast<std::uint16_t>(u);
}

// The exact decode GEMV the forward uses (forward_gpu.cu
// row_gemm_dequant_kernel): coalesced block-per-row with a 16-byte (uint4)
// vector path, falling back to a 4-element uint32 path then scalar.
__global__ void row_gemm_dequant_kernel(const std::uint16_t* a, const std::uint8_t* B,
                                        float scale, int N, int K, std::uint16_t* out) {
  const int i = blockIdx.x;
  if (i >= N) return;
  const int t = threadIdx.x, nt = blockDim.x;
  const std::uint8_t* row = B + static_cast<std::size_t>(i) * K;
  const std::uint32_t* a4 = reinterpret_cast<const std::uint32_t*>(a);
  float acc = 0.0f;
  if ((K & 15) == 0 && (reinterpret_cast<std::uintptr_t>(row) & 15u) == 0 &&
      (reinterpret_cast<std::uintptr_t>(a) & 3u) == 0) {
    const int K16 = K >> 4;
    const uint4* row16 = reinterpret_cast<const uint4*>(row);
    for (int g = t; g < K16; g += nt) {
      const uint4 w = row16[g];
      const std::uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int e = 16 * g + 4 * q;
        const std::uint32_t wq = ws[q];
        const std::uint32_t y0 = a4[e / 2];
        const std::uint32_t y1 = a4[e / 2 + 1];
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
    const std::uint32_t* row4 = reinterpret_cast<const std::uint32_t*>(row);
    const int K4 = K >> 2;
    for (int k4 = t; k4 < K4; k4 += nt) {
      const std::uint32_t b = row4[k4];
      const std::uint32_t x0 = a4[2 * k4];
      const std::uint32_t x1 = a4[2 * k4 + 1];
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
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  __shared__ float red[32];
  if ((t & 31) == 0) red[t >> 5] = acc;
  __syncthreads();
  if (t == 0) {
    float s = red[0];
    for (int j = 1; j < (nt >> 5); ++j) s += red[j];
    out[i] = f32_to_bf16(scale * s);
  }
}

}  // namespace
}  // namespace granite4

// L2 flusher: evicts the ~128 MB L2 (RTX 4070 Ti Super) by streaming a
// 512 MB buffer, so each timed GEMV call reads its weights from HBM - the
// decode regime (each weight byte is read exactly once per token, never a
// second time within the step).
__global__ void l2_flush(float* buf, std::size_t n4) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) buf[i] = 0.0f;
}

int main(int argc, char** argv) {
  int iters = 200;
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);

  // Decode per-layer GEMV shapes (Granite 4.2 8B: H=4096, I=12800, Hq=32,
  // Hk=8, d=128, V=100352, L=40). name, N (rows), K (reduction), count
  // (times per token: 40 layers, or 1 for lm_head).
  struct Shape {
    const char* name;
    int N, K, count;
  };
  const Shape shapes[] = {
      {"q", 32 * 128, 4096, 40},    // Hq*d x H
      {"k", 8 * 128, 4096, 40},     // Hk*d x H
      {"v", 8 * 128, 4096, 40},     // Hk*d x H
      {"o", 4096, 4096, 40},        // H x H
      {"gate", 12800, 4096, 40},    // I x H
      {"up", 12800, 4096, 40},      // I x H
      {"down", 4096, 12800, 40},    // H x I
      {"lm_head", 100352, 4096, 1}, // V x H
  };

  double total_ms_per_token = 0.0;
  double total_bytes_per_token = 0.0;
  std::printf("{\"schema\":\"granite4.gemvbench.v1\",\"shapes\":[\n");
  bool first = true;
  for (const auto& s : shapes) {
    std::uint16_t* dA;
    std::uint8_t* dB;
    std::uint16_t* dOut;
    if (cudaMalloc(&dA, s.K * 2) != cudaSuccess ||
        cudaMalloc(&dB, (std::size_t)s.N * s.K) != cudaSuccess ||
        cudaMalloc(&dOut, (std::size_t)s.N * 2) != cudaSuccess) {
      std::fprintf(stderr, "cudaMalloc failed\n");
      return 1;
    }
    cudaMemset(dA, 0x3f, s.K * 2);
    cudaMemset(dB, 0x3f, (std::size_t)s.N * s.K);
    // L2 flush buffer (512 MB >> 128 MB L2) so each timed call reads its
    // weights from HBM - the decode regime (each weight byte is read exactly
    // once per token, never twice within a step). The unflushed loop would
    // report the L2-resident rate, roughly 2x faster, which does not bound
    // decode.
    const std::size_t flush_elems = 512u * 1024u * 1024u / sizeof(float);
    float* dFlush;
    cudaMalloc(&dFlush, flush_elems * sizeof(float));
    cudaMemset(dFlush, 0, flush_elems * sizeof(float));
    cudaDeviceSynchronize();
    // Warmup.
    granite4::row_gemm_dequant_kernel<<<s.N, 256>>>(dA, dB, 1e-3f, s.N, s.K,
                                                    dOut);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double ms_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      l2_flush<<<flush_elems / 256, 256>>>(dFlush, flush_elems);
      cudaDeviceSynchronize();
      cudaEventRecord(e0);
      granite4::row_gemm_dequant_kernel<<<s.N, 256>>>(dA, dB, 1e-3f, s.N, s.K,
                                                       dOut);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float one = 0.0f;
      cudaEventElapsedTime(&one, e0, e1);
      ms_sum += one;
    }
    const float ms = static_cast<float>(ms_sum / iters);
    const double bytes = (double)s.N * s.K;  // B bytes read per call (A is K bf16)
    const double gbps = bytes / (ms * 1e-3) / 1e9;
    const double layer_ms = ms * s.count;
    const double layer_bytes = bytes * s.count;
    total_ms_per_token += layer_ms;
    total_bytes_per_token += layer_bytes;
    if (!first) std::printf(",\n");
    first = false;
    std::printf(
        "  {\"name\":\"%s\",\"N\":%d,\"K\":%d,\"count\":%d,\"ms_per_call\":%.3f,"
        "\"gbps\":%.1f,\"ms_per_token\":%.2f,\"bytes_per_token\":%.0f}",
        s.name, s.N, s.K, s.count, ms, gbps, layer_ms, layer_bytes);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dOut);
    cudaFree(dFlush);
  }
  // Roofline reference: the measured read bandwidth from
  // granite4-kernel-bench on the RTX 4070 Ti Super (631.9 GB/s).
  const double roofline_gbps = 631.9;
  std::printf("\n],\n\"total\":{\"ms_per_token\":%.2f,\"bytes_per_token\":%.0f,"
              "\"roofline_gbps\":%.1f,\"effective_tps_76\":%.1f}\n}\n",
              total_ms_per_token, total_bytes_per_token, roofline_gbps,
              total_bytes_per_token / 1e9 / roofline_gbps / 1e-3);
  return 0;
}
