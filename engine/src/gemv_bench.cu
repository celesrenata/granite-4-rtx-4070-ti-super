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

#include <cuda_runtime.h>

// The production decode GEMV launcher (forward_gpu.cu): the bench measures
// exactly the kernel and launch width the forward uses.
extern "C" int granite4_kernel_row_gemm_dequant(const std::uint16_t* d_a,
                                                const std::uint8_t* d_B,
                                                float scale, int N, int K,
                                                std::uint16_t* d_out);

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
    granite4_kernel_row_gemm_dequant(dA, dB, 1e-3f, s.N, s.K, dOut);
    cudaDeviceSynchronize();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double ms_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      l2_flush<<<flush_elems / 256, 256>>>(dFlush, flush_elems);
      cudaDeviceSynchronize();
      cudaEventRecord(e0);
      granite4_kernel_row_gemm_dequant(dA, dB, 1e-3f, s.N, s.K, dOut);
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
