// A/B benchmark for speculative verification GEMV.
// Compares k separate M=1 launches with one k-token fused launch while
// keeping the model-shaped FP8 weight matrix resident on device.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

extern "C" int granite4_verify_gemv(const std::uint16_t*, const std::uint8_t*, float,
                                     int, int, int, std::uint16_t*);

static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  int iters = 100;
  if (argc == 3 && std::string(argv[1]) == "--iters") iters = std::atoi(argv[2]);
  constexpr int K = 4096;
  constexpr int rows = 4096;
  constexpr int max_tokens = 4;
  constexpr float scale = 1.0f;
  std::uint16_t* a = nullptr;
  std::uint8_t* b = nullptr;
  std::uint16_t* out = nullptr;
  check(cudaMalloc(&a, static_cast<std::size_t>(max_tokens) * K * 2), "cudaMalloc a");
  check(cudaMalloc(&b, static_cast<std::size_t>(rows) * K), "cudaMalloc b");
  check(cudaMalloc(&out, static_cast<std::size_t>(max_tokens) * rows * 2), "cudaMalloc out");
  check(cudaMemset(a, 0x3f, static_cast<std::size_t>(max_tokens) * K * 2), "memset a");
  check(cudaMemset(b, 0x21, static_cast<std::size_t>(rows) * K), "memset b");

  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin), "event create");
  check(cudaEventCreate(&end), "event create");
  check(cudaDeviceSynchronize(), "warmup sync");
  // Warm up both launch forms and force compilation/cache setup.
  for (int t = 1; t <= max_tokens; ++t) {
    for (int i = 0; i < 10; ++i)
      if (granite4_verify_gemv(a, b, scale, 1, rows, K, out) != 0) return 3;
    if (granite4_verify_gemv(a, b, scale, t, rows, K, out) != 0) return 3;
  }
  check(cudaDeviceSynchronize(), "warmup sync");

  for (int tokens = 1; tokens <= max_tokens; ++tokens) {
    check(cudaEventRecord(begin), "record begin");
    for (int i = 0; i < iters; ++i)
      for (int t = 0; t < tokens; ++t)
        if (granite4_verify_gemv(a + static_cast<std::size_t>(t) * K, b, scale,
                                 1, rows, K, out + static_cast<std::size_t>(t) * rows) != 0)
          return 3;
    check(cudaEventRecord(end), "record end");
    check(cudaEventSynchronize(end), "sync repeated");
    float repeated_ms = 0.0f;
    check(cudaEventElapsedTime(&repeated_ms, begin, end), "elapsed repeated");

    check(cudaEventRecord(begin), "record begin");
    for (int i = 0; i < iters; ++i)
      if (granite4_verify_gemv(a, b, scale, tokens, rows, K, out) != 0) return 3;
    check(cudaEventRecord(end), "record end");
    check(cudaEventSynchronize(end), "sync fused");
    float fused_ms = 0.0f;
    check(cudaEventElapsedTime(&fused_ms, begin, end), "elapsed fused");

    std::printf("tokens=%d repeated_ms=%.3f fused_ms=%.3f speedup=%.2fx "
                "repeated_per_token_us=%.2f fused_per_token_us=%.2f\n",
                tokens, repeated_ms / iters, fused_ms / iters,
                repeated_ms / fused_ms, repeated_ms * 1000.0f / iters / tokens,
                fused_ms * 1000.0f / iters / tokens);
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
  return 0;
}
