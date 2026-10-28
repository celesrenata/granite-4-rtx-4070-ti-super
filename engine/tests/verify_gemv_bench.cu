// A/B benchmark for speculative verification GEMV.
//
// Compares k separate M=1 launches with one k-token fused launch while
// keeping the model-shaped FP8 weight matrix resident on device. The L2
// cache is flushed before every timed launch so each call reads its weights
// from HBM, matching decode (8.38 GB of weights >> 128 MB L2).
//
//   granite4-verify-gemv-bench [--iters n]          one 4096x4096 shape
//   granite4-verify-gemv-bench --model [--iters n]  real Granite 4.2 shapes
//
// The --model mode prints the aggregate cost of one k-token verify round
// (weights streamed once) against k single-token rounds.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int granite4_verify_gemv(const std::uint16_t*, const std::uint8_t*,
                                    float, int, int, int, std::uint16_t*);

namespace {
constexpr int kMaxTokens = 4;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

// 512 MB streamed buffer >> 128 MB L2.
__global__ void l2_flush(float* buf, std::size_t n4) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) buf[i] = 0.0f;
}

struct ShapeResult {
  float repeated_ms[kMaxTokens + 1] = {};
  float fused_ms[kMaxTokens + 1] = {};
};

int bench_shape(const char* name, int rows, int K, int iters, float* flush,
                std::size_t flush_elems, ShapeResult* result) {
  std::uint16_t* a = nullptr;
  std::uint8_t* b = nullptr;
  std::uint16_t* out = nullptr;
  check(cudaMalloc(&a, static_cast<std::size_t>(kMaxTokens) * K * 2),
        "cudaMalloc a");
  check(cudaMalloc(&b, static_cast<std::size_t>(rows) * K), "cudaMalloc b");
  check(cudaMalloc(&out, static_cast<std::size_t>(kMaxTokens) * rows * 2),
        "cudaMalloc out");
  check(cudaMemset(a, 0x3f, static_cast<std::size_t>(kMaxTokens) * K * 2),
        "memset a");
  check(cudaMemset(b, 0x21, static_cast<std::size_t>(rows) * K), "memset b");

  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin), "event create");
  check(cudaEventCreate(&end), "event create");
  for (int t = 1; t <= kMaxTokens; ++t) {
    for (int i = 0; i < 4; ++i)
      if (granite4_verify_gemv(a, b, 1.0f, 1, rows, K, out) != 0) return 3;
    if (granite4_verify_gemv(a, b, 1.0f, t, rows, K, out) != 0) return 3;
  }
  check(cudaDeviceSynchronize(), "warmup sync");

  for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
    double repeated_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      // k separate M=1 launches, each after its own L2 flush.
      for (int t = 0; t < tokens; ++t) {
        l2_flush<<<static_cast<unsigned>(flush_elems / 256), 256>>>(flush,
                                                                   flush_elems);
        check(cudaDeviceSynchronize(), "flush sync");
        check(cudaEventRecord(begin), "record begin");
        if (granite4_verify_gemv(a, b, 1.0f, 1, rows, K, out) != 0) return 3;
        check(cudaEventRecord(end), "record end");
        check(cudaEventSynchronize(end), "sync single");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, begin, end), "elapsed single");
        repeated_sum += ms;
      }
    }
    double fused_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
      l2_flush<<<static_cast<unsigned>(flush_elems / 256), 256>>>(flush,
                                                                 flush_elems);
      check(cudaDeviceSynchronize(), "flush sync");
      check(cudaEventRecord(begin), "record begin");
      if (granite4_verify_gemv(a, b, 1.0f, tokens, rows, K, out) != 0) return 3;
      check(cudaEventRecord(end), "record end");
      check(cudaEventSynchronize(end), "sync fused");
      float ms = 0.0f;
      check(cudaEventElapsedTime(&ms, begin, end), "elapsed fused");
      fused_sum += ms;
    }
    result->repeated_ms[tokens] = static_cast<float>(repeated_sum / iters);
    result->fused_ms[tokens] = static_cast<float>(fused_sum / iters);
    if (name != nullptr) {
      const double bytes = static_cast<double>(rows) * K;
      std::printf(
          "{\"name\":\"%s\",\"rows\":%d,\"K\":%d,\"tokens\":%d,"
          "\"repeated_ms\":%.3f,\"fused_ms\":%.3f,\"speedup\":%.2f,"
          "\"fused_gbps\":%.1f}\n",
          name, rows, K, tokens, result->repeated_ms[tokens],
          result->fused_ms[tokens],
          result->repeated_ms[tokens] / result->fused_ms[tokens],
          bytes / (result->fused_ms[tokens] * 1e-3) / 1e9);
    }
  }
  check(cudaEventDestroy(begin), "event destroy");
  check(cudaEventDestroy(end), "event destroy");
  cudaFree(a);
  cudaFree(b);
  cudaFree(out);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  int iters = 30;
  bool model = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--model") == 0)
      model = true;
  }

  const std::size_t flush_elems = 512u * 1024u * 1024u / sizeof(float);
  float* flush = nullptr;
  check(cudaMalloc(&flush, flush_elems * sizeof(float)), "cudaMalloc flush");
  check(cudaMemset(flush, 0, flush_elems * sizeof(float)), "memset flush");

  if (!model) {
    ShapeResult r;
    const int rc = bench_shape("4096x4096", 4096, 4096, iters, flush,
                               flush_elems, &r);
    cudaFree(flush);
    if (rc != 0) return rc;
    for (int tokens = 1; tokens <= kMaxTokens; ++tokens) {
      std::printf(
          "tokens=%d repeated_ms=%.3f fused_ms=%.3f speedup=%.2fx "
          "repeated_per_token_us=%.2f fused_per_token_us=%.2f\n",
          tokens, r.repeated_ms[tokens], r.fused_ms[tokens],
          r.repeated_ms[tokens] / r.fused_ms[tokens],
          r.repeated_ms[tokens] * 1000.0f / tokens,
          r.fused_ms[tokens] * 1000.0f / tokens);
    }
    return 0;
  }

  // Real Granite 4.2 8B decode shapes: H=4096, I=12800, Hq=32, Hk=8, d=128,
  // V=100352, L=40. `count` is launches per token.
  struct Shape {
    const char* name;
    int rows, K, count;
  };
  const Shape shapes[] = {
      {"q", 32 * 128, 4096, 40},
      {"k", 8 * 128, 4096, 40},
      {"v", 8 * 128, 4096, 40},
      {"o", 4096, 4096, 40},
      {"gate", 12800, 4096, 40},
      {"up", 12800, 4096, 40},
      {"down", 4096, 12800, 40},
      {"lm_head", 100352, 4096, 1},
  };
  double fused4_total = 0.0, repeated4_total = 0.0, fused1_total = 0.0;
  double bytes_total = 0.0;
  for (const Shape& s : shapes) {
    ShapeResult r;
    const int rc = bench_shape(s.name, s.rows, s.K, iters, flush, flush_elems,
                               &r);
    if (rc != 0) return rc;
    fused4_total += r.fused_ms[4] * s.count;
    repeated4_total += r.repeated_ms[4] * s.count;
    fused1_total += r.fused_ms[1] * s.count;
    bytes_total += static_cast<double>(s.rows) * s.K * s.count;
  }
  cudaFree(flush);
  std::printf(
      "model_aggregate fused_k4_ms_per_round=%.2f repeated_k4_ms=%.2f "
      "speedup=%.2fx fused_k4_ms_per_token=%.2f fused_k1_ms_per_token=%.2f "
      "bytes_per_token=%.0f fused_k4_gbps=%.1f\n",
      fused4_total, repeated4_total,
      repeated4_total > 0.0 ? repeated4_total / fused4_total : 0.0,
      fused4_total / 4.0, fused1_total, bytes_total,
      bytes_total / (fused4_total * 1e-3) / 1e9);
  return 0;
}
