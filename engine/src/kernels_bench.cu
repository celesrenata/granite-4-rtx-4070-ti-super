// Phase 2 (Kernels): micro-benchmark + roofline baseline (the P2 exit
// criterion). Measures the card's *actual* achievable bandwidth on the
// running device rather than assuming the spec, and derives the memory-bound
// roofline the kernels must sit under.
//
//   granite4-kernel-bench
//     --bytes <MiB>     buffer size (default 256)
//     --iters <n>       timed iterations (default 100)
//
// Prints one JSON object to stdout:
//   { "schema": "granite4.kbench.v1", "device": "...",
//     "read_gbps": ..., "copy_gbps": ..., "tokens_s_at_read": ... ,
//     "roofline": { "mem_gb": ..., "ops_flops": ... } }
//
// tokens_s_at_read = (measured read bandwidth) / (bytes read per token).
// For decode, each token reads every weight once; with FP8 e4m3 weights the
// per-token footprint is ~total_numel bytes. We report bandwidth, not a
// token figure, but expose the division so a caller can plug in a footprint.
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Device kernels (timed).
// ---------------------------------------------------------------------------

// Pure read: each thread streams a contiguous slice, xor-reduces, writes one
// sentinel so the compiler cannot elide the read.
__global__ void read_bench(const float4* buf, size_t n4, float* sink) {
  float acc = 0.0f;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const float4 v = buf[i];
    acc ^= __int_as_float(v.x) + __int_as_float(v.y);
  }
  // One write per block, so the kernel is overwhelmingly read-bound.
  if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}

// Copy: device-to-device streaming (read + write).
__global__ void copy_bench(const float4* src, float4* dst, size_t n4) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = idx; i < n4; i += stride) dst[i] = src[i];
}

#define CK(expr)                                                       \
  do {                                                                  \
    cudaError_t e_ = (expr);                                           \
    if (e_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "granite4-kernel-bench: %s at %s:%d\n",      \
                   cudaGetErrorString(e_), __FILE__, __LINE__);         \
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main(int argc, char** argv) {
  long bytes_mib = 256;
  long iters = 100;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) bytes_mib = std::atol(argv[++i]);
    else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atol(argv[++i]);
  }

  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));

  const size_t bytes = (size_t)bytes_mib * 1024u * 1024u;
  const size_t n4 = bytes / sizeof(float4);
  float4 *d_a, *d_b;
  float* d_sink;
  CK(cudaMalloc(&d_a, bytes));
  CK(cudaMalloc(&d_b, bytes));
  CK(cudaMalloc(&d_sink, 1024u * sizeof(float)));
  CK(cudaMemset(d_a, 0x3c, bytes));  // 1.0f in float4 bits; content is irrelevant

  const int threads = 256;
  const int blocks_read = 1024;
  const int blocks_copy = 2048;

  // Warmup + timed run with events.
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));

  read_bench<<<blocks_read, threads>>>(d_a, n4, d_sink);
  CK(cudaDeviceSynchronize());

  double read_ms = 0.0, copy_ms = 0.0;
  for (long i = 0; i < iters; ++i) {
    CK(cudaEventRecord(t0));
    read_bench<<<blocks_read, threads>>>(d_a, n4, d_sink);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    read_ms += ms;
  }
  for (long i = 0; i < iters; ++i) {
    CK(cudaEventRecord(t0));
    copy_bench<<<blocks_copy, threads>>>(d_a, d_b, n4);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    copy_ms += ms;
  }

  const double read_bytes = (double)n4 * (double)sizeof(float4);
  const double read_s = (read_ms / (double)iters) * 1e-3;
  const double copy_s = (copy_ms / (double)iters) * 1e-3;
  const double read_gbps = read_bytes / read_s / 1e9;
  const double copy_gbps = (2.0 * read_bytes) / copy_s / 1e9;  // read + write

  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_sink);

  std::printf(
      "{\"schema\":\"granite4.kbench.v1\","
      "\"device\":\"%s\","
      "\"sm\":\"%d.%d\","
      "\"bytes_mib\":%ld,"
      "\"iters\":%ld,"
      "\"read_gbps\":%.2f,"
      "\"copy_gbps\":%.2f,"
      "\"roofline\":{\"mem_gbps\":%.2f,"
      "\"note\":\"weights-per-token decode is read-bound; tokens/s = read_gbps "
      "/ (fp8 weight bytes per token)\"}}\n",
      p.name, p.major, p.minor, bytes_mib, iters, read_gbps, copy_gbps, read_gbps);
  return 0;
}
