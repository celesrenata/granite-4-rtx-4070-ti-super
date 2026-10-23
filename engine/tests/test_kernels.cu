// Phase 2 (Kernels): device correctness tests.
//
// Every kernel is run on the GPU with seeded random inputs and compared
// against the host reference in kernel_math.h (same RNG, same formulas).
// BF16 outputs are compared with a 2-ULP tolerance: the kernel and the
// reference use the same arithmetic but different summation orders, so
// single-ulp ties are permitted.
//
// Also verifies the arch probe: the binary must have been compiled for
// sm_89 and be running on it (the probe returns 1 only under __CUDA_ARCH__
// == 890).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kernel_math.h"

using namespace granite4;

// Declared in kernels.cu (the arch probe) and kernels_phase2.cu (launchers).
extern "C" void granite4_cuda_arch_probe(int* out);
extern "C" int granite4_kernel_rmsnorm(const std::uint16_t* d_x, const std::uint16_t* d_w,
                                       float eps, int rows, int cols, std::uint16_t* d_out);
extern "C" int granite4_kernel_rope(const std::uint16_t* d_x, const float* d_cos,
                                    const float* d_sin, int rows, int head_dim,
                                    std::uint16_t* d_out);
extern "C" int granite4_kernel_softmax(const std::uint16_t* d_x, float scale, int rows,
                                       int seq_len, std::uint16_t* d_out);
extern "C" int granite4_kernel_gemm_dequant(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                            float scale, int M, int N, int K, std::uint16_t* d_c);
extern "C" int granite4_rope_tables_build(double theta, int head_dim, int max_pos,
                                          float** d_cos, float** d_sin);

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

#define CUDA_CHECK(expr)                                                 \
  do {                                                                  \
    cudaError_t e_ = (expr);                                            \
    if (e_ != cudaSuccess) {                                            \
      std::printf("FAIL: CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                  cudaGetErrorString(e_));                              \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

// BF16 ULP tolerance: allow the device to differ from the reference by at
// most 2 ULP (summation-order differences).
bool bf16_close(std::uint16_t got, std::uint16_t want, int ulps = 2) {
  auto step = [](std::uint16_t bits, int dir) {
    return static_cast<std::uint16_t>(bits + dir);
  };
  int diff = 0;
  while (got != want && diff < 1024) {
    if (got > want) got = step(got, -1);
    else got = step(got, 1);
    ++diff;
  }
  return diff <= ulps;
}

std::uint16_t gen_bf16(Xorshift64& rng) {
  return f32_to_bf16_host(rng.next_float_range(4.0));
}

}  // namespace

static void test_arch_probe() {
  int out = -1;
  int* d_out;
  if (cudaMalloc(&d_out, sizeof(int)) != cudaSuccess) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_out, &out, sizeof(int), cudaMemcpyHostToDevice));
  granite4_cuda_arch_probe(d_out);
  int got = -99;
  CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(int), cudaMemcpyDeviceToHost));
  expect(got == 1, "arch probe: compiled and running on sm_89");
  CUDA_CHECK(cudaFree(d_out));
}

static void test_rmsnorm_device() {
  const int rows = 64, cols = 256;
  const float eps = 1e-5f;
  Xorshift64 rng(0x4242);
  std::vector<std::uint16_t> hx(rows * cols), hw(cols), hout(rows * cols);
  for (auto& v : hx) v = gen_bf16(rng);
  for (auto& v : hw) v = gen_bf16(rng);

  std::uint16_t* d_x, *d_w, *d_out;
  if (cudaMalloc(&d_x, hx.size() * 2) || cudaMalloc(&d_w, hw.size() * 2) ||
      cudaMalloc(&d_out, hout.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice));

  expect(granite4_kernel_rmsnorm(d_x, d_w, eps, rows, cols, d_out) == 0, "rmsnorm launch");
  CUDA_CHECK(cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> xr(cols), wr(cols), ref(cols), gotf(cols);
    for (int i = 0; i < cols; ++i) {
      xr[i] = bf16_to_f32_host(hx[r * cols + i]);
      wr[i] = bf16_to_f32_host(hw[i]);
      gotf[i] = bf16_to_f32_host(hout[r * cols + i]);
    }
    rmsnorm_f32(xr.data(), wr.data(), eps, cols, ref.data());
    for (int i = 0; i < cols; ++i)
      if (!bf16_close(hout[r * cols + i], f32_to_bf16_host(ref[i]))) ++bad;
  }
  expect(bad == 0, "rmsnorm: device matches host reference (2 ULP)");
  cudaFree(d_x);
  cudaFree(d_w);
  cudaFree(d_out);
}

static void test_rope_device() {
  const int rows = 128;  // positions
  const int d = 128;     // head_dim
  Xorshift64 rng(0x1337);
  std::vector<std::uint16_t> hx(rows * d), hout(rows * d);
  for (auto& v : hx) v = gen_bf16(rng);
  float *d_cos, *d_sin;
  expect(granite4_rope_tables_build(1e7, d, rows, &d_cos, &d_sin) == 0, "rope tables build");

  std::uint16_t* d_x, *d_out;
  if (cudaMalloc(&d_x, hx.size() * 2) || cudaMalloc(&d_out, hout.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_rope(d_x, d_cos, d_sin, rows, d, d_out) == 0, "rope launch");
  CUDA_CHECK(cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost));

  std::vector<float> ct, st;
  rope_freqs_f32(1e7, d, rows, &ct, &st);
  int bad = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> xr(d), ref(d), gotf(d);
    for (int i = 0; i < d; ++i) xr[i] = bf16_to_f32_host(hx[r * d + i]);
    rope_half_f32(xr.data(), &ct[r * (d / 2)], &st[r * (d / 2)], d, ref.data());
    for (int i = 0; i < d; ++i)
      if (!bf16_close(hout[r * d + i], f32_to_bf16_host(ref[i]))) ++bad;
  }
  expect(bad == 0, "rope: device matches host reference (2 ULP)");
  cudaFree(d_x);
  cudaFree(d_out);
  cudaFree(d_cos);
  cudaFree(d_sin);
}

static void test_softmax_device() {
  const int rows = 32, seq = 512;
  const float scale = 1.0f / 128.0f;  // Granite attention multiplier
  Xorshift64 rng(0xbeef);
  std::vector<std::uint16_t> hx(rows * seq), hout(rows * seq);
  for (auto& v : hx) v = gen_bf16(rng);

  std::uint16_t* d_x, *d_out;
  if (cudaMalloc(&d_x, hx.size() * 2) || cudaMalloc(&d_out, hout.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_softmax(d_x, scale, rows, seq, d_out) == 0, "softmax launch");
  CUDA_CHECK(cudaMemcpy(hout.data(), d_out, hout.size() * 2, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> xr(seq), ref(seq), gotf(seq);
    for (int i = 0; i < seq; ++i) xr[i] = bf16_to_f32_host(hx[r * seq + i]);
    softmax_f32(xr.data(), scale, seq, ref.data());
    for (int i = 0; i < seq; ++i)
      if (!bf16_close(hout[r * seq + i], f32_to_bf16_host(ref[i]), 3)) ++bad;
  }
  expect(bad == 0, "softmax: device matches host reference (3 ULP)");
  cudaFree(d_x);
  cudaFree(d_out);
}

static void test_gemm_device() {
  const int M = 32, N = 64, K = 128;
  const float scale = 0.001f;
  Xorshift64 rng(0xdead);
  std::vector<std::uint16_t> hA(M * K);
  std::vector<std::uint8_t> hB(N * K);
  std::vector<std::uint16_t> hC(M * N);
  for (auto& v : hA) v = gen_bf16(rng);
  for (auto& v : hB) v = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);

  std::uint16_t* dA, *dC;
  std::uint8_t* dB;
  if (cudaMalloc(&dA, hA.size() * 2) || cudaMalloc(&dB, hB.size()) ||
      cudaMalloc(&dC, hC.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size(), cudaMemcpyHostToDevice));
  expect(granite4_kernel_gemm_dequant(dA, dB, scale, M, N, K, dC) == 0, "gemm launch");
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));

  std::vector<float> ref;
  gemm_dequant_f32(hA.data(), hB.data(), scale, M, N, K, &ref);
  int bad = 0;
  for (std::size_t i = 0; i < hC.size(); ++i) {
    // Accumulation over K = 128 fp8*bf16 products in float: a few ULP is
    // expected for large magnitudes.
    if (!bf16_close(hC[i], f32_to_bf16_host(ref[i]), 4)) ++bad;
  }
  expect(bad == 0, "gemm dequant: device matches host reference (4 ULP)");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }
  test_arch_probe();
  test_rmsnorm_device();
  test_rope_device();
  test_softmax_device();
  test_gemm_device();
  cudaDeviceSynchronize();
  if (g_failures == 0) std::printf("all device kernel tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
