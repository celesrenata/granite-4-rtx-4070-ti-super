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
extern "C" int granite4_fp8_decode_all(float** d_out);
extern "C" int granite4_kernel_rmsnorm(const std::uint16_t* d_x, const std::uint16_t* d_w,
                                       float eps, int rows, int cols, std::uint16_t* d_out);
extern "C" int granite4_kernel_rope(const std::uint16_t* d_x, const float* d_cos,
                                    const float* d_sin, int rows, int head_dim,
                                    std::uint16_t* d_out);
extern "C" int granite4_kernel_softmax(const std::uint16_t* d_x, float scale, int rows,
                                       int seq_len, std::uint16_t* d_out);
extern "C" int granite4_kernel_gemm_dequant(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                            float scale, int M, int N, int K, std::uint16_t* d_c);
extern "C" int granite4_kernel_gqa_logits(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                          float scale, int Hq, int Hk, int d, int kv_len,
                                          std::uint16_t* d_logits);
extern "C" int granite4_kernel_gqa_value(const std::uint16_t* d_p, const std::uint16_t* d_v,
                                         int Hq, int Hk, int d, int kv_len, std::uint16_t* d_out);
extern "C" int granite4_kernel_gqa_fused(const std::uint16_t* d_q, const std::uint16_t* d_k,
                                         const std::uint16_t* d_v, float att_scale, int Hq,
                                         int Hk, int d, int kv_len, int k_stride,
                                         std::uint16_t* d_out);
extern "C" int granite4_kernel_silu_gate_up(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                           const std::uint16_t* d_x, float scale_g, float scale_u,
                                           int N, int K, std::uint16_t* d_out);
extern "C" int granite4_kernel_row_gemm_dequant(const std::uint16_t* d_a,
                                                const std::uint8_t* d_B, float scale, int N,
                                                int K, std::uint16_t* d_out);
extern "C" int granite4_kernel_row_gemm_resid_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_B, float scale,
    const std::uint16_t* d_residual, float residual_scale, int tokens, int N, int K,
    std::uint16_t* d_out);
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a,
                                     const std::uint8_t* d_b, float scale,
                                     int tokens, int rows, int k,
                                     std::uint16_t* d_c);
extern "C" int granite4_kernel_qkv_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int nq, int nk, int k,
    std::uint16_t* d_q, std::uint16_t* d_kk, std::uint16_t* d_v);
extern "C" int granite4_verify_qkv(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int tokens, int nq,
    int nk, int k, std::uint16_t* d_q, std::uint16_t* d_k, std::uint16_t* d_v);
extern "C" int granite4_kernel_qkv_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int nq, int nk, int k,
    std::uint16_t* d_q, std::uint16_t* d_kk, std::uint16_t* d_v);
extern "C" int granite4_kernel_sample_greedy(const std::uint16_t* d_p, int rows, int n, int* d_out);
extern "C" int granite4_kernel_sample_top_p(const std::uint16_t* d_p, float nucleus,
                                            const float* d_u, int rows, int n, int* d_out);
extern "C" int granite4_rope_tables_build(double theta, int head_dim, int max_pos,
                                          float** d_cos, float** d_sin);
// P3 paged KV compaction (kernels_phase3.cu).
extern "C" int granite4_kernel_paged_kv_compact(
    const std::uint8_t* d_paged, const std::size_t* d_block_table, int kv_len,
    int tokens_per_page, int num_layers, int Hk, int d, int layer, int which,
    std::uint16_t* d_out);
extern "C" int granite4_kernel_silu_gate_up(
    const std::uint8_t* d_wg, const std::uint8_t* d_wu,
    const std::uint16_t* d_x, float scale_g, float scale_u, int N, int K,
    std::uint16_t* d_out);
extern "C" int granite4_kernel_silu_gate_up_batched(
    const std::uint8_t* d_wg, const std::uint8_t* d_wu,
    const std::uint16_t* d_x, float scale_g, float scale_u, int N, int K,
    int rows, std::uint16_t* d_out);
extern "C" int granite4_kernel_rmsnorm_batched(
    const std::uint16_t* d_x, const std::uint16_t* d_w, float eps, int rows,
    int cols, std::uint16_t* d_out);
extern "C" int granite4_kernel_rope_at_pos(const std::uint16_t* d_x,
                                           const float* d_cos,
                                           const float* d_sin, int heads, int d,
                                           std::uint16_t* d_out);
extern "C" int granite4_kernel_rope_at_pos_batched(
    const std::uint16_t* d_x, const float* d_cos, const float* d_sin, int rows,
    int heads, int d, std::uint16_t* d_out);
extern "C" int granite4_kernel_verify_qkv_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    const float* d_cos, const float* d_sin, int base_pos, int rows, int Hq, int Hk,
    int d, int cache_stride, std::uint16_t* d_q_rot, std::uint16_t* d_k_rot,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v);
extern "C" int granite4_kernel_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    const float* d_cos, const float* d_sin, float att_scale, int base_pos, int rows,
    int Hq, int Hk, int d, int cache_stride, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_out);
extern "C" int granite4_kernel_kv_append_batched(
    const std::uint16_t* d_src_k, const std::uint16_t* d_src_v,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v, int base_pos, int rows,
    int stride, int Hk, int d);
extern "C" int granite4_kernel_gqa_fused_batched(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    float att_scale, int Hq, int Hk, int d, int base_kv, int rows, int k_stride,
    std::uint16_t* d_out);

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

// Exhaustive e4m3 decode: the branchless fp8_to_f32 must be bit-exact against
// the arithmetic reference on all 256 codepoints (subnormals, the e==15
// top-exponent + sNaN, and both sign bits included). The reference mirrors the
// pre-existing arithmetic decode exactly (sNaN 0x7f/0xff -> +0.0, as the old
// device code did), which is the P2 ground truth the kernels are verified
// against - NOT fp8_to_f32_host, whose fp8_e4m3_to_bf16 maps sNaN to the max
// value instead.
static float fp8_ref(std::uint8_t x) {
  const bool neg = (x & 0x80u) != 0;
  const unsigned e = (x >> 3) & 0xFu, m = x & 0x7u;
  if (e == 15 && m == 7) return 0.0f;  // sNaN -> +0.0 (old behavior)
  float v;
  if (e == 0) v = static_cast<float>(m) * std::ldexp(1.0f, -9);
  else if (e == 15) v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, 8);
  else v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, static_cast<int>(e) - 7);
  return neg ? -v : v;
}

static void test_fp8_decode_all() {
  float* d_out = nullptr;
  if (granite4_fp8_decode_all(&d_out) != 0) {
    ++g_failures;
    return;
  }
  std::vector<float> h(256);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, 256 * sizeof(float),
                       cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  int bad = 0;
  for (int x = 0; x < 256; ++x) {
    std::uint32_t a, b;
    const float ref = fp8_ref(static_cast<std::uint8_t>(x));
    std::memcpy(&a, &h[x], 4);
    std::memcpy(&b, &ref, 4);
    if (a != b) {
      if (bad < 8)
        std::printf("  fp8 decode mismatch x=%d got=%08x ref=%08x\n", x, a, b);
      ++bad;
    }
  }
  expect(bad == 0, "fp8_to_f32 branchless decode bit-exact on all 256 codepoints");
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

static void test_row_gemm_device() {
  // The forward's GEMV shape: a = 1xK, B = N x K, K > 256 (the coalesced
  // strided path), N not a multiple of the block dim (bounds-checking).
  const int N = 97, K = 512;
  const float scale = 0.0007f;
  Xorshift64 rng(0xb077);
  std::vector<std::uint16_t> hA(K);
  std::vector<std::uint8_t> hB(N * K);
  std::vector<std::uint16_t> hC(N);
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
  expect(granite4_kernel_row_gemm_dequant(dA, dB, scale, N, K, dC) == 0,
         "row-gemm launch");
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 2, cudaMemcpyDeviceToHost));

  std::vector<float> ref;
  gemm_dequant_f32(hA.data(), hB.data(), scale, /*M=*/1, N, K, &ref);
  int bad = 0;
  for (std::size_t i = 0; i < hC.size(); ++i)
    if (!bf16_close(hC[i], f32_to_bf16_host(ref[i]), 4)) ++bad;
  expect(bad == 0, "row gemm (GEMV): device matches host reference (4 ULP)");
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

static void test_verify_gemv_production_parity_device() {
  const int tokens = 4, rows = 17, k = 512;
  Xorshift64 rng(0x9494);
  std::vector<std::uint16_t> a(static_cast<std::size_t>(tokens) * k);
  std::vector<std::uint8_t> b(static_cast<std::size_t>(rows) * k);
  for (auto& v : a) v = gen_bf16(rng);
  for (auto& v : b) v = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  std::uint16_t *da, *dv, *dr;
  std::uint8_t *db;
  if (cudaMalloc(&da, a.size() * 2) || cudaMalloc(&db, b.size()) ||
      cudaMalloc(&dv, static_cast<std::size_t>(tokens) * rows * 2) ||
      cudaMalloc(&dr, static_cast<std::size_t>(tokens) * rows * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
  expect(granite4_verify_gemv(da, db, 0.0007f, tokens, rows, k, dv) == 0,
         "verify GEMV: launch");
  for (int t = 0; t < tokens; ++t)
    expect(granite4_kernel_row_gemm_dequant(
               da + static_cast<std::size_t>(t) * k, db, 0.0007f, rows, k,
               dr + static_cast<std::size_t>(t) * rows) == 0,
           "verify GEMV: production reference launch");
  std::vector<std::uint16_t> v(static_cast<std::size_t>(tokens) * rows),
      r(v.size());
  CUDA_CHECK(cudaMemcpy(v.data(), dv, v.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.data(), dr, r.size() * 2, cudaMemcpyDeviceToHost));
  int parity_bad = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    if (!bf16_close(v[i], r[i], 2)) ++parity_bad;
  expect(parity_bad == 0, "verify GEMV: within 2 ULP of production GEMV");
  cudaFree(da); cudaFree(db); cudaFree(dv); cudaFree(dr);
}

static void test_batched_qkv_parity_device() {
  const int tokens = 4, nq = 32, nk = 8, k = 256;
  Xorshift64 rng(0xa5a5);
  std::vector<std::uint16_t> a(static_cast<std::size_t>(tokens) * k);
  std::vector<std::uint8_t> wq(static_cast<std::size_t>(nq) * k),
      wk(static_cast<std::size_t>(nk) * k), wv(static_cast<std::size_t>(nk) * k);
  for (auto& x : a) x = gen_bf16(rng);
  for (auto& x : wq) x = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  for (auto& x : wk) x = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  for (auto& x : wv) x = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  const std::size_t qn = static_cast<std::size_t>(tokens) * nq;
  const std::size_t kn = static_cast<std::size_t>(tokens) * nk;
  std::uint16_t *da, *dq, *dk, *dv, *rq, *rk, *rv;
  std::uint8_t *dwq, *dwk, *dwv;
  if (cudaMalloc(&da, a.size() * 2) || cudaMalloc(&dwq, wq.size()) ||
      cudaMalloc(&dwk, wk.size()) || cudaMalloc(&dwv, wv.size()) ||
      cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, kn * 2) || cudaMalloc(&dv, kn * 2) ||
      cudaMalloc(&rq, qn * 2) || cudaMalloc(&rk, kn * 2) || cudaMalloc(&rv, kn * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwq, wq.data(), wq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwk, wk.data(), wk.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwv, wv.data(), wv.size(), cudaMemcpyHostToDevice));
  expect(granite4_verify_qkv(da, dwq, dwk, dwv, .0007f, .0008f, .0009f,
                             tokens, nq, nk, k, dq, dk, dv) == 0,
         "batched QKV: launch");
  for (int t = 0; t < tokens; ++t)
    expect(granite4_kernel_qkv_dequant(
               da + static_cast<std::size_t>(t) * k, dwq, dwk, dwv,
               .0007f, .0008f, .0009f, nq, nk, k,
               rq + static_cast<std::size_t>(t) * nq,
               rk + static_cast<std::size_t>(t) * nk,
               rv + static_cast<std::size_t>(t) * nk) == 0,
           "batched QKV: production reference launch");
  std::vector<std::uint16_t> gotq(qn), gotk(kn), gotv(kn), refq(qn), refk(kn), refv(kn);
  CUDA_CHECK(cudaMemcpy(gotq.data(), dq, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gotk.data(), dk, kn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gotv.data(), dv, kn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(refq.data(), rq, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(refk.data(), rk, kn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(refv.data(), rv, kn * 2, cudaMemcpyDeviceToHost));
  int parity_bad = 0;
  for (std::size_t i = 0; i < gotq.size(); ++i) if (!bf16_close(gotq[i], refq[i], 2)) ++parity_bad;
  for (std::size_t i = 0; i < gotk.size(); ++i) if (!bf16_close(gotk[i], refk[i], 2)) ++parity_bad;
  for (std::size_t i = 0; i < gotv.size(); ++i) if (!bf16_close(gotv[i], refv[i], 2)) ++parity_bad;
  expect(parity_bad == 0, "batched QKV: within 2 ULP of production calls");
  cudaFree(da); cudaFree(dwq); cudaFree(dwk); cudaFree(dwv);
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(rq); cudaFree(rk); cudaFree(rv);
}

static void test_batched_residual_projection_parity_device() {
  const int tokens = 4, N = 96, K = 384;
  Xorshift64 rng(0x5151);
  std::vector<std::uint16_t> a(static_cast<std::size_t>(tokens) * K);
  std::vector<std::uint16_t> residual(static_cast<std::size_t>(tokens) * N);
  std::vector<std::uint8_t> weights(static_cast<std::size_t>(N) * K);
  for (auto& x : a) x = gen_bf16(rng);
  for (auto& x : residual) x = gen_bf16(rng);
  for (auto& x : weights) x = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  const std::size_t count = static_cast<std::size_t>(tokens) * N;
  std::uint16_t *da, *dr, *do_batched, *do_single;
  std::uint8_t* dw;
  if (cudaMalloc(&da, a.size() * 2) || cudaMalloc(&dr, residual.size() * 2) ||
      cudaMalloc(&dw, weights.size()) || cudaMalloc(&do_batched, count * 2) ||
      cudaMalloc(&do_single, count * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dr, residual.data(), residual.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, weights.data(), weights.size(), cudaMemcpyHostToDevice));
  expect(granite4_kernel_row_gemm_resid_dequant(
             da, dw, .0006f, dr, .75f, tokens, N, K, do_batched) == 0,
         "batched residual projection: launch");
  for (int t = 0; t < tokens; ++t) {
    expect(granite4_kernel_row_gemm_dequant(
               da + static_cast<std::size_t>(t) * K, dw, .0006f, N, K,
               do_single + static_cast<std::size_t>(t) * N) == 0,
           "residual projection: production launch");
  }
  std::vector<std::uint16_t> got(count), ref(count);
  CUDA_CHECK(cudaMemcpy(got.data(), do_batched, count * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref.data(), do_single, count * 2, cudaMemcpyDeviceToHost));
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < N; ++i) {
      const float expected = bf16_to_f32_host(residual[static_cast<std::size_t>(t) * N + i]) +
                             .75f * bf16_to_f32_host(ref[static_cast<std::size_t>(t) * N + i]);
      expect(bf16_close(got[static_cast<std::size_t>(t) * N + i],
                        f32_to_bf16_host(expected), 2),
             "batched residual projection: residual arithmetic within 2 ULP");
    }
  }
  cudaFree(da); cudaFree(dr); cudaFree(dw); cudaFree(do_batched); cudaFree(do_single);
}

static void test_verify_qkv_stage_device() {
  const int base = 5, rows = 4, Hq = 8, Hk = 2, d = 32, stride = 32, half = d / 2;
  Xorshift64 rng(0x9191);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t kn = static_cast<std::size_t>(rows) * Hk * d;
  std::vector<std::uint16_t> q(qn), k(kn), v(kn), qrot(qn), krot(kn);
  std::vector<float> cs(static_cast<std::size_t>(rows) * half), sn(cs.size());
  for (auto& x : q) x = gen_bf16(rng);
  for (auto& x : k) x = gen_bf16(rng);
  for (auto& x : v) x = gen_bf16(rng);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < half; ++i) {
      const float th = 1.0f / std::pow(10000.0f, 2.0f * i / d);
      cs[r * half + i] = std::cos((base + r) * th);
      sn[r * half + i] = std::sin((base + r) * th);
    }
  const std::size_t cache_n = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> ck(cache_n), cv(cache_n);
  std::uint16_t *dq, *dk, *dv, *dqr, *dkr, *dck, *dcv;
  float *dc, *ds;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, kn * 2) || cudaMalloc(&dv, kn * 2) ||
      cudaMalloc(&dqr, qn * 2) || cudaMalloc(&dkr, kn * 2) ||
      cudaMalloc(&dck, cache_n * 2) || cudaMalloc(&dcv, cache_n * 2) ||
      cudaMalloc(&dc, cs.size() * 4) || cudaMalloc(&ds, sn.size() * 4)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, k.data(), kn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, v.data(), kn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
  expect(granite4_kernel_verify_qkv_stage(dq, dk, dv, dc, ds, base, rows, Hq, Hk, d,
                                          stride, dqr, dkr, dck, dcv) == 0,
         "verify QKV stage: launch");
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(qrot.data(), dqr, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(krot.data(), dkr, kn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ck.data(), dck, cache_n * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(cv.data(), dcv, cache_n * 2, cudaMemcpyDeviceToHost));
  for (int r = 0; r < rows; ++r) for (int h = 0; h < Hk; ++h) for (int i = 0; i < d; ++i) {
    const std::size_t src = (static_cast<std::size_t>(r) * Hk + h) * d + i;
    const std::size_t dst = (static_cast<std::size_t>(h) * stride + base + r) * d + i;
    expect(ck[dst] == krot[src] && cv[dst] == v[src], "verify QKV stage: cache append");
  }
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dqr); cudaFree(dkr);
  cudaFree(dck); cudaFree(dcv); cudaFree(dc); cudaFree(ds);
}

static void test_verify_attention_stage_device() {
  const int base = 5, rows = 4, Hq = 8, Hk = 2, d = 32, stride = 32, half = d / 2;
  Xorshift64 rng(0xa1a1);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t kn = static_cast<std::size_t>(rows) * Hk * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), k(kn), v(kn), ck(cn), cv(cn), batch(qn), ref(qn);
  std::vector<float> cs(static_cast<std::size_t>(rows) * half), sn(cs.size());
  for (auto& x : q) x = gen_bf16(rng);
  for (auto& x : k) x = gen_bf16(rng);
  for (auto& x : v) x = gen_bf16(rng);
  for (auto& x : ck) x = gen_bf16(rng);
  for (auto& x : cv) x = gen_bf16(rng);
  for (int r = 0; r < rows; ++r) for (int i = 0; i < half; ++i) {
    const float th = 1.0f / std::pow(10000.0f, 2.0f * i / d);
    cs[r * half + i] = std::cos((base + r) * th);
    sn[r * half + i] = std::sin((base + r) * th);
  }
  std::uint16_t *dq, *dk, *dv, *dck, *dcv, *dqr, *dkr, *do_batch, *do_ref;
  float *dc, *ds;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, kn * 2) || cudaMalloc(&dv, kn * 2) ||
      cudaMalloc(&dck, cn * 2) || cudaMalloc(&dcv, cn * 2) || cudaMalloc(&dqr, qn * 2) ||
      cudaMalloc(&dkr, kn * 2) || cudaMalloc(&do_batch, qn * 2) || cudaMalloc(&do_ref, qn * 2) ||
      cudaMalloc(&dc, cs.size() * 4) || cudaMalloc(&ds, sn.size() * 4)) { ++g_failures; return; }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, k.data(), kn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, v.data(), kn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dck, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(ds, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
  const float scale = 0.177f;
  expect(granite4_kernel_verify_attention_stage(dq, dk, dv, dc, ds, scale, base, rows,
      Hq, Hk, d, stride, dqr, dkr, dck, dcv, do_batch) == 0,
      "verify attention stage: launch");
  for (int r = 0; r < rows; ++r)
    expect(granite4_kernel_gqa_fused(dqr + static_cast<std::size_t>(r) * Hq * d,
      dck, dcv, scale, Hq, Hk, d, base + r + 1, stride,
      do_ref + static_cast<std::size_t>(r) * Hq * d) == 0,
      "verify attention stage: single-row reference");
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(batch.data(), do_batch, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(ref.data(), do_ref, qn * 2, cudaMemcpyDeviceToHost));
  expect(batch == ref, "verify attention stage: batched GQA parity");
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dck); cudaFree(dcv);
  cudaFree(dqr); cudaFree(dkr); cudaFree(do_batch); cudaFree(do_ref); cudaFree(dc); cudaFree(ds);
}

static void test_attention_device() {
  // GQA decode: 8 Q heads, 2 KV heads (gh = 4), head_dim 16, kv_len 32.
  const int Hq = 8, Hk = 2, d = 16, kv = 32;
  Xorshift64 rng(0x4a7c);
  std::vector<std::uint16_t> hq(Hq * d), hk(Hk * kv * d), hv(Hk * kv * d);
  for (auto& t : hq) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : hk) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : hv) t = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 1.0f / 128.0f;

  std::uint16_t *dq, *dk, *dv, *dlog, *dp, *dout;
  if (cudaMalloc(&dq, hq.size() * 2) || cudaMalloc(&dk, hk.size() * 2) ||
      cudaMalloc(&dv, hv.size() * 2) || cudaMalloc(&dlog, Hq * kv * 2) ||
      cudaMalloc(&dp, Hq * kv * 2) || cudaMalloc(&dout, Hq * d * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));

  expect(granite4_kernel_gqa_logits(dq, dk, scale, Hq, Hk, d, kv, dlog) == 0, "attn logits launch");
  // softmax over the logits (rows = Hq, seq = kv), scale already in the logits.
  expect(granite4_kernel_softmax(dlog, 1.0f, Hq, kv, dp) == 0, "attn softmax launch");
  expect(granite4_kernel_gqa_value(dp, dv, Hq, Hk, d, kv, dout) == 0, "attn value launch");

  std::vector<std::uint16_t> hout(Hq * d);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size() * 2, cudaMemcpyDeviceToHost));

  std::vector<std::uint16_t> ref;
  gqa_attention_bf16(hq.data(), hk.data(), hv.data(), scale, Hq, Hk, d, kv, &ref);
  int bad = 0;
  for (std::size_t i = 0; i < ref.size(); ++i)
    if (!bf16_close(hout[i], ref[i], 4)) ++bad;
  expect(bad == 0, "attention: device pipeline matches host reference (4 ULP)");
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dlog);
  cudaFree(dp);
  cudaFree(dout);
}

static void test_gqa_fused_device() {
  // The fused decode GQA kernel must be bit-identical to the three-kernel
  // pipeline (logits -> softmax(1.0) -> value) at the real head shape.
  const int Hq = 32, Hk = 8, d = 128, kv = 29;  // step 29 of the A/B prompt
  Xorshift64 rng(0x77e1);
  std::vector<std::uint16_t> hq(Hq * d), hk(Hk * kv * d), hv(Hk * kv * d);
  for (auto& t : hq) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : hk) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : hv) t = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 0.114456f;  // 1.0 / sqrt(128), the Granite value

  std::uint16_t *dq, *dk, *dv, *dlog, *dp, *dout, *dout2;
  if (cudaMalloc(&dq, hq.size() * 2) || cudaMalloc(&dk, hk.size() * 2) ||
      cudaMalloc(&dv, hv.size() * 2) || cudaMalloc(&dlog, Hq * kv * 2) ||
      cudaMalloc(&dp, Hq * kv * 2) || cudaMalloc(&dout, Hq * d * 2) ||
      cudaMalloc(&dout2, Hq * d * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, hv.data(), hv.size() * 2, cudaMemcpyHostToDevice));

  expect(granite4_kernel_gqa_logits(dq, dk, scale, Hq, Hk, d, kv, dlog) == 0,
         "fused: logits launch");
  expect(granite4_kernel_softmax(dlog, 1.0f, Hq, kv, dp) == 0,
         "fused: softmax launch");
  expect(granite4_kernel_gqa_value(dp, dv, Hq, Hk, d, kv, dout) == 0,
         "fused: value launch");
  expect(granite4_kernel_gqa_fused(dq, dk, dv, scale, Hq, Hk, d, kv, kv, dout2) == 0,
         "fused: gqa_fused launch");

  std::vector<std::uint16_t> ref(Hq * d), alt(Hq * d);
  CUDA_CHECK(cudaMemcpy(ref.data(), dout, ref.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(alt.data(), dout2, alt.size() * 2, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (std::size_t i = 0; i < ref.size(); ++i)
    if (ref[i] != alt[i]) ++bad;
  expect(bad == 0, "gqa_fused is bit-identical to the logits/softmax/value pipeline");

  // Direct-cache form: the same rows with a padded row stride (the persistent
  // cache is [Hk][max_pos][d], max_pos >= kv_len) must give identical output.
  const int pad = 7;  // max_pos - kv_len in the padded cache
  std::uint16_t* dkc = nullptr;
  std::uint16_t* dvc = nullptr;
  std::uint16_t* dout3 = nullptr;
  if (cudaMalloc(&dkc, static_cast<std::size_t>(Hk) * (kv + pad) * d * 2) ||
      cudaMalloc(&dvc, static_cast<std::size_t>(Hk) * (kv + pad) * d * 2) ||
      cudaMalloc(&dout3, Hq * d * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemset(dkc, 0xff, static_cast<std::size_t>(Hk) * (kv + pad) * d * 2));
  CUDA_CHECK(cudaMemset(dvc, 0xff, static_cast<std::size_t>(Hk) * (kv + pad) * d * 2));
  for (int h = 0; h < Hk; ++h) {
    CUDA_CHECK(cudaMemcpy(dkc + static_cast<std::size_t>(h) * (kv + pad) * d,
                          hk.data() + static_cast<std::size_t>(h) * kv * d,
                          static_cast<std::size_t>(kv) * d * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dvc + static_cast<std::size_t>(h) * (kv + pad) * d,
                          hv.data() + static_cast<std::size_t>(h) * kv * d,
                          static_cast<std::size_t>(kv) * d * 2, cudaMemcpyHostToDevice));
  }
  expect(granite4_kernel_gqa_fused(dq, dkc, dvc, scale, Hq, Hk, d, kv, kv + pad, dout3) == 0,
         "fused: gqa_fused direct-cache launch");
  std::vector<std::uint16_t> alt3(Hq * d);
  CUDA_CHECK(cudaMemcpy(alt3.data(), dout3, alt3.size() * 2, cudaMemcpyDeviceToHost));
  bad = 0;
  for (std::size_t i = 0; i < ref.size(); ++i)
    if (ref[i] != alt3[i]) ++bad;
  expect(bad == 0, "gqa_fused direct-cache stride is bit-identical to the compact form");
  cudaFree(dkc);
  cudaFree(dvc);
  cudaFree(dout3);
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dlog);
  cudaFree(dp);
  cudaFree(dout);
  cudaFree(dout2);
}

static void test_batched_verify_attention_device() {
  // P4: the batched verify kernels must be bit-identical, row by row, to the
  // single-row path. Row r appends its KV at position base+r and attends the
  // causal prefix [0, base+r+1) - the speculative verify contract.
  const int Hq = 32, Hk = 8, d = 128, base = 5, rows = 4, stride = 64;
  Xorshift64 rng(0x4242);
  std::vector<std::uint16_t> q(static_cast<std::size_t>(rows) * Hq * d);
  for (auto& t : q) t = f32_to_bf16_host(rng.next_float_range(2.0));
  const std::size_t cache_elems = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> ck(cache_elems), cv(cache_elems);
  for (auto& t : ck) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : cv) t = f32_to_bf16_host(rng.next_float_range(2.0));
  std::vector<std::uint16_t> sk(static_cast<std::size_t>(rows) * Hk * d);
  std::vector<std::uint16_t> sv(sk.size());
  for (auto& t : sk) t = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& t : sv) t = f32_to_bf16_host(rng.next_float_range(2.0));

  // Host expectation for the append: positions base..base+rows-1 get the
  // per-row projection, every other cache slot stays untouched.
  std::vector<std::uint16_t> ek = ck, ev = cv;
  for (int r = 0; r < rows; ++r)
    for (int hkv = 0; hkv < Hk; ++hkv)
      for (int i = 0; i < d; ++i) {
        const std::size_t dst = static_cast<std::size_t>(hkv) * stride * d +
                                static_cast<std::size_t>(base + r) * d + i;
        const std::size_t src =
            static_cast<std::size_t>(r) * Hk * d + hkv * d + i;
        ek[dst] = sk[src];
        ev[dst] = sv[src];
      }

  std::uint16_t *dq, *dsk, *dsv, *dak, *dav, *drk, *drv, *dout, *dref;
  if (cudaMalloc(&dq, q.size() * 2) ||
      cudaMalloc(&dsk, sk.size() * 2) || cudaMalloc(&dsv, sv.size() * 2) ||
      cudaMalloc(&dak, ck.size() * 2) || cudaMalloc(&dav, cv.size() * 2) ||
      cudaMalloc(&drk, ek.size() * 2) || cudaMalloc(&drv, ev.size() * 2) ||
      cudaMalloc(&dout, static_cast<std::size_t>(rows) * Hq * d * 2) ||
      cudaMalloc(&dref, static_cast<std::size_t>(rows) * Hq * d * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsk, sk.data(), sk.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsv, sv.data(), sv.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dak, ck.data(), ck.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dav, cv.data(), cv.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(drk, ek.data(), ek.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(drv, ev.data(), ev.size() * 2, cudaMemcpyHostToDevice));

  expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, rows,
                                           stride, Hk, d) == 0,
         "batched kv append: launch");
  expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, 5, stride,
                                           Hk, d) == -1,
         "batched kv append: rejects rows > 4");
  expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, rows,
                                           base + rows - 1, Hk, d) == -1,
         "batched kv append: rejects stride < base+rows");

  std::vector<std::uint16_t> gak(ck.size()), gav(cv.size());
  CUDA_CHECK(cudaMemcpy(gak.data(), dak, gak.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gav.data(), dav, gav.size() * 2, cudaMemcpyDeviceToHost));
  expect(gak == ek, "batched kv append: K cache bit-identical");
  expect(gav == ev, "batched kv append: V cache bit-identical");

  const float scale = 0.114456f;  // 1.0 / sqrt(128), the Granite value
  expect(granite4_kernel_gqa_fused_batched(dq, drk, drv, scale, Hq, Hk, d, base,
                                           rows, stride, dout) == 0,
         "batched gqa: launch");
  expect(granite4_kernel_gqa_fused_batched(dq, drk, drv, scale, Hq, Hk, d, base,
                                           rows, base + rows - 1, dout) == -1,
         "batched gqa: rejects k_stride < base+rows");
  for (int r = 0; r < rows; ++r)
    expect(granite4_kernel_gqa_fused(
               dq + static_cast<std::size_t>(r) * Hq * d, drk, drv, scale, Hq,
               Hk, d, base + r + 1, stride,
               dref + static_cast<std::size_t>(r) * Hq * d) == 0,
           "batched gqa: single-row reference launch");

  std::vector<std::uint16_t> bout(static_cast<std::size_t>(rows) * Hq * d),
      bref(bout.size());
  CUDA_CHECK(cudaMemcpy(bout.data(), dout, bout.size() * 2,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bref.data(), dref, bref.size() * 2,
                        cudaMemcpyDeviceToHost));
  expect(bout == bref,
         "batched gqa: bit-identical to per-row single-row calls");

  cudaFree(dq);
  cudaFree(dsk);
  cudaFree(dsv);
  cudaFree(dak);
  cudaFree(dav);
  cudaFree(drk);
  cudaFree(drv);
  cudaFree(dout);
  cudaFree(dref);
}

static void test_batched_silu_gate_up_device() {
  const int N = 257, K = 4096, rows = 4;
  Xorshift64 rng(0x5151);
  std::vector<std::uint8_t> wg(static_cast<std::size_t>(N) * K),
      wu(static_cast<std::size_t>(N) * K);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(rows) * K);
  for (auto& v : wg) v = static_cast<std::uint8_t>(rng.next());
  for (auto& v : wu) v = static_cast<std::uint8_t>(rng.next());
  for (auto& v : x) v = f32_to_bf16_host(rng.next_float_range(2.0));
  std::uint8_t *dwg, *dwu;
  std::uint16_t *dx, *db, *ds;
  if (cudaMalloc(&dwg, wg.size()) || cudaMalloc(&dwu, wu.size()) ||
      cudaMalloc(&dx, x.size() * 2) || cudaMalloc(&db, static_cast<std::size_t>(rows) * N * 2) ||
      cudaMalloc(&ds, static_cast<std::size_t>(rows) * N * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dwg, wg.data(), wg.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwu, wu.data(), wu.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  const float sg = 0.03125f, su = 0.02734375f;
  expect(granite4_kernel_silu_gate_up_batched(dwg, dwu, dx, sg, su, N, K,
                                              rows, db) == 0,
         "batched silu: launch");
  expect(granite4_kernel_silu_gate_up_batched(dwg, dwu, dx, sg, su, N, K, 5,
                                              db) == -1,
         "batched silu: rejects rows > 4");
  for (int r = 0; r < rows; ++r)
    expect(granite4_kernel_silu_gate_up(
               dwg, dwu, dx + static_cast<std::size_t>(r) * K, sg, su, N, K,
               ds + static_cast<std::size_t>(r) * N) == 0,
           "batched silu: single-row reference launch");
  std::vector<std::uint16_t> batch(static_cast<std::size_t>(rows) * N), single(batch.size());
  CUDA_CHECK(cudaMemcpy(batch.data(), db, batch.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(single.data(), ds, single.size() * 2, cudaMemcpyDeviceToHost));
  expect(batch == single, "batched silu: bit-identical to single-row calls");
  cudaFree(dwg); cudaFree(dwu); cudaFree(dx); cudaFree(db); cudaFree(ds);
}

static void test_batched_rmsnorm_device() {
  const int rows = 4, cols = 4096;
  Xorshift64 rng(0x6262);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(rows) * cols), w(cols);
  for (auto& v : x) v = f32_to_bf16_host(rng.next_float_range(4.0));
  for (auto& v : w) v = f32_to_bf16_host(rng.next_float_range(2.0));
  std::uint16_t *dx, *dw, *db, *ds;
  if (cudaMalloc(&dx, x.size() * 2) || cudaMalloc(&dw, w.size() * 2) ||
      cudaMalloc(&db, x.size() * 2) || cudaMalloc(&ds, x.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  const float eps = 1e-5f;
  expect(granite4_kernel_rmsnorm_batched(dx, dw, eps, rows, cols, db) == 0,
         "batched rmsnorm: launch");
  expect(granite4_kernel_rmsnorm_batched(dx, dw, eps, 5, cols, db) == -1,
         "batched rmsnorm: rejects rows > 4");
  for (int r = 0; r < rows; ++r)
    expect(granite4_kernel_rmsnorm(dx + static_cast<std::size_t>(r) * cols, dw,
                                   eps, 1, cols,
                                   ds + static_cast<std::size_t>(r) * cols) == 0,
           "batched rmsnorm: single-row reference launch");
  std::vector<std::uint16_t> batch(x.size()), single(x.size());
  CUDA_CHECK(cudaMemcpy(batch.data(), db, batch.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(single.data(), ds, single.size() * 2, cudaMemcpyDeviceToHost));
  expect(batch == single, "batched rmsnorm: bit-identical to single-row calls");
  cudaFree(dx); cudaFree(dw); cudaFree(db); cudaFree(ds);
}

static void test_batched_rope_device() {
  // P4: batched RoPE over consecutive verify positions must be bit-identical
  // to one rope_at_pos call per position (the production decode primitive).
  const int rows = 4, heads = 8, d = 128, half = d / 2;
  Xorshift64 rng(0x7272);
  std::vector<std::uint16_t> x(static_cast<std::size_t>(rows) * heads * d);
  for (auto& v : x) v = f32_to_bf16_host(rng.next_float_range(2.0));
  // Verify positions base..base+rows-1: row r uses that position's table row.
  std::vector<float> cs(static_cast<std::size_t>(rows) * half),
      sn(static_cast<std::size_t>(rows) * half);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < half; ++i) {
      const float th = 1.0f / std::pow(10000.0f, 2.0f * i / d);
      cs[r * half + i] = std::cos((r + 3) * th);
      sn[r * half + i] = std::sin((r + 3) * th);
    }
  std::uint16_t *dx, *db, *ds;
  float *dc, *dsin;
  if (cudaMalloc(&dx, x.size() * 2) || cudaMalloc(&db, x.size() * 2) ||
      cudaMalloc(&ds, x.size() * 2) || cudaMalloc(&dc, cs.size() * 4) ||
      cudaMalloc(&dsin, sn.size() * 4)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, cs.data(), cs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsin, sn.data(), sn.size() * 4, cudaMemcpyHostToDevice));
  expect(granite4_kernel_rope_at_pos_batched(dx, dc, dsin, rows, heads, d,
                                             db) == 0,
         "batched rope: launch");
  expect(granite4_kernel_rope_at_pos_batched(dx, dc, dsin, 5, heads, d,
                                             db) == -1,
         "batched rope: rejects rows > 4");
  for (int r = 0; r < rows; ++r)
    expect(granite4_kernel_rope_at_pos(
               dx + static_cast<std::size_t>(r) * heads * d,
               dc + static_cast<std::size_t>(r) * half,
               dsin + static_cast<std::size_t>(r) * half, heads, d,
               ds + static_cast<std::size_t>(r) * heads * d) == 0,
           "batched rope: single-position reference launch");
  std::vector<std::uint16_t> batch(x.size()), single(x.size());
  CUDA_CHECK(cudaMemcpy(batch.data(), db, batch.size() * 2,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(single.data(), ds, single.size() * 2,
                        cudaMemcpyDeviceToHost));
  expect(batch == single, "batched rope: bit-identical to per-position calls");
  cudaFree(dx); cudaFree(db); cudaFree(ds); cudaFree(dc); cudaFree(dsin);
}

static void test_paged_kv_compact_device() {
  // P3: gather one layer's K/V out of the paged FP8 pool into contiguous
  // [Hk, kv_len, d] BF16 and check it bit-exactly against the host reference
  // (FP8->BF16 is exact). Hk=2, d=8, B=2, layers=3, kv_len=6 (3 pages).
  const int Hk = 2, d = 8, B = 2, L = 3, kv_len = 6;
  const std::size_t per_token = (std::size_t)L * 2 * Hk * d;
  const std::size_t page_bytes = (std::size_t)B * per_token;
  const std::size_t num_pages = (kv_len + B - 1) / B;  // 3
  // A permuting block table so "use page id instead of block_table[t]" fails.
  const std::vector<std::size_t> bt = {1, 2, 0};
  Xorshift64 rng(0x0512);
  std::vector<std::uint8_t> paged(num_pages * page_bytes, 0);
  for (auto& b : paged) b = static_cast<std::uint8_t>(rng.next());

  std::uint8_t* d_paged;
  std::size_t* d_bt;
  std::uint16_t* d_out;
  const std::size_t out_elems = (std::size_t)Hk * kv_len * d;
  if (cudaMalloc(&d_paged, paged.size()) ||
      cudaMalloc(&d_bt, bt.size() * sizeof(std::size_t)) ||
      cudaMalloc(&d_out, out_elems * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_paged, paged.data(), paged.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bt, bt.data(), bt.size() * sizeof(std::size_t), cudaMemcpyHostToDevice));

  for (int which = 0; which < 2; ++which) {
    expect(granite4_kernel_paged_kv_compact(d_paged, d_bt, kv_len, B, L, Hk, d, 0, which,
                                            d_out) == 0,
           "paged kv compact launch");
    std::vector<std::uint16_t> got(out_elems);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, out_elems * 2, cudaMemcpyDeviceToHost));

    std::vector<std::uint16_t> ref;
    paged_kv_compact_bf16(paged.data(), bt, kv_len, B, L, Hk, d, 0, which, &ref);
    int bad = 0;
    for (std::size_t i = 0; i < out_elems; ++i)
      if (got[i] != ref[i]) ++bad;
    expect(bad == 0, "paged kv compact: device matches host reference (bit-exact)");
  }

  cudaFree(d_paged);
  cudaFree(d_bt);
  cudaFree(d_out);
}

static void test_silu_gate_up_device() {
  const int N = 64, K = 256;
  Xorshift64 rng(0x9c3b);
  std::vector<std::uint8_t> hg(N * K), hu(N * K);
  std::vector<std::uint16_t> hx(K), hout(N);
  for (auto& t : hg) t = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  for (auto& t : hu) t = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  for (auto& t : hx) t = f32_to_bf16_host(rng.next_float_range(2.0));

  std::uint8_t *dg, *du;
  std::uint16_t *dx, *dout;
  if (cudaMalloc(&dg, hg.size()) || cudaMalloc(&du, hu.size()) ||
      cudaMalloc(&dx, hx.size() * 2) || cudaMalloc(&dout, hout.size() * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dg, hg.data(), hg.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(du, hu.data(), hu.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_silu_gate_up(dg, du, dx, 1.0f, 1.0f, N, K, dout) == 0, "swiglu launch");
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, hout.size() * 2, cudaMemcpyDeviceToHost));

  std::vector<std::uint16_t> ref;
  silu_gate_up_bf16(hg.data(), hu.data(), hx.data(), 1.0f, 1.0f, N, K, &ref);
  int bad = 0;
  for (int i = 0; i < N; ++i)
    // 1 ULP: the device computes x*(1/sigmoid denom) while the reference
    // divides; the two differ by at most one rounding step.
    if (!bf16_close(hout[i], ref[i], 1)) ++bad;
  expect(bad == 0, "swiglu: device matches host reference (1 ULP)");
  cudaFree(dg);
  cudaFree(du);
  cudaFree(dx);
  cudaFree(dout);
}

static void test_sampling_device() {
  const int rows = 32, n = 512;
  Xorshift64 rng(0xa11ce);
  // Softmax-ish probability rows (BF16, sums ~ 1): build, normalize, round.
  std::vector<std::uint16_t> hp(rows * n);
  std::vector<float> u(rows);
  for (int r = 0; r < rows; ++r) {
    std::vector<float> w(n);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      w[i] = std::exp(rng.next_float_range(2.0));
      s += w[i];
    }
    for (int i = 0; i < n; ++i) hp[r * n + i] = f32_to_bf16_host(static_cast<float>(w[i] / s));
    u[r] = rng.next_float();
  }
  std::uint16_t* dp;
  int* dg;
  int* dtp;
  float* du;
  if (cudaMalloc(&dp, hp.size() * 2) || cudaMalloc(&dg, rows * 4) ||
      cudaMalloc(&dtp, rows * 4) || cudaMalloc(&du, rows * 4)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dp, hp.data(), hp.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(du, u.data(), u.size() * 4, cudaMemcpyHostToDevice));
  expect(granite4_kernel_sample_greedy(dp, rows, n, dg) == 0, "greedy launch");
  expect(granite4_kernel_sample_top_p(dp, 0.9f, du, rows, n, dtp) == 0, "top-p launch");
  std::vector<int> gre(rows), tpo(rows);
  CUDA_CHECK(cudaMemcpy(gre.data(), dg, rows * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(tpo.data(), dtp, rows * 4, cudaMemcpyDeviceToHost));

  int bad_greedy = 0, bad_topp = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<float> pr(n);
    for (int i = 0; i < n; ++i) pr[i] = bf16_to_f32_host(hp[r * n + i]);
    if (gre[r] != sample_greedy_f32(pr.data(), n)) ++bad_greedy;
    if (tpo[r] != sample_top_p_f32(pr.data(), 0.9f, u[r], n)) ++bad_topp;
  }
  expect(bad_greedy == 0, "sampling: device greedy matches host reference");
  expect(bad_topp == 0, "sampling: device top-p matches host reference");

  // Full production vocabulary (100352): greedy must scale to any size
  // (top-p is bounded by kTopPVocabMax in this phase).
  const int full = 100352;
  std::vector<std::uint16_t> hpf(2 * full);
  for (auto& t : hpf) t = f32_to_bf16_host(rng.next_float_range(2.0));
  std::uint16_t* dpf;
  int* dfg;
  if (cudaMalloc(&dpf, hpf.size() * 2) || cudaMalloc(&dfg, 2 * 4)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dpf, hpf.data(), hpf.size() * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_sample_greedy(dpf, 2, full, dfg) == 0, "greedy full-vocab launch");
  std::vector<int> gref(2);
  CUDA_CHECK(cudaMemcpy(gref.data(), dfg, 2 * 4, cudaMemcpyDeviceToHost));
  int bad_full = 0;
  for (int r = 0; r < 2; ++r) {
    std::vector<float> pr(full);
    for (int i = 0; i < full; ++i) pr[i] = bf16_to_f32_host(hpf[r * full + i]);
    if (gref[r] != sample_greedy_f32(pr.data(), full)) ++bad_full;
  }
  expect(bad_full == 0, "greedy: full 100352-vocab rows match host reference");
  cudaFree(dpf);
  cudaFree(dfg);
  cudaFree(dp);
  cudaFree(dg);
  cudaFree(dtp);
  cudaFree(du);
}

int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }
  test_arch_probe();
  test_fp8_decode_all();
  test_rmsnorm_device();
  test_rope_device();
  test_softmax_device();
  test_gemm_device();
  test_row_gemm_device();
  test_verify_gemv_production_parity_device();
  test_batched_qkv_parity_device();
  test_batched_residual_projection_parity_device();
  test_verify_qkv_stage_device();
  test_verify_attention_stage_device();
  test_attention_device();
  test_gqa_fused_device();
  test_batched_verify_attention_device();
  test_batched_silu_gate_up_device();
  test_batched_rmsnorm_device();
  test_batched_rope_device();
  test_paged_kv_compact_device();
  test_silu_gate_up_device();
  test_sampling_device();
  cudaDeviceSynchronize();
  if (g_failures == 0) std::printf("all device kernel tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
