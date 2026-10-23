// Phase 2 (Kernels): host reference math tests (no CUDA).
//
// kernel_math.h is the source of truth the device kernels are checked
// against (tests/test_kernels.cu), so its own properties are pinned here on
// every platform. All expected values are derived, not assumed.
#include "kernel_math.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace granite4;

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
static void test_bf16() {
  // Powers of two (and the bf16 min subnormal, 2^-133) round-trip exactly.
  for (float v : {0.5f, 1.0f, 2.0f, -1.0f, 128.0f, -128.0f,
                  std::ldexp(1.0f, -133)}) {
    if (bf16_to_f32_host(f32_to_bf16_host(v)) != v)
      expect(false, "bf16 exact round-trip");
  }
  // RNE tie at 1.0: 1 + 2^-8 is exactly halfway between 1.0 (k=0, even
  // mantissa) and 1 + 2^-7 (k=1, odd) -> rounds to even = 1.0.
  expect(f32_to_bf16_host(1.0f + std::ldexp(1.0f, -8)) == f32_to_bf16_host(1.0f),
         "bf16 RNE tie -> even (1.0)");
  // 1 + 2^-7 is one ULP above 1.0 and is exactly representable.
  expect(bf16_to_f32_host(f32_to_bf16_host(1.0f + std::ldexp(1.0f, -7))) ==
             1.0f + std::ldexp(1.0f, -7),
         "bf16 1+2^-7 exact");
  // -0.0 keeps its sign bit.
  expect((f32_to_bf16_host(-0.0f) & 0x8000u) != 0, "bf16 -0.0 keeps sign");
  // An inexact value: 1 + 3.5*2^-7 is exactly halfway between k=3 (odd)
  // and k=4 (even) -> RNE rounds to even = 1 + 4 ULP.
  const float a = 1.0f + 3.5f * std::ldexp(1.0f, -7);
  const float r = bf16_to_f32_host(f32_to_bf16_host(a));
  expect(r == 1.0f + 4.0f * std::ldexp(1.0f, -7), "bf16 RNE tie at 3.5 ULPs -> even");
}

static void test_rmsnorm() {
  const int n = 8;
  // x = 2 everywhere: mean(x^2) = 4, so r = 1/sqrt(4 + eps), out = 2*r*w.
  std::vector<float> x(n, 2.0f), w(n, 1.0f);
  // eps = 0 -> r = 1/2 exactly -> out = 1.0 exactly.
  std::vector<float> out(n, 0.0f);
  rmsnorm_f32(x.data(), w.data(), 0.0f, n, out.data());
  for (int i = 0; i < n; ++i) expect(out[i] == 1.0f, "rmsnorm eps=0 constant -> unit");
  // eps = 1e-5 -> out = 2 / sqrt(4 + 1e-5), within 1e-5 of 1.0.
  std::vector<float> out2(n, 0.0f);
  rmsnorm_f32(x.data(), w.data(), 1e-5f, n, out2.data());
  for (int i = 0; i < n; ++i)
    expect(std::fabs(out2[i] - 2.0f / std::sqrt(4.0f + 1e-5f)) < 1e-7f,
           "rmsnorm eps applied");
  // weight scaling: w = 2 doubles every element.
  std::vector<float> w2(n, 2.0f), out3(n, 0.0f);
  rmsnorm_f32(x.data(), w2.data(), 0.0f, n, out3.data());
  for (int i = 0; i < n; ++i) expect(out3[i] == 2.0f, "rmsnorm weight scales row");
  // Zero row: 0 * rsqrt(0 + eps) * w == 0, not NaN.
  std::vector<float> z(n, 0.0f), outz(n, 1.0f);
  rmsnorm_f32(z.data(), w.data(), 1e-5f, n, outz.data());
  for (int i = 0; i < n; ++i) expect(outz[i] == 0.0f, "rmsnorm zero row -> zero, not NaN");
}

static void test_rope() {
  const int d = 8;
  std::vector<float> c, s;
  rope_freqs_f32(1e7, d, 1, &c, &s);
  // pos 0: angle 0 -> cos = 1, sin = 0 for every frequency.
  for (int i = 0; i < d / 2; ++i) {
    expect(c[i] == 1.0f, "rope cos(0) = 1");
    expect(s[i] == 0.0f, "rope sin(0) = 0");
  }
  // Direct rotation against a hand-built table at angle 0.5 rad.
  // Llama half-split: x = [a0, a1, b0, b1], pairs are (a_i, b_i) =
  // (i, i + d/2):  a' = a*c - b*s,  b' = b*c + a*s.
  const int d2 = 4;
  const float cc = std::cos(0.5f), ss = std::sin(0.5f);
  const float hcos[2] = {cc, cc}, hsin[2] = {ss, ss};
  const float x[4] = {2.0f, 4.0f, 1.0f, 3.0f};  // a = {2, 4}, b = {1, 3}
  std::vector<float> out(4, 0.0f);
  rope_half_f32(x, hcos, hsin, d2, out.data());
  const double cc_d = std::cos(0.5), ss_d = std::sin(0.5);
  expect(std::fabs(out[0] - (2.0 * cc_d - 1.0 * ss_d)) < 1e-5, "rope a0' = a c - b s");
  expect(std::fabs(out[2] - (1.0 * cc_d + 2.0 * ss_d)) < 1e-5, "rope b0' = b c + a s");
  expect(std::fabs(out[1] - (4.0 * cc_d - 3.0 * ss_d)) < 1e-5, "rope a1' = a c - b s");
  expect(std::fabs(out[3] - (3.0 * cc_d + 4.0 * ss_d)) < 1e-5, "rope b1' = b c + a s");
  // Inverse property: the negative-angle rotation restores the input.
  const float hcos_n[2] = {cc, cc}, hsin_n[2] = {-ss, -ss};
  std::vector<float> back(4, 0.0f);
  rope_half_f32(out.data(), hcos_n, hsin_n, d2, back.data());
  for (int i = 0; i < d2; ++i)
    expect(std::fabs(back[i] - x[i]) < 1e-5, "rope inverse rotation restores input");
  // Frequency table: freq 0 has inv_freq = 1, so at pos 1 the angle is 1.0 rad.
  rope_freqs_f32(1e7, d, 2, &c, &s);
  expect(std::fabs(c[(d / 2) + 0] - std::cos(1.0)) < 1e-5f, "rope table cos(1) at pos 1");
  expect(std::fabs(s[(d / 2) + 0] - std::sin(1.0)) < 1e-5f, "rope table sin(1) at pos 1");
}

static void test_softmax() {
  // Row always sums to 1.
  std::vector<float> x = {1.0f, 2.0f, 3.0f};
  std::vector<float> p(3);
  softmax_f32(x.data(), 1.0f, 3, p.data());
  double s = 0.0;
  for (float v : p) s += v;
  expect(std::fabs(s - 1.0) < 1e-6, "softmax row sums to 1");
  // Constant row -> uniform.
  std::vector<float> k(8, 5.0f), uk(8, 0.0f);
  softmax_f32(k.data(), 1.0f, 8, uk.data());
  for (int i = 0; i < 8; ++i)
    expect(std::fabs(uk[i] - 1.0f / 8.0f) < 1e-6f, "softmax constant -> uniform");
  // Scale invariance of the relative ordering: larger logits -> larger p.
  expect(p[2] > p[1] && p[1] > p[0], "softmax monotone in logits");
  // The Granite attention scale (1/128) checked against double arithmetic.
  std::vector<float> lg = {0.0f, 0.01f, -0.02f, 0.005f};
  std::vector<float> o(4);
  softmax_f32(lg.data(), 1.0f / 128.0f, 4, o.data());
  double mx = -std::numeric_limits<double>::infinity();
  for (float v : lg) mx = std::max(mx, v / 128.0);
  double sum = 0.0;
  for (float v : lg) sum += std::exp(v / 128.0 - mx);
  for (int i = 0; i < 4; ++i) {
    const double want = std::exp(lg[i] / 128.0 - mx) / sum;
    expect(std::fabs(o[i] - static_cast<float>(want)) < 1e-7, "softmax 1/128 scale matches");
  }
}

static void test_silu() {
  expect(silu_f32(0.0f) == 0.0f, "silu(0) = 0");
  const double want1 = 1.0 / (1.0 + std::exp(-1.0));  // 1 * sigmoid(1)
  expect(std::fabs(silu_f32(1.0f) - static_cast<float>(want1)) < 1e-7f, "silu(1) exact");
  expect(silu_f32(-0.5f) < 0.0f, "silu keeps sign");
  // silu_mul(x) = silu(x) * x: same sign as x^2 for x != 0, so always >= 0.
  expect(silu_mul_f32(1.0f) > 0.0f && silu_mul_f32(-0.5f) > 0.0f,
         "silu_mul non-negative");
  expect(std::fabs(silu_mul_f32(1.0f) - static_cast<float>(want1)) < 1e-7f, "silu_mul(1)");
}

static void test_gemm_dequant() {
  // A[2,2] (bf16): [[1, 2], [3, 4]]   B[2,2] (fp8): [[1, -1], [2, 0.5]]
  // C[i,j] = scale * sum_k A[i,k] * B[j,k],  scale = 2.
  std::vector<std::uint16_t> A = {f32_to_bf16_host(1.0f), f32_to_bf16_host(2.0f),
                                  f32_to_bf16_host(3.0f), f32_to_bf16_host(4.0f)};
  std::vector<std::uint8_t> B = {fp8_e4m3_encode(1.0, nullptr),
                                 fp8_e4m3_encode(-1.0, nullptr),
                                 fp8_e4m3_encode(2.0, nullptr),
                                 fp8_e4m3_encode(0.5, nullptr)};
  const float scale = 2.0f;
  std::vector<float> C;
  gemm_dequant_f32(A.data(), B.data(), scale, 2, 2, 2, &C);
  // C[0,0] = 2*(1*1 + 2*(-1)) = -2
  // C[0,1] = 2*(1*2 + 2*0.5)  =  6
  // C[1,0] = 2*(3*1 + 4*(-1)) = -2
  // C[1,1] = 2*(3*2 + 4*0.5)  = 16
  const float want[4] = {-2.0f, 6.0f, -2.0f, 16.0f};
  for (int i = 0; i < 4; ++i)
    expect(std::fabs(C[i] - want[i]) < 1e-4f, "gemm dequant element");
}

static void test_rng() {
  Xorshift64 a(123), b(123), c(456);
  for (int i = 0; i < 1000; ++i) expect(a.next() == b.next(), "rng same seed same stream");
  bool differs = false;
  for (int i = 0; i < 1000; ++i)
    if (a.next() != c.next()) {
      differs = true;
      break;
    }
  expect(differs, "rng different seeds differ");
  for (int i = 0; i < 200000; ++i) {
    const float f = a.next_float();
    if (f < 0.0f || f >= 1.0f) {
      expect(false, "rng float in [0, 1)");
      break;
    }
  }
  for (int i = 0; i < 200000; ++i) {
    const float f = a.next_float_range(2.0);
    if (f < -2.0f || f >= 2.0f) {
      expect(false, "rng range float in [-2, 2)");
      break;
    }
  }
}

int main() {
  test_bf16();
  test_rmsnorm();
  test_rope();
  test_softmax();
  test_silu();
  test_gemm_dequant();
  test_rng();
  if (g_failures == 0) std::printf("all kernel-math reference tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
