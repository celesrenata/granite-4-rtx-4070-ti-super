// Phase 2 (Kernels): portable reference math, host-only, stdlib only.
//
// Every P2 device kernel has a CPU reference implemented here so the CUDA
// test can compare device output against an exact-arithmetic reference with
// the same seed. The reference runs in float (BF16 inputs are decoded to
// float, results re-rounded to BF16) - the same arithmetic the kernels use
// internally, so a kernel that deviates from these formulas fails the test.
//
// Granite 4.2 8B conventions (see reference/granite/contract.py and the
// checkpoint config.json):
//   * RMSNorm:      y = x * rsqrt(mean(x^2) + eps) * weight
//   * RoPE:         Llama-style half split, theta = 1e7 (base), no scaling
//   * Attention:    logits = Q . K^T * attention_multiplier (1/128, NOT
//                   1/sqrt(d_head)) then softmax; the multiplier replaces
//                   the inverse-sqrt-d scaling
//   * FFN:          silu(gate) * up, then down
//   * Residuals:    hidden = hidden + residual_multiplier * block_out (1.0)
//
// Only deterministic, seedable math: the xorshift64* RNG below is the
// single source of test inputs so host and device runs match bit-for-bit.
#ifndef GRANITE4_KERNEL_MATH_H
#define GRANITE4_KERNEL_MATH_H

#include "checkpoint.h"  // Bf16ToFloat, fp8_e4m3_to_bf16

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace granite4 {

// ---------------------------------------------------------------------------
// BF16 <-> float (round-to-nearest-even).
// ---------------------------------------------------------------------------

inline float bf16_to_f32_host(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// Round float to BF16 (round-to-nearest-even on the low 16-bit group:
// add 0x7fff + ((low >> 15) & 1) so a half-tie rounds toward the even
// exponent bit).
inline std::uint16_t f32_to_bf16_host(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, sizeof u);
  u += 0x7fffu + ((u >> 16) & 1u);
  u >>= 16;
  return static_cast<std::uint16_t>(u);
}

// Decode an FP8 e4m3 byte to float (exact: every finite e4m3 value is
// representable in BF16, hence in float).
inline float fp8_to_f32_host(std::uint8_t x) {
  return bf16_to_f32_host(fp8_e4m3_to_bf16(x));
}

// ---------------------------------------------------------------------------
// RMSNorm (float reference). n is the hidden size (one row).
// ---------------------------------------------------------------------------

inline void rmsnorm_f32(const float* x, const float* weight, float eps, int n,
                        float* out) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += static_cast<double>(x[i]) * x[i];
  const float r = 1.0f / std::sqrt(static_cast<float>(s / n) + eps);
  for (int i = 0; i < n; ++i) out[i] = x[i] * r * weight[i];
}

// ---------------------------------------------------------------------------
// RoPE (Llama-style half split). head_dim d, pairs (i, i + d/2), i in 0..d/2.
// cos/sin tables are [pos, d/2], precomputed via rope_freqs_f32.
// ---------------------------------------------------------------------------

inline void rope_freqs_f32(double theta, int head_dim, int max_pos,
                           std::vector<float>* cos_table,
                           std::vector<float>* sin_table) {
  const int half = head_dim / 2;
  const int rows = max_pos;
  cos_table->assign(static_cast<std::size_t>(rows) * half, 0.0f);
  sin_table->assign(static_cast<std::size_t>(rows) * half, 0.0f);
  for (int i = 0; i < half; ++i) {
    const double inv_freq = 1.0 / std::pow(theta, 2.0 * static_cast<double>(i) / head_dim);
    for (int pos = 0; pos < max_pos; ++pos) {
      const double angle = static_cast<double>(pos) * inv_freq;
      (*cos_table)[pos * half + i] = static_cast<float>(std::cos(angle));
      (*sin_table)[pos * half + i] = static_cast<float>(std::sin(angle));
    }
  }
}

inline void rope_half_f32(const float* x, const float* cos_row, const float* sin_row,
                          int head_dim, float* out) {
  const int half = head_dim / 2;
  for (int i = 0; i < half; ++i) {
    const float a = x[i];
    const float b = x[i + half];
    const float c = cos_row[i];
    const float s = sin_row[i];
    out[i] = a * c - b * s;
    out[i + half] = b * c + a * s;
  }
}

// ---------------------------------------------------------------------------
// Softmax with a logit scale (the Granite attention multiplier).
// p = softmax(scale * x), 2-pass (max, then sum) in float.
// ---------------------------------------------------------------------------

inline void softmax_f32(const float* x, float scale, int n, float* out) {
  float m = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) m = std::max(m, scale * x[i]);
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += std::exp(static_cast<double>(scale * x[i] - m));
  for (int i = 0; i < n; ++i)
    out[i] = static_cast<float>(std::exp(static_cast<double>(scale * x[i] - m)) / sum);
}

// ---------------------------------------------------------------------------
// Silu: silu(x) = x * sigmoid(x); silu_mul(x) = silu(x) * x.
// ---------------------------------------------------------------------------

inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
inline float silu_mul_f32(float x) { return silu_f32(x) * x; }

// ---------------------------------------------------------------------------
// FP8 e4m3 x BF16 GEMM reference (the Phase-1 storage path):
//   C[M, N] = scale * (A[M, K] @ B[N, K]^T)
// A is BF16 row-major, B is FP8 e4m3 row-major (weight rows), C is BF16.
// The reference decodes each element exactly and accumulates in float, the
// same as the kernel.
// ---------------------------------------------------------------------------

inline void gemm_dequant_f32(const std::uint16_t* A, const std::uint8_t* B,
                             float scale, int M, int N, int K,
                             std::vector<float>* C_f32) {
  C_f32->assign(static_cast<std::size_t>(M) * N, 0.0f);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += fp8_to_f32_host(B[j * K + k]) * bf16_to_f32_host(A[i * K + k]);
      (*C_f32)[i * N + j] = scale * acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64*) - the only test input source, so host and
// device runs with the same seed produce identical data.
// ---------------------------------------------------------------------------

struct Xorshift64 {
  std::uint64_t s;
  explicit Xorshift64(std::uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
  std::uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s * 0x2545F4914F6CDD1Dull;
  }
  // Uniform float in [0, 1): the top 23 bits of the 64-bit state give a
  // 23-bit significand; the max is (2^23-1)*2^-23 < 1.0f, so it never
  // rounds to 1.
  float next_float() {
    const std::uint32_t u = static_cast<std::uint32_t>(next() >> 41);
    return static_cast<float>(u) * 0x1p-23f;
  }
  // Uniform float in [-a, a).
  float next_float_range(double a) {
    const float u = next_float();
    return (u * 2.0f - 1.0f) * static_cast<float>(a);
  }
};

}  // namespace granite4

#endif  // GRANITE4_KERNEL_MATH_H
