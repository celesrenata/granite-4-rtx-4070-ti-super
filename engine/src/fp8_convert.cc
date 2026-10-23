// FP8 e4m3 / BF16 conversions for the Phase 1 quantization pass.
//
// All rounding is round-to-nearest-even in exact integer arithmetic on the
// input's binary significand - no intermediate float arithmetic - so the
// encoder is deterministic and exact for values the format can represent.
//
// e4m3 (OCP microscaling) layout: S | EEEE | MMM (sign, 4-bit exponent bias
// 7, 3-bit mantissa):
//   subnormals (E=0):    m * 2^-9,             m in 1..7
//   normals (E in 1..14): (1 + m/8) * 2^(E-7),  m in 0..7
//   top (E=15):          (1 + m/8) * 2^8,      m in 0..6   (m=7 is sNaN 0x7F)
//   max finite:          448                   (E=15, m=6 -> 0x7E)
#include "checkpoint.h"

#include <cmath>
#include <cstring>

namespace granite4 {

namespace {

uint64_t double_bits(double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof b);
  return b;
}

// Round RNE: num / 2^den, num >= 0, den in [0, 62].
int64_t round_rne(int64_t num, int den) {
  const int64_t scale = den == 0 ? 1 : (int64_t(1) << den);
  int64_t whole = num / scale;
  const int64_t rem = num - whole * scale;  // rem in [0, scale)
  if (2 * rem > scale) ++whole;
  else if (2 * rem == scale && (whole & 1) != 0) ++whole;
  return whole;
}

}  // namespace

float Bf16ToFloat::from_bits(std::uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

std::uint8_t fp8_e4m3_encode(double value, bool* saturated) {
  if (saturated) *saturated = false;
  if (value != value) {
    if (saturated) *saturated = true;
    return kFp8E4M3SNaN;  // NaN -> sNaN (0x7F)
  }
  // signbit rather than (value < 0.0): -0.0 must keep its sign (encodes to
  // -0), and signbit is well defined for NaN, which we handle below.
  const bool negative = std::signbit(value);
  const double ad = negative ? -value : value;
  if (ad == 0.0) return negative ? static_cast<std::uint8_t>(0x80u) : 0x00u;
  if (std::isinf(ad)) {
    if (saturated) *saturated = true;
    return negative ? static_cast<std::uint8_t>(0x7Eu | 0x80u) : 0x7Eu;
  }
  // Clip at the midpoint between 448 and the (reserved) 480 grid point:
  // |value| >= 464 saturates at +-448.
  if (ad >= 464.0) {
    if (saturated) *saturated = true;
    return negative ? static_cast<std::uint8_t>(0x7Eu | 0x80u) : 0x7Eu;
  }

  const uint64_t bits = double_bits(ad);
  const uint64_t exp_field = (bits >> 52) & 0x7FFu;
  if (exp_field == 0) {
    // Subnormal double: value < 2^-1022 << smallest e4m3 subnormal (2^-9).
    return negative ? static_cast<std::uint8_t>(0x80u) : 0x00u;
  }
  const uint64_t M = (1ull << 52) | (bits & 0x000FFFFFFFFFFFFFull);  // 53-bit
  const int e = static_cast<int>(exp_field) - 1023;  // value = M * 2^(e-52)

  // Since ad < 464 < 2^9, e <= 8.
  if (e <= -7) {
    // Subnormal region: value in (0, 2^-6). m = RNE(value / 2^-9) =
    // RNE(M / 2^(43-e)); e <= -11 gives value < 2^-10 -> 0.
    if (e <= -11) return negative ? static_cast<std::uint8_t>(0x80u) : 0x00u;
    const int64_t m = round_rne(static_cast<int64_t>(M), 43 - e);  // den 50..53
    if (m <= 0) return negative ? static_cast<std::uint8_t>(0x80u) : 0x00u;
    if (m >= 8) {
      // Rounded up to 2^-6, the first normal (E=1, m=0).
      std::uint8_t out = 0x08u;
      if (negative) out |= 0x80u;
      return out;
    }
    std::uint8_t out = static_cast<std::uint8_t>(m & 0x7u);
    if (negative) out |= 0x80u;
    return out;
  }
  // Normal region: value in [2^-6, 464), e in [-6, 8]. The decade is forced
  // by the exponent: value in [2^(E-7), 2^(E-6)) (or the top decade
  // [256, 464)) with E = e + 7, and value in [2^e, 2^(e+1)).
  const int E = e + 7;  // 1..15
  // m = RNE(fraction of value / 2^(E-7) scaled by 8) = RNE(F / 2^49) where
  // F = M - 2^52 is the 52-bit significand fraction (value/2^(E-7) =
  // M / 2^52 = 1 + F/2^52).
  const int64_t F = static_cast<int64_t>(M) - (1ll << 52);  // in [0, 2^52)
  int64_t m = round_rne(F, 49);
  if (m < 0) return negative ? static_cast<std::uint8_t>(0x80u) : 0x00u;
  if (m >= 8) {
    if (E >= 15) {
      // 0x7F is the sNaN slot: unreachable given the 464 clip, but never
      // emit it - clip to 448 instead.
      if (saturated) *saturated = true;
      return negative ? static_cast<std::uint8_t>(0x7Eu | 0x80u) : 0x7Eu;
    }
    // Carried into the first grid point of the next decade.
    std::uint8_t out = static_cast<std::uint8_t>((static_cast<uint32_t>(E + 1) & 0x0Fu) << 3);
    if (negative) out |= 0x80u;
    return out;
  }
  std::uint8_t out =
      static_cast<std::uint8_t>(((static_cast<uint32_t>(E) & 0x0Fu) << 3) |
                                (static_cast<uint32_t>(m) & 0x7u));
  if (negative) out |= 0x80u;
  return out;
}

std::uint8_t bf16_to_fp8_e4m3(std::uint16_t bits, bool* saturated) {
  return fp8_e4m3_encode(Bf16ToFloat::from_bits(bits), saturated);
}

std::uint16_t fp8_e4m3_to_bf16(std::uint8_t x) {
  const bool negative = (x & 0x80u) != 0;
  const uint32_t e = (x >> 3) & 0x0Fu;
  const uint32_t m = x & 0x7u;
  auto pack = [&](int e16, uint32_t frac7) {
    return static_cast<std::uint16_t>((negative ? 0x8000u : 0u) |
                                      (static_cast<uint32_t>(e16) << 7) | (frac7 & 0x7Fu));
  };
  if (e == 0x0Fu) {
    if (m == 0x7u) return static_cast<std::uint16_t>(negative ? 0xFF80u : 0x7F80u);  // sNaN -> qNaN
    // Top: value = (1 + m/8) * 2^8, m in 0..6.
    return pack(135, m << 4);
  }
  if (e == 0) {
    if (m == 0) return negative ? 0x8000u : 0x0000u;  // +-0
    // value = m * 2^-9, m in 1..7. Normalize: 2^k <= m < 2^(k+1), k in 0..2
    // (m <= 7 = 111b). mf = m * 2^k in 4..7, so
    // value = (mf / 4) * 2^(2 - k - 9) = (mf / 4) * 2^(-k - 7);
    // e16 = -k - 7 + 127, frac7 = (mf - 4) << 5.
    uint32_t mf = m;
    int k = 0;
    while (k < 2 && (mf & 0x4u) == 0) {
      mf <<= 1;
      ++k;
    }
    return pack(120 - k, (mf & 0x3u) << 5);
  }
  // normal: (1 + m/8) * 2^(E - 7), E in 1..14.
  return pack(e - 7 + 127, m << 4);
}

}  // namespace granite4
