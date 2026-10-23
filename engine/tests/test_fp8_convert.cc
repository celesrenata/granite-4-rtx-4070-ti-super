// Host-only tests for the FP8 e4m3 <-> BF16 conversion pass.
//
// The encoder is checked against an independent reference: the exact 255-point
// e4m3 magnitude grid with round-to-nearest-even resolved by exact distance
// comparison (Sterbenz-safe), plus the 464 saturation clip and the special
// codes. The decoder is checked for exactness (every finite e4m3 value is
// exactly representable in BF16) and round-trip (encode(decode(code)) == code
// for all finite codes).
//
// No CUDA required; run via ctest after a host-only build.
#include "checkpoint.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using granite4::Bf16ToFloat;
using granite4::fp8_e4m3_encode;
using granite4::fp8_e4m3_to_bf16;

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

float bf16f(std::uint16_t b) { return Bf16ToFloat::from_bits(b); }

// ---------------------------------------------------------------------------
// Reference: exact e4m3 grid + nearest-even encoder.
// ---------------------------------------------------------------------------

double code_value(std::uint8_t code) {
  const int e = (code >> 3) & 0xF;
  const int m = code & 0x7;
  if (e == 0xF) {
    if (m == 7) return std::nan("");  // unreachable: callers skip 0x7F
    return (1.0 + m / 8.0) * std::ldexp(1.0, 8);
  }
  if (e == 0) {
    if (m == 0) return 0.0;
    return m * std::ldexp(1.0, -9);
  }
  return (1.0 + m / 8.0) * std::ldexp(1.0, e - 7);
}

struct GridEntry {
  double v;
  std::uint8_t code;
};

GridEntry g_grid[255];
int g_grid_n = 0;

void build_grid() {
  if (g_grid_n) return;
  for (std::uint8_t code = 0; code < 0x80; ++code) {
    if (code == 0x7F) continue;
    g_grid[g_grid_n++] = {code_value(code), code};
  }
  for (int i = 1; i < g_grid_n; ++i) {  // insertion sort: already nearly sorted
    for (int j = i; j > 0 && g_grid[j].v < g_grid[j - 1].v; --j) {
      const GridEntry t = g_grid[j];
      g_grid[j] = g_grid[j - 1];
      g_grid[j - 1] = t;
    }
  }
}

std::uint8_t ref_encode_magnitude(double a) {
  // a >= 0, a < 464 (saturation handled by caller)
  int lo = 0, hi = g_grid_n;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (g_grid[mid].v < a) lo = mid + 1;
    else hi = mid;
  }
  int best = -1;
  double bd = 0.0;
  for (int c = lo - 1; c <= lo && c < g_grid_n; ++c) {
    if (c < 0) continue;
    const double d = g_grid[c].v >= a ? g_grid[c].v - a : a - g_grid[c].v;
    if (best < 0 || d < bd) {
      best = c;
      bd = d;
    } else if (d == bd && (g_grid[c].code & 0x1) == 0 && (g_grid[best].code & 0x1)) {
      best = c;  // exact tie -> even code
    }
  }
  return g_grid[best].code;
}

std::uint8_t ref_encode(double v) {
  if (v != v) return 0x7F;
  const bool neg = std::signbit(v);  // -0.0 keeps its sign (encodes to 0x80)
  const double a = neg ? -v : v;
  if (std::isinf(a) || a >= 464.0) return neg ? 0xFE : 0x7E;
  std::uint8_t code = ref_encode_magnitude(a);
  return neg ? static_cast<std::uint8_t>(code | 0x80) : code;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Exhaustive BF16 sweep: all 65536 codes, encoder vs reference.
// ---------------------------------------------------------------------------
static void test_bf16_sweep() {
  build_grid();
  int fails = 0;
  for (int b = 0; b < 0x10000; ++b) {
    const double f = static_cast<double>(bf16f(b));
    const std::uint8_t my = fp8_e4m3_encode(f, nullptr);
    const std::uint8_t ref = ref_encode(f);
    if (my != ref) {
      ++fails;
      if (fails <= 10)
        std::printf("FAIL sweep: bf16=%#05x f=%g enc=%#04x ref=%#04x\n",
                    static_cast<unsigned>(b), f, my, ref);
    }
  }
  expect(fails == 0, "bf16 exhaustive sweep: encoder == reference (65536)");
}

// ---------------------------------------------------------------------------
// 2. Random doubles (biased to subnormal / edge magnitudes) vs reference.
// ---------------------------------------------------------------------------
static void test_random_doubles() {
  build_grid();
  std::mt19937_64 rng(0xC0FFEE);
  int fails = 0;
  const int kN = 300000;
  for (int i = 0; i < kN; ++i) {
    double v;
    const int r = static_cast<int>(rng() & 0x7);
    if (r == 0) {
      v = static_cast<double>(rng()) * std::ldexp(1.0, static_cast<int>(rng() % 24) - 24);
    } else if (r == 1) {
      // Random bit pattern with a low exponent -> subnormal / tiny region.
      std::uint64_t bits = (1ull << 52) | (rng() & 0x000FFFFFFFFFFFFFull);
      bits |= static_cast<std::uint64_t>(4 + (rng() % 12)) << 52;  // exp 4..15
      double t;
      std::memcpy(&t, &bits, sizeof t);
      v = t;
    } else if (r == 2) {
      v = std::ldexp(1.0, static_cast<int>(rng() % 24) - 12);  // powers of two
    } else {
      v = static_cast<double>(rng()) * std::ldexp(1.0, static_cast<int>(rng() % 26) - 12);
    }
    if (rng() & 0x7FF) v = -v;
    const std::uint8_t my = fp8_e4m3_encode(v, nullptr);
    const std::uint8_t ref = ref_encode(v);
    if (my != ref) {
      ++fails;
      if (fails <= 10)
        std::printf("FAIL rand: v=%a enc=%#04x ref=%#04x\n", v, my, ref);
    }
  }
  expect(fails == 0, "random doubles: encoder == reference (300000)");
}

// ---------------------------------------------------------------------------
// 3. Exact midpoints between adjacent grid points: tie -> even code.
// ---------------------------------------------------------------------------
static void test_ties_to_even() {
  build_grid();
  int fails = 0;
  for (int i = 0; i + 1 < g_grid_n; ++i) {
    const double mid = g_grid[i].v + (g_grid[i + 1].v - g_grid[i].v) / 2.0;
    const std::uint8_t ref = ref_encode(mid);
    expect((ref & 0x1) == 0, "tie reference is even");  // sanity of reference
    for (const double s : {mid, -mid}) {
      const std::uint8_t my = fp8_e4m3_encode(s, nullptr);
      const std::uint8_t want = ref_encode(s);
      if (my != want) {
        ++fails;
        if (fails <= 10)
          std::printf("FAIL tie: v=%a enc=%#04x ref=%#04x\n", s, my, want);
      }
    }
  }
  expect(fails == 0, "midpoint ties: encoder == reference (even code)");
}

// ---------------------------------------------------------------------------
// 4. Special values + saturation flag.
// ---------------------------------------------------------------------------
static void test_special_and_saturation() {
  const double nan_v = std::nan("");
  const double inf_v = std::numeric_limits<double>::infinity();
  struct Case {
    double v;
    std::uint8_t want;
    bool want_sat;
  };
  const Case cases[] = {
      {0.0, 0x00, false},
      {-0.0, 0x80, false},
      {448.0, 0x7E, false},
      {-448.0, 0xFE, false},
      {464.0, 0x7E, true},          // exact clip boundary
      {-464.0, 0xFE, true},
      {463.9999999999999, 0x7E, false},  // just inside
      {inf_v, 0x7E, true},
      {-inf_v, 0xFE, true},
      {nan_v, 0x7F, true},
      {-nan_v, 0x7F, true},
      {std::ldexp(1.0, -9), 0x01, false},          // min subnormal, exact
      {2.0 * std::ldexp(1.0, -9), 0x02, false},
      {std::ldexp(1.0, -6), 0x08, false},          // min normal, exact
      {std::ldexp(1.0, -7) * 1.5, 0x06, false},  // = 6*2^-9 exactly
  };
  for (const Case& c : cases) {
    bool sat = false;
    const std::uint8_t got = fp8_e4m3_encode(c.v, &sat);
    char msg[128];
    std::snprintf(msg, sizeof msg, "encode(%a) == %#04x", c.v, c.want);
    expect(got == c.want, msg);
    std::snprintf(msg, sizeof msg, "saturation flag for %a", c.v);
    expect(sat == c.want_sat, msg);
  }
  // The 7-bit magnitude code of 448 is the max finite value; values in
  // (448, 464) round to 448 without the saturation flag.
  bool sat = true;
  const std::uint8_t code = fp8_e4m3_encode(450.0, &sat);
  expect(code == 0x7E && !sat, "450 rounds to 448 without saturation");
}

// ---------------------------------------------------------------------------
// 5. Decoder exactness: every finite code decodes to the exact grid value;
//    special codes map to the documented BF16 bits.
// ---------------------------------------------------------------------------
static void test_decoder_exact() {
  int fails = 0;
  for (std::uint8_t code = 0; code < 0x80; ++code) {
    if (code == 0x7F) continue;
    const double want = code_value(code);
    const float f = bf16f(fp8_e4m3_to_bf16(code));
    if (want == 0.0) {
      if (f != 0.0) {
        ++fails;
        if (fails <= 10)
          std::printf("FAIL dec: code=%#04x want 0 got %a\n", code,
                      static_cast<double>(f));
      }
    } else if (static_cast<double>(f) != want) {
      ++fails;
      if (fails <= 10)
        std::printf("FAIL dec: code=%#04x want %a got %a\n", code, want,
                    static_cast<double>(f));
    }
  }
  // Signed mirrors: neg = pos with sign bit (0x8000) set, magnitude equal.
  for (std::uint8_t code = 0x01; code < 0x80; ++code) {
    if (code == 0x7F) continue;
    const std::uint16_t p = fp8_e4m3_to_bf16(code);
    const std::uint16_t n = fp8_e4m3_to_bf16(static_cast<std::uint8_t>(code | 0x80));
    if ((p & 0x8000) != 0 || (n & 0x8000) == 0 || (p & 0x7FFF) != (n & 0x7FFF)) {
      ++fails;
      if (fails <= 10)
        std::printf("FAIL dec sign: code=%#04x pos=%#05x neg=%#05x\n", code, p, n);
    }
  }
  expect(fails == 0, "decoder: every finite code is exact, signs consistent");
  // Special codes.
  expect(fp8_e4m3_to_bf16(0x7F) == 0x7F80, "0x7F (sNaN) decodes to BF16 qNaN 0x7F80");
  expect(fp8_e4m3_to_bf16(0xFF) == 0xFF80, "0xFF (-sNaN) decodes to BF16 qNaN 0xFF80");
  expect(bf16f(fp8_e4m3_to_bf16(0x7E)) == 448.0f, "0x7E decodes to +448");
  expect(bf16f(fp8_e4m3_to_bf16(0xFE)) == -448.0f, "0xFE decodes to -448");
  expect(fp8_e4m3_to_bf16(0x00) == 0x0000, "0x00 decodes to +0");
  expect(fp8_e4m3_to_bf16(0x80) == 0x8000, "0x80 decodes to -0");
}

// ---------------------------------------------------------------------------
// 6. Round-trip: encode(decode(code)) == code for all finite codes.
// ---------------------------------------------------------------------------
static void test_round_trip() {
  int fails = 0;
  for (int code = 0; code < 0x100; ++code) {
    if ((code & 0x7F) == 0x7F) continue;  // NaN codes do not round-trip
    const double want_val = code & 0x80 ? -code_value(code & 0x7F)
                                        : code_value(code & 0x7F);
    const float f = bf16f(fp8_e4m3_to_bf16(code));
    const std::uint8_t re = fp8_e4m3_encode(static_cast<double>(f), nullptr);
    if (re != code) {
      ++fails;
      if (fails <= 10)
        std::printf("FAIL rt: code=%#04x val=%a -> re=%#04x\n", code, want_val, re);
    }
  }
  expect(fails == 0, "round-trip: encode(decode(code)) == code (254 finite)");
}

int main() {
  test_bf16_sweep();
  test_random_doubles();
  test_ties_to_even();
  test_special_and_saturation();
  test_decoder_exact();
  test_round_trip();
  if (g_failures == 0) std::printf("all fp8 tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
