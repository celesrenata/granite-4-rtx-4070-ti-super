// Device scalar conversions shared by every CUDA translation unit.
//
// One definition of the BF16 / FP8 e4m3 conversions, so the forward, the
// verify GEMV, the benches and the kernel tests cannot drift apart.
//
//   f32_to_bf16  - round-to-nearest-even (the host twin is f32_to_bf16_host).
//   bf16_to_f32  - exact.
//   fp8_to_f32   - exact branchless e4m3 decode; the 0x7f/0xff NaN codes
//                  decode to +0.0 (the P2 reference behaviour, checked on all
//                  256 codepoints by test_kernels.cu).
//   fp8x4_to_f32 - four packed e4m3 bytes via the sm_89 hardware converter
//                  (cvt.rn.f16x2.e4m3x2, then exact f16 -> f32). Every finite
//                  e4m3 value is exact in f16, so this is bit-identical to
//                  fp8_to_f32 on all 254 finite codes; only the NaN codes
//                  differ (NaN instead of 0). The packer never emits them:
//                  fp8_e4m3_encode clips finite input to +-448 and only a NaN
//                  input maps to 0x7f. It replaces ~10 integer ops per weight
//                  with one conversion per pair, which matters because the
//                  decode GEMVs are close to integer-issue bound on Ada.
#ifndef GRANITE4_DEVICE_MATH_CUH
#define GRANITE4_DEVICE_MATH_CUH

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace granite4 {
namespace dev {

__device__ __forceinline__ std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u = __float_as_uint(f);
  u += 0x7fffu + ((u >> 16) & 1u);  // RNE
  return static_cast<std::uint16_t>(u >> 16);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

// Low / high bf16 of a packed pair.
__device__ __forceinline__ float bf16_lo(std::uint32_t w) {
  return __uint_as_float(w << 16);
}
__device__ __forceinline__ float bf16_hi(std::uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const std::uint32_t s = static_cast<std::uint32_t>(x & 0x80u) << 24;
  const std::uint32_t y = x & 0x7Fu;
  const std::uint32_t e = y >> 3;
  const std::uint32_t m = y & 7u;
  // 1 <= e <= 14: (1 + m/8) * 2^(e-7) -> f32 exponent 120 + e, mantissa m.
  std::uint32_t bits = ((120u + e) << 23) | (m << 20);
  // e == 15: (1 + m/8) * 2^8; m == 7 is the NaN code.
  const std::uint32_t top = (m == 7u) ? 0u : (0x43800000u | (m << 20));
  bits = (e == 15u) ? top : bits;
  // e == 0: subnormal m * 2^-9 (exact in f32).
  bits = (e == 0u) ? __float_as_uint(static_cast<float>(m) * 0.001953125f) : bits;
  // NaN codes decode to +0.0 (sign dropped), the P2 reference behaviour.
  return (e == 15u && m == 7u) ? 0.0f : __uint_as_float(bits | s);
}

// Two e4m3 bytes (low byte first) -> two floats.
__device__ __forceinline__ float2 fp8x2_to_f32(std::uint16_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3);
  return make_float2(__half2float(__half(__half_raw{h.x})),
                     __half2float(__half(__half_raw{h.y})));
}

// Four e4m3 bytes of a little-endian word: byte j -> element j.
__device__ __forceinline__ void fp8x4_to_f32(std::uint32_t w, float& b0, float& b1,
                                             float& b2, float& b3) {
  const float2 lo = fp8x2_to_f32(static_cast<std::uint16_t>(w & 0xffffu));
  const float2 hi = fp8x2_to_f32(static_cast<std::uint16_t>(w >> 16));
  b0 = lo.x;
  b1 = lo.y;
  b2 = hi.x;
  b3 = hi.y;
}

// ---------------------------------------------------------------------------
// KV8 (Q8_0) int8 KV cache math.
//
// The persistent KV cache (and the EAGLE head's own cache) is stored int8:
// each (head, position) row of d elements shares one float scale, s, chosen
// from the row's absolute maximum so the largest magnitude maps to the
// largest int8 code (127). Quantize: q_i = clamp(round(x_i / s), -127, 127).
// Dequant: x_i ~= q_i * s.
//
// Round-trip error is bounded by half a quantization step, |x_i - q_i*s| <=
// s/2, so the worst-case relative error is (1/127)/2 ~= 0.4% ~= 2^-8. That
// is below BF16 precision but enough to move a 2-ULP teacher-forced parity
// gate, so KV8 is opt-in (G4_KV8) and the BF16 cache stays the default the
// production gate runs on.
//
// The device and host twins below stay bit-identical: IEEE RNE division
// (x / s), RNE round-to-even for the int code, then clamp to [-127, 127].
// A zero row keeps s == 0 and quantizes to all-zero codes (dequant exactly
// 0.0; no NaN from dividing by zero).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float kv8_scale(float amax) {
  // s = amax / 127 (IEEE RNE division). amax is a max of absolute values.
  return amax / 127.0f;
}

// Round-half-away-from-zero to a long, bit-identical to the host twin's
// std::round(float). The IEEE division feeding it is correctly rounded on
// both sides, so the code matches. RNE (__float2ll_rn) and away-from-zero
// differ only at exact +/-0.5 fractions, where RNE keeps the even side; the
// fmodf correction catches exactly those. (floorf(x + 0.5f) would NOT
// match: the float addition itself RNE-rounds, e.g. 0.5 - 2^-25 + 0.5f ==
// 1.0f.)
__device__ __forceinline__ long kv8_round_away(float r) {
  long i = __float2ll_rn(r);
  const float fr = fmodf(r, 1.0f);  // exact fractional part for finite r
  if (fr == 0.5f) i = static_cast<long>(ceilf(r));
  else if (fr == -0.5f) i = static_cast<long>(floorf(r));
  return i;
}

__device__ __forceinline__ std::int8_t kv8_quant(float x, float scale) {
  if (scale == 0.0f) return 0;
  const long r = kv8_round_away(x / scale);
  const long clamped = r > 127 ? 127 : (r < -127 ? -127 : r);
  return static_cast<std::int8_t>(clamped);
}

__device__ __forceinline__ float kv8_dequant(std::int8_t q, float scale) {
  return scale * static_cast<float>(q);
}

// ---------------------------------------------------------------------------
// int4 KV (per-row scale, 2 codes/byte).
//
// Q4_0-style: s = amax / 7 (IEEE RNE); code = clamp(round_away(x / s), -7, 7)
// + 8, so it lives in 0..15 and packs two per byte (lower nibble = the even
// dim, upper nibble = the odd dim). Dequant = s * (code - 8). A (head,position)
// row is d/2 code bytes + one float scale: for d == 128 that is 64 B + 4 B,
// 4x the density of the int8 cache. Rounding reuses kv8_round_away so the
// device/host twins stay bit-identical (the same guarantee as int8).
// ---------------------------------------------------------------------------
__device__ __forceinline__ float q4_scale(float amax) {
  // s = amax / 7 (IEEE RNE division). amax is a max of absolute values.
  return amax / 7.0f;
}

// Quantize one value to a 0..15 code (caller places it in a byte nibble).
// Rounding follows the int8 math (half away from zero) so the device/host
// twins stay bit-identical; scale == 0 -> 8 (zero).
__device__ __forceinline__ unsigned char q4_quant(float x, float scale) {
  if (scale == 0.0f) return 8;
  const long r = kv8_round_away(x / scale);
  const long clamped = r > 7 ? 7 : (r < -7 ? -7 : r);
  return static_cast<unsigned char>(clamped + 8);
}

// Unpack one byte into the two dequantized values (even dim, odd dim).
__device__ __forceinline__ void q4_dequant_pair(unsigned char byte, float scale, float& lo,
                                               float& hi) {
  const float l = static_cast<float>(static_cast<int>(byte & 0x0fu) - 8);
  const float h = static_cast<float>(static_cast<int>((byte >> 4) & 0x0fu) - 8);
  lo = scale * l;
  hi = scale * h;
}

// ---------------------------------------------------------------------------
// K5 (5-bit K cache), device twin of k5_*_host in kernel_math.h (design: docs/kv-k5v4.md).
//
// s = amax / 15; code = clamp(round_away(x / s), -15, 15) + 16 (1..31); dequant = s * (code - 16).
// A (head, position) K row is 80 bytes: bytes [0, 64) hold the LOW 4 bits of each code exactly
// like int4 (dim 2j in the low nibble, 2j+1 in the high nibble); bytes [64, 80) hold the HIGH
// bit of each code, bit (d % 8) of byte 64 + d / 8. V stays int4 (q4_*).
// ---------------------------------------------------------------------------
constexpr int kK5RowBytes = 80;
constexpr int kK5NibbleBytes = 64;

__device__ __forceinline__ float k5_scale(float amax) {
  return amax / 15.0f;
}

// Quantize one value to a 0..31 code; scale == 0 -> 16 (zero).
__device__ __forceinline__ unsigned char k5_quant(float x, float scale) {
  if (scale == 0.0f) return 16;
  const long r = kv8_round_away(x / scale);
  const long clamped = r > 15 ? 15 : (r < -15 ? -15 : r);
  return static_cast<unsigned char>(clamped + 16);
}

// Dequantize the 5-bit codes of dims (2j, 2j+1) from one nibble byte and their two high bits.
__device__ __forceinline__ void k5_dequant_pair(unsigned char nibbles, unsigned hi_even,
                                               unsigned hi_odd, float scale, float& lo, float& hi) {
  const int c0 = static_cast<int>(nibbles & 0x0fu) | static_cast<int>((hi_even & 1u) << 4);
  const int c1 = static_cast<int>((nibbles >> 4) & 0x0fu) | static_cast<int>((hi_odd & 1u) << 4);
  lo = scale * static_cast<float>(c0 - 16);
  hi = scale * static_cast<float>(c1 - 16);
}

// Writes the 16-byte high-bit plane of one K5 row. The caller has stored each dim's code >> 4 in
// hi[0..127] and synchronized the block; thread w < 16 packs dims [8w, 8w + 8) into byte w
// (bit d % 8 of byte d / 8), matching k5_pack_row_host.
__device__ __forceinline__ void k5_write_plane(const unsigned char* hi, unsigned char* row, int w) {
  unsigned plane = 0;
#pragma unroll
  for (int bit = 0; bit < 8; ++bit) plane |= static_cast<unsigned>(hi[8 * w + bit] & 1u) << bit;
  row[kK5NibbleBytes + w] = static_cast<unsigned char>(plane);
}

}  // namespace dev
}  // namespace granite4

#endif  // GRANITE4_DEVICE_MATH_CUH
