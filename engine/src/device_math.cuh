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

}  // namespace dev
}  // namespace granite4

#endif  // GRANITE4_DEVICE_MATH_CUH
