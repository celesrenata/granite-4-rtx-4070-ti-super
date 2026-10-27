// Correctness test for the contained experimental batched verify GEMV API.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" int granite4_verify_gemv(const std::uint16_t*, const std::uint8_t*, float,
                                     int, int, int, std::uint16_t*);
static float bf(std::uint16_t x) { std::uint32_t u = static_cast<std::uint32_t>(x) << 16; float f; __builtin_memcpy(&f, &u, 4); return f; }
static float fp8(std::uint8_t x) {
  const int e = (x & 0x7f) >> 3, m = x & 7;
  if (e == 15 && m == 7) return 0;
  float v = e ? std::ldexp(1.0f + m / 8.0f, e - 7) : std::ldexp(m, -9);
  return (x & 128) ? -v : v;
}
static std::uint16_t bf16(float x) {
  std::uint32_t u; __builtin_memcpy(&u, &x, 4);
  return static_cast<std::uint16_t>((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
int main() {
  constexpr int T = 4, R = 9, K = 256; constexpr float scale = 0.375f;
  std::vector<std::uint16_t> a(T*K), out(T*R), got(T*R);
  std::vector<std::uint8_t> b(R*K);
  for (int i = 0; i < T*K; ++i) a[i] = bf16(std::sin(i * .071f) * .5f);
  for (int i = 0; i < R*K; ++i) b[i] = static_cast<std::uint8_t>((i * 37 + 11) & 0x7f);
  std::vector<std::uint16_t> ref(T*R);
  for (int t = 0; t < T; ++t) for (int r = 0; r < R; ++r) {
    float sum = 0;
    for (int k = 0; k < K; ++k) sum = std::fmaf(bf(a[t*K+k]), fp8(b[r*K+k]), sum);
    ref[t*R+r] = bf16(sum * scale);
  }
  std::uint16_t *da, *dc; std::uint8_t *db;
  if (cudaMalloc(&da, a.size()*2) || cudaMalloc(&db, b.size()) || cudaMalloc(&dc, out.size()*2)) return 2;
  cudaMemcpy(da, a.data(), a.size()*2, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  if (granite4_verify_gemv(da, db, scale, T, R, K, dc) || cudaDeviceSynchronize()) return 3;
  cudaMemcpy(got.data(), dc, got.size()*2, cudaMemcpyDeviceToHost);
  int errors = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const int delta = static_cast<int>(got[i]) - static_cast<int>(ref[i]);
    if (delta < -1 || delta > 1) ++errors;
  }
  cudaFree(da); cudaFree(db); cudaFree(dc);
  if (errors) { std::fprintf(stderr, "verify_gemv mismatches: %d\n", errors); return 1; }
  std::puts("verify_gemv: PASS (4 tokens x 9 rows x 256 K)"); return 0;
}
