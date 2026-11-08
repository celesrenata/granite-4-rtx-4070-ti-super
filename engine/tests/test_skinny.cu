// Skinny tensor-core GEMM (skinny_gemm.cu): correctness, row invariance, speed.
//
//   granite4-skinny-test            correctness + T-invariance (exit 1 on failure)
//   granite4-skinny-test --bench    + timing of a model-shaped layer per T through
//                                   the granite4_verify_* entry points (run with
//                                   G4_SKINNY=0 / 1 for the A/B)
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "checkpoint.h"   // fp8_e4m3_encode
#include "kernel_math.h"  // fp8_to_f32_host

extern "C" int granite4_skinny_gemv(const std::uint16_t*, const std::uint8_t*, float, int, int,
                                    int, std::uint16_t*);
extern "C" int granite4_skinny_gemv_resid(const std::uint16_t*, const std::uint8_t*, float,
                                          const std::uint16_t*, float, int, int, int,
                                          std::uint16_t*);
extern "C" int granite4_skinny_qkv(const std::uint16_t*, const std::uint8_t*,
                                   const std::uint8_t*, const std::uint8_t*, float, float, float,
                                   int, int, int, int, int, std::uint16_t*, std::uint16_t*,
                                   std::uint16_t*);
extern "C" int granite4_skinny_silu(const std::uint8_t*, const std::uint8_t*,
                                    const std::uint16_t*, float, float, int, int, int,
                                    std::uint16_t*);
extern "C" int granite4_verify_gemv(const std::uint16_t*, const std::uint8_t*, float, int, int,
                                    int, std::uint16_t*);
extern "C" int granite4_verify_silu(const std::uint8_t*, const std::uint8_t*,
                                    const std::uint16_t*, float, float, int, int, int,
                                    std::uint16_t*);
extern "C" int granite4_verify_skinny_enabled();
extern "C" int granite4_verify_qkv(const std::uint16_t*, const std::uint8_t*, const std::uint8_t*,
                                   const std::uint8_t*, float, float, float, int, int, int, int,
                                   int, std::uint16_t*, std::uint16_t*, std::uint16_t*);

namespace {
int g_fail = 0;
void expect(bool c, const char* what) {
  if (!c) {
    std::printf("FAIL: %s\n", what);
    ++g_fail;
  }
}
#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      std::printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      std::exit(2);                                                              \
    }                                                                            \
  } while (0)
std::uint64_t rng = 0x2545F4914F6CDD1DULL;
double urand() {
  rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
  return static_cast<double>(rng >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}
float bf(std::uint16_t h) {
  std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
std::uint16_t tobf(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<std::uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
float f8(std::uint8_t b) { return granite4::fp8_to_f32_host(b); }

struct Mat {
  int n, k;
  std::vector<std::uint8_t> h;
  std::uint8_t* d = nullptr;
  Mat(int n_, int k_) : n(n_), k(k_), h(static_cast<std::size_t>(n_) * k_) {
    for (auto& b : h) b = granite4::fp8_e4m3_encode(urand() * 2.0, nullptr);
    CK(cudaMalloc(&d, h.size()));
    CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice));
  }
  ~Mat() { cudaFree(d); }
  double dot(int row, const std::vector<std::uint16_t>& a, int t) const {
    double s = 0;
    for (int i = 0; i < k; ++i)
      s += static_cast<double>(f8(h[static_cast<std::size_t>(row) * k + i])) *
           bf(a[static_cast<std::size_t>(t) * k + i]);
    return s;
  }
};
struct Act {
  int t, k;
  std::vector<std::uint16_t> h;
  std::uint16_t* d = nullptr;
  Act(int t_, int k_) : t(t_), k(k_), h(static_cast<std::size_t>(t_) * k_) {
    for (auto& x : h) x = tobf(static_cast<float>(urand()));
    CK(cudaMalloc(&d, h.size() * 2));
    CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  }
  ~Act() { cudaFree(d); }
};
std::vector<std::uint16_t> fetch(const std::uint16_t* d, std::size_t n) {
  std::vector<std::uint16_t> h(n);
  CK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
  return h;
}
// |got - want| within a few bf16 steps of the larger magnitude.
bool close(float got, double want, double mag) {
  return std::fabs(got - want) <= 0.02 * std::fabs(want) + 4e-3 * mag + 1e-6;
}

void test_out_and_invariance() {
  const int N = 320, K = 4096;
  Mat w(N, K);
  for (int T : {1, 2, 7, 14, 16}) {
    Act a(T, K);
    std::uint16_t *out, *one;
    CK(cudaMalloc(&out, static_cast<std::size_t>(T) * N * 2));
    CK(cudaMalloc(&one, static_cast<std::size_t>(N) * 2));
    expect(granite4_skinny_gemv(a.d, w.d, 0.01f, T, N, K, out) == 0, "skinny launch");
    CK(cudaDeviceSynchronize());
    const auto got = fetch(out, static_cast<std::size_t>(T) * N);
    bool ok = true, inv = true;
    for (int t = 0; t < T; ++t) {
      for (int n = 0; n < N; ++n) {
        const double want = 0.01 * w.dot(n, a.h, t);
        ok = ok && close(bf(got[static_cast<std::size_t>(t) * N + n]), want, 0.01 * std::sqrt(K));
      }
      // Row t alone must give the same bits as row t of the T-row call.
      expect(granite4_skinny_gemv(a.d + static_cast<std::size_t>(t) * K, w.d, 0.01f, 1, N, K,
                                  one) == 0,
             "skinny single-row launch");
      CK(cudaDeviceSynchronize());
      const auto g1 = fetch(one, N);
      inv = inv && std::memcmp(g1.data(), &got[static_cast<std::size_t>(t) * N], N * 2) == 0;
    }
    expect(ok, "skinny out matches the fp64 reference (bf16 tolerance)");
    expect(inv, "skinny T-row call == T single-row calls (bit-identical)");
    cudaFree(out);
    cudaFree(one);
  }
}

void test_modes() {
  const int K = 12800, N = 256, T = 7;
  Mat g(N, K), u(N, K);
  Act a(T, K);
  std::uint16_t* out;
  CK(cudaMalloc(&out, static_cast<std::size_t>(T) * N * 2));
  // silu
  expect(granite4_skinny_silu(g.d, u.d, a.d, 0.02f, 0.03f, N, K, T, out) == 0, "silu launch");
  CK(cudaDeviceSynchronize());
  auto got = fetch(out, static_cast<std::size_t>(T) * N);
  bool ok = true;
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < N; ++n) {
      const double gs = 0.02 * g.dot(n, a.h, t), us = 0.03 * u.dot(n, a.h, t);
      const double want = gs / (1.0 + std::exp(-gs)) * us;
      ok = ok && close(bf(got[static_cast<std::size_t>(t) * N + n]), want,
                       std::fabs(gs) * 0.03 * std::sqrt(K) + 0.02 * std::sqrt(K) * std::fabs(us));
    }
  expect(ok, "skinny silu matches the fp64 reference");
  // resid (in place)
  std::vector<std::uint16_t> res(static_cast<std::size_t>(T) * N);
  for (auto& x : res) x = tobf(static_cast<float>(urand() * 4.0));
  CK(cudaMemcpy(out, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
  expect(granite4_skinny_gemv_resid(a.d, g.d, 0.01f, out, 0.5f, T, N, K, out) == 0,
         "resid launch");
  CK(cudaDeviceSynchronize());
  got = fetch(out, res.size());
  ok = true;
  for (int t = 0; t < T; ++t)
    for (int n = 0; n < N; ++n) {
      const std::size_t i = static_cast<std::size_t>(t) * N + n;
      const double want = bf(res[i]) + 0.5 * 0.01 * g.dot(n, a.h, t);
      ok = ok && close(bf(got[i]), want, 0.01 * std::sqrt(K) + 4.0);
    }
  expect(ok, "skinny resid (in place) matches the fp64 reference");
  cudaFree(out);
  // qkv
  const int H = 512, nq = 512, nk = 128;
  Mat wq(nq, H), wk(nk, H), wv(nk, H);
  Act x(T, H);
  std::uint16_t *oq, *ok_, *ov;
  CK(cudaMalloc(&oq, static_cast<std::size_t>(T) * nq * 2));
  CK(cudaMalloc(&ok_, static_cast<std::size_t>(T) * nk * 2));
  CK(cudaMalloc(&ov, static_cast<std::size_t>(T) * nk * 2));
  expect(granite4_skinny_qkv(x.d, wq.d, wk.d, wv.d, 0.1f, 0.2f, 0.3f, T, nq, nk, nk, H, oq, ok_,
                             ov) == 0,
         "qkv launch");
  CK(cudaDeviceSynchronize());
  const auto gq = fetch(oq, static_cast<std::size_t>(T) * nq);
  const auto gk = fetch(ok_, static_cast<std::size_t>(T) * nk);
  const auto gv = fetch(ov, static_cast<std::size_t>(T) * nk);
  ok = true;
  for (int t = 0; t < T; ++t) {
    for (int n = 0; n < nq; ++n)
      ok = ok && close(bf(gq[static_cast<std::size_t>(t) * nq + n]), 0.1 * wq.dot(n, x.h, t), 3.0);
    for (int n = 0; n < nk; ++n) {
      ok = ok && close(bf(gk[static_cast<std::size_t>(t) * nk + n]), 0.2 * wk.dot(n, x.h, t), 5.0);
      ok = ok && close(bf(gv[static_cast<std::size_t>(t) * nk + n]), 0.3 * wv.dot(n, x.h, t), 7.0);
    }
  }
  expect(ok, "skinny qkv matches the fp64 reference");
  cudaFree(oq);
  cudaFree(ok_);
  cudaFree(ov);
}

// One decoder layer's GEMMs at Granite 4.2 8B shapes + lm_head, per T.
void bench() {
  const int H = 4096, I = 12800, KV = 1024, V = 100352;
  Mat q(H, H), k(KV, H), v(KV, H), o(H, H), g(I, H), u(I, H), dn(H, I), lm(V, H);
  float* flush;
  const std::size_t fl = 256u << 20;
  CK(cudaMalloc(&flush, fl));
  std::uint16_t* out;
  CK(cudaMalloc(&out, static_cast<std::size_t>(16) * V * 2));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::printf("  mode=%s\n", granite4_verify_skinny_enabled() ? "skinny (tensor cores)"
                                                              : "per-row GEMV");
  for (int T : {1, 2, 7, 14, 16}) {
    Act a(T, I);
    float total = 0;
    const double bytes = 2.0 * H * H + 2.0 * KV * H + 3.0 * I * H;
    for (int rep = 0; rep < 3; ++rep) {
      CK(cudaMemset(flush, rep, fl));
      cudaEventRecord(e0);
      // Fused Q|K|V, as the forward runs it.
      granite4_verify_qkv(a.d, q.d, k.d, v.d, 1.0f, 1.0f, 1.0f, T, H, KV, KV, H, out,
                          out + static_cast<std::size_t>(16) * H,
                          out + static_cast<std::size_t>(16) * (H + KV));
      granite4_verify_gemv(a.d, o.d, 1.0f, T, H, H, out);
      granite4_verify_silu(g.d, u.d, a.d, 1.0f, 1.0f, I, H, T, out);
      granite4_verify_gemv(a.d, dn.d, 1.0f, T, H, I, out);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      if (rep > 0) total += ms;
    }
    const float layer = total / 2;
    // Per GEMM, each from a cold L2: where the layer loses bandwidth.
    if (T == 1 || T == 16) {
      auto one = [&](const char* name, double mb, auto&& fn) {
        float best = 1e9f;
        for (int rep = 0; rep < 3; ++rep) {
          CK(cudaMemset(flush, rep, fl));
          cudaEventRecord(e0);
          fn();
          cudaEventRecord(e1);
          cudaEventSynchronize(e1);
          float ms = 0;
          cudaEventElapsedTime(&ms, e0, e1);
          best = std::min(best, ms);
        }
        std::printf("    T=%2d %-6s %6.1f MB %7.1f us %4.0f GB/s\n", T, name, mb, best * 1e3,
                    mb / best);  // MB per ms = GB/s
      };
      one("qkv", (H + 2.0 * KV) * H / 1e6, [&] {
        granite4_verify_qkv(a.d, q.d, k.d, v.d, 1.0f, 1.0f, 1.0f, T, H, KV, KV, H, out,
                            out + static_cast<std::size_t>(16) * H,
                            out + static_cast<std::size_t>(16) * (H + KV));
      });
      one("o", 1.0 * H * H / 1e6, [&] { granite4_verify_gemv(a.d, o.d, 1.0f, T, H, H, out); });
      one("gateup", 2.0 * I * H / 1e6,
          [&] { granite4_verify_silu(g.d, u.d, a.d, 1.0f, 1.0f, I, H, T, out); });
      one("down", 1.0 * I * H / 1e6, [&] { granite4_verify_gemv(a.d, dn.d, 1.0f, T, H, I, out); });
    }
    CK(cudaMemset(flush, 7, fl));
    cudaEventRecord(e0);
    granite4_verify_gemv(a.d, lm.d, 1.0f, T, V, H, out);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float lmms = 0;
    cudaEventElapsedTime(&lmms, e0, e1);
    std::printf("  T=%2d  layer %.3f ms (%.0f GB/s)  x40 + lm_head %.3f ms = %.2f ms/step\n", T,
                layer, bytes / (layer * 1e6), lmms, 40 * layer + lmms);
  }
  cudaFree(flush);
  cudaFree(out);
}
}  // namespace

int main(int argc, char** argv) {
  test_out_and_invariance();
  test_modes();
  if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) bench();
  std::printf(g_fail ? "skinny: %d FAILED\n" : "skinny: all tests passed\n", g_fail);
  return g_fail ? 1 : 0;
}
