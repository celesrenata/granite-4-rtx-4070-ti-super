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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "device_math.cuh"
#include "kernel_math.h"
#include "forward_gpu.h"    // GpuWeightsLayout, kGpuScalesPerLayer
#include "forward_gpu_api.h" // public gpu_ctx_* / gpu_forward_range
#include "model_config.h"   // ModelConfig, ForwardSpec

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
                                         std::uint16_t* d_out, const int* d_pt = nullptr);
extern "C" int granite4_kernel_gqa_fused_pos(const std::uint16_t* d_q,
                                             const std::uint16_t* d_k,
                                             const std::uint16_t* d_v, float att_scale,
                                             int Hq, int Hk, int d, int kv_len,
                                             int k_stride, const int* d_pos,
                                             std::uint16_t* d_out,
                                             const int* d_pt = nullptr);
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
extern "C" int granite4_verify_skinny_enabled();
extern "C" int granite4_verify_gemv(const std::uint16_t* d_a,
                                     const std::uint8_t* d_b, float scale,
                                     int tokens, int rows, int k,
                                     std::uint16_t* d_c);
extern "C" int granite4_verify_gemv_resid(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                          float scale, const std::uint16_t* d_residual,
                                          float residual_scale, int tokens, int rows, int k,
                                          std::uint16_t* d_out);
extern "C" int granite4_verify_qkv(const std::uint16_t* d_a, const std::uint8_t* d_wq,
                                   const std::uint8_t* d_wk, const std::uint8_t* d_wv,
                                   float sq, float sk, float sv, int tokens, int nq, int nk,
                                   int nv, int k, std::uint16_t* d_q, std::uint16_t* d_k,
                                   std::uint16_t* d_v);
extern "C" int granite4_verify_silu(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                    const std::uint16_t* d_x, float sg, float su, int n,
                                    int k, int tokens, std::uint16_t* d_out);
extern "C" int granite4_kernel_qkv_dequant(
    const std::uint16_t* d_a, const std::uint8_t* d_wq, const std::uint8_t* d_wk,
    const std::uint8_t* d_wv, float sq, float sk, float sv, int nq, int nk, int k,
    std::uint16_t* d_q, std::uint16_t* d_kk, std::uint16_t* d_v);

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
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
     const int* d_pt = nullptr);
extern "C" int granite4_kernel_verify_attention_stage(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    const float* d_cos, const float* d_sin, float att_scale, int base_pos, int rows,
    int Hq, int Hk, int d, int cache_stride, std::uint16_t* d_q_rot,
    std::uint16_t* d_k_rot, std::uint16_t* d_cache_k, std::uint16_t* d_cache_v,
    std::uint16_t* d_out, const int* d_pt = nullptr);
extern "C" int granite4_kernel_kv_append_batched(
    const std::uint16_t* d_src_k, const std::uint16_t* d_src_v,
    std::uint16_t* d_cache_k, std::uint16_t* d_cache_v, int base_pos, int rows,
    int stride, int Hk, int d, const int* d_pt);
namespace granite4 {
int gpu_verify_attention_layer(
    const std::uint16_t*, const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    float, float, float, const float*, const float*, float, int, int, int, int, int,
    int, int, std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*,
    std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*, const int*);
int gpu_verify_attention_layer_batched(
    const std::uint16_t*, const std::uint8_t*, const std::uint8_t*, const std::uint8_t*,
    float, float, float, const float*, const float*, float, int, int, int, int, int,
    int, int, std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*,
    std::uint16_t*, std::uint16_t*, std::uint16_t*, std::uint16_t*, const int*);
}
extern "C" int granite4_kernel_gqa_fused_batched(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    float att_scale, int Hq, int Hk, int d, int base_kv, int rows, int k_stride,
    std::uint16_t* d_out, const int* d_pt = nullptr);

extern "C" int granite4_kernel_gqa_fused_rows(
    const std::uint16_t* d_q, const std::uint16_t* d_k, const std::uint16_t* d_v,
    float att_scale, int Hq, int Hk, int d, int base_kv, int rows, int k_stride,
    std::uint16_t* d_out, const int* d_pt = nullptr);
extern "C" int granite4_prefill_gemm_bf16(const std::uint16_t* d_a, const std::uint8_t* d_w,
                                          float scale, int M, int N, int K,
                                          std::uint16_t* d_c);
namespace granite4 {
std::size_t attn_split_scratch_bytes(int Hq, int rows, int stride);
int attn_split(const std::uint16_t* q, const std::uint16_t* cache_k,
               const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
               int kv_base, const int* d_pos, int rows, int stride, void* scratch,
               std::uint16_t* out, const std::int8_t* cache_k8 = nullptr,
               const std::int8_t* cache_v8 = nullptr,
               const float* cache_k_scale = nullptr,
               const float* cache_v_scale = nullptr, bool int8 = false,
               const std::uint8_t* cache_k4 = nullptr,
               const std::uint8_t* cache_v4 = nullptr, bool int4 = false,
               const int* d_pt = nullptr, int max_len = 0,
               const int* row_pos = nullptr, const int* const* row_pt = nullptr,
               const int* row_grp = nullptr, int ngroups = 0,
               const int* row_tbase = nullptr, const int* row_tail = nullptr,
               int kv_hint = 0);
int prefill_attention(const std::uint16_t* q_rot, const std::uint16_t* cache_k,
                      const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
                      int base_pos, int rows, int stride, std::uint16_t* out,
                      const std::int8_t* cache_k8 = nullptr,
                      const std::int8_t* cache_v8 = nullptr,
                      const float* k_scale = nullptr, const float* v_scale = nullptr,
                      bool int8 = false, const std::uint8_t* cache_k4 = nullptr,
                      const std::uint8_t* cache_v4 = nullptr, bool int4 = false,
                      const int* d_pt = nullptr);
}

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

// The sm_89 hardware decode (fp8x4_to_f32) used by every decode GEMV must
// agree bit-for-bit with the branchless reference on every finite code; only
// the two NaN codes (never produced by the packer) may differ.
__global__ void fp8_hw_vs_sw_kernel(int* bad) {
  const std::uint32_t x = threadIdx.x;  // one code per thread
  const std::uint32_t word = x | ((x ^ 0x5au) << 8) | ((x ^ 0xa5u) << 16) | ((x ^ 0xffu) << 24);
  float h[4];
  granite4::dev::fp8x4_to_f32(word, h[0], h[1], h[2], h[3]);
  for (int j = 0; j < 4; ++j) {
    const std::uint8_t code = static_cast<std::uint8_t>(word >> (8 * j));
    if ((code & 0x7fu) == 0x7fu) continue;
    if (__float_as_uint(h[j]) != __float_as_uint(granite4::dev::fp8_to_f32(code)))
      atomicAdd(bad, 1);
  }
}

static void test_fp8_hw_decode() {
  int* d_bad = nullptr;
  if (cudaMalloc(&d_bad, sizeof(int)) != cudaSuccess) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(int)));
  fp8_hw_vs_sw_kernel<<<1, 256>>>(d_bad);
  int bad = -1;
  CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_bad));
  expect(bad == 0, "fp8x4_to_f32 hardware decode bit-exact on all finite codes");
}

// ---------------------------------------------------------------------------
// KV8 (Q8_0) int8 KV cache: the device quantize/dequant (dev::kv8_*) must match
// the host twin bit-for-bit (identical int codes, identical dequant) and the
// row round-trip must stay within half a quantization step.
// ---------------------------------------------------------------------------

// One thread per row: quantize a bf16 row (amax -> scale -> codes) on the
// device and write codes + scale. The host recomputes the same and compares.
__global__ void kv8_row_quant_kernel(const std::uint16_t* x, std::int8_t* q,
                                    float* scale, int rows, int d) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const std::uint16_t* row = x + static_cast<std::size_t>(r) * d;
  float amax = 0.0f;
  for (int i = 0; i < d; ++i)
    amax = fmaxf(amax, fabsf(granite4::dev::bf16_to_f32(row[i])));
  const float s = granite4::dev::kv8_scale(amax);
  scale[r] = s;
  std::int8_t* codes = q + static_cast<std::size_t>(r) * d;
  for (int i = 0; i < d; ++i)
    codes[i] = granite4::dev::kv8_quant(granite4::dev::bf16_to_f32(row[i]), s);
}

static void test_kv8_device() {
  const int rows = 256, d = 128;  // head_dim of the model
  Xorshift64 rng(0x8abc);
  // A mix of magnitudes so the scale is exercised across its range: small,
  // typical (O(1)) and large, plus an all-zero row (scale == 0) and a
  // near-tie row.
  std::vector<std::uint16_t> hx(static_cast<std::size_t>(rows) * d);
  for (int r = 0; r < rows; ++r) {
    float scale = 1.0f;
    if (r == 0) scale = 0.0f;                        // all-zero row
    else if (r == 1) scale = 1e-5f;                  // subnormal-ish amax
    else if (r % 4 == 0) scale = 0.25f;
    else if (r % 4 == 1) scale = 3.0f;
    else if (r % 4 == 2) scale = 32.0f;
    else scale = 256.0f;
    for (int i = 0; i < d; ++i) {
      const float v = rng.next_float_range(scale);
      hx[static_cast<std::size_t>(r) * d + i] = f32_to_bf16_host(v);
    }
  }
  std::uint16_t* d_x = nullptr;
  std::int8_t* d_q = nullptr;
  float* d_s = nullptr;
  if (cudaMalloc(&d_x, hx.size() * 2) != cudaSuccess ||
      cudaMalloc(&d_q, hx.size()) != cudaSuccess ||
      cudaMalloc(&d_s, rows * sizeof(float)) != cudaSuccess ||
      cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    ++g_failures;
    if (d_x) cudaFree(d_x);
    if (d_q) cudaFree(d_q);
    if (d_s) cudaFree(d_s);
    return;
  }
  kv8_row_quant_kernel<<<(rows + 255) / 256, 256>>>(d_x, d_q, d_s, rows, d);
  std::vector<std::int8_t> hq(hx.size());
  std::vector<float> hs(rows);
  CUDA_CHECK(cudaMemcpy(hq.data(), d_q, hx.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs.data(), d_s, rows * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_x);
  cudaFree(d_q);
  cudaFree(d_s);

  int bad_code = 0, bad_scale = 0, bad_dequant = 0, bad_bound = 0;
  float worst_step_frac = 0.0f;
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int i = 0; i < d; ++i)
      amax = std::max(amax, std::fabs(bf16_to_f32_host(hx[static_cast<std::size_t>(r) * d + i])));
    const float s_ref = kv8_scale_host(amax);
    if (hs[r] != s_ref && bad_scale < 4) {
      std::printf("  kv8 scale mismatch row=%d dev=%a ref=%a\n", r, hs[r], s_ref);
      ++bad_scale;
    }
    for (int i = 0; i < d; ++i) {
      const float x = bf16_to_f32_host(hx[static_cast<std::size_t>(r) * d + i]);
      const std::int8_t q_ref = kv8_quant_host(x, s_ref);
      const std::int8_t q_dev = hq[static_cast<std::size_t>(r) * d + i];
      if (q_dev != q_ref && bad_code < 8) {
        std::printf("  kv8 code mismatch row=%d i=%d x=%a dev=%d ref=%d\n", r, i, x,
                    q_dev, q_ref);
        ++bad_code;
      }
      // dequant bit-exact + round-trip error bound.
      const float dq_dev = hs[r] * static_cast<float>(q_dev);
      const float dq_ref = kv8_dequant_host(q_ref, s_ref);
      std::uint32_t bits_dev, bits_ref;
      std::memcpy(&bits_dev, &dq_dev, 4);
      std::memcpy(&bits_ref, &dq_ref, 4);
      if (bits_dev != bits_ref) {
        if (bad_dequant < 8)
          std::printf("  kv8 dequant mismatch row=%d i=%d dev=%a ref=%a\n", r, i, dq_dev,
                      dq_ref);
        ++bad_dequant;
      }
      // Absolute error is bounded by half a quantization step (s/2) plus one
      // ULP of the dequant product (s*q) for the multiplication rounding.
      if (hs[r] != 0.0f) {
        const float err = std::fabs(x - dq_dev);
        const float half_step = hs[r] * 0.5f;
        worst_step_frac = std::max(worst_step_frac, err / half_step);
        // Tolerance: s/2 + one ULP at the dequant magnitude (s * 127).
        const float tol = half_step + hs[r] * 127.0f * 1.19e-7f;  // 2^-23
        if (err > tol) ++bad_bound;
      }
    }
  }
  expect(bad_code == 0, "kv8 int8 quantize codes bit-exact device vs host");
  expect(bad_scale == 0, "kv8 row scale bit-exact device vs host");
  expect(bad_dequant == 0, "kv8 dequant bit-exact device vs host");
  expect(bad_bound == 0, "kv8 round-trip within half a step (abs error)");
  if (g_failures == 0)
    std::printf("  kv8 worst abs error = %.4f of a half step (bound 1.0)\n",
                worst_step_frac);
}

// ---------------------------------------------------------------------------
// KV4 (Q4_0) int4 KV cache: same guarantee as KV8 - the device quantize
// (dev::q4_*) must match the host twin bit-for-bit and the row round-trip must
// stay within half a quantization step. The packing differs (two codes per
// byte), so the device kernel emits a d/2-byte row.
// ---------------------------------------------------------------------------

// One thread per row: quantize a bf16 row into packed int4 codes (two per
// byte, even dim = lower nibble) + a per-row scale. The host recomputes and
// compares. d must be even.
__global__ void kv4_row_quant_kernel(const std::uint16_t* x, std::uint8_t* q,
                                    float* scale, int rows, int d) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const std::uint16_t* row = x + static_cast<std::size_t>(r) * d;
  float amax = 0.0f;
  for (int i = 0; i < d; ++i)
    amax = fmaxf(amax, fabsf(granite4::dev::bf16_to_f32(row[i])));
  const float s = granite4::dev::q4_scale(amax);
  scale[r] = s;
  std::uint8_t* codes = q + static_cast<std::size_t>(r) * (d / 2);
  for (int w = 0; w < d / 2; ++w) {
    const float x0 = granite4::dev::bf16_to_f32(row[2 * w]);
    const float x1 = granite4::dev::bf16_to_f32(row[2 * w + 1]);
    codes[w] = static_cast<std::uint8_t>(
        (granite4::dev::q4_quant(x0, s) & 0x0fu) |
        (granite4::dev::q4_quant(x1, s) << 4));
  }
}

static void test_kv4_device() {
  const int rows = 256, d = 128;  // head_dim of the model
  Xorshift64 rng(0x4e8a);
  std::vector<std::uint16_t> hx(static_cast<std::size_t>(rows) * d);
  for (int r = 0; r < rows; ++r) {
    float scale = 1.0f;
    if (r == 0) scale = 0.0f;                        // all-zero row
    else if (r == 1) scale = 1e-5f;                  // subnormal-ish amax
    else if (r % 4 == 0) scale = 0.25f;
    else if (r % 4 == 1) scale = 3.0f;
    else if (r % 4 == 2) scale = 32.0f;
    else scale = 256.0f;
    for (int i = 0; i < d; ++i) {
      const float v = rng.next_float_range(scale);
      hx[static_cast<std::size_t>(r) * d + i] = f32_to_bf16_host(v);
    }
  }
  const std::size_t codes = static_cast<std::size_t>(rows) * (d / 2);
  std::uint16_t* d_x = nullptr;
  std::uint8_t* d_q = nullptr;
  float* d_s = nullptr;
  if (cudaMalloc(&d_x, hx.size() * 2) != cudaSuccess ||
      cudaMalloc(&d_q, codes) != cudaSuccess ||
      cudaMalloc(&d_s, rows * sizeof(float)) != cudaSuccess ||
      cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    ++g_failures;
    if (d_x) cudaFree(d_x);
    if (d_q) cudaFree(d_q);
    if (d_s) cudaFree(d_s);
    return;
  }
  kv4_row_quant_kernel<<<(rows + 255) / 256, 256>>>(d_x, d_q, d_s, rows, d);
  std::vector<std::uint8_t> hq(codes);
  std::vector<float> hs(rows);
  CUDA_CHECK(cudaMemcpy(hq.data(), d_q, codes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs.data(), d_s, rows * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_x);
  cudaFree(d_q);
  cudaFree(d_s);

  int bad_code = 0, bad_scale = 0, bad_dequant = 0, bad_bound = 0;
  float worst_step_frac = 0.0f;
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int i = 0; i < d; ++i)
      amax = std::max(amax, std::fabs(bf16_to_f32_host(hx[static_cast<std::size_t>(r) * d + i])));
    const float s_ref = q4_scale_host(amax);
    if (hs[r] != s_ref && bad_scale < 4) {
      std::printf("  kv4 scale mismatch row=%d dev=%a ref=%a\n", r, hs[r], s_ref);
      ++bad_scale;
    }
    for (int w = 0; w < d / 2; ++w) {
      const std::size_t i0 = static_cast<std::size_t>(r) * d + 2 * w;
      const float x0 = bf16_to_f32_host(hx[i0]);
      const float x1 = bf16_to_f32_host(hx[i0 + 1]);
      const unsigned char byte_ref =
          static_cast<unsigned char>((q4_quant_host(x0, s_ref) & 0x0fu) |
                                     (q4_quant_host(x1, s_ref) << 4));
      const unsigned char byte_dev = hq[static_cast<std::size_t>(r) * (d / 2) + w];
      if (byte_dev != byte_ref && bad_code < 8) {
        std::printf("  kv4 code mismatch row=%d w=%d byte dev=%02x ref=%02x\n", r, w,
                    byte_dev, byte_ref);
        ++bad_code;
      }
      // dequant bit-exact + round-trip error bound (even + odd dim in the byte).
      float lo_dev, hi_dev, lo_ref, hi_ref;
      q4_dequant_pair_host(byte_dev, hs[r], lo_dev, hi_dev);
      q4_dequant_pair_host(byte_ref, s_ref, lo_ref, hi_ref);
      std::uint32_t bl, bh, brl, brh;
      std::memcpy(&bl, &lo_dev, 4);
      std::memcpy(&bh, &hi_dev, 4);
      std::memcpy(&brl, &lo_ref, 4);
      std::memcpy(&brh, &hi_ref, 4);
      if (bl != brl || bh != brh) {
        if (bad_dequant < 8)
          std::printf("  kv4 dequant mismatch row=%d w=%d\n", r, w);
        ++bad_dequant;
      }
      const float half_step = hs[r] * 0.5f;
      if (hs[r] != 0.0f) {
        const float tol = half_step + hs[r] * 7.0f * 1.19e-7f;  // 2^-23
        const float e0 = std::fabs(x0 - lo_dev);
        const float e1 = std::fabs(x1 - hi_dev);
        if (e0 > tol || e1 > tol) ++bad_bound;
        worst_step_frac = std::max(worst_step_frac,
                                   std::max(e0, e1) / half_step);
      }
    }
  }
  expect(bad_code == 0, "kv4 int4 quantize codes bit-exact device vs host");
  expect(bad_scale == 0, "kv4 row scale bit-exact device vs host");
  expect(bad_dequant == 0, "kv4 dequant bit-exact device vs host");
  expect(bad_bound == 0, "kv4 round-trip within half a step (abs error)");
  if (g_failures == 0)
    std::printf("  kv4 worst abs error = %.4f of a half step (bound 1.0)\n",
                worst_step_frac);
}

// KV4 attention on the persistent int4 cache: quantize a bf16 K/V cache to the
// int4 layout (two codes/byte + a per-row scale, dev::q4_*/q4_*_host), run the
// split-KV attention with int4 dequant-on-load, and check it stays within the
// int4 dequantization noise of the bf16 fused-GQA reference. (The int8 path
// has no attention-level test; this is the KV4 analogue of test_attn_split
// exercised on the quantized cache.)
static void test_kv4_attn_split_device() {
  const int Hq = 8, Hk = 2, d = 128, stride = 512, base = 200, rows = 8;
  Xorshift64 rng(0x4a11);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));

  // Build the int4 cache from the bf16 K/V on the host (the host twin of the
  // device append), keeping the per-row scales so the attention test reads
  // exactly what the device wrote.
  std::vector<std::uint8_t> c4k(Hk * stride * (d / 2)), c4v(Hk * stride * (d / 2));
  std::vector<float> sk(Hk * stride), sv(Hk * stride);
  for (int h = 0; h < Hk; ++h)
    for (int p = 0; p < stride; ++p) {
      const std::size_t row = static_cast<std::size_t>(h) * stride + p;
      const std::size_t src = row * d;
      float ak = 0.0f, av = 0.0f;
      for (int i = 0; i < d; ++i) {
        ak = std::max(ak, std::fabs(bf16_to_f32_host(ck[src + i])));
        av = std::max(av, std::fabs(bf16_to_f32_host(cv[src + i])));
      }
      const float sks = q4_scale_host(ak), svs = q4_scale_host(av);
      sk[row] = sks;
      sv[row] = svs;
      for (int w = 0; w < d / 2; ++w) {
        c4k[row * (d / 2) + w] =
            static_cast<std::uint8_t>((q4_quant_host(bf16_to_f32_host(ck[src + 2 * w]),
                                                    sks) &
                                       0x0fu) |
                                      (q4_quant_host(bf16_to_f32_host(ck[src + 2 * w + 1]),
                                                     sks) <<
                                       4));
        c4v[row * (d / 2) + w] =
            static_cast<std::uint8_t>((q4_quant_host(bf16_to_f32_host(cv[src + 2 * w]),
                                                    svs) &
                                       0x0fu) |
                                      (q4_quant_host(bf16_to_f32_host(cv[src + 2 * w + 1]),
                                                     svs) <<
                                       4));
      }
    }

  std::uint16_t *dq, *dk, *dv, *dref, *dgot;
  std::uint8_t *dc4k, *dc4v;
  float *dsk, *dsv;
  void* scratch;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, cn * 2) || cudaMalloc(&dv, cn * 2) ||
      cudaMalloc(&dref, qn * 2) || cudaMalloc(&dgot, qn * 2) ||
      cudaMalloc(&dc4k, c4k.size()) || cudaMalloc(&dc4v, c4v.size()) ||
      cudaMalloc(&dsk, sk.size() * sizeof(float)) ||
      cudaMalloc(&dsv, sv.size() * sizeof(float)) ||
      cudaMalloc(&scratch, granite4::attn_split_scratch_bytes(Hq, rows, stride))) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc4k, c4k.data(), c4k.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc4v, c4v.data(), c4v.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsk, sk.data(), sk.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dsv, sv.data(), sv.size() * sizeof(float), cudaMemcpyHostToDevice));

  const float scale = 0.0078125f;  // model's attention scale (smoother softmax)
  expect(granite4_kernel_gqa_fused_rows(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                        dref) == 0,
         "kv4 attention: reference launch");
  expect(granite4::attn_split(dq, nullptr, nullptr, scale, Hq, Hk, d, base, nullptr, rows,
                             stride, scratch, dgot, nullptr, nullptr, dsk, dsv, false, dc4k,
                             dc4v, true) == 0,
         "kv4 attention: launch");
  std::vector<std::uint16_t> ref(qn), got(qn);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got.data(), dgot, qn * 2, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (std::size_t i = 0; i < qn; ++i)
    max_err = std::max(
        max_err,
        static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) - bf16_to_f32_host(ref[i]))));
  std::printf("  kv4 split-KV attention: max |delta| = %.5f\n", max_err);
  expect(max_err < 0.5, "kv4 attention: int4 dequant noise within bound");
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dref);
  cudaFree(dgot);
  cudaFree(dc4k);
  cudaFree(dc4v);
  cudaFree(dsk);
  cudaFree(dsv);
  cudaFree(scratch);
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
    if (v[i] != r[i]) ++parity_bad;
  expect(parity_bad == 0, "verify GEMV: bit-identical to production GEMV");
  cudaFree(da); cudaFree(db); cudaFree(dv); cudaFree(dr);
}

// The multi-token verify GEMVs against their single-token decode kernels at
// every width 1..16, on decode-like K (4096: two 16-element chunks per lane;
// 12800: an uneven 6.25 per lane), bit for bit.
static void test_verify_kernels_bit_identical_device() {
  constexpr int kMaxT = 16;
  Xorshift64 rng(0x7e57);
  for (const int K : {4096, 12800}) {
    const int N = 72, nq = 48, nk = 12;  // nq + 2 nk == N rows of dw
    const std::size_t outn = static_cast<std::size_t>(kMaxT) * N;
    std::vector<std::uint16_t> a(static_cast<std::size_t>(kMaxT) * K), res(outn);
    std::vector<std::uint8_t> w(static_cast<std::size_t>(N) * K), w2(w.size());
    for (auto& v : a) v = gen_bf16(rng);
    for (auto& v : res) v = gen_bf16(rng);
    for (auto& v : w) v = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
    for (auto& v : w2) v = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
    std::uint16_t *da, *dres, *dgot, *dref, *dq, *dk, *dv, *rq, *rk, *rv;
    std::uint8_t *dw, *dw2;
    if (cudaMalloc(&da, a.size() * 2) || cudaMalloc(&dres, res.size() * 2) ||
        cudaMalloc(&dw, w.size()) || cudaMalloc(&dw2, w2.size()) ||
        cudaMalloc(&dgot, outn * 2) || cudaMalloc(&dref, outn * 2) ||
        cudaMalloc(&dq, outn * 2) || cudaMalloc(&dk, outn * 2) || cudaMalloc(&dv, outn * 2) ||
        cudaMalloc(&rq, outn * 2) || cudaMalloc(&rk, outn * 2) || cudaMalloc(&rv, outn * 2)) {
      ++g_failures;
      return;
    }
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dres, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw2, w2.data(), w2.size(), cudaMemcpyHostToDevice));
    auto same = [](const std::uint16_t* x, const std::uint16_t* y, std::size_t n) {
      std::vector<std::uint16_t> hx(n), hy(n);
      CUDA_CHECK(cudaMemcpy(hx.data(), x, n * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
      return hx == hy;
    };
    auto at = [](auto* base, int t, int pitch) { return base + static_cast<std::size_t>(t) * pitch; };
    const std::uint8_t* const wk = dw + static_cast<std::size_t>(nq) * K;
    const std::uint8_t* const wv = wk + static_cast<std::size_t>(nk) * K;
    bool launch_ok = true, gemv_ok = true, resid_ok = true, qkv_ok = true, silu_ok = true;
    // The reference row is what decode computes: with the skinny kernels on
    // (G4_SKINNY, default) decode calls these same entry points with one
    // token; with them off it runs the per-row decode kernels.
    const bool sk = granite4_verify_skinny_enabled() != 0;
    for (int T = 1; T <= kMaxT; ++T) {
      const std::size_t n = static_cast<std::size_t>(T) * N;
      launch_ok &= granite4_verify_gemv(da, dw, 0.0007f, T, N, K, dgot) == 0;
      for (int t = 0; t < T; ++t)
        launch_ok &= (sk ? granite4_verify_gemv(at(da, t, K), dw, 0.0007f, 1, N, K, at(dref, t, N))
                         : granite4_kernel_row_gemm_dequant(at(da, t, K), dw, 0.0007f, N, K,
                                                            at(dref, t, N))) == 0;
      gemv_ok &= same(dgot, dref, n);

      launch_ok &= granite4_verify_gemv_resid(da, dw, 0.0006f, dres, 0.75f, T, N, K, dgot) == 0;
      for (int t = 0; t < T; ++t)
        launch_ok &= (sk ? granite4_verify_gemv_resid(at(da, t, K), dw, 0.0006f, at(dres, t, N),
                                                      0.75f, 1, N, K, at(dref, t, N))
                         : granite4_kernel_row_gemm_resid_dequant(at(da, t, K), dw, 0.0006f,
                                                                  at(dres, t, N), 0.75f, 1, N,
                                                                  K, at(dref, t, N))) == 0;
      resid_ok &= same(dgot, dref, n);

      launch_ok &= granite4_verify_qkv(da, dw, wk, wv, .0007f, .0008f, .0009f, T, nq, nk, nk,
                                       K, dq, dk, dv) == 0;
      for (int t = 0; t < T; ++t)
        launch_ok &= (sk ? granite4_verify_qkv(at(da, t, K), dw, wk, wv, .0007f, .0008f, .0009f,
                                               1, nq, nk, nk, K, at(rq, t, nq), at(rk, t, nk),
                                               at(rv, t, nk))
                         : granite4_kernel_qkv_dequant(at(da, t, K), dw, wk, wv, .0007f, .0008f,
                                                       .0009f, nq, nk, K, at(rq, t, nq),
                                                       at(rk, t, nk), at(rv, t, nk))) == 0;
      qkv_ok &= same(dq, rq, static_cast<std::size_t>(T) * nq) &&
                same(dk, rk, static_cast<std::size_t>(T) * nk) &&
                same(dv, rv, static_cast<std::size_t>(T) * nk);

      launch_ok &= granite4_verify_silu(dw, dw2, da, 0.03125f, 0.02734375f, N, K, T, dgot) == 0;
      for (int t = 0; t < T; ++t)
        launch_ok &= (sk ? granite4_verify_silu(dw, dw2, at(da, t, K), 0.03125f, 0.02734375f, N,
                                                K, 1, at(dref, t, N))
                         : granite4_kernel_silu_gate_up(dw, dw2, at(da, t, K), 0.03125f,
                                                        0.02734375f, N, K, at(dref, t, N))) == 0;
      silu_ok &= same(dgot, dref, n);
    }
    expect(launch_ok, "verify kernels: launches");
    expect(gemv_ok, "verify kernels: GEMV bit-identical to decode (1..16 tokens)");
    expect(resid_ok, "verify kernels: residual GEMV bit-identical to decode (1..16 tokens)");
    expect(qkv_ok, "verify kernels: QKV bit-identical to decode (1..16 tokens)");
    expect(silu_ok, "verify kernels: SwiGLU bit-identical to decode (1..16 tokens)");
    expect(granite4_verify_gemv(da, dw, 1.0f, 17, N, K, dgot) == -1 &&
               granite4_verify_gemv(da, dw, 1.0f, 0, N, K, dgot) == -1,
           "verify kernels: reject widths outside 1..16");
    cudaFree(da); cudaFree(dres); cudaFree(dw); cudaFree(dw2); cudaFree(dgot); cudaFree(dref);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(rq); cudaFree(rk); cudaFree(rv);
  }
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
  expect(granite4_verify_gemv(da, dwq, .0007f, tokens, nq, k, dq) == 0,
         "batched QKV: Q launch");
  expect(granite4_verify_gemv(da, dwk, .0008f, tokens, nk, k, dk) == 0,
         "batched QKV: K launch");
  expect(granite4_verify_gemv(da, dwv, .0009f, tokens, nk, k, dv) == 0,
         "batched QKV: V launch");
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

static void test_verify_attention_validation_device() {
  expect(granite4_kernel_verify_qkv_stage(
              nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 2, 1, 32,
              32, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify stage: rejects null pointers");
  expect(granite4_kernel_verify_qkv_stage(
              nullptr, nullptr, nullptr, nullptr, nullptr, -1, 1, 2, 1, 32,
              32, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify stage: rejects negative position");
  expect(granite4_kernel_verify_qkv_stage(
              nullptr, nullptr, nullptr, nullptr, nullptr, 0, 1, 1, 2, 32,
              32, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify stage: rejects invalid GQA topology");
  expect(granite4_kernel_verify_attention_stage(
              nullptr, nullptr, nullptr, nullptr, nullptr, NAN, 0, 1, 2, 1,
              32, 32, nullptr, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify attention: rejects non-finite scale");
  expect(granite4_kernel_verify_attention_stage(
              nullptr, nullptr, nullptr, nullptr, nullptr, 1.0f, 0, 17, 2, 1,
              32, 32, nullptr, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify attention: rejects oversized batch");
}

static void test_verify_attention_layer_validation_device() {
  expect(gpu_verify_attention_layer_batched(
              nullptr, nullptr, nullptr, nullptr, NAN, 1.0f, 1.0f, nullptr,
              nullptr, 1.0f, 0, 1, 2, 1, 32, 64, 32, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify layer: rejects non-finite Q scale");
  expect(gpu_verify_attention_layer_batched(
              nullptr, nullptr, nullptr, nullptr, 1.0f, INFINITY, 1.0f, nullptr,
              nullptr, 1.0f, 0, 1, 2, 1, 32, 64, 32, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify layer: rejects non-finite K scale");
  expect(gpu_verify_attention_layer_batched(
              nullptr, nullptr, nullptr, nullptr, 1.0f, 1.0f, 1.0f, nullptr,
              nullptr, NAN, 0, 1, 2, 1, 32, 64, 32, nullptr, nullptr,
              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == -1,
         "verify layer: rejects non-finite attention scale");
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

// Tensor-core prefill GEMM vs the exact host GEMM (f32 sum order differs:
// within bf16 rounding). Shapes exercise partial M and N tiles.
static void test_prefill_gemm_device() {
  const int M = 70, N = 136, K = 256;
  Xorshift64 rng(0x6e44);
  std::vector<std::uint16_t> a(static_cast<std::size_t>(M) * K);
  std::vector<std::uint8_t> w(static_cast<std::size_t>(N) * K);
  for (auto& x : a) x = gen_bf16(rng);
  for (auto& x : w) x = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  const float scale = 0.0123f;
  std::uint16_t *da = nullptr, *dc = nullptr;
  std::uint8_t* dw = nullptr;
  if (cudaMalloc(&da, a.size() * 2) || cudaMalloc(&dw, w.size()) ||
      cudaMalloc(&dc, static_cast<std::size_t>(M) * N * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice));
  expect(granite4_prefill_gemm_bf16(da, dw, scale, M, N, K, dc) == 0, "prefill gemm: launch");
  std::vector<std::uint16_t> got(static_cast<std::size_t>(M) * N);
  CUDA_CHECK(cudaMemcpy(got.data(), dc, got.size() * 2, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = 0.0;
      for (int k = 0; k < K; ++k)
        ref += static_cast<double>(bf16_to_f32_host(a[static_cast<std::size_t>(m) * K + k])) *
               fp8_to_f32_host(w[static_cast<std::size_t>(n) * K + k]);
      if (!bf16_close(got[static_cast<std::size_t>(m) * N + n],
                      f32_to_bf16_host(static_cast<float>(scale * ref)), 2))
        ++bad;
    }
  expect(bad == 0, "prefill gemm: within 2 ULP of the exact product");
  cudaFree(da); cudaFree(dw); cudaFree(dc);
}

// Tensor-core causal prefill attention vs the fused GQA reference (bf16
// logits, double softmax) on the same cache: per element within a small
// absolute tolerance relative to the output scale.
static void test_prefill_attention_device() {
  const int Hq = 8, Hk = 2, d = 128, base = 37, rows = 75, stride = 128;
  Xorshift64 rng(0xa77e);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  std::uint16_t *dq, *dk, *dv, *dref, *dgot;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, cn * 2) || cudaMalloc(&dv, cn * 2) ||
      cudaMalloc(&dref, qn * 2) || cudaMalloc(&dgot, qn * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  const float scale = 0.0078125f * 8.0f;  // sharper than the model's, harder
  expect(granite4_kernel_gqa_fused_rows(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                        dref) == 0,
         "prefill attention: reference launch");
  expect(granite4::prefill_attention(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                     dgot) == 0,
         "prefill attention: launch");
  std::vector<std::uint16_t> ref(qn), got(qn);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got.data(), dgot, qn * 2, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (std::size_t i = 0; i < qn; ++i)
    max_err = std::max(max_err, static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                              bf16_to_f32_host(ref[i]))));
  std::printf("  prefill attention: max |delta| = %.5f\n", max_err);
  expect(max_err < 0.03, "prefill attention: matches fused GQA within bf16 tolerance");
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dref); cudaFree(dgot);
}

// Paged prefill: the tensor-core causal attention gathers each tile row
// through the same position->row table the split and fused kernels use, so a
// physically permuted arena must read bit-identically to the contiguous one.
// All three KV formats are checked because the quantized paths also dequantize
// with the row's scale - the scale travels with the physical row, not the
// logical position, so a kernel that reads it at the logical position is
// bit-correct only under an identity table.
static void test_paged_prefill_device() {
  const int Hq = 8, Hk = 2, d = 128, base = 37, rows = 75, stride = 128;
  Xorshift64 rng(0x51f3);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 0.0078125f * 8.0f;  // sharp, exercises the softmax

  // A fixed permutation of the arena rows: perm[pos] is the physical row
  // holding logical position pos. Fisher-Yates makes it a real bijection.
  std::vector<int> perm(stride);
  for (int p = 0; p < stride; ++p) perm[p] = p;
  for (int i = stride - 1; i > 0; --i) {
    const int j = static_cast<int>(rng.next() % (i + 1));
    std::swap(perm[i], perm[j]);
  }

  std::uint16_t *dq, *dref, *dgotp;
  int *dpt;
  std::uint16_t *dk, *dv, *dkp, *dvp;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dref, qn * 2) ||
      cudaMalloc(&dgotp, qn * 2) || cudaMalloc(&dpt, sizeof(int) * stride) ||
      cudaMalloc(&dk, cn * 2) || cudaMalloc(&dv, cn * 2) ||
      cudaMalloc(&dkp, cn * 2) || cudaMalloc(&dvp, cn * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpt, perm.data(), sizeof(int) * stride, cudaMemcpyHostToDevice));

  // ---- bf16: permuted arena, d_pt, must equal the contiguous reference ----
  std::vector<std::uint16_t> ckp(cn), cvp(cn);
  for (int hkv = 0; hkv < Hk; ++hkv)
    for (int pos = 0; pos < stride; ++pos) {
      const std::size_t src = static_cast<std::size_t>(hkv) * stride * d + pos * d;
      const std::size_t dst =
          static_cast<std::size_t>(hkv) * stride * d + static_cast<std::size_t>(perm[pos]) * d;
      std::copy(ck.begin() + src, ck.begin() + src + d, ckp.begin() + dst);
      std::copy(cv.begin() + src, cv.begin() + src + d, cvp.begin() + dst);
    }
  CUDA_CHECK(cudaMemcpy(dkp, ckp.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvp, cvp.data(), cn * 2, cudaMemcpyHostToDevice));
  expect(granite4::prefill_attention(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                     dref) == 0,
         "paged prefill: contiguous bf16 reference");
  expect(granite4::prefill_attention(dq, dkp, dvp, scale, Hq, Hk, d, base, rows, stride,
                                     dgotp, nullptr, nullptr, nullptr, nullptr, false,
                                     nullptr, nullptr, false, dpt) == 0,
         "paged prefill: permuted bf16 launch");
  std::vector<std::uint16_t> ref(qn), got(qn);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got.data(), dgotp, qn * 2, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (std::size_t i = 0; i < qn; ++i)
    max_err = std::max(max_err,
                       static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                    bf16_to_f32_host(ref[i]))));
  expect(max_err == 0.0, "paged prefill (bf16): bit-identical to contiguous");

  // ---- KV8: codes + scales travel with the row; d_pt must pick up both ----
  {
    std::vector<std::int8_t> c8k(cn), c8v(cn);
    std::vector<float> sk(Hk * stride), sv(Hk * stride);
    for (int h = 0; h < Hk; ++h)
      for (int p = 0; p < stride; ++p) {
        const std::size_t row = static_cast<std::size_t>(h) * stride + p;
        const std::size_t src = row * d;
        float ak = 0.0f, av = 0.0f;
        for (int i = 0; i < d; ++i) {
          ak = std::max(ak, std::fabs(bf16_to_f32_host(ck[src + i])));
          av = std::max(av, std::fabs(bf16_to_f32_host(cv[src + i])));
        }
        const float sks = kv8_scale_host(ak), svs = kv8_scale_host(av);
        sk[row] = sks;
        sv[row] = svs;
        for (int i = 0; i < d; ++i) {
          c8k[row * d + i] = kv8_quant_host(bf16_to_f32_host(ck[src + i]), sks);
          c8v[row * d + i] = kv8_quant_host(bf16_to_f32_host(cv[src + i]), svs);
        }
      }
    // permuted: code rows and their scales both move to perm[pos].
    std::vector<std::int8_t> c8kp(cn), c8vp(cn);
    std::vector<float> skp(Hk * stride), svp(Hk * stride);
    for (int hkv = 0; hkv < Hk; ++hkv)
      for (int pos = 0; pos < stride; ++pos) {
        const std::size_t src = static_cast<std::size_t>(hkv) * stride + pos;
        const std::size_t dst =
            static_cast<std::size_t>(hkv) * stride + static_cast<std::size_t>(perm[pos]);
        std::copy(c8k.begin() + src * d, c8k.begin() + src * d + d, c8kp.begin() + dst * d);
        std::copy(c8v.begin() + src * d, c8v.begin() + src * d + d, c8vp.begin() + dst * d);
        skp[dst] = sk[src];
        svp[dst] = sv[src];
      }
    std::int8_t *dc8k, *dc8v, *dc8kp, *dc8vp;
    float *dsk, *dsv, *dskp, *dsvp;
    if (cudaMalloc(&dc8k, cn) || cudaMalloc(&dc8v, cn) ||
        cudaMalloc(&dc8kp, cn) || cudaMalloc(&dc8vp, cn) ||
        cudaMalloc(&dsk, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dsv, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dskp, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dsvp, sizeof(float) * Hk * stride)) {
      ++g_failures;
      return;
    }
    CUDA_CHECK(cudaMemcpy(dc8k, c8k.data(), cn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc8v, c8v.data(), cn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc8kp, c8kp.data(), cn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc8vp, c8vp.data(), cn, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsk, sk.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsv, sv.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dskp, skp.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsvp, svp.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    expect(granite4::prefill_attention(dq, nullptr, nullptr, scale, Hq, Hk, d, base, rows,
                                       stride, dref, dc8k, dc8v, dsk, dsv, true,
                                       nullptr, nullptr, false) == 0,
           "paged prefill: contiguous kv8 reference");
    expect(granite4::prefill_attention(dq, nullptr, nullptr, scale, Hq, Hk, d, base, rows,
                                       stride, dgotp, dc8kp, dc8vp, dskp, dsvp, true,
                                       nullptr, nullptr, false, dpt) == 0,
           "paged prefill: permuted kv8 launch");
    CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.data(), dgotp, qn * 2, cudaMemcpyDeviceToHost));
    double max_err8 = 0.0;
    for (std::size_t i = 0; i < qn; ++i)
      max_err8 = std::max(max_err8,
                           static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                       bf16_to_f32_host(ref[i]))));
    expect(max_err8 == 0.0, "paged prefill (kv8): bit-identical to contiguous");
    cudaFree(dc8k);
    cudaFree(dc8v);
    cudaFree(dc8kp);
    cudaFree(dc8vp);
    cudaFree(dsk);
    cudaFree(dsv);
    cudaFree(dskp);
    cudaFree(dsvp);
  }

  // ---- KV4: same contract, two codes per byte ----
  {
    std::vector<std::uint8_t> c4k(Hk * stride * (d / 2)), c4v(Hk * stride * (d / 2));
    std::vector<float> sk(Hk * stride), sv(Hk * stride);
    for (int h = 0; h < Hk; ++h)
      for (int p = 0; p < stride; ++p) {
        const std::size_t row = static_cast<std::size_t>(h) * stride + p;
        const std::size_t src = row * d;
        float ak = 0.0f, av = 0.0f;
        for (int i = 0; i < d; ++i) {
          ak = std::max(ak, std::fabs(bf16_to_f32_host(ck[src + i])));
          av = std::max(av, std::fabs(bf16_to_f32_host(cv[src + i])));
        }
        const float sks = q4_scale_host(ak), svs = q4_scale_host(av);
        sk[row] = sks;
        sv[row] = svs;
        for (int w = 0; w < d / 2; ++w) {
          c4k[row * (d / 2) + w] = static_cast<std::uint8_t>(
              (q4_quant_host(bf16_to_f32_host(ck[src + 2 * w]), sks) & 0x0fu) |
              (q4_quant_host(bf16_to_f32_host(ck[src + 2 * w + 1]), sks) << 4));
          c4v[row * (d / 2) + w] = static_cast<std::uint8_t>(
              (q4_quant_host(bf16_to_f32_host(cv[src + 2 * w]), svs) & 0x0fu) |
              (q4_quant_host(bf16_to_f32_host(cv[src + 2 * w + 1]), svs) << 4));
        }
      }
    std::vector<std::uint8_t> c4kp(Hk * stride * (d / 2)), c4vp(Hk * stride * (d / 2));
    std::vector<float> skp(Hk * stride), svp(Hk * stride);
    for (int hkv = 0; hkv < Hk; ++hkv)
      for (int pos = 0; pos < stride; ++pos) {
        const std::size_t src = static_cast<std::size_t>(hkv) * stride + pos;
        const std::size_t dst =
            static_cast<std::size_t>(hkv) * stride + static_cast<std::size_t>(perm[pos]);
        std::copy(c4k.begin() + src * (d / 2), c4k.begin() + src * (d / 2) + d / 2,
                  c4kp.begin() + dst * (d / 2));
        std::copy(c4v.begin() + src * (d / 2), c4v.begin() + src * (d / 2) + d / 2,
                  c4vp.begin() + dst * (d / 2));
        skp[dst] = sk[src];
        svp[dst] = sv[src];
      }
    std::uint8_t *dc4k, *dc4v, *dc4kp, *dc4vp;
    float *dsk, *dsv, *dskp, *dsvp;
    if (cudaMalloc(&dc4k, c4k.size()) || cudaMalloc(&dc4v, c4v.size()) ||
        cudaMalloc(&dc4kp, c4kp.size()) || cudaMalloc(&dc4vp, c4vp.size()) ||
        cudaMalloc(&dsk, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dsv, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dskp, sizeof(float) * Hk * stride) ||
        cudaMalloc(&dsvp, sizeof(float) * Hk * stride)) {
      ++g_failures;
      return;
    }
    CUDA_CHECK(cudaMemcpy(dc4k, c4k.data(), c4k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc4v, c4v.data(), c4v.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc4kp, c4kp.data(), c4kp.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc4vp, c4vp.data(), c4vp.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsk, sk.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsv, sv.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dskp, skp.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dsvp, svp.data(), sizeof(float) * Hk * stride, cudaMemcpyHostToDevice));
    expect(granite4::prefill_attention(dq, nullptr, nullptr, scale, Hq, Hk, d, base, rows,
                                       stride, dref, nullptr, nullptr, dsk, dsv, false,
                                       dc4k, dc4v, true) == 0,
           "paged prefill: contiguous kv4 reference");
    expect(granite4::prefill_attention(dq, nullptr, nullptr, scale, Hq, Hk, d, base, rows,
                                       stride, dgotp, nullptr, nullptr, dskp, dsvp, false,
                                       dc4kp, dc4vp, true, dpt) == 0,
           "paged prefill: permuted kv4 launch");
    CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got.data(), dgotp, qn * 2, cudaMemcpyDeviceToHost));
    double max_err4 = 0.0;
    for (std::size_t i = 0; i < qn; ++i)
      max_err4 = std::max(max_err4,
                           static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                       bf16_to_f32_host(ref[i]))));
    expect(max_err4 == 0.0, "paged prefill (kv4): bit-identical to contiguous");
    cudaFree(dc4k);
    cudaFree(dc4v);
    cudaFree(dc4kp);
    cudaFree(dc4vp);
    cudaFree(dsk);
    cudaFree(dsv);
    cudaFree(dskp);
    cudaFree(dsvp);
  }

  cudaFree(dq);
  cudaFree(dref);
  cudaFree(dgotp);
  cudaFree(dpt);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dkp);
  cudaFree(dvp);
}

// Split-KV attention vs fused GQA on long prefixes (several splits), both
// with a host kv_base and with the position read from device memory; the
// device-position single row must equal the corresponding batched row
// bit for bit (decode and verify share one layout).
static void test_attn_split_device() {
  const int Hq = 8, Hk = 2, d = 128, stride = 1500, base = 1203, rows = 4;
  Xorshift64 rng(0x5917);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 0.0078125f * 8.0f;
  std::uint16_t *dq, *dk, *dv, *dref, *dgot, *done;
  int* dpos;
  void* scratch;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dk, cn * 2) || cudaMalloc(&dv, cn * 2) ||
      cudaMalloc(&dref, qn * 2) || cudaMalloc(&dgot, qn * 2) || cudaMalloc(&done, qn * 2) ||
      cudaMalloc(&dpos, sizeof(int)) ||
      cudaMalloc(&scratch, granite4::attn_split_scratch_bytes(Hq, rows, stride))) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_gqa_fused_rows(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                        dref) == 0,
         "split attention: reference launch");
  expect(granite4::attn_split(dq, dk, dv, scale, Hq, Hk, d, base, nullptr, rows, stride,
                              scratch, dgot) == 0,
         "split attention: batched launch");
  const int pos = base + 2;  // row 2 as a device-position decode row
  CUDA_CHECK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
  const std::size_t r2 = static_cast<std::size_t>(2) * Hq * d;
  expect(granite4::attn_split(dq + r2, dk, dv, scale, Hq, Hk, d, 0, dpos, 1, stride, scratch,
                              done) == 0,
         "split attention: device-position launch");
  std::vector<std::uint16_t> ref(qn), got(qn), one(static_cast<std::size_t>(Hq) * d);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got.data(), dgot, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(one.data(), done, one.size() * 2, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  for (std::size_t i = 0; i < qn; ++i)
    max_err = std::max(max_err, static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                              bf16_to_f32_host(ref[i]))));
  std::printf("  split attention: max |delta| = %.5f\n", max_err);
  expect(max_err < 0.03, "split attention: matches fused GQA within bf16 tolerance");
  bool same = true;
  for (std::size_t i = 0; i < one.size(); ++i) same = same && one[i] == got[r2 + i];
  expect(same, "split attention: decode row bit-identical to the batched row");
  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dref); cudaFree(dgot); cudaFree(done);
  cudaFree(dpos); cudaFree(scratch);
}

// Phase 4: paged decode attention. The KV arena need not be contiguous per
// sequence - shared-prefix pages can sit anywhere in the arena, so the row
// holding a logical position is gathered from a device position->row table
// (d_pt). With d_pt null the kernel is the contiguous identity layout; with
// a permutation d_pt the SAME logical cache is read out of a physically
// permuted arena. The paged and contiguous results are bit-identical (same
// split-KV kernel, same logical K/V in the same order), and both match the
// fused-GQA reference within bf16 tolerance (the split path has a per-split
// combine pass, so it is not bit-identical to the fused one).
static void test_paged_attn_split_device() {
  const int Hq = 8, Hk = 2, d = 128, stride = 1500, base = 1203, rows = 4;
  Xorshift64 rng(0x9e57);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 0.0078125f * 8.0f;

  // A fixed permutation of the arena rows: perm[pos] is the physical row that
  // holds logical position pos. Fisher-Yates makes it a real (non-identity)
  // bijection, so the permuted arena is not trivially re-sorted.
  std::vector<int> perm(stride);
  for (int p = 0; p < stride; ++p) perm[p] = p;
  for (int i = stride - 1; i > 0; --i) {
    const int j = static_cast<int>(rng.next() % (i + 1));
    std::swap(perm[i], perm[j]);
  }

  std::uint16_t *dq, *dref, *dgot, *dgotp;
  int *dpos, *dpt;
  void* scratch;
  if (cudaMalloc(&dq, qn * 2) ||
      cudaMalloc(&dref, qn * 2) || cudaMalloc(&dgot, qn * 2) ||
      cudaMalloc(&dgotp, qn * 2) || cudaMalloc(&dpos, sizeof(int)) ||
      cudaMalloc(&dpt, sizeof(int) * stride) ||
      cudaMalloc(&scratch, granite4::attn_split_scratch_bytes(Hq, rows, stride))) {
    ++g_failures;
    return;
  }

  // Contiguous reference: the original arena, no row table.
  std::uint16_t *dk, *dv;
  if (cudaMalloc(&dk, cn * 2) || cudaMalloc(&dv, cn * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_gqa_fused_rows(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                         dref) == 0,
         "paged attention: reference launch");
  expect(granite4::attn_split(dq, dk, dv, scale, Hq, Hk, d, base, nullptr, rows, stride,
                               scratch, dgot) == 0,
         "paged attention: contiguous launch");

  // Build the permuted arena: physical row perm[pos] holds logical position pos.
  std::vector<std::uint16_t> ckp(cn), cvp(cn);
  for (int hkv = 0; hkv < Hk; ++hkv)
    for (int pos = 0; pos < stride; ++pos) {
      const std::size_t src = static_cast<std::size_t>(hkv) * stride * d + static_cast<std::size_t>(pos) * d;
      const std::size_t dst = static_cast<std::size_t>(hkv) * stride * d +
                              static_cast<std::size_t>(perm[pos]) * d;
      std::copy(ck.begin() + src, ck.begin() + src + d, ckp.begin() + dst);
      std::copy(cv.begin() + src, cv.begin() + src + d, cvp.begin() + dst);
    }
  std::uint16_t *dkp, *dvp;
  if (cudaMalloc(&dkp, cn * 2) || cudaMalloc(&dvp, cn * 2)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dkp, ckp.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvp, cvp.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpt, perm.data(), sizeof(int) * stride, cudaMemcpyHostToDevice));

  // Paged: the permuted arena read through the row table must match the
  // contiguous reference bit-for-bit.
  expect(granite4::attn_split(dq, dkp, dvp, scale, Hq, Hk, d, base, nullptr, rows, stride,
                               scratch, dgotp, nullptr, nullptr, nullptr, nullptr, false,
                               nullptr, nullptr, false, dpt) == 0,
         "paged attention: paged launch");

  // The split-KV path and the fused-GQA reference are different algorithms
  // (the split path has a per-split max/sum combine), so they are only
  // bf16-close, not bit-identical - same tolerance as test_attn_split_device.
  // The paged-vs-contiguous comparison, however, is between two calls of the
  // SAME split kernel (one with a null row table, one with a permutation), so
  // it must be bit-identical: permuting the arena through d_pt gathers the
  // same logical K/V rows in the same order, with the same arithmetic.
  std::vector<std::uint16_t> ref(qn), got(qn), gotp(qn);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got.data(), dgot, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gotp.data(), dgotp, qn * 2, cudaMemcpyDeviceToHost));
  double max_err = 0.0;
  bool paged_ok = true;
  for (std::size_t i = 0; i < qn; ++i) {
    max_err = std::max(max_err, static_cast<double>(std::fabs(bf16_to_f32_host(got[i]) -
                                                              bf16_to_f32_host(ref[i]))));
    if (gotp[i] != got[i]) paged_ok = false;
  }
  std::printf("  paged attention: contiguous vs fused max |delta| = %.5f\n", max_err);
  expect(max_err < 0.03, "paged attention: contiguous split matches fused GQA (bf16 tolerance)");
  expect(paged_ok, "paged attention: permuted arena via d_pt == contiguous (bit-identical)");

  cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dkp); cudaFree(dvp);
  cudaFree(dref); cudaFree(dgot); cudaFree(dgotp); cudaFree(dpos); cudaFree(dpt);
  cudaFree(scratch);
}

static void test_paged_gqa_fused_device() {
  // Phase 4: the fused-GQA kernels read KV rows through the same row table
  // (d_pt) the split attention and the batched append use, so a paged KV pool
  // can be read by EITHER attention path. The arena is physically permuted
  // (physical row perm[pos] holds logical position pos) and the row table maps
  // logical position -> physical row. The kernel then gathers the same logical
  // K/V rows in the same order as the contiguous launch, so paged vs
  // contiguous must be bit-identical (same kernel, same arithmetic; the only
  // difference is an extra load from the table). If d_pt were ignored the
  // permuted arena would read different rows and this would fail.
  const int Hq = 8, Hk = 2, d = 128, stride = 1500, base = 1203, rows = 4;
  Xorshift64 rng(0xf1a2);
  const std::size_t qn = static_cast<std::size_t>(rows) * Hq * d;
  const std::size_t cn = static_cast<std::size_t>(Hk) * stride * d;
  std::vector<std::uint16_t> q(qn), ck(cn), cv(cn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  const float scale = 0.0078125f * 8.0f;

  std::vector<int> perm(stride);
  for (int p = 0; p < stride; ++p) perm[p] = p;
  for (int i = stride - 1; i > 0; --i) {
    const int j = static_cast<int>(rng.next() % (i + 1));
    std::swap(perm[i], perm[j]);
  }

  std::uint16_t *dq, *dref, *dgotp;
  std::uint16_t *dk, *dv, *dkp, *dvp;
  int *dpt, *dpos;
  int pos_h = base;
  if (cudaMalloc(&dq, qn * 2) || cudaMalloc(&dref, qn * 2) ||
      cudaMalloc(&dgotp, qn * 2) || cudaMalloc(&dk, cn * 2) ||
      cudaMalloc(&dv, cn * 2) || cudaMalloc(&dkp, cn * 2) ||
      cudaMalloc(&dvp, cn * 2) || cudaMalloc(&dpt, sizeof(int) * stride) ||
      cudaMalloc(&dpos, sizeof(int))) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpt, perm.data(), sizeof(int) * stride, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpos, &pos_h, sizeof(int), cudaMemcpyHostToDevice));
  expect(granite4_kernel_gqa_fused_rows(dq, dk, dv, scale, Hq, Hk, d, base, rows, stride,
                                        dref) == 0,
         "paged fused: contiguous reference launch");

  // Build the permuted arena, same construction as the paged split test.
  std::vector<std::uint16_t> ckp(cn), cvp(cn);
  for (int hkv = 0; hkv < Hk; ++hkv)
    for (int pos = 0; pos < stride; ++pos) {
      const std::size_t src = static_cast<std::size_t>(hkv) * stride * d +
                              static_cast<std::size_t>(pos) * d;
      const std::size_t dst = static_cast<std::size_t>(hkv) * stride * d +
                              static_cast<std::size_t>(perm[pos]) * d;
      std::copy(ck.begin() + src, ck.begin() + src + d, ckp.begin() + dst);
      std::copy(cv.begin() + src, cv.begin() + src + d, cvp.begin() + dst);
    }
  CUDA_CHECK(cudaMemcpy(dkp, ckp.data(), cn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvp, cvp.data(), cn * 2, cudaMemcpyHostToDevice));
  expect(granite4_kernel_gqa_fused_rows(dq, dkp, dvp, scale, Hq, Hk, d, base, rows, stride,
                                        dgotp, dpt) == 0,
         "paged fused: paged launch (rows)");

  std::vector<std::uint16_t> ref(qn), gotp(qn);
  CUDA_CHECK(cudaMemcpy(ref.data(), dref, qn * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gotp.data(), dgotp, qn * 2, cudaMemcpyDeviceToHost));
  bool paged_ok = true;
  for (std::size_t i = 0; i < qn; ++i)
    if (gotp[i] != ref[i]) paged_ok = false;
  expect(paged_ok,
         "paged fused: permuted arena via d_pt == contiguous (bit-identical, rows)");

  // Single-row decode form (d_pos + d_pt): the captured-graph replay path.
  // One query row at position base; kv_len bounds the shared memory.
  std::uint16_t *dref1, *dgot1;
  if (cudaMalloc(&dref1, static_cast<std::size_t>(Hq) * d * 2) ||
      cudaMalloc(&dgot1, static_cast<std::size_t>(Hq) * d * 2)) {
    cudaFree(dq); cudaFree(dref); cudaFree(dgotp);
    cudaFree(dk); cudaFree(dv); cudaFree(dkp); cudaFree(dvp);
    cudaFree(dpt); cudaFree(dpos);
    ++g_failures;
    return;
  }
  const int kv1 = base + 1;
  expect(granite4_kernel_gqa_fused(dq, dk, dv, scale, Hq, Hk, d, kv1, stride,
                                   dref1) == 0,
         "paged fused: single-row contiguous launch");
  expect(granite4_kernel_gqa_fused_pos(dq, dkp, dvp, scale, Hq, Hk, d, kv1, stride,
                                       dpos, dgot1, dpt) == 0,
         "paged fused: single-row paged launch (d_pos + d_pt)");
  const std::size_t n1 = static_cast<std::size_t>(Hq) * d;
  std::vector<std::uint16_t> ref1(n1), got1(n1);
  CUDA_CHECK(cudaMemcpy(ref1.data(), dref1, n1 * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got1.data(), dgot1, n1 * 2, cudaMemcpyDeviceToHost));
  bool paged1_ok = true;
  for (std::size_t i = 0; i < n1; ++i)
    if (got1[i] != ref1[i]) paged1_ok = false;
  expect(paged1_ok,
         "paged fused: permuted arena via d_pt == contiguous (bit-identical, d_pos)");

  cudaFree(dq); cudaFree(dref); cudaFree(dgotp);
  cudaFree(dref1); cudaFree(dgot1);
  cudaFree(dk); cudaFree(dv); cudaFree(dkp); cudaFree(dvp);
  cudaFree(dpt); cudaFree(dpos);
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
                                           stride, Hk, d, nullptr) == 0,
         "batched kv append: launch");
  expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, 17, stride + 64,
                                           Hk, d, nullptr) == -1,
         "batched kv append: rejects rows > 16");
  expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, rows,
                                           base + rows - 1, Hk, d, nullptr) == -1,
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

static void test_paged_kv_append_batched_device() {
   // Phase 4: batched append must write into arena rows d_pt[pos] (not the
   // contiguous pos), so that a paged KV pool can share a physical arena
   // across sequences while the logical cache remains a contiguous view
   // through the row table. The row table is the same one attention reads
   // through, so the append lands where a later read looks.
   //
   // We compare two launches of the SAME batched kernel on the SAME source
   // KV:
   //   (a) d_pt == nullptr (contiguous identity layout)
   //   (b) d_pt == perm (a physical permutation)
   // The kernel writes logical position pos to arena row d_pt[pos] == perm[pos],
   // so the paged cache's physical row perm[pos] must equal the contiguous
   // cache's physical row pos. Un-permuting the paged result via the same
   // table yields a bit-identical result.
   const int Hk = 8, d = 128, base = 5, rows = 4, stride = 64;
   Xorshift64 rng(0x5050);
   const std::size_t src_elems = static_cast<std::size_t>(rows) * Hk * d;
   std::vector<std::uint16_t> sk(src_elems), sv(src_elems);
   for (auto& t : sk) t = f32_to_bf16_host(rng.next_float_range(2.0));
   for (auto& t : sv) t = f32_to_bf16_host(rng.next_float_range(2.0));

   const int max_pos = 64;
   std::vector<int> perm(max_pos);
   for (int r = 0; r < max_pos; ++r) perm[r] = r;
   std::swap(perm[base], perm[base + rows - 1]);
   std::swap(perm[base + 1], perm[base + rows - 2]);

   const std::size_t cache_elems = static_cast<std::size_t>(Hk) * max_pos * d;
   std::vector<std::uint16_t> host_c_contig(cache_elems, 0), host_c_paged(cache_elems, 0);
   std::vector<std::uint16_t> host_c_contig_v(cache_elems, 0), host_c_paged_v(cache_elems, 0);
   std::uint16_t *dsk, *dsv, *dak, *dav, *dcp, *dcp_v;
   int* dperm;
   const std::size_t bytes = cache_elems * sizeof(std::uint16_t);
   CUDA_CHECK(cudaMalloc(&dsk, src_elems * sizeof(std::uint16_t)));
   CUDA_CHECK(cudaMalloc(&dsv, src_elems * sizeof(std::uint16_t)));
   CUDA_CHECK(cudaMalloc(&dak, bytes));
   CUDA_CHECK(cudaMalloc(&dav, bytes));
   CUDA_CHECK(cudaMalloc(&dcp, bytes));
   CUDA_CHECK(cudaMalloc(&dcp_v, bytes));
   CUDA_CHECK(cudaMalloc(&dperm, max_pos * sizeof(int)));
   CUDA_CHECK(cudaMemset(dak, 0, bytes));
   CUDA_CHECK(cudaMemset(dav, 0, bytes));
   CUDA_CHECK(cudaMemset(dcp, 0, bytes));
   CUDA_CHECK(cudaMemset(dcp_v, 0, bytes));
   CUDA_CHECK(cudaMemcpy(dsk, sk.data(), src_elems * 2, cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(dsv, sv.data(), src_elems * 2, cudaMemcpyHostToDevice));
   CUDA_CHECK(cudaMemcpy(dperm, perm.data(), max_pos * sizeof(int),
                         cudaMemcpyHostToDevice));

   expect(granite4_kernel_kv_append_batched(dsk, dsv, dak, dav, base, rows, stride,
                                            Hk, d, nullptr) == 0,
          "paged append: contiguous launch");
   expect(granite4_kernel_kv_append_batched(dsk, dsv, dcp, dcp_v, base, rows, stride,
                                            Hk, d, dperm) == 0,
          "paged append: paged launch");

   CUDA_CHECK(cudaMemcpy(host_c_contig.data(), dak, bytes, cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(host_c_contig_v.data(), dav, bytes, cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(host_c_paged.data(), dcp, bytes, cudaMemcpyDeviceToHost));
   CUDA_CHECK(cudaMemcpy(host_c_paged_v.data(), dcp_v, bytes, cudaMemcpyDeviceToHost));
   bool ok = true;
   for (int hkv = 0; hkv < Hk; ++hkv) {
     for (int pos = 0; pos < stride; ++pos) {
       const int phys = perm[pos];
       const std::size_t ci = static_cast<std::size_t>(hkv) * stride * d +
                              static_cast<std::size_t>(pos) * d;
       const std::size_t pi = static_cast<std::size_t>(hkv) * stride * d +
                              static_cast<std::size_t>(phys) * d;
       for (int i = 0; i < d; ++i) {
         if (host_c_paged[pi + i] != host_c_contig[ci + i]) ok = false;
         if (host_c_paged_v[pi + i] != host_c_contig_v[ci + i]) ok = false;
       }
     }
   }
   expect(ok,
          "paged append: permuted arena via d_pt == contiguous (bit-identical)");

   cudaFree(dsk); cudaFree(dsv); cudaFree(dak); cudaFree(dav);
   cudaFree(dcp); cudaFree(dcp_v); cudaFree(dperm);
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
  expect(granite4_kernel_rmsnorm_batched(dx, dw, eps, 17, cols, db) == -1,
         "batched rmsnorm: rejects rows > 16");
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
  expect(granite4_kernel_rope_at_pos_batched(dx, dc, dsin, 17, heads, d,
                                             db) == -1,
         "batched rope: rejects rows > 16");
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

// Full-model parity: the batched suffix forward (forward_range_batched_impl,
// auto-selected for 2-4-row incremental suffixes) must be bit-identical to
// the per-position single-token loop, which the draft-verify loop relies on
// for an exact greedy gate. Covers:
//   * batch widths 3, 4, 3, 2 in one ctx with single-token decode,
//   * batched attention reading the persistent KV written by single-token
//     prefills (mixed KV provenance),
//   * truncate + recompute: batched rows appended then rolled back and
//     rewritten by a later forward (the draft-loop rejection path).
static void test_batched_forward_range_parity_device() {
  ModelConfig model;
  model.num_layers = 2;
  model.hidden_size = 128;
  model.intermediate_size = 96;
  model.num_attention_heads = 4;
  model.num_key_value_heads = 2;
  model.head_dim = 32;
  model.vocab_size = 512;
  model.max_context_length = 131072;
  ForwardSpec spec;  // defaults

  const int L = static_cast<int>(model.num_layers);
  const int H = static_cast<int>(model.hidden_size);
  const int I = static_cast<int>(model.intermediate_size);
  const int Hq = static_cast<int>(model.num_attention_heads);
  const int Hk = static_cast<int>(model.num_key_value_heads);
  const int d = static_cast<int>(model.head_dim);
  const int V = static_cast<int>(model.vocab_size);
  const int max_pos = 32;
  const int seq_len = 30;

  const auto z = [](int v) { return static_cast<std::size_t>(v); };
  const GpuWeightsLayout lay{z(H), z(I), z(V), z(L), z(Hq), z(Hk), z(d)};
  Xorshift64 rng(0xF0F0);
  auto gen_bf16v = [&]() { return gen_bf16(rng); };
  auto gen_fp8 = [&]() { return fp8_e4m3_encode(rng.next_float_range(2.0), nullptr); };

  // Packed host weights in GpuWeightsLayout order.
  std::vector<std::uint8_t> weights(lay.total_bytes());
  auto fill_bf16 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i) {
      std::uint16_t v = gen_bf16v();
      weights[off + 2 * i] = static_cast<std::uint8_t>(v & 0xffu);
      weights[off + 2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
  };
  auto fill_fp8 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i) weights[off + i] = gen_fp8();
  };
  // embedding
  fill_bf16(0, static_cast<std::size_t>(V) * H);
  for (int layer = 0; layer < L; ++layer) {
    const std::size_t base =
        lay.embed_bytes() + static_cast<std::size_t>(layer) * lay.layer_bytes();
    fill_bf16(base + lay.off_attn_norm(), H);
    fill_bf16(base + lay.off_ffn_norm(), H);
    fill_fp8(base + lay.off_q(), static_cast<std::size_t>(H) * H);
    fill_fp8(base + lay.off_k(), static_cast<std::size_t>(Hk) * d * H);
    fill_fp8(base + lay.off_v(), static_cast<std::size_t>(Hk) * d * H);
    fill_fp8(base + lay.off_o(), static_cast<std::size_t>(H) * H);
    fill_fp8(base + lay.off_gate(), static_cast<std::size_t>(I) * H);
    fill_fp8(base + lay.off_up(), static_cast<std::size_t>(I) * H);
    fill_fp8(base + lay.off_down(), static_cast<std::size_t>(H) * I);
  }
  fill_bf16(lay.final_norm_off(), H);
  fill_fp8(lay.lm_head_off(), static_cast<std::size_t>(V) * H);
  std::vector<float> scales(lay.num_scales());
  for (auto& s : scales) s = 0.1f + 0.9f * rng.next_float_range(1.0);

  // Token sequence (ids < V).
  std::vector<std::int32_t> tokens(seq_len);
  for (int i = 0; i < seq_len; ++i) tokens[i] = (i * 97 + 13) % V;

  std::int32_t* d_tokens;
  if (cudaMalloc(&d_tokens, seq_len * sizeof(std::int32_t)) != cudaSuccess) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), seq_len * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice));

  // The per-position loop is the reference; the batched suffix forward is
  // what the draft-verify loop relies on. G4_BATCH_VERIFY is read at forward
  // time and is toggled around each forward in the step loop below.
  GpuCtx* ref_ctx = gpu_ctx_create(model, spec, max_pos);
  expect(ref_ctx != nullptr, "parity: ref ctx created");
  char err[256];
  if (ref_ctx != nullptr)
    expect(gpu_weights_load(ref_ctx, weights.data(), weights.size(),
                            scales.data(), scales.size(), err, sizeof(err)) == 0,
           "parity: ref weights loaded");
  GpuCtx* batch_ctx = gpu_ctx_create(model, spec, max_pos);
  expect(batch_ctx != nullptr, "parity: batch ctx created");
  if (batch_ctx != nullptr)
    expect(gpu_weights_load(batch_ctx, weights.data(), weights.size(),
                            scales.data(), scales.size(), err, sizeof(err)) == 0,
           "parity: batch weights loaded");
  if (ref_ctx == nullptr || batch_ctx == nullptr) {
    gpu_ctx_free(ref_ctx);
    gpu_ctx_free(batch_ctx);
    cudaFree(d_tokens);
    return;
  }

  std::uint16_t* d_logits;
  if (cudaMalloc(&d_logits, static_cast<std::size_t>(max_pos) * V * 2) != cudaSuccess) {
    ++g_failures;
    gpu_ctx_free(ref_ctx);
    gpu_ctx_free(batch_ctx);
    cudaFree(d_tokens);
    return;
  }

  // Step plan mirroring the draft loop: a short prefill (end 3, the
  // watermark the verify loop starts from), then incremental suffix forwards
  // of widths 4, 3, 2 (end 7, 10, 12), a rows==1 decode (end 13) that stays
  // on the per-position path in both ctxs, the widest batched suffix (16
  // rows, end 29) and one more decode. Both ctxs share one d_logits buffer,
  // so capture each step's fresh rows immediately after the forward that
  // wrote them (before the other ctx overwrites them).
  const int ends[] = {3, 7, 10, 12, 13, 29, 30};
  constexpr int kSteps = static_cast<int>(sizeof(ends) / sizeof(ends[0]));
  std::vector<std::uint16_t> ref_rows, batch_rows;  // accumulated logit rows
  for (int e = 0; e < kSteps; ++e) {
    const int end = ends[e];
    const int prev = e > 0 ? ends[e - 1] : 0;
    const std::size_t rows = static_cast<std::size_t>(end - prev) * V;
    std::vector<std::uint16_t> chunk(rows);
    // G4_BATCH_VERIFY is read at forward time (forward_range_impl), so
    // toggle it per call: '0' forces the per-position loop for the ref ctx.
    setenv("G4_BATCH_VERIFY", "0", 1);
    const bool ref_ok =
        gpu_forward_range(ref_ctx, d_tokens, end, d_logits, err, sizeof(err)) == 0;
    expect(ref_ok, "parity: ref forward step");
    if (ref_ok && rows > 0) {
      CUDA_CHECK(cudaMemcpy(chunk.data(), d_logits + prev * V, rows * 2,
                            cudaMemcpyDeviceToHost));
      ref_rows.insert(ref_rows.end(), chunk.begin(), chunk.end());
    }
    // Back to the default (batched) for the batch ctx.
    unsetenv("G4_BATCH_VERIFY");
    const bool batch_ok =
        gpu_forward_range(batch_ctx, d_tokens, end, d_logits, err, sizeof(err)) == 0;
    expect(batch_ok, "parity: batch forward step");
    if (batch_ok && rows > 0) {
      CUDA_CHECK(cudaMemcpy(chunk.data(), d_logits + prev * V, rows * 2,
                            cudaMemcpyDeviceToHost));
      batch_rows.insert(batch_rows.end(), chunk.begin(), chunk.end());
    }
  }

  // The batched suffix forwards (widths 2..16) must reproduce the
  // per-position loop's logits bit for bit: every stage of the verify
  // forward shares its decode twin's per-row arithmetic (verify_gemv.cu,
  // same RMSNorm / RoPE / attention kernels), which is what makes the greedy
  // verify gate exact.
  constexpr int kUlpTol = 0;
  expect(ref_rows.size() == batch_rows.size(), "parity: row counts match");
  if (ref_rows.size() == batch_rows.size()) {
    std::size_t bad = 0, bad_argmax = 0;
    const int ulp = kUlpTol;
    auto argmax = [](const std::uint16_t* p, int V) {
      int bi = 0; float bv = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < V; ++i) {
        const float f = bf16_to_f32_host(p[i]);
        if (f > bv || (f == bv && i < bi)) { bv = f; bi = i; }
      }
      return bi;
    };
    for (int r = 0; r < static_cast<int>(ref_rows.size() / V); ++r) {
      const std::uint16_t* a = ref_rows.data() + static_cast<std::size_t>(r) * V;
      const std::uint16_t* b = batch_rows.data() + static_cast<std::size_t>(r) * V;
      for (int i = 0; i < V; ++i)
        if (!bf16_close(a[i], b[i], ulp)) ++bad;
      if (argmax(a, V) != argmax(b, V)) ++bad_argmax;
    }
    expect(bad == 0, "parity: batched suffix forward bit-identical to per-position");
    expect(bad_argmax == 0,
           "parity: greedy argmax stable (batched vs per-position)");
  }

  // Rollback: drop the KV rows [10, 13) the previous step wrote and
  // re-forward [10, 12) - the batch ctx takes the batched path again
  // (rows == 2), overwriting its stale KV, exactly the draft-loop rejection
  // sequence. Then roll back further (to 9) and re-forward the width-3
  // suffix [9, 12) the same way.
  expect(gpu_ctx_truncate(ref_ctx, 10) == 0, "parity: ref truncate to 10");
  expect(gpu_ctx_truncate(batch_ctx, 10) == 0, "parity: batch truncate to 10");
  std::vector<std::uint16_t> ref_redo(2 * V), batch_redo(2 * V);
  setenv("G4_BATCH_VERIFY", "0", 1);
  expect(gpu_forward_range(ref_ctx, d_tokens, 12, d_logits, err, sizeof(err)) == 0,
         "parity: ref recompute to 12");
  unsetenv("G4_BATCH_VERIFY");
  expect(gpu_forward_range(batch_ctx, d_tokens, 12, d_logits, err, sizeof(err)) == 0,
         "parity: batch recompute to 12");
  CUDA_CHECK(cudaMemcpy(ref_redo.data(), d_logits + 10 * V, 2 * V * 2,
                         cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(batch_redo.data(), d_logits + 10 * V, 2 * V * 2,
                         cudaMemcpyDeviceToHost));
  {
    int bad = 0;
    for (int i = 0; i < 2 * V; ++i)
      if (!bf16_close(ref_redo[i], batch_redo[i], kUlpTol)) ++bad;
    expect(bad == 0,
           "parity: truncate + batched recompute (width 2) bit-identical");
  }
  expect(gpu_ctx_truncate(ref_ctx, 9) == 0, "parity: ref truncate to 9");
  expect(gpu_ctx_truncate(batch_ctx, 9) == 0, "parity: batch truncate to 9");
  setenv("G4_BATCH_VERIFY", "0", 1);
  expect(gpu_forward_range(ref_ctx, d_tokens, 12, d_logits, err, sizeof(err)) == 0,
         "parity: ref recompute to 12 (width 3)");
  unsetenv("G4_BATCH_VERIFY");
  expect(gpu_forward_range(batch_ctx, d_tokens, 12, d_logits, err, sizeof(err)) == 0,
         "parity: batch recompute to 12 (width 3)");
  CUDA_CHECK(cudaMemcpy(ref_redo.data(), d_logits + 10 * V, 2 * V * 2,
                         cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(batch_redo.data(), d_logits + 10 * V, 2 * V * 2,
                         cudaMemcpyDeviceToHost));
  {
    int bad = 0;
    for (int i = 0; i < 2 * V; ++i)
      if (!bf16_close(ref_redo[i], batch_redo[i], kUlpTol)) ++bad;
    expect(bad == 0,
           "parity: truncate + batched recompute (width 3) bit-identical");
  }

  expect(gpu_ctx_written(ref_ctx) == 12, "parity: ref watermark");
  expect(gpu_ctx_written(batch_ctx) == 12, "parity: batch watermark");

  cudaFree(d_logits);
  cudaFree(d_tokens);
  gpu_ctx_free(ref_ctx);
  gpu_ctx_free(batch_ctx);
}

// Two sessions on ONE shared KV arena (P4): each context's KV lives in
// interleaved, non-contiguous rows of the same pool, the contexts are driven
// alternately (prefill, decode, a batched verify width), and every logits row
// must equal - bit for bit - what a private [max_pos] cache produces for the
// same tokens. Covers all three persistent-KV formats (bf16, KV8, KV4), the
// unmapped-row guard, row-table validation, and a shared prefill scratch.
static void test_shared_arena_sessions_device() {
  ModelConfig model;
  model.num_layers = 2;
  model.hidden_size = 512;  // == Hq * d, as the synthetic q projection assumes
  model.intermediate_size = 256;
  model.num_attention_heads = 4;
  model.num_key_value_heads = 1;
  model.head_dim = 128;  // split-KV attention shape (d == 128, Hq == 4 * Hk)
  model.vocab_size = 512;
  model.max_context_length = 131072;
  ForwardSpec spec;
  const int L = 2, H = 512, I = 256, Hq = 4, Hk = 1, d = 128, V = 512;
  const int max_pos = 64, arena_rows = 192;
  const auto z = [](int v) { return static_cast<std::size_t>(v); };
  const GpuWeightsLayout lay{z(H), z(I), z(V), z(L), z(Hq), z(Hk), z(d)};
  Xorshift64 rng(0xA7E4A);
  std::vector<std::uint8_t> weights(lay.total_bytes());
  auto fill_bf16 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i) {
      const std::uint16_t v = gen_bf16(rng);
      weights[off + 2 * i] = static_cast<std::uint8_t>(v & 0xffu);
      weights[off + 2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
  };
  auto fill_fp8 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i)
      weights[off + i] = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  };
  fill_bf16(0, z(V) * H);
  for (int layer = 0; layer < L; ++layer) {
    const std::size_t base = lay.embed_bytes() + z(layer) * lay.layer_bytes();
    fill_bf16(base + lay.off_attn_norm(), H);
    fill_bf16(base + lay.off_ffn_norm(), H);
    fill_fp8(base + lay.off_q(), z(H) * H);
    fill_fp8(base + lay.off_k(), z(Hk) * d * H);
    fill_fp8(base + lay.off_v(), z(Hk) * d * H);
    fill_fp8(base + lay.off_o(), z(H) * H);
    fill_fp8(base + lay.off_gate(), z(I) * H);
    fill_fp8(base + lay.off_up(), z(I) * H);
    fill_fp8(base + lay.off_down(), z(H) * I);
  }
  fill_bf16(lay.final_norm_off(), H);
  fill_fp8(lay.lm_head_off(), z(V) * H);
  std::vector<float> scales(lay.num_scales());
  for (auto& s : scales) s = 0.1f + 0.9f * rng.next_float_range(1.0);

  const int seq_len = 32;
  std::vector<std::int32_t> tok[2];
  for (int s = 0; s < 2; ++s) {
    tok[s].resize(seq_len);
    for (int i = 0; i < seq_len; ++i) tok[s][i] = (i * (s == 0 ? 97 : 57) + (s == 0 ? 13 : 201)) % V;
  }
  std::int32_t* d_tok[2] = {nullptr, nullptr};
  std::uint16_t* d_logits = nullptr;
  bool alloc_ok = cudaMalloc(&d_logits, z(max_pos) * V * 2) == cudaSuccess;
  for (int s = 0; s < 2; ++s) {
    alloc_ok = alloc_ok && cudaMalloc(&d_tok[s], z(seq_len) * 4) == cudaSuccess;
    if (alloc_ok)
      CUDA_CHECK(cudaMemcpy(d_tok[s], tok[s].data(), z(seq_len) * 4, cudaMemcpyHostToDevice));
  }
  if (!alloc_ok) {
    ++g_failures;
    return;
  }
  // 20-row prefill, two single decodes, an 8-row batched verify, a decode.
  const int ends[] = {20, 21, 22, 30, 31};
  constexpr int kSteps = 5;
  char err[256];
  const char* const modes[] = {"0", "1", "4"};
  const char* const mode_name[] = {"bf16", "KV8", "KV4"};
  for (int m = 0; m < 3; ++m) {
    setenv("G4_KV8", modes[m], 1);
    std::string tag = std::string("shared arena [") + mode_name[m] + "]: ";
    // Reference: one private-cache context, run once per token sequence.
    GpuCtx* ref = gpu_ctx_create(model, spec, max_pos);
    expect(ref != nullptr, (tag + "ref ctx").c_str());
    if (ref == nullptr ||
        gpu_weights_load(ref, weights.data(), weights.size(), scales.data(), scales.size(), err,
                         sizeof(err)) != 0) {
      ++g_failures;
      gpu_ctx_free(ref);
      continue;
    }
    std::vector<std::uint16_t> want[2];
    for (int s = 0; s < 2; ++s) {
      expect(gpu_ctx_truncate(ref, 0) == 0, (tag + "ref reset").c_str());
      for (int e = 0; e < kSteps; ++e) {
        const int prev = e > 0 ? ends[e - 1] : 0;
        const bool ok = gpu_forward_range(ref, d_tok[s], ends[e], d_logits, err, sizeof(err)) == 0;
        expect(ok, (tag + "ref forward").c_str());
        if (!ok) break;
        std::vector<std::uint16_t> rows(z(ends[e] - prev) * V);
        CUDA_CHECK(cudaMemcpy(rows.data(), d_logits + z(prev) * V, rows.size() * 2,
                              cudaMemcpyDeviceToHost));
        want[s].insert(want[s].end(), rows.begin(), rows.end());
      }
    }
    gpu_ctx_free(ref);
    // The shared arena, one weight object, and two contexts on them.
    GpuKvArena* arena = gpu_kv_arena_create(model, arena_rows, err, sizeof(err));
    GpuWeights* wobj = gpu_weights_create();
    expect(arena != nullptr && wobj != nullptr, (tag + "arena + weights").c_str());
    if (arena == nullptr || wobj == nullptr ||
        gpu_weights_load(wobj, weights.data(), weights.size(), scales.data(), scales.size(), err,
                         sizeof(err)) != 0) {
      ++g_failures;
      gpu_kv_arena_free(arena);
      gpu_weights_free_obj(wobj);
      continue;
    }
    expect(gpu_ctx_create_paged(model, spec, arena_rows + 1, arena) == nullptr,
           (tag + "a context longer than the pool is rejected").c_str());
    GpuCtx* c[2];
    for (int s = 0; s < 2; ++s) {
      c[s] = gpu_ctx_create_paged(model, spec, max_pos, arena);
      expect(c[s] != nullptr && gpu_ctx_use_weights(c[s], wobj) == 0,
             (tag + "paged ctx").c_str());
    }
    if (c[0] == nullptr || c[1] == nullptr) {
      ++g_failures;
      gpu_ctx_free(c[0]);
      gpu_ctx_free(c[1]);
      gpu_kv_arena_free(arena);
      gpu_weights_free_obj(wobj);
      continue;
    }
    expect(gpu_ctx_share_prefill_scratch(c[1], c[0]) == 0, (tag + "share prefill scratch").c_str());
    // Unmapped contexts must refuse to run.
    expect(gpu_forward_range(c[0], d_tok[0], 5, d_logits, err, sizeof(err)) != 0,
           (tag + "forward before mapping rows is rejected").c_str());
    // Row tables: A on the even rows, B on the odd rows in descending order -
    // interleaved and non-contiguous inside one arena.
    std::vector<int> rows_a(max_pos), rows_b(max_pos);
    for (int i = 0; i < max_pos; ++i) {
      rows_a[i] = 2 * i;
      rows_b[i] = 2 * (max_pos - 1 - i) + 1;
    }
    std::vector<int> dup = rows_a;
    dup[7] = dup[3];
    std::vector<int> oob = rows_a;
    oob[5] = arena_rows;
    expect(gpu_ctx_set_kv_rows_n(c[0], dup.data(), max_pos) != 0,
           (tag + "duplicate rows rejected").c_str());
    expect(gpu_ctx_set_kv_rows_n(c[0], oob.data(), max_pos) != 0,
           (tag + "out-of-pool row rejected").c_str());
    expect(gpu_ctx_set_kv_rows_n(c[0], rows_a.data(), max_pos + 1) != 0,
           (tag + "more rows than max_pos rejected").c_str());
    expect(gpu_ctx_mapped_rows(c[0]) == 0, (tag + "failed maps leave it unmapped").c_str());
    expect(gpu_ctx_set_kv_rows_n(c[0], rows_a.data(), max_pos) == 0 &&
               gpu_ctx_set_kv_rows_n(c[1], rows_b.data(), 40) == 0,
           (tag + "map rows").c_str());
    expect(gpu_forward_range(c[1], d_tok[1], 41, d_logits, err, sizeof(err)) != 0,
           (tag + "forward past the mapped rows is rejected").c_str());
    expect(gpu_ctx_set_kv_rows_n(c[1], rows_b.data(), max_pos) == 0, (tag + "remap B").c_str());
    // Alternate the two sessions step by step; capture each fresh logits row
    // block right after the forward that wrote it.
    std::vector<std::uint16_t> got[2];
    bool ran = true;
    for (int e = 0; e < kSteps && ran; ++e) {
      const int prev = e > 0 ? ends[e - 1] : 0;
      for (int s = 0; s < 2; ++s) {
        ran = gpu_forward_range(c[s], d_tok[s], ends[e], d_logits, err, sizeof(err)) == 0;
        expect(ran, (tag + "paged forward").c_str());
        if (!ran) break;
        std::vector<std::uint16_t> rows(z(ends[e] - prev) * V);
        CUDA_CHECK(cudaMemcpy(rows.data(), d_logits + z(prev) * V, rows.size() * 2,
                              cudaMemcpyDeviceToHost));
        got[s].insert(got[s].end(), rows.begin(), rows.end());
      }
    }
    bool all_same = true;
    for (int s = 0; s < 2; ++s) {
      const bool same = got[s].size() == want[s].size() && !want[s].empty() &&
                        std::memcmp(got[s].data(), want[s].data(), want[s].size() * 2) == 0;
      expect(same, (tag + "session " + std::to_string(s) +
                    " logits are bit-identical to a private cache").c_str());
      all_same = all_same && same;
    }
    if (all_same)
      std::printf("  shared arena [%s]: 2 interleaved sessions (%zu logits each) "
                  "bit-identical to private caches\n",
                  mode_name[m], got[0].size());
    // Rollback + recompute on a shared arena (the draft-loop rejection path):
    // truncate A, re-forward the same range, and B's rows must be untouched.
    if (ran) {
      std::vector<std::uint16_t> before(z(V)), after(z(V));
      expect(gpu_ctx_truncate(c[0], 25) == 0, (tag + "truncate A").c_str());
      expect(gpu_forward_range(c[0], d_tok[0], 31, d_logits, err, sizeof(err)) == 0,
             (tag + "recompute A").c_str());
      CUDA_CHECK(cudaMemcpy(after.data(), d_logits + z(30) * V, z(V) * 2, cudaMemcpyDeviceToHost));
      std::vector<std::uint16_t> want_last(want[0].end() - V, want[0].end());
      expect(std::memcmp(after.data(), want_last.data(), z(V) * 2) == 0,
             (tag + "recompute after truncate matches").c_str());
      expect(gpu_ctx_truncate(c[1], 25) == 0, (tag + "truncate B").c_str());
      expect(gpu_forward_range(c[1], d_tok[1], 31, d_logits, err, sizeof(err)) == 0,
             (tag + "recompute B").c_str());
      CUDA_CHECK(cudaMemcpy(after.data(), d_logits + z(30) * V, z(V) * 2, cudaMemcpyDeviceToHost));
      std::vector<std::uint16_t> want_last_b(want[1].end() - V, want[1].end());
      expect(std::memcmp(after.data(), want_last_b.data(), z(V) * 2) == 0,
             (tag + "B unaffected by A's recompute").c_str());
    }
    gpu_ctx_free(c[1]);  // sharer before the scratch owner
    gpu_ctx_free(c[0]);
    gpu_kv_arena_free(arena);
    gpu_weights_free_obj(wobj);
  }
  unsetenv("G4_KV8");
  cudaFree(d_logits);
  cudaFree(d_tok[0]);
  cudaFree(d_tok[1]);
}
// Multi-row split attention == per-row single-row calls, bit for bit. The
// batched verify (EAGLE / draft-lookup) is only exact if every verify row
// equals the decode step at that position, so a K-row attn_split must equal K
// rows=1 calls - in every KV format, through a paged row table, on a pool
// wider than the sequence bound (max_len < stride), across row groups.
static void test_attn_split_rows_bit_identical_device() {
  const int Hq = 8, Hk = 2, d = 128, stride = 3000, max_len = 2100;
  Xorshift64 rng(0x5eed7);
  const std::size_t rowsz = static_cast<std::size_t>(Hk) * stride;  // (head, row) slots
  std::vector<std::uint16_t> ck(rowsz * d), cv(rowsz * d);
  for (auto& x : ck) x = f32_to_bf16_host(rng.next_float_range(2.0));
  for (auto& x : cv) x = f32_to_bf16_host(rng.next_float_range(2.0));
  std::vector<std::int8_t> k8(rowsz * d), v8(rowsz * d);
  for (auto& x : k8) x = static_cast<std::int8_t>(static_cast<int>(rng.next() % 255) - 127);
  for (auto& x : v8) x = static_cast<std::int8_t>(static_cast<int>(rng.next() % 255) - 127);
  std::vector<std::uint8_t> k4(rowsz * d / 2), v4(rowsz * d / 2);
  for (auto& x : k4) x = static_cast<std::uint8_t>(rng.next() & 0xffu);
  for (auto& x : v4) x = static_cast<std::uint8_t>(rng.next() & 0xffu);
  std::vector<float> ks(rowsz), vs(rowsz);
  for (auto& x : ks) x = 0.002f + 0.02f * rng.next_float_range(1.0);
  for (auto& x : vs) x = 0.002f + 0.02f * rng.next_float_range(1.0);
  // Paged table: logical positions scattered over the wider pool.
  std::vector<int> pool(stride);
  for (int i = 0; i < stride; ++i) pool[i] = i;
  for (int i = stride - 1; i > 0; --i) std::swap(pool[i], pool[rng.next() % (i + 1)]);
  std::vector<int> pt(pool.begin(), pool.begin() + max_len);
  const int kMaxR = 13;
  const std::size_t qn = static_cast<std::size_t>(kMaxR) * Hq * d;
  std::vector<std::uint16_t> q(qn);
  for (auto& x : q) x = f32_to_bf16_host(rng.next_float_range(2.0));
  std::uint16_t *dk, *dv, *dq, *dmulti, *dsingle;
  std::int8_t *dk8, *dv8;
  std::uint8_t *dk4, *dv4;
  float *dks, *dvs;
  int* dpt;
  void* scratch;
  const std::size_t scr = granite4::attn_split_scratch_bytes(Hq, kMaxR, max_len);
  if (cudaMalloc(&dk, ck.size() * 2) || cudaMalloc(&dv, cv.size() * 2) ||
      cudaMalloc(&dk8, k8.size()) || cudaMalloc(&dv8, v8.size()) ||
      cudaMalloc(&dk4, k4.size()) || cudaMalloc(&dv4, v4.size()) ||
      cudaMalloc(&dks, ks.size() * 4) || cudaMalloc(&dvs, vs.size() * 4) ||
      cudaMalloc(&dq, qn * 2) || cudaMalloc(&dmulti, qn * 2) || cudaMalloc(&dsingle, qn * 2) ||
      cudaMalloc(&dpt, pt.size() * 4) || cudaMalloc(&scratch, scr)) {
    ++g_failures;
    return;
  }
  CUDA_CHECK(cudaMemcpy(dk, ck.data(), ck.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, cv.data(), cv.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk8, k8.data(), k8.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk4, k4.data(), k4.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv4, v4.data(), v4.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dks, ks.data(), ks.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvs, vs.data(), vs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq, q.data(), qn * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dpt, pt.data(), pt.size() * 4, cudaMemcpyHostToDevice));
  const float scale = 0.0078125f * 8.0f;
  const char* const names[] = {"bf16", "KV8", "KV4"};
  for (int fmt = 0; fmt < 3; ++fmt) {
    for (const int base : {100, 1777}) {  // short (one row per block) and long contexts
    for (const int R : {2, 7, 13}) {
      for (const bool paged : {false, true}) {
        const int* t = paged ? dpt : nullptr;
        auto call = [&](const std::uint16_t* qp, int kvb, int rows, std::uint16_t* out) {
          if (fmt == 0)
            return granite4::attn_split(qp, dk, dv, scale, Hq, Hk, d, kvb, nullptr, rows, stride,
                                        scratch, out, nullptr, nullptr, nullptr, nullptr, false,
                                        nullptr, nullptr, false, t, max_len);
          if (fmt == 1)
            return granite4::attn_split(qp, nullptr, nullptr, scale, Hq, Hk, d, kvb, nullptr,
                                        rows, stride, scratch, out, dk8, dv8, dks, dvs, true,
                                        nullptr, nullptr, false, t, max_len);
          return granite4::attn_split(qp, nullptr, nullptr, scale, Hq, Hk, d, kvb, nullptr, rows,
                                      stride, scratch, out, nullptr, nullptr, dks, dvs, false,
                                      dk4, dv4, true, t, max_len);
        };
        const std::string tag = std::string("attn_split rows [") + names[fmt] + " R=" +
                                std::to_string(R) + " base=" + std::to_string(base) +
                                (paged ? " paged" : "") + "]: ";
        expect(call(dq, base, R, dmulti) == 0, (tag + "multi-row launch").c_str());
        bool launches = true;
        for (int r = 0; r < R; ++r)
          launches = launches &&
                     call(dq + static_cast<std::size_t>(r) * Hq * d, base + r, 1,
                          dsingle + static_cast<std::size_t>(r) * Hq * d) == 0;
        expect(launches, (tag + "single-row launches").c_str());
        std::vector<std::uint16_t> a(static_cast<std::size_t>(R) * Hq * d), b(a.size());
        CUDA_CHECK(cudaMemcpy(a.data(), dmulti, a.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(b.data(), dsingle, b.size() * 2, cudaMemcpyDeviceToHost));
        expect(std::memcmp(a.data(), b.data(), a.size() * 2) == 0,
               (tag + "R-row call bit-identical to R single-row calls").c_str());
      }
    }
    }
  }
  std::printf("  attn_split rows: multi-row == single-row (bf16/KV8/KV4, R=2/7/13, paged)\n");
  // Multi-sequence rows grouped per sequence (row_grp) == one row per block:
  // sequence A rows at 1500..1503 (table pt), B at 1800..1804 (table ptB).
  {
    std::vector<int> ptB(pool.rbegin(), pool.rbegin() + max_len);
    int* dptB;
    int *drpos, *dgrp;
    const int** drpt;
    const int nA = 4, nB = 5, n = nA + nB;
    CUDA_CHECK(cudaMalloc(&dptB, ptB.size() * 4));
    CUDA_CHECK(cudaMalloc(&drpos, n * 4));
    CUDA_CHECK(cudaMalloc(&dgrp, 2 * 4));
    CUDA_CHECK(cudaMalloc(&drpt, n * sizeof(int*)));
    CUDA_CHECK(cudaMemcpy(dptB, ptB.data(), ptB.size() * 4, cudaMemcpyHostToDevice));
    std::vector<int> rpos(n);
    std::vector<const int*> rpt(n);
    for (int r = 0; r < n; ++r) {
      rpos[r] = r < nA ? 1500 + r : 1800 + (r - nA);
      rpt[r] = r < nA ? dpt : dptB;
    }
    const int grp[2] = {0 | (nA << 8), nA | (nB << 8)};
    CUDA_CHECK(cudaMemcpy(drpos, rpos.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(drpt, rpt.data(), n * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dgrp, grp, sizeof grp, cudaMemcpyHostToDevice));
    for (int fmt = 0; fmt < 3; ++fmt) {
      auto call = [&](const int* g, int ng, std::uint16_t* out) {
        if (fmt == 0)
          return granite4::attn_split(dq, dk, dv, scale, Hq, Hk, d, 0, nullptr, n, stride,
                                      scratch, out, nullptr, nullptr, nullptr, nullptr, false,
                                      nullptr, nullptr, false, nullptr, max_len, drpos, drpt, g,
                                      ng);
        if (fmt == 1)
          return granite4::attn_split(dq, nullptr, nullptr, scale, Hq, Hk, d, 0, nullptr, n,
                                      stride, scratch, out, dk8, dv8, dks, dvs, true, nullptr,
                                      nullptr, false, nullptr, max_len, drpos, drpt, g, ng);
        return granite4::attn_split(dq, nullptr, nullptr, scale, Hq, Hk, d, 0, nullptr, n,
                                    stride, scratch, out, nullptr, nullptr, dks, dvs, false, dk4,
                                    dv4, true, nullptr, max_len, drpos, drpt, g, ng);
      };
      const std::string tag = std::string("attn_split groups [") + names[fmt] + "]: ";
      expect(call(nullptr, 0, dsingle) == 0, (tag + "per-row launch").c_str());
      expect(call(dgrp, 2, dmulti) == 0, (tag + "grouped launch").c_str());
      std::vector<std::uint16_t> a(static_cast<std::size_t>(n) * Hq * d), b(a.size());
      CUDA_CHECK(cudaMemcpy(a.data(), dmulti, a.size() * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(b.data(), dsingle, b.size() * 2, cudaMemcpyDeviceToHost));
      expect(std::memcmp(a.data(), b.data(), a.size() * 2) == 0,
             (tag + "grouped multi-sequence rows bit-identical to one row per block").c_str());
    }
    cudaFree(dptB); cudaFree(drpos); cudaFree(dgrp); cudaFree(drpt);
    std::printf("  attn_split groups: 2 sequences x 4/5 rows grouped == per-row (bf16/KV8/KV4)\n");
  }
  cudaFree(dk); cudaFree(dv); cudaFree(dk8); cudaFree(dv8); cudaFree(dk4); cudaFree(dv4);
  cudaFree(dks); cudaFree(dvs); cudaFree(dq); cudaFree(dmulti); cudaFree(dsingle);
  cudaFree(dpt); cudaFree(scratch);
}
// Multi-session decode: gpu_decode_multi over two sessions of one shared
// arena == each session decoding alone (gpu_decode_greedy on a private
// cache), token for token and bit for bit in the last step's logits.
static void test_decode_multi_device() {
  ModelConfig model;
  model.num_layers = 2;
  model.hidden_size = 512;
  model.intermediate_size = 256;
  model.num_attention_heads = 4;
  model.num_key_value_heads = 1;
  model.head_dim = 128;
  model.vocab_size = 512;
  model.max_context_length = 131072;
  ForwardSpec spec;
  const int L = 2, H = 512, I = 256, Hq = 4, Hk = 1, d = 128, V = 512;
  const int max_pos = 64, arena_rows = 192, steps = 9;
  const auto z = [](int v) { return static_cast<std::size_t>(v); };
  const GpuWeightsLayout lay{z(H), z(I), z(V), z(L), z(Hq), z(Hk), z(d)};
  Xorshift64 rng(0x3317);
  std::vector<std::uint8_t> weights(lay.total_bytes());
  auto fill_bf16 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i) {
      const std::uint16_t v = gen_bf16(rng);
      weights[off + 2 * i] = static_cast<std::uint8_t>(v & 0xffu);
      weights[off + 2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
  };
  auto fill_fp8 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i)
      weights[off + i] = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  };
  fill_bf16(0, z(V) * H);
  for (int layer = 0; layer < L; ++layer) {
    const std::size_t base = lay.embed_bytes() + z(layer) * lay.layer_bytes();
    fill_bf16(base + lay.off_attn_norm(), H);
    fill_bf16(base + lay.off_ffn_norm(), H);
    fill_fp8(base + lay.off_q(), z(H) * H);
    fill_fp8(base + lay.off_k(), z(Hk) * d * H);
    fill_fp8(base + lay.off_v(), z(Hk) * d * H);
    fill_fp8(base + lay.off_o(), z(H) * H);
    fill_fp8(base + lay.off_gate(), z(I) * H);
    fill_fp8(base + lay.off_up(), z(I) * H);
    fill_fp8(base + lay.off_down(), z(H) * I);
  }
  fill_bf16(lay.final_norm_off(), H);
  fill_fp8(lay.lm_head_off(), z(V) * H);
  std::vector<float> scales(lay.num_scales());
  for (auto& s : scales) s = 0.1f + 0.9f * rng.next_float_range(1.0);
  const int plen[2] = {13, 21};  // different positions per session
  char err[256];
  const char* const modes[] = {"0", "1", "4"};
  const char* const names[] = {"bf16", "KV8", "KV4"};
  for (int m = 0; m < 3; ++m) {
    setenv("G4_KV8", modes[m], 1);
    const std::string tag = std::string("decode multi [") + names[m] + "]: ";
    std::int32_t* tok[2][2];  // [run][session]
    std::uint16_t *lg_ref, *lg_multi;
    bool ok = cudaMalloc(&lg_ref, z(V) * 2) == cudaSuccess &&
              cudaMalloc(&lg_multi, z(2) * V * 2) == cudaSuccess;
    for (int r = 0; r < 2 && ok; ++r)
      for (int s = 0; s < 2 && ok; ++s) {
        ok = cudaMalloc(&tok[r][s], z(max_pos) * 4) == cudaSuccess;
        std::vector<std::int32_t> t(max_pos, 0);
        for (int i = 0; i < plen[s]; ++i) t[i] = (i * (s ? 41 : 97) + (s ? 7 : 13)) % V;
        if (ok) CUDA_CHECK(cudaMemcpy(tok[r][s], t.data(), z(max_pos) * 4, cudaMemcpyHostToDevice));
      }
    if (!ok) {
      ++g_failures;
      continue;
    }
    // Reference: each session alone on a private cache.
    std::vector<std::int32_t> want[2];
    std::vector<std::uint16_t> want_lg[2];
    for (int s = 0; s < 2; ++s) {
      GpuCtx* c = gpu_ctx_create(model, spec, max_pos);
      expect(c != nullptr && gpu_weights_load(c, weights.data(), weights.size(), scales.data(),
                                              scales.size(), err, sizeof err) == 0,
             (tag + "ref ctx").c_str());
      GpuForwardOut out;
      out.d_logits = lg_ref;
      out.logits_from = plen[s] - 1;
      out.d_argmax = tok[0][s] + plen[s];
      expect(gpu_forward(c, tok[0][s], plen[s], out, err, sizeof err) == 0 &&
                 gpu_decode_greedy(c, tok[0][s], steps, lg_ref, err, sizeof err) == 0,
             (tag + "ref prefill + decode").c_str());
      want[s].resize(z(steps + 1));
      want_lg[s].resize(z(V));
      CUDA_CHECK(cudaMemcpy(want[s].data(), tok[0][s] + plen[s], want[s].size() * 4,
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(want_lg[s].data(), lg_ref, z(V) * 2, cudaMemcpyDeviceToHost));
      gpu_ctx_free(c);
    }
    // Multi: both sessions on one arena, decoded together.
    GpuKvArena* arena = gpu_kv_arena_create(model, arena_rows, err, sizeof err);
    GpuWeights* w = gpu_weights_create();
    ok = arena != nullptr && w != nullptr &&
         gpu_weights_load(w, weights.data(), weights.size(), scales.data(), scales.size(), err,
                          sizeof err) == 0;
    GpuCtx* c[2] = {nullptr, nullptr};
    for (int s = 0; s < 2 && ok; ++s) {
      c[s] = gpu_ctx_create_paged(model, spec, max_pos, arena);
      std::vector<int> rows(max_pos);
      for (int i = 0; i < max_pos; ++i) rows[i] = s == 0 ? 3 * i : 3 * (max_pos - 1 - i) + 1;
      ok = c[s] != nullptr && gpu_ctx_use_weights(c[s], w) == 0 &&
           gpu_ctx_set_kv_rows_n(c[s], rows.data(), max_pos) == 0;
      GpuForwardOut out;
      out.d_logits = lg_ref;
      out.logits_from = plen[s] - 1;
      out.d_argmax = tok[1][s] + plen[s];
      ok = ok && gpu_forward(c[s], tok[1][s], plen[s], out, err, sizeof err) == 0;
    }
    expect(ok, (tag + "paged sessions prefilled").c_str());
    std::int32_t* toks[2] = {tok[1][0], tok[1][1]};
    // Two calls (5 + 4 steps): the second replays the captured graph.
    ok = ok && gpu_decode_multi(c, toks, 2, 5, lg_multi, err, sizeof err) == 0 &&
         gpu_decode_multi(c, toks, 2, steps - 5, lg_multi, err, sizeof err) == 0;
    expect(ok, (tag + "gpu_decode_multi").c_str());
    if (ok) {
      bool same = true;
      for (int s = 0; s < 2; ++s) {
        std::vector<std::int32_t> got(z(steps + 1));
        std::vector<std::uint16_t> lg(z(V));
        CUDA_CHECK(cudaMemcpy(got.data(), tok[1][s] + plen[s], got.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(lg.data(), lg_multi + z(s) * V, z(V) * 2, cudaMemcpyDeviceToHost));
        same = same && got == want[s] && lg == want_lg[s] &&
               gpu_ctx_written(c[s]) == plen[s] + steps;
      }
      expect(same, (tag + "2 sessions decoded together == each decoded alone (tokens + logits)").c_str());
      if (same)
        std::printf("  decode multi [%s]: 2 sessions x %d steps bit-identical to solo decode\n",
                    names[m], steps);
    }
    gpu_ctx_free(c[1]);
    gpu_ctx_free(c[0]);
    gpu_kv_arena_free(arena);
    gpu_weights_free_obj(w);
    cudaFree(lg_ref);
    cudaFree(lg_multi);
    for (auto& r : tok)
      for (auto* p : r) cudaFree(p);
  }
  unsetenv("G4_KV8");
}
// Draft-tree verify: gpu_forward_tree over an anchor + 5 nodes (depths
// 1,1,2,2,3) of a paged session == decoding each node's path one position at
// a time on a private cache (logits, bit for bit); after gpu_ctx_kv_move of
// the accepted path 1 -> 3 -> 5 and a truncate, the next decoded position
// matches the sequential reference too.
static void test_tree_verify_device() {
  ModelConfig model;
  model.num_layers = 2;
  model.hidden_size = 512;
  model.intermediate_size = 256;
  model.num_attention_heads = 4;
  model.num_key_value_heads = 1;
  model.head_dim = 128;
  model.vocab_size = 512;
  model.max_context_length = 131072;
  ForwardSpec spec;
  const int L = 2, H = 512, I = 256, Hq = 4, Hk = 1, d = 128, V = 512;
  const auto z = [](int v) { return static_cast<std::size_t>(v); };
  const GpuWeightsLayout lay{z(H), z(I), z(V), z(L), z(Hq), z(Hk), z(d)};
  Xorshift64 rng(0x7ee5);
  std::vector<std::uint8_t> weights(lay.total_bytes());
  auto fill_bf16 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i) {
      const std::uint16_t v = gen_bf16(rng);
      weights[off + 2 * i] = static_cast<std::uint8_t>(v & 0xffu);
      weights[off + 2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
  };
  auto fill_fp8 = [&](std::size_t off, std::size_t elems) {
    for (std::size_t i = 0; i < elems; ++i)
      weights[off + i] = fp8_e4m3_encode(rng.next_float_range(2.0), nullptr);
  };
  fill_bf16(0, z(V) * H);
  for (int layer = 0; layer < L; ++layer) {
    const std::size_t base = lay.embed_bytes() + z(layer) * lay.layer_bytes();
    fill_bf16(base + lay.off_attn_norm(), H);
    fill_bf16(base + lay.off_ffn_norm(), H);
    fill_fp8(base + lay.off_q(), z(H) * H);
    fill_fp8(base + lay.off_k(), z(Hk) * d * H);
    fill_fp8(base + lay.off_v(), z(Hk) * d * H);
    fill_fp8(base + lay.off_o(), z(H) * H);
    fill_fp8(base + lay.off_gate(), z(I) * H);
    fill_fp8(base + lay.off_up(), z(I) * H);
    fill_fp8(base + lay.off_down(), z(H) * I);
  }
  fill_bf16(lay.final_norm_off(), H);
  fill_fp8(lay.lm_head_off(), z(V) * H);
  std::vector<float> scales(lay.num_scales());
  for (auto& sc : scales) sc = 0.1f + 0.9f * rng.next_float_range(1.0);
  const int N = 5;
  const int parent[N + 1] = {-1, 0, 0, 1, 1, 3};
  const int depth[N + 1] = {0, 1, 1, 2, 2, 3};
  std::vector<std::int32_t> node_tok(z(N + 1));
  for (int n = 1; n <= N; ++n) node_tok[z(n)] = (n * 131 + 29) % V;
  // (max_pos, prompt P; the anchor sits at P): one tile; tails crossing a
  // tile (= split, 1 tile per split) boundary; the production split width
  // (max_pos > 16K: 2 tiles per split) with tails crossing a tile inside a
  // split, and a split boundary.
  const int kCases[4][2] = {{64, 17}, {512, 62}, {20000, 12861}, {20000, 12924}};
  for (const auto& cs : kCases) {
  const int max_pos = cs[0], P = cs[1], arena_rows = 2 * max_pos + 8;
  std::vector<std::int32_t> prompt(z(P));
  for (int i = 0; i < P; ++i) prompt[z(i)] = (i * 53 + 11) % V;
  char err[256];
  const char* const modes[] = {"0", "1", "4"};
  const char* const names[] = {"bf16", "KV8", "KV4"};
  for (int m = 0; m < 3; ++m) {
    setenv("G4_KV8", modes[m], 1);
    const std::string tag = std::string("tree verify [") + names[m] + " P=" +
                            std::to_string(P) + "]: ";
    std::int32_t *dtok = nullptr, *dref = nullptr, *dids = nullptr;
    std::uint16_t *lg = nullptr, *lg1 = nullptr;
    if (cudaMalloc(&dtok, z(max_pos) * 4) || cudaMalloc(&dref, z(max_pos) * 4) ||
        cudaMalloc(&dids, z(kMaxVerifyRows) * 4) ||
        cudaMalloc(&lg, z(kMaxVerifyRows) * V * 2) || cudaMalloc(&lg1, z(V) * 2)) {
      ++g_failures;
      continue;
    }
    std::vector<std::int32_t> t(z(max_pos), 0);
    std::copy(prompt.begin(), prompt.end(), t.begin());
    CUDA_CHECK(cudaMemcpy(dtok, t.data(), z(max_pos) * 4, cudaMemcpyHostToDevice));
    // Paged session: prefill the prompt (its argmax is the anchor at P).
    GpuKvArena* arena = gpu_kv_arena_create(model, arena_rows, err, sizeof err);
    GpuWeights* w = gpu_weights_create();
    bool ok = arena != nullptr && w != nullptr &&
              gpu_weights_load(w, weights.data(), weights.size(), scales.data(), scales.size(),
                               err, sizeof err) == 0;
    GpuCtx* c = ok ? gpu_ctx_create_paged(model, spec, max_pos, arena) : nullptr;
    std::vector<int> rows(z(max_pos));
    for (int i = 0; i < max_pos; ++i) rows[z(i)] = 2 * (max_pos - 1 - i) + 1;
    ok = c != nullptr && gpu_ctx_use_weights(c, w) == 0 &&
         gpu_ctx_set_kv_rows_n(c, rows.data(), max_pos) == 0;
    GpuForwardOut out;
    out.d_logits = lg1;
    out.logits_from = P - 1;
    out.d_argmax = dtok + P;
    ok = ok && gpu_forward(c, dtok, P, out, err, sizeof err) == 0;
    std::int32_t anchor = 0;
    if (ok) CUDA_CHECK(cudaMemcpy(&anchor, dtok + P, 4, cudaMemcpyDeviceToHost));
    // Node tokens at their slots P + n.
    if (ok)
      CUDA_CHECK(cudaMemcpy(dtok + P + 1, node_tok.data() + 1, z(N) * 4, cudaMemcpyHostToDevice));
    GpuTreeRow tr[N + 1];
    for (int n = 0; n <= N; ++n) {
      tr[n] = GpuTreeRow{};
      tr[n].ctx = c;
      tr[n].tokens = dtok;
      tr[n].pos = P + depth[n];
      tr[n].slot = P + n;
      tr[n].depth = depth[n];
      std::vector<int> anc;
      for (int a = n; a > 0; a = parent[a]) anc.push_back(a);
      for (int j = 0; j < depth[n]; ++j) tr[n].tail[j] = P + anc[anc.size() - 1 - z(j)];
    }
    ok = ok && gpu_forward_tree(tr, N + 1, lg, dids, nullptr, err, sizeof err) == 0;
    expect(ok, (tag + "gpu_forward_tree").c_str());
    // Reference: a private cache decoding each node's path position by position.
    auto ref_path = [&](const std::vector<std::int32_t>& path, std::vector<std::uint16_t>* lo) {
      GpuCtx* r = gpu_ctx_create(model, spec, max_pos);
      bool good = r != nullptr && gpu_weights_load(r, weights.data(), weights.size(),
                                                   scales.data(), scales.size(), err,
                                                   sizeof err) == 0;
      std::vector<std::int32_t> seq(prompt);
      seq.insert(seq.end(), path.begin(), path.end());
      seq.resize(z(max_pos), 0);
      CUDA_CHECK(cudaMemcpy(dref, seq.data(), z(max_pos) * 4, cudaMemcpyHostToDevice));
      GpuForwardOut o;
      o.d_logits = lg1;
      o.logits_from = P - 1;
      good = good && gpu_forward(r, dref, P, o, err, sizeof err) == 0;
      for (std::size_t i = 0; good && i < path.size(); ++i) {
        GpuForwardOut o1;
        o1.d_logits = lg1;
        o1.logits_from = P + static_cast<int>(i);
        good = gpu_forward(r, dref, P + static_cast<int>(i) + 1, o1, err, sizeof err) == 0;
      }
      lo->resize(z(V));
      if (good) CUDA_CHECK(cudaMemcpy(lo->data(), lg1, z(V) * 2, cudaMemcpyDeviceToHost));
      gpu_ctx_free(r);
      return good;
    };
    bool same = ok;
    for (int n = 0; ok && n <= N; ++n) {
      std::vector<std::int32_t> path{anchor};
      std::vector<int> anc;
      for (int a = n; a > 0; a = parent[a]) anc.push_back(a);
      for (auto it = anc.rbegin(); it != anc.rend(); ++it) path.push_back(node_tok[z(*it)]);
      std::vector<std::uint16_t> want, got(z(V));
      const bool good = ref_path(path, &want);
      CUDA_CHECK(cudaMemcpy(got.data(), lg + z(n) * V, z(V) * 2, cudaMemcpyDeviceToHost));
      same = same && good && want == got;
    }
    expect(same, (tag + "every tree row bit-identical to decoding its path").c_str());
    // Accept 1 -> 3 -> 5: move slots P+3 -> P+2, P+5 -> P+3, truncate, decode one more.
    const int src[2] = {P + 3, P + 5}, dst[2] = {P + 2, P + 3};
    ok = ok && gpu_ctx_kv_move(c, src, dst, 2) == 0 && gpu_ctx_truncate(c, P + 4) == 0;
    const std::int32_t acc_tok[3] = {node_tok[1], node_tok[3], node_tok[5]};
    const std::int32_t next = 77;
    if (ok) {
      CUDA_CHECK(cudaMemcpy(dtok + P + 1, acc_tok, 3 * 4, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dtok + P + 4, &next, 4, cudaMemcpyHostToDevice));
    }
    GpuForwardOut o2;
    o2.d_logits = lg1;
    o2.logits_from = P + 4;
    ok = ok && gpu_forward(c, dtok, P + 5, o2, err, sizeof err) == 0;
    std::vector<std::uint16_t> got2(z(V)), want2;
    if (ok) CUDA_CHECK(cudaMemcpy(got2.data(), lg1, z(V) * 2, cudaMemcpyDeviceToHost));
    const bool good2 = ok && ref_path({anchor, acc_tok[0], acc_tok[1], acc_tok[2], next}, &want2);
    expect(good2 && got2 == want2,
           (tag + "accepted path moved into place: next position == sequential decode").c_str());
    if (same && good2 && got2 == want2)
      std::printf("  tree verify [%s P=%d]: 6 rows == path decodes; KV move + continue exact\n",
                  names[m], P);
    gpu_ctx_free(c);
    gpu_kv_arena_free(arena);
    gpu_weights_free_obj(w);
    cudaFree(dtok); cudaFree(dref); cudaFree(dids); cudaFree(lg); cudaFree(lg1);
  }
  }  // cases
  unsetenv("G4_KV8");
}
int main() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("FAIL: no CUDA device\n");
    return 1;
  }
  test_arch_probe();
  test_fp8_decode_all();
  test_fp8_hw_decode();
  test_kv8_device();
  test_kv4_device();
  test_kv4_attn_split_device();
  test_rmsnorm_device();
  test_rope_device();
  test_softmax_device();
  test_gemm_device();
  test_row_gemm_device();
  test_verify_gemv_production_parity_device();
  test_verify_kernels_bit_identical_device();
  test_batched_qkv_parity_device();
  test_batched_residual_projection_parity_device();
  test_verify_attention_validation_device();
  test_verify_attention_layer_validation_device();
  test_verify_qkv_stage_device();
  test_verify_attention_stage_device();
  test_attention_device();
  test_gqa_fused_device();
  test_batched_verify_attention_device();
  test_prefill_gemm_device();
  test_prefill_attention_device();
  test_paged_prefill_device();
  test_attn_split_device();
  test_paged_attn_split_device();
  test_paged_gqa_fused_device();
  test_paged_kv_append_batched_device();
  test_batched_silu_gate_up_device();
  test_batched_rmsnorm_device();
  test_batched_rope_device();
  test_batched_forward_range_parity_device();
  test_shared_arena_sessions_device();
  test_attn_split_rows_bit_identical_device();
  test_decode_multi_device();
  test_tree_verify_device();
  test_paged_kv_compact_device();
  test_silu_gate_up_device();
  test_sampling_device();
  cudaDeviceSynchronize();
  if (g_failures == 0) std::printf("all device kernel tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
