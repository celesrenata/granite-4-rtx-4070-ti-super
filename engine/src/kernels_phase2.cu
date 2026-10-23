// Phase 2 (Kernels): device kernels for Granite 4.2 8B on sm_89.
//
// Conventions (shared with the host reference in kernel_math.h):
//   * BF16 in / BF16 out, float (double for reductions) accumulation.
//   * RMSNorm:  y = x * rsqrt(mean(x^2) + eps) * w
//   * RoPE:     Llama-style half split (pairs i, i+d/2), precomputed
//               [max_pos, d/2] cos/sin tables, theta = 1e7
//   * Softmax:  logit scale = attention_multiplier (1/128, replaces
//               1/sqrt(d_head)); max-subtracted 2-pass, exp in double
//   * GEMM:     FP8 e4m3 weight rows x BF16 activation rows, per-tensor
//               dequant scale (Ada has no block-scaled MMA; per-tensor is
//               the P1 storage decision)
//
// Correctness-first tiling (one thread per output element); the
// micro-benchmark pass (P2 exit criterion) lands in the next milestone.
#include <cmath>
#include <cstdint>
#include <cstdio>

#define GRANITE_CUDA 1
#include "kernel_math.h"
#include "model_config.h"

namespace granite4 {

namespace {

__device__ __forceinline__ std::uint16_t f32_to_bf16(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, sizeof u);
  u += 0x7fffu + ((u >> 16) & 1u);  // RNE
  u >>= 16;
  return static_cast<std::uint16_t>(u);
}

__device__ __forceinline__ float bf16_to_f32(std::uint16_t bits) {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof f);
  return f;
}

// Exact e4m3 decode (matches checkpoint.cc fp8_e4m3_to_bf16, device-side).
__device__ __forceinline__ float fp8_to_f32(std::uint8_t x) {
  const bool neg = (x & 0x80u) != 0;
  const unsigned e = (x >> 3) & 0xFu;
  const unsigned m = x & 0x7u;
  if (e == 0) {
    // Subnormal: m * 2^-9 (m == 0 is the zero values).
    const float v = static_cast<float>(m) * std::ldexp(1.0f, -9);
    return neg ? -v : v;
  }
  if (e == 15) {
    // Top exponent: (1 + m/8) * 2^8; m == 7 is sNaN (map to 0.0; upstream
    // saturation never produces it for real weights).
    if (m == 7) return 0.0f;
    const float v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, 8);
    return neg ? -v : v;
  }
  const float v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, static_cast<int>(e) - 7);
  return neg ? -v : v;
}

}  // namespace

// ---------------------------------------------------------------------------
// RMSNorm: one thread per element; the row reduction (mean of squares) is
// re-read per thread (correct; the fast variant tiles over rows).
// ---------------------------------------------------------------------------

__global__ void rmsnorm_kernel(const std::uint16_t* x, const std::uint16_t* w,
                               float eps, int rows, int cols,
                               std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  const int row = idx / cols;
  const int col = idx % cols;
  const int base = row * cols;
  double s = 0.0;
  for (int i = 0; i < cols; ++i) {
    const float v = bf16_to_f32(x[base + i]);
    s += static_cast<double>(v) * v;
  }
  const float r = 1.0f / std::sqrt(static_cast<float>(s / cols) + eps);
  out[idx] = f32_to_bf16(bf16_to_f32(x[base + col]) * r * bf16_to_f32(w[col]));
}

// ---------------------------------------------------------------------------
// RoPE: one thread per (row, col) of the head vector.
// ---------------------------------------------------------------------------

__global__ void rope_kernel(const std::uint16_t* x, const float* cos_table,
                            const float* sin_table, int rows, int head_dim,
                            std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * head_dim) return;
  const int row = idx / head_dim;
  const int col = idx % head_dim;
  const int half = head_dim / 2;
  const int base = row * head_dim;
  const int i = col < half ? col : col - half;
  const float c = cos_table[row * half + i];
  const float s = sin_table[row * half + i];
  float y;
  if (col < half) {
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col + half]);
    y = a * c - b * s;
  } else {
    // Back half: out[col] = x[col]*c + x[col-half]*s (Llama convention).
    const float a = bf16_to_f32(x[base + col]);
    const float b = bf16_to_f32(x[base + col - half]);
    y = a * c + b * s;
  }
  out[idx] = f32_to_bf16(y);
}

// ---------------------------------------------------------------------------
// Scaled softmax: one thread per element, two passes per row
// (max, then sum - same arithmetic as the host reference).
// ---------------------------------------------------------------------------

__global__ void softmax_kernel(const std::uint16_t* x, float scale, int rows,
                               int seq_len, std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * seq_len) return;
  const int row = idx / seq_len;
  const int col = idx % seq_len;
  const int base = row * seq_len;
  float m = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < seq_len; ++i)
    m = fmaxf(m, scale * bf16_to_f32(x[base + i]));
  double sum = 0.0;
  for (int i = 0; i < seq_len; ++i)
    sum += std::exp(static_cast<double>(scale * bf16_to_f32(x[base + i]) - m));
  out[idx] = f32_to_bf16(
      static_cast<float>(std::exp(static_cast<double>(scale * bf16_to_f32(x[base + col]) - m)) / sum));
}

// ---------------------------------------------------------------------------
// GEMM: C[i, j] = scale * sum_k A[i, k] * fp8_to_f32(B[j, k]).
// One thread per output element.
// ---------------------------------------------------------------------------

__global__ void gemm_dequant_kernel(const std::uint16_t* A, const std::uint8_t* B,
                                    float scale, int M, int N, int K,
                                    std::uint16_t* C) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  const int i = idx / N;
  const int j = idx % N;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k)
    acc += fp8_to_f32(B[j * K + k]) * bf16_to_f32(A[i * K + k]);
  C[idx] = f32_to_bf16(scale * acc);
}

// ---------------------------------------------------------------------------
// GQA attention (decode, batch 1): logits = scale * Q K^T with the K-head
// for Q-head hq being hq/gh (gh = Hq/Hk); value accumulation over p * V.
// Logits are stored BF16 (the softmax kernel consumes them as-is) and the
// softmax probabilities are consumed BF16 by the value kernel - matching
// the reference pipeline exactly.
// ---------------------------------------------------------------------------

__global__ void gqa_logits_kernel(const std::uint16_t* q, const std::uint16_t* k,
                                  float scale, int Hq, int Hk, int d, int kv_len,
                                  std::uint16_t* logits) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hq * kv_len) return;
  const int hq = idx / kv_len;
  const int kv = idx % kv_len;
  const int hkv = hq / (Hq / Hk);
  float acc = 0.0f;
  const int kb = (hkv * kv_len + kv) * d;
  for (int i = 0; i < d; ++i)
    acc += bf16_to_f32(k[kb + i]) * bf16_to_f32(q[hq * d + i]);
  logits[idx] = f32_to_bf16(scale * acc);
}

__global__ void gqa_value_kernel(const std::uint16_t* p, const std::uint16_t* v,
                                 int Hq, int Hk, int d, int kv_len,
                                 std::uint16_t* out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Hq * d) return;
  const int hq = idx / d;
  const int i = idx % d;
  const int hkv = hq / (Hq / Hk);
  float acc = 0.0f;
  for (int kv = 0; kv < kv_len; ++kv)
    acc += bf16_to_f32(p[hq * kv_len + kv]) * bf16_to_f32(v[(hkv * kv_len + kv) * d + i]);
  out[idx] = f32_to_bf16(acc);
}

// ---------------------------------------------------------------------------
// FFN gate/up (SwiGLU): y[r] = silu(sg * Wg[r].x) * (su * Wu[r].x).
// ---------------------------------------------------------------------------

__global__ void silu_gate_up_kernel(const std::uint8_t* Wg, const std::uint8_t* Wu,
                                    const std::uint16_t* x, float sg, float su,
                                    int N, int K, std::uint16_t* out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= N) return;
  float g = 0.0f, u = 0.0f;
  for (int i = 0; i < K; ++i) {
    const float xv = bf16_to_f32(x[i]);
    g += fp8_to_f32(Wg[r * K + i]) * xv;
    u += fp8_to_f32(Wu[r * K + i]) * xv;
  }
  const float sig = 1.0f / (1.0f + std::exp(-(sg * g)));
  out[r] = f32_to_bf16(sg * g * sig * (su * u));
}

// ---------------------------------------------------------------------------
// Sampling. Greedy: one thread per row (any vocab size). Top-p: one block
// per row with a shared-memory stable sort; vocab <= 4096 (the production
// 100352 path uses the same per-row loop with a chunked argsort - a P3
// concern; the math is identical).
// ---------------------------------------------------------------------------

__global__ void sample_greedy_kernel(const std::uint16_t* p, int rows, int n,
                                     int* out) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int best = 0;
  float bv = bf16_to_f32(p[row * n]);
  for (int i = 1; i < n; ++i) {
    const float v = bf16_to_f32(p[row * n + i]);
    if (v > bv) {
      bv = v;
      best = i;
    }
  }
  out[row] = best;
}

constexpr int kTopPVocabMax = 4096;

__global__ void sample_top_p_kernel(const std::uint16_t* p, float nucleus,
                                    const float* u_in, int n, int* out) {
  // One block per row: the shared arrays hold the row being sorted.
  const int row = blockIdx.x;
  if (n > kTopPVocabMax) return;
  __shared__ float sp[kTopPVocabMax];
  __shared__ int st[kTopPVocabMax];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    sp[i] = bf16_to_f32(p[row * n + i]);
    st[i] = i;
  }
  __syncthreads();
  // One thread: stable insertion sort (probability descending, token id
  // ascending on ties), then nucleus truncation and prefix-sum pick.
  // Single-threaded because the sort is in-place over the shared arrays;
  // O(n^2) is fine at the tested sizes (a faster argsort is a P3 concern).
  if (threadIdx.x == 0) {
    for (int i = 0; i < n; ++i) {
      float v = sp[i];
      int id = st[i];
      int j = i - 1;
      while (j >= 0 && (sp[j] < v || (sp[j] == v && st[j] > id))) {
        sp[j + 1] = sp[j];
        st[j + 1] = st[j];
        --j;
      }
      sp[j + 1] = v;
      st[j + 1] = id;
    }
    double acc = 0.0;
    int cut = n;
    for (int j = 0; j < n; ++j) {
      acc += sp[j];
      cut = j + 1;
      if (acc >= nucleus) break;
    }
    const double target = u_in[row] * acc;
    double cum = 0.0;
    int pick = st[cut - 1];
    for (int j = 0; j < cut; ++j) {
      cum += sp[j];
      if (cum >= target) {
        pick = st[j];
        break;
      }
    }
    out[row] = pick;
  }
}

}  // namespace granite4

// ---------------------------------------------------------------------------
// Host launchers (C ABI surface for the P3 scheduler). 0 on success, -1 on
// CUDA error (diagnostic on stderr).
// ---------------------------------------------------------------------------

#define G4_CUDA_CHECK(expr)                                                 \
  do {                                                                      \
    cudaError_t e_ = (expr);                                               \
    if (e_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "granite4: CUDA error at %s:%d: %s\n", __FILE__, \
                   __LINE__, cudaGetErrorString(e_));                      \
      return -1;                                                            \
    }                                                                       \
  } while (0)

extern "C" {

int granite4_rope_tables_build(double theta, int head_dim, int max_pos,
                               float** d_cos, float** d_sin) {
  std::vector<float> h_cos, h_sin;
  granite4::rope_freqs_f32(theta, head_dim, max_pos, &h_cos, &h_sin);
  const std::size_t bytes = h_cos.size() * sizeof(float);
  G4_CUDA_CHECK(cudaMalloc(d_cos, bytes));
  G4_CUDA_CHECK(cudaMalloc(d_sin, bytes));
  G4_CUDA_CHECK(cudaMemcpy(*d_cos, h_cos.data(), bytes, cudaMemcpyHostToDevice));
  G4_CUDA_CHECK(cudaMemcpy(*d_sin, h_sin.data(), bytes, cudaMemcpyHostToDevice));
  return 0;
}

int granite4_kernel_rmsnorm(const std::uint16_t* d_x, const std::uint16_t* d_w,
                            float eps, int rows, int cols, std::uint16_t* d_out) {
  granite4::rmsnorm_kernel<<<(rows * cols + 255) / 256, 256>>>(d_x, d_w, eps,
                                                              rows, cols, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_rope(const std::uint16_t* d_x, const float* d_cos,
                         const float* d_sin, int rows, int head_dim,
                         std::uint16_t* d_out) {
  granite4::rope_kernel<<<(rows * head_dim + 255) / 256, 256>>>(
      d_x, d_cos, d_sin, rows, head_dim, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_softmax(const std::uint16_t* d_x, float scale, int rows,
                           int seq_len, std::uint16_t* d_out) {
  granite4::softmax_kernel<<<(rows * seq_len + 255) / 256, 256>>>(
      d_x, scale, rows, seq_len, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gemm_dequant(const std::uint16_t* d_a, const std::uint8_t* d_b,
                                 float scale, int M, int N, int K, std::uint16_t* d_c) {
  granite4::gemm_dequant_kernel<<<(M * N + 255) / 256, 256>>>(
      d_a, d_b, scale, M, N, K, d_c);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_logits(const std::uint16_t* d_q, const std::uint16_t* d_k,
                               float scale, int Hq, int Hk, int d, int kv_len,
                               std::uint16_t* d_logits) {
  granite4::gqa_logits_kernel<<<(Hq * kv_len + 255) / 256, 256>>>(
      d_q, d_k, scale, Hq, Hk, d, kv_len, d_logits);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_gqa_value(const std::uint16_t* d_p, const std::uint16_t* d_v,
                              int Hq, int Hk, int d, int kv_len, std::uint16_t* d_out) {
  granite4::gqa_value_kernel<<<(Hq * d + 255) / 256, 256>>>(
      d_p, d_v, Hq, Hk, d, kv_len, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_silu_gate_up(const std::uint8_t* d_wg, const std::uint8_t* d_wu,
                                 const std::uint16_t* d_x, float scale_g, float scale_u,
                                 int N, int K, std::uint16_t* d_out) {
  granite4::silu_gate_up_kernel<<<(N + 255) / 256, 256>>>(
      d_wg, d_wu, d_x, scale_g, scale_u, N, K, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_sample_greedy(const std::uint16_t* d_p, int rows, int n, int* d_out) {
  granite4::sample_greedy_kernel<<<(rows + 255) / 256, 256>>>(d_p, rows, n, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

int granite4_kernel_sample_top_p(const std::uint16_t* d_p, float nucleus,
                                 const float* d_u, int rows, int n, int* d_out) {
  if (n > granite4::kTopPVocabMax) return -1;
  // One block per row: the shared-memory arrays hold the row being sorted.
  granite4::sample_top_p_kernel<<<rows, 256>>>(d_p, nucleus, d_u, n, d_out);
  G4_CUDA_CHECK(cudaGetLastError());
  return 0;
}

}  // extern "C"
