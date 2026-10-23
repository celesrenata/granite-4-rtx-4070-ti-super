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

#include <algorithm>
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
// GQA attention, decode step (batch 1), Llama half-split RoPE applied before
// the call. q/k/v are BF16:
//   logits[hq, kv] = scale * dot(q[hq], k[hq/gh, kv]),  gh = Hq / Hk
//   p = softmax(logits)                  (scale is applied to the logits)
//   out[hq, i] = sum_kv p[hq, kv] * v[hq/gh, kv, i]
// The reference mirrors the 3-kernel pipeline exactly: logits are rounded
// to BF16 before the softmax, and p is rounded to BF16 before the value
// accumulation (matching what the device kernels store).
// ---------------------------------------------------------------------------

// logits[hq, kv] = scale * dot(q[hq], k[hq/gh, kv]), rounded to BF16.
inline void attention_logits_bf16(const std::uint16_t* q, const std::uint16_t* k,
                                  float scale, int Hq, int Hk, int d, int kv_len,
                                  std::vector<std::uint16_t>* logits) {
  const int gh = Hq / Hk;
  logits->assign(static_cast<std::size_t>(Hq) * kv_len, 0);
  for (int hq = 0; hq < Hq; ++hq) {
    const int hkv = hq / gh;
    for (int kv = 0; kv < kv_len; ++kv) {
      float acc = 0.0f;
      for (int i = 0; i < d; ++i)
        acc += bf16_to_f32_host(k[(hkv * kv_len + kv) * d + i]) *
               bf16_to_f32_host(q[hq * d + i]);
      (*logits)[hq * kv_len + kv] = f32_to_bf16_host(scale * acc);
    }
  }
}

// out[hq, i] = sum_kv p[hq, kv] * v[hq/gh, kv, i] (p is BF16, per kernel).
inline void attention_value_bf16(const std::uint16_t* p, const std::uint16_t* v,
                                 int Hq, int Hk, int d, int kv_len,
                                 std::vector<std::uint16_t>* out) {
  const int gh = Hq / Hk;
  out->assign(static_cast<std::size_t>(Hq) * d, 0);
  for (int hq = 0; hq < Hq; ++hq) {
    const int hkv = hq / gh;
    for (int i = 0; i < d; ++i) {
      float acc = 0.0f;
      for (int kv = 0; kv < kv_len; ++kv)
        acc += bf16_to_f32_host(p[hq * kv_len + kv]) *
               bf16_to_f32_host(v[(hkv * kv_len + kv) * d + i]);
      (*out)[hq * d + i] = f32_to_bf16_host(acc);
    }
  }
}

// Full attention: logits -> (BF16) -> softmax -> (BF16) -> value accumulation.
inline void gqa_attention_bf16(const std::uint16_t* q, const std::uint16_t* k,
                               const std::uint16_t* v, float scale, int Hq, int Hk,
                               int d, int kv_len, std::vector<std::uint16_t>* out) {
  std::vector<std::uint16_t> logits;
  attention_logits_bf16(q, k, scale, Hq, Hk, d, kv_len, &logits);
  std::vector<float> pf(static_cast<std::size_t>(Hq) * kv_len);
  for (int hq = 0; hq < Hq; ++hq) {
    std::vector<float> lf(kv_len);
    for (int kv = 0; kv < kv_len; ++kv)
      lf[kv] = bf16_to_f32_host(logits[static_cast<std::size_t>(hq) * kv_len + kv]);
    std::vector<float> sf(kv_len);
    softmax_f32(lf.data(), 1.0f, kv_len, sf.data());
    for (int kv = 0; kv < kv_len; ++kv)
      pf[static_cast<std::size_t>(hq) * kv_len + kv] = bf16_to_f32_host(f32_to_bf16_host(sf[kv]));
  }
  std::vector<std::uint16_t> pbf;
  pbf.reserve(pf.size());
  for (float f : pf) pbf.push_back(f32_to_bf16_host(f));
  attention_value_bf16(pbf.data(), v, Hq, Hk, d, kv_len, out);
}

// ---------------------------------------------------------------------------
// Paged KV compaction (P3): gather one layer's KV from the paged FP8 pool
// into the contiguous [Hk, kv_len, d] BF16 layout the P2 attention kernels
// consume (see kv_pool.h KvLayout). The paged layout is token-major:
//   page = [token_local][layer][k|v][kv_head][head_dim]  (FP8 e4m3, 1 B/elt)
//   page_bytes = tokens_per_page * 2 * num_layers * Hk * d
// and a token's page is block_table[token / tokens_per_page]. FP8->BF16 is
// exact (every finite e4m3 value is representable in BF16), so the device
// compaction kernel is checked against this reference ULP-exactly.
// `which` selects K (0) or V (1) within a token's per-layer KV.
// ---------------------------------------------------------------------------
inline void paged_kv_compact_bf16(const std::uint8_t* paged,
                                  const std::vector<std::size_t>& block_table,
                                  int kv_len, int tokens_per_page, int num_layers,
                                  int Hk, int d, int layer, int which,
                                  std::vector<std::uint16_t>* out) {
  const std::size_t B = tokens_per_page;
  const std::size_t per_token = num_layers * 2ull * Hk * d;
  const std::size_t page_bytes = B * per_token;
  out->assign(static_cast<std::size_t>(Hk) * kv_len * d, 0);
  for (int token = 0; token < kv_len; ++token) {
    const std::size_t page_id = block_table[token / B];
    const std::size_t token_local = token % B;
    // Base of this token's per-layer KV within the page.
    const std::size_t base =
        page_id * page_bytes +
        token_local * per_token +
        (static_cast<std::size_t>(layer) * 2 + static_cast<std::size_t>(which)) *
            Hk * d;
    for (int head = 0; head < Hk; ++head) {
      for (int i = 0; i < d; ++i) {
        const std::uint16_t bits =
            fp8_e4m3_to_bf16(paged[base + head * d + i]);
        (*out)[(static_cast<std::size_t>(head) * kv_len + token) * d + i] = bits;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Deterministic forward pass (P3): token ids -> logits, all layers, float
// arithmetic. This is the algorithmic reference the GPU forward pass and the
// golden parity gate are validated against, so it composes only the already
// device-verified primitives (rmsnorm_f32, rope_half_f32, gqa_attention_bf16,
// silu_f32). The weights are the *dequantized* FP8 weights (the exact values
// the FP8 pool holds, per kv_pool.h / fp8_convert), so the parity gate
// isolates arithmetic, not quantization noise.
//
// Weights are laid out as contiguous per-layer buffers in the safetensors
// tensor order (checkpoint.h expected_tensors):
//   [input_ln(H) | post_ln(H) | q(H,H) | k(KV,H) | v(KV,H) | o(H,H) |
//    gate(I,H) | up(I,H) | down(H,I)]
// where KV = num_kv_heads * head_dim. A per-layer pointer `layer` is the
// head of that block; the next layer is at layer + bytes_per_layer.
//
// The decode loop is:  embed -> (per layer: rmsnorm -> attention -> +residual
// -> rmsnorm -> swiglu-FFN -> +residual) -> rmsnorm -> lm_head -> logits.
// Q/K use the Llama half-split RoPE (theta from the spec); V does not.
// ---------------------------------------------------------------------------
inline void forward_logits_f32(const std::vector<std::int32_t>& tokens,
                               const std::vector<float>& embed,
                               const std::vector<float>& layers,
                               const float* final_norm, const float* lm_head,
                               const ModelConfig& cfg, const ForwardSpec& spec,
                               std::vector<float>* logits) {
  const int H = static_cast<int>(cfg.hidden_size);
  const int I = static_cast<int>(cfg.intermediate_size);
  const int Hq = static_cast<int>(cfg.num_attention_heads);
  const int Hk = static_cast<int>(cfg.num_key_value_heads);
  const int d = static_cast<int>(cfg.head_dim);
  const int V = static_cast<int>(cfg.vocab_size);
  const int L = static_cast<int>(cfg.num_layers);
  const int KV = Hk * d;
  const int half = d / 2;

  // Per-layer block layout (offsets from the block head), matching the
  // safetensors tensor order in checkpoint.h expected_tensors.
  const std::size_t q_size = (std::size_t)H * H;
  const std::size_t kv_size = (std::size_t)KV * H;
  const std::size_t io_size = (std::size_t)I * H;
  const std::size_t o_size = (std::size_t)H * H;
  const std::size_t down_size = (std::size_t)H * I;
  const std::size_t off_post_ln = (std::size_t)H;
  const std::size_t off_q = 2 * (std::size_t)H;
  const std::size_t off_k = off_q + q_size;
  const std::size_t off_v = off_k + kv_size;
  const std::size_t off_o = off_v + kv_size;
  const std::size_t off_gate = off_o + o_size;
  const std::size_t off_up = off_gate + io_size;
  const std::size_t off_down = off_up + io_size;
  const std::size_t layer_bytes = off_down + down_size;

  const int n = static_cast<int>(tokens.size());
  // Positional cos/sin for every position used, all d/2 RoPE pairs.
  std::vector<float> cos_tbl, sin_tbl;
  rope_freqs_f32(spec.rope_theta, d, n, &cos_tbl, &sin_tbl);

  // KV history per layer: K/V are [Hk][pos][d] (the P2 attention layout), one
  // d-slice appended per position. The attention block at position t attends
  // over positions 0..t (causal), so the history up to t is the context.
  std::vector<std::vector<float>> k_hist(L,
                                         std::vector<float>((std::size_t)Hk * n * d));
  std::vector<std::vector<float>> v_hist(L,
                                         std::vector<float>((std::size_t)Hk * n * d));

  std::vector<float> residual(H), normed(H), normed2(H), o_out(H), ffn_out(H);
  std::vector<float> qf((std::size_t)Hq * d), k_new((std::size_t)Hk * d),
      v_new((std::size_t)Hk * d);
  std::vector<std::uint16_t> qbf, attn_out;
  qbf.resize(qf.size());
  std::vector<float> g(I), u(I);

  // logits[t][v] for every position (the parity gate compares a full sequence).
  logits->assign((std::size_t)n * V, 0.0f);

  for (int t = 0; t < n; ++t) {
    // Fresh hidden state: the embedding of this token (scaled per spec).
    const float* emb_row =
        embed.data() + static_cast<std::size_t>(tokens[t]) * H;
    for (int j = 0; j < H; ++j)
      residual[j] = spec.embedding_multiplier * emb_row[j];

    for (int layer = 0; layer < L; ++layer) {
      const float* const lb = layers.data() + (std::size_t)layer * layer_bytes;
      std::vector<float>& kh = k_hist[layer];
      std::vector<float>& vh = v_hist[layer];

      // ---- attention block (causal over positions 0..t) ----
      rmsnorm_f32(residual.data(), lb, spec.rms_norm_eps, H, normed.data());
      // q = Q @ normed: Q is [Hq][d][H] (row hq, d-slice, H cols).
      for (int hq = 0; hq < Hq; ++hq)
        for (int i = 0; i < d; ++i) {
          float acc = 0.0f;
          const float* w = lb + off_q + (std::size_t)hq * H + i;
          for (int j = 0; j < H; ++j) acc += w[j] * normed[j];
          qf[(std::size_t)hq * d + i] = acc;
        }
      // k/v for this position = K/V @ normed: [Hk][d][H].
      for (int hkv = 0; hkv < Hk; ++hkv)
        for (int i = 0; i < d; ++i) {
          float a = 0.0f, a2 = 0.0f;
          const float* wk = lb + off_k + (std::size_t)hkv * H + i;
          const float* wv = lb + off_v + (std::size_t)hkv * H + i;
          for (int j = 0; j < H; ++j) {
            a += wk[j] * normed[j];
            a2 += wv[j] * normed[j];
          }
          k_new[(std::size_t)hkv * d + i] = a;
          v_new[(std::size_t)hkv * d + i] = a2;
        }
      // RoPE (Llama half split) on q and the new k at position t; V is raw.
      const float* cos_t = &cos_tbl[(std::size_t)t * half];
      const float* sin_t = &sin_tbl[(std::size_t)t * half];
      for (int hq = 0; hq < Hq; ++hq)
        rope_half_f32(&qf[(std::size_t)hq * d], cos_t, sin_t, d,
                      &qf[(std::size_t)hq * d]);
      for (int hkv = 0; hkv < Hk; ++hkv)
        rope_half_f32(&k_new[(std::size_t)hkv * d], cos_t, sin_t, d,
                      &k_new[(std::size_t)hkv * d]);
      // Append this position's RoPE'd k and raw v to the per-layer history.
      for (int hkv = 0; hkv < Hk; ++hkv) {
        std::copy(&k_new[(std::size_t)hkv * d], &k_new[(std::size_t)hkv * d + d],
                  &kh[(std::size_t)hkv * n * d + (std::size_t)t * d]);
        std::copy(&v_new[(std::size_t)hkv * d], &v_new[(std::size_t)hkv * d + d],
                  &vh[(std::size_t)hkv * n * d + (std::size_t)t * d]);
      }
      // Gather the causal context (positions 0..t) into a contiguous
      // [Hk][t+1][d] block (the history is [Hk][n][d]; copy the prefix).
      const int kv_len = t + 1;
      std::vector<std::uint16_t> kb((std::size_t)Hk * kv_len * d),
          vb((std::size_t)Hk * kv_len * d);
      for (int hkv = 0; hkv < Hk; ++hkv)
        for (int pos = 0; pos < kv_len; ++pos) {
          for (int i = 0; i < d; ++i)
            kb[(std::size_t)hkv * kv_len * d + (std::size_t)pos * d + i] =
                f32_to_bf16_host(kh[(std::size_t)hkv * n * d + (std::size_t)pos * d + i]);
          for (int i = 0; i < d; ++i)
            vb[(std::size_t)hkv * kv_len * d + (std::size_t)pos * d + i] =
                f32_to_bf16_host(vh[(std::size_t)hkv * n * d + (std::size_t)pos * d + i]);
        }
      for (std::size_t i = 0; i < qf.size(); ++i) qbf[i] = f32_to_bf16_host(qf[i]);
      attn_out.clear();
      gqa_attention_bf16(qbf.data(), kb.data(), vb.data(),
                         spec.attention_multiplier, Hq, Hk, d, kv_len, &attn_out);
      // o = O @ attn (attn is [Hq][d] == H wide, O is [H][H]). attn_out is
      // BF16, so decode before the matvec.
      for (int r = 0; r < H; ++r) {
        float acc = 0.0f;
        const float* w = lb + off_o + (std::size_t)r * H;
        for (int j = 0; j < H; ++j)
          acc += w[j] * bf16_to_f32_host(attn_out[(std::size_t)j]);
        o_out[r] = acc;
      }
      for (int r = 0; r < H; ++r)
        residual[r] += spec.residual_multiplier * o_out[r];

      // ---- FFN block (SwiGLU) ----
      rmsnorm_f32(residual.data(), lb + off_post_ln, spec.rms_norm_eps, H,
                  normed2.data());
      for (int r = 0; r < I; ++r) {
        float ag = 0.0f, au = 0.0f;
        const float* wg = lb + off_gate + (std::size_t)r * H;
        const float* wu = lb + off_up + (std::size_t)r * H;
        for (int j = 0; j < H; ++j) {
          ag += wg[j] * normed2[j];
          au += wu[j] * normed2[j];
        }
        g[r] = silu_f32(ag);
        u[r] = au;
      }
      for (int r = 0; r < H; ++r) {
        float acc = 0.0f;
        const float* w = lb + off_down + (std::size_t)r * I;
        for (int j = 0; j < I; ++j) acc += w[j] * (g[j] * u[j]);
        ffn_out[r] = acc;
      }
      for (int r = 0; r < H; ++r)
        residual[r] += spec.residual_multiplier * ffn_out[r];
    }

    // Final RMSNorm, then lm_head (vocab x hidden, row-major) -> logits[t].
    rmsnorm_f32(residual.data(), final_norm, spec.rms_norm_eps, H, normed.data());
    float* logits_t = logits->data() + (std::size_t)t * V;
    for (int i = 0; i < V; ++i) {
      float acc = 0.0f;
      const float* w = lm_head + (std::size_t)i * H;
      for (int j = 0; j < H; ++j) acc += w[j] * normed[j];
      logits_t[i] = spec.logits_scaling * acc;
    }
  }
}

// ---------------------------------------------------------------------------
// FFN gate/up (SwiGLU): y[r] = silu(scale_g * Wg[r] . x) * (scale_u * Wu[r] . x)
// Wg/Wu are FP8 e4m3 rows (Phase-1 storage), x is BF16, y is BF16.
// ---------------------------------------------------------------------------

inline void silu_gate_up_bf16(const std::uint8_t* Wg, const std::uint8_t* Wu,
                             const std::uint16_t* x, float scale_g, float scale_u,
                             int N, int K, std::vector<std::uint16_t>* out) {
  out->assign(N, 0);
  for (int r = 0; r < N; ++r) {
    float g = 0.0f, u = 0.0f;
    for (int i = 0; i < K; ++i) {
      const float xv = bf16_to_f32_host(x[i]);
      g += fp8_to_f32_host(Wg[r * K + i]) * xv;
      u += fp8_to_f32_host(Wu[r * K + i]) * xv;
    }
    (*out)[r] = f32_to_bf16_host(silu_f32(scale_g * g) * (scale_u * u));
  }
}

// ---------------------------------------------------------------------------
// Sampling (deterministic in (p, u); the RNG only produces u).
// p is a normalized probability row (sums to 1), n tokens.
// ---------------------------------------------------------------------------

// Greedy: argmax, ties go to the smallest token id.
inline int sample_greedy_f32(const float* p, int n) {
  int best = 0;
  float bv = p[0];
  for (int i = 1; i < n; ++i)
    if (p[i] > bv) {
      bv = p[i];
      best = i;
    }
  return best;
}

// Nucleus (top-p): the nucleus is the smallest prefix of tokens sorted by
// probability descending (ties by ascending token id) whose cumulative mass
// reaches `nucleus`. Sampling picks the first prefix position whose running
// sum reaches u * nucleus_mass, so a fixed u is fully deterministic.
// Double arithmetic: at 100352-vocab rows a float cumulative sum drifts
// enough to flip the picked prefix position.
inline int sample_top_p_f32(const float* p, float nucleus, float u, int n) {
  std::vector<int> idx(n);
  for (int i = 0; i < n; ++i) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [p](int a, int b) {
    if (p[a] != p[b]) return p[a] > p[b];
    return a < b;
  });
  double acc = 0.0;
  int cut = n;
  for (int j = 0; j < n; ++j) {
    acc += static_cast<double>(p[idx[j]]);
    cut = j + 1;
    if (acc >= static_cast<double>(nucleus)) break;
  }
  const double target = static_cast<double>(u) * acc;
  double cum = 0.0;
  for (int j = 0; j < cut; ++j) {
    cum += static_cast<double>(p[idx[j]]);
    if (cum >= target) return idx[j];
  }
  return idx[cut - 1];
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
