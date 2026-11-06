// Tensor-core prefill primitives (prefill_gemm.cu). Device pointers only;
// every launcher returns 0 on success and -1 on bad arguments or a launch
// error. Compiled only with GRANITE_ENABLE_CUDA=ON.
#ifndef GRANITE4_PREFILL_GEMM_H
#define GRANITE4_PREFILL_GEMM_H

#include <cstddef>
#include <cstdint>

namespace granite4 {

enum class GemmOut {
  kBf16,      // out_bf16[m][n] = bf16(scale(n) * acc)
  kF32,       // out_f32[m][n]  = scale(n) * acc
  kResidual,  // out_bf16[m][n] = bf16(residual[m][n] + resid_scale * scale(n) * acc)
};

// scale(n) = scale0 for n < seg1, scale1 for n < seg2, else scale2: one GEMM
// covers weight tensors stored back to back with separate per-tensor scales.
struct GemmEpilogue {
  float scale0 = 1.0f, scale1 = 1.0f, scale2 = 1.0f;
  int seg1 = INT32_MAX, seg2 = INT32_MAX;
  std::uint16_t* out_bf16 = nullptr;
  float* out_f32 = nullptr;
  const std::uint16_t* residual = nullptr;  // may alias out_bf16
  float resid_scale = 1.0f;
  int ldc = 0;  // output (and residual) row pitch, elements
};

// C[M][N] = A[M][K] (bf16, row pitch lda) x W[N][K]^T (fp8 e4m3, row-major).
// Needs K % 32 == 0, lda % 8 == 0, even N / ldc, 16-byte aligned A and W.
// With a workspace (f32, ws_bytes), GEMMs with too few output tiles to fill
// the card split K across blocks and reduce deterministically.
int prefill_gemm(GemmOut out, const std::uint16_t* A, int lda, const std::uint8_t* W,
                 int M, int N, int K, const GemmEpilogue& ep, float* ws = nullptr,
                 std::size_t ws_bytes = 0);

// Workspace that lets prefill_gemm use its largest split for an M x N GEMM.
std::size_t prefill_gemm_workspace_bytes(int M, int N);

// out[r][:] = mult * emb[tokens[start + r]][:], r in [0, rows).
int prefill_embed_rows(const std::int32_t* d_tokens, int start, int rows,
                       const std::uint16_t* emb, float mult, std::uint16_t* out, int H);

// Split token-major q|k|v rows at positions [base_pos, base_pos + rows):
// RoPE q into q_rot [rows][Hq*d]; RoPE k and copy v into the [Hk][stride][d]
// caches.
int prefill_rope_split_append(const std::uint16_t* qkv, const float* cos_tbl,
                              const float* sin_tbl, int base_pos, int rows, int Hq,
                              int Hk, int d, int stride, std::uint16_t* q_rot,
                              std::uint16_t* cache_k, std::uint16_t* cache_v);

// KV8: append the prefill chunk's K/V to the int8 persistent cache - K is
// rotated then quantized per (head, position) row, V is quantized as-is - with
// a per-row float scale (Q8_0). Q is rotated to bf16 as in the bf16 path.
// Requires d == 128. Returns -1 on bad args or a launch error.
int prefill_rope_split_append_int8(const std::uint16_t* qkv, const float* cos_tbl,
                                   const float* sin_tbl, int base_pos, int rows,
                                   int Hq, int Hk, int d, int stride,
                                   std::uint16_t* q_rot, std::int8_t* cache_k,
                                   std::int8_t* cache_v, float* cache_k_scale,
                                   float* cache_v_scale);

// KV4: append the prefill chunk's K/V to the int4 persistent cache - K is
// rotated then quantized per (head, position) row as two-per-byte int4
// codes, V is quantized as-is - with a per-row float scale (Q4_0). Q is
// rotated to bf16 as in the bf16 path. Requires d == 128. Returns -1 on
// bad args or a launch error.
int prefill_rope_split_append_int4(const std::uint16_t* qkv, const float* cos_tbl,
                                   const float* sin_tbl, int base_pos, int rows,
                                   int Hq, int Hk, int d, int stride,
                                   std::uint16_t* q_rot, std::uint8_t* cache_k,
                                   std::uint8_t* cache_v, float* cache_k_scale,
                                   float* cache_v_scale);

// Causal attention for query rows at positions [base_pos, base_pos + rows)
// (q_rot [rows][Hq*d]) over the cache positions [0, position]; out is
// [rows][Hq*d]. Tensor cores; head_dim 128 and Hq / Hk == 4 only (-1
// otherwise, so the caller can fall back to the fused GQA kernel). With
// int8/int4, the KV cache is read as int8/int4 codes + per-row scales
// (cache_k8/cache_v8 [Hk][stride][d], cache_k4/cache_v4 [Hk][stride][d/2],
// *_scale [Hk][stride]) and dequantized on load.
int prefill_attention(const std::uint16_t* q_rot, const std::uint16_t* cache_k,
                       const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
                       int base_pos, int rows, int stride, std::uint16_t* out,
                       const std::int8_t* cache_k8, const std::int8_t* cache_v8,
                       const float* k_scale, const float* v_scale, bool int8,
                       const std::uint8_t* cache_k4, const std::uint8_t* cache_v4,
                       bool int4);

// act[r][i] = bf16(silu(gu[r][i]) * gu[r][I + i]).
int prefill_silu_mul(const float* gu, int I, int rows, std::uint16_t* act);

}  // namespace granite4

#endif  // GRANITE4_PREFILL_GEMM_H
