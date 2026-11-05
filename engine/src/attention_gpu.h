// Split-KV decode / verify attention (attention_split.cu). Compiled only
// with GRANITE_ENABLE_CUDA=ON.
#ifndef GRANITE4_ATTENTION_GPU_H
#define GRANITE4_ATTENTION_GPU_H

#include <cstddef>
#include <cstdint>

namespace granite4 {

// Scratch bytes attn_split needs for `rows` query rows at cache stride
// `stride`.
std::size_t attn_split_scratch_bytes(int Hq, int rows, int stride);

// Causal GQA for `rows` (1..4) consecutive query rows q [rows][Hq*d]: row r
// sits at position p + r, p = *d_pos when d_pos is non-null (read on the
// device) else kv_base, and attends to cache positions [0, p + r]. The
// caches are [Hk][stride][d]; out is [rows][Hq*d]. Head dim 128 and
// Hq == 4 * Hk only; returns -1 otherwise (callers fall back to fused GQA).
//
// KV8 (Q8_0): when int8 is true the persistent cache is read as int8 codes
// (cache_k8/cache_v8, [Hk][stride][d]) plus a per-row scale
// (cache_k_scale/cache_v_scale, [Hk][stride]) and dequantized into the same
// bf16 tiles; cache_k/cache_v are then ignored (pass nullptr). The bf16
// path is bit-identical when int8 is false (the default).
int attn_split(const std::uint16_t* q, const std::uint16_t* cache_k,
               const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
               int kv_base, const int* d_pos, int rows, int stride, void* scratch,
               std::uint16_t* out, const std::int8_t* cache_k8 = nullptr,
               const std::int8_t* cache_v8 = nullptr,
               const float* cache_k_scale = nullptr,
               const float* cache_v_scale = nullptr, bool int8 = false);

}  // namespace granite4

#endif  // GRANITE4_ATTENTION_GPU_H
