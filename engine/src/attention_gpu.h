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
// d_pt (default nullptr): a device `position -> physical KV row` table
// covering cache positions [0, stride]. When null the layout is contiguous
// (row == position). When set, each cache position is gathered through the
// table, so a paged arena (shared-prefix pages that are not physically
// contiguous) reads the same bytes as the contiguous layout. The split-KV
// path honors d_pt for both bf16 and quantized (int8/int4) caches; the
// fused-GQA rows fallback (rows > kMaxVerifyRows) does not, so a caller that
// needs a paged layout must keep rows <= kMaxVerifyRows.
//
// max_len (default 0 = stride): the longest logical sequence the caller will
// attend over. `stride` is the row stride of the cache arena (rows per head),
// which for a shared paged arena is the whole pool; the split layout, the
// grid and the scratch are sized from max_len instead, so a pool much larger
// than any one sequence does not widen the launch. max_len <= stride.
//
// row_pos / row_pt (default null; both or neither): multi-sequence rows. Row r
// is its own sequence at position row_pos[r] (attending to [0, row_pos[r]])
// through its own position -> arena-row table row_pt[r]; kv_base, d_pos and
// d_pt are ignored. Device arrays of `rows` entries. Used to decode several
// sessions of one shared KV arena in one step; each row's arithmetic is that
// of a single-row call at its position.
// row_grp (device, ngroups entries, with row_pos): block z owns rows
// [g & 0xff, + (g >> 8)) of ONE sequence at consecutive positions, so those
// rows read each K/V tile once (bit-identical to one row per block).
//
// KV8 (Q8_0): when int8 is true the persistent cache is read as int8 codes
// int4, the KV cache is read as int4 codes (cache_k4/cache_v4,
// [Hk][stride][d/2], two per byte) plus a per-row scale
// (cache_k_scale/cache_v_scale, [Hk][stride]) and dequantized into the same
// bf16 tiles; cache_k/cache_v are then ignored (pass nullptr). The bf16
// path is bit-identical when int8 and int4 are both false (the default).
// Tree rows: row_tbase[r] / row_tail[r * kAttnTailStride + j] remap logical
// positions above tbase to ancestor slots (one row per block, no groups).
constexpr int kAttnTailStride = 8;
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
               int kv_hint = 0, bool k5 = false);

}  // namespace granite4

#endif  // GRANITE4_ATTENTION_GPU_H
