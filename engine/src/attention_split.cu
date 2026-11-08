// Split-KV attention for decode and speculative verify (flash-decoding).
//
// The fused GQA kernel (kernels_phase2.cu) runs one block per query head and
// accumulates the value sum serially over the whole prefix: 32 blocks on a
// 66-SM card, and a dependent chain of kv_len FMAs per output. That is cheap
// at a few hundred positions and dominant at a few thousand (190 us/layer at
// 2 K context). Here the prefix is cut into 64-position tiles:
//
//   attn_split_kernel    grid (Hk, splits, row groups), 128 threads. One block
//                        owns one KV head, its G = 4 query heads (one warp
//                        each), a contiguous run of tiles, and up to 8 query
//                        rows that all reuse each staged tile. K/V tiles are staged
//                        in shared memory once for all four heads; each warp
//                        computes bf16-rounded logits (the reference
//                        rounding), an online f32 softmax, and an
//                        unnormalized partial output.
//   attn_combine_kernel  one block per (row, head): rescales and sums the
//                        split partials, normalizes, rounds to bf16.
//
// The split layout depends only on the cache stride (max positions), never
// on the call, so the decode step and the batched verify produce identical
// rows. The causal length may come from device memory (*d_pos + 1 + row), so
// the decode graph replays it at every position; the grid is sized for the
// full stride and blocks past the prefix exit at once.
//
// Numerics: logits are rounded to bf16 exactly as in the reference; the
// softmax runs in f32 (expf) and the probabilities are not rounded to bf16
// before the value product, so outputs agree with the fused kernel within
// bf16 rounding of the result (test_kernels.cu).
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include <cuda_runtime.h>

#include "attention_gpu.h"
#include "device_math.cuh"
#include "model_config.h"

namespace granite4 {
namespace {

using dev::bf16_to_f32;
using dev::f32_to_bf16;

constexpr int kTile = 64;
constexpr int kD = 128;
constexpr int kG = 4;  // query heads per KV head == warps per block
constexpr int kThreads = kG * 32;
// Shared K row pitch: 65 words, so lane j reading row j hits bank (j + i)
// mod 32 - conflict free. Rows are therefore only 4-byte aligned.
constexpr int kPitchWords = kD / 2 + 1;
constexpr int kMaxSplits = 256;
// Verify rows sharing one block's K/V tiles (rows > 1).
constexpr int kRowsPerBlock = 8;
constexpr int kShortCtx = 1024;

int tiles_per_split(int stride) {
  const int tiles = (stride + kTile - 1) / kTile;
  return (tiles + kMaxSplits - 1) / kMaxSplits;
}

// RB query rows share each block (RB = 1 for decode, kRowsPerBlock for a
// batched verify): the K/V tile is staged once and every row of the block
// attends to it, so a K-row verify reads the cache once instead of K times.
// Each row runs exactly the per-row arithmetic of the single-row kernel (same
// tiles, same op order), so outputs are bit-identical whatever RB is.
//
// Int8 (Q8_0) / Int4 (Q4_0): the tile fill dequantizes each row into the same
// bf16 Ks/Vs tiles the bf16 path writes (16 / 32 codes per thread, vector
// loads), so everything after the fill is byte-identical between formats.
template <bool Int8, bool Int4, int RB>
__global__ void __launch_bounds__(kThreads)
    attn_split_kernel(const std::uint16_t* __restrict__ q,
                      const std::uint16_t* __restrict__ k,
                      const std::uint16_t* __restrict__ v, float scale, int Hq,
                      int kv_base, const int* __restrict__ d_pos, int stride, int tps,
                      int splits, int rows, float2* __restrict__ part_ml,
                      float* __restrict__ part_o,
                      const std::int8_t* __restrict__ cache_k8,
                      const std::int8_t* __restrict__ cache_v8,
                      const float* __restrict__ k_scale,
                      const float* __restrict__ v_scale,
                      const std::uint8_t* __restrict__ cache_k4,
                      const std::uint8_t* __restrict__ cache_v4,
                      const int* __restrict__ d_pt,
                      const int* __restrict__ row_pos,
                      const int* const* __restrict__ row_pt,
                      const int* __restrict__ row_grp,
                      const int* __restrict__ row_tbase,
                      const int* __restrict__ row_tail) {
  const int hkv = blockIdx.x;
  const int split = blockIdx.y;
  // row_grp (multi-sequence rows): block z owns rows [r0, r0 + nr) of ONE
  // sequence at consecutive positions (encoded r0 | nr << 8, nr <= RB).
  const int gd = row_grp != nullptr ? row_grp[blockIdx.z] : 0;
  const int r0 = row_grp != nullptr ? (gd & 0xff) : blockIdx.z * RB;
  const int nr = row_grp != nullptr ? (gd >> 8) : min(RB, rows - r0);
  // Multi-sequence rows: row r0 is at position row_pos[r0] with row table
  // row_pt[r0] (a group's rows continue it); base is chosen so that the
  // shared per-row formula base + r0 + rr + 1 is each row's causal length.
  const int base = row_pos != nullptr ? row_pos[r0] - r0
                                      : (d_pos != nullptr ? *d_pos : kv_base);
  const int t_begin = split * tps * kTile;
  // Row rr attends to [0, row_len(rr)): consecutive rows (decode / chain
  // verify) or each row's own position (multi-sequence and tree rows).
  auto row_len = [&](int rr) {
    return row_pos != nullptr ? row_pos[r0 + rr] + 1 : base + r0 + rr + 1;
  };
  int kv_max = 0;
  for (int rr = 0; rr < nr; ++rr) kv_max = max(kv_max, row_len(rr));
  if (t_begin >= kv_max) return;
  const int t_end_max = min(kv_max, t_begin + tps * kTile);

  __shared__ std::uint32_t Ks[kTile][kPitchWords];
  __shared__ __align__(16) std::uint16_t Vs[kTile][kD];
  __shared__ std::uint32_t qs[RB][kG][kD / 2];  // bf16 pairs (exact as f32)
  __shared__ float ps[kG][kTile];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = hkv * kG + warp;
  for (int r = 0; r < nr; ++r) {
    const std::uint32_t* const qrow = reinterpret_cast<const std::uint32_t*>(
        q + (static_cast<std::size_t>(r0 + r) * Hq + h) * kD);
    for (int i = lane; i < kD / 2; i += 32) qs[r][warp][i] = qrow[i];
  }

  const bool quantized = Int8 || Int4;
  const std::uint16_t* const kb =
      quantized ? nullptr : k + static_cast<std::size_t>(hkv) * stride * kD;
  const std::uint16_t* const vb =
      quantized ? nullptr : v + static_cast<std::size_t>(hkv) * stride * kD;
  const std::int8_t* const k8 =
      Int8 ? cache_k8 + static_cast<std::size_t>(hkv) * stride * kD : nullptr;
  const std::int8_t* const v8 =
      Int8 ? cache_v8 + static_cast<std::size_t>(hkv) * stride * kD : nullptr;
  const std::uint8_t* const k4 =
      Int4 ? cache_k4 + static_cast<std::size_t>(hkv) * stride * (kD / 2) : nullptr;
  const std::uint8_t* const v4 =
      Int4 ? cache_v4 + static_cast<std::size_t>(hkv) * stride * (kD / 2) : nullptr;
  const float* const ksc =
      quantized ? k_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const float* const vsc =
      quantized ? v_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  float m[RB], l[RB], o[RB][4];  // dims 4 * lane ... 4 * lane + 3
#pragma unroll
  for (int r = 0; r < RB; ++r) {
    m[r] = -INFINITY;
    l[r] = 0.0f;
    o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.0f;
  }

  // Logical position -> physical arena row (identity when d_pt is null).
  // Tree rows (row_tbase, one row per block): positions above tbase are this
  // row's ancestors' slots (row_tail), the keys of the path it continues.
  const int* const pt = row_pt != nullptr ? row_pt[r0] : d_pt;
  // A group of tree rows shares tbase (one session's draft tree): tiles at or
  // below it are filled once for the group; a tile reaching past it is
  // filled and attended once per row with that row's own tail (so every row
  // still sees exactly its sequential tiles).
  const int tb = row_tbase != nullptr ? row_tbase[r0] : INT_MAX;
  auto phys = [&](int rr, int lp) {
    if (lp > tb) lp = row_tail[(r0 + rr) * kAttnTailStride + lp - tb - 1];
    return pt != nullptr ? pt[lp] : lp;
  };
  for (int t0 = t_begin; t0 < t_end_max; t0 += kTile) {
    const int n = min(kTile, t_end_max - t0);
    const bool specific = row_tail != nullptr && t0 + n - 1 > tb;
    const int passes = specific ? nr : 1;
    for (int fr = 0; fr < passes; ++fr) {
    __syncthreads();  // previous tile's readers (and the q preload) are done
    if (!quantized) {
      for (int idx = threadIdx.x; idx < n * (kD / 8); idx += kThreads) {
        const int r = idx / (kD / 8), c = (idx % (kD / 8)) * 8;
        const int row = phys(fr, t0 + r);
        const std::size_t off = static_cast<std::size_t>(row) * kD + c;
        const uint4 kw = *reinterpret_cast<const uint4*>(kb + off);
        std::uint32_t* const kd = &Ks[r][c / 2];
        kd[0] = kw.x;
        kd[1] = kw.y;
        kd[2] = kw.z;
        kd[3] = kw.w;
        *reinterpret_cast<uint4*>(&Vs[r][c]) = *reinterpret_cast<const uint4*>(vb + off);
      }
    } else if (Int4) {
      // KV4: one thread per (row, 16 code bytes = 32 dims).
      for (int idx = threadIdx.x; idx < n * (kD / 32); idx += kThreads) {
        const int r = idx / (kD / 32), c = (idx % (kD / 32)) * 16;  // code-byte offset
        const int pos = phys(fr, t0 + r);
        const std::size_t cbase = static_cast<std::size_t>(pos) * (kD / 2) + c;
        const float sk = ksc[pos], sv = vsc[pos];
        const uint4 kraw = *reinterpret_cast<const uint4*>(k4 + cbase);
        const uint4 vraw = *reinterpret_cast<const uint4*>(v4 + cbase);
        const unsigned char* const kc = reinterpret_cast<const unsigned char*>(&kraw);
        const unsigned char* const vc = reinterpret_cast<const unsigned char*>(&vraw);
        std::uint16_t vout[32];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          float klo, khi, vlo, vhi;
          dev::q4_dequant_pair(kc[e], sk, klo, khi);
          dev::q4_dequant_pair(vc[e], sv, vlo, vhi);
          Ks[r][c + e] = static_cast<std::uint32_t>(f32_to_bf16(klo)) |
                         (static_cast<std::uint32_t>(f32_to_bf16(khi)) << 16);
          vout[2 * e] = f32_to_bf16(vlo);
          vout[2 * e + 1] = f32_to_bf16(vhi);
        }
#pragma unroll
        for (int e = 0; e < 4; ++e)
          *reinterpret_cast<uint4*>(&Vs[r][2 * c + 8 * e]) =
              *reinterpret_cast<const uint4*>(&vout[8 * e]);
      }
    } else {
      // KV8: one thread per (row, 16 codes): two 16-byte loads, then the same
      // per-element dequant (scale * code -> bf16) the scalar fill did.
      for (int idx = threadIdx.x; idx < n * (kD / 16); idx += kThreads) {
        const int r = idx / (kD / 16), c = (idx % (kD / 16)) * 16;  // dim offset
        const int pos = phys(fr, t0 + r);
        const std::size_t base8 = static_cast<std::size_t>(pos) * kD + c;
        const float sk = ksc[pos], sv = vsc[pos];
        const uint4 kraw = *reinterpret_cast<const uint4*>(k8 + base8);
        const uint4 vraw = *reinterpret_cast<const uint4*>(v8 + base8);
        const std::int8_t* const kc = reinterpret_cast<const std::int8_t*>(&kraw);
        const std::int8_t* const vc = reinterpret_cast<const std::int8_t*>(&vraw);
        std::uint16_t vout[16];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const std::uint16_t klo = f32_to_bf16(dev::kv8_dequant(kc[2 * e], sk));
          const std::uint16_t khi = f32_to_bf16(dev::kv8_dequant(kc[2 * e + 1], sk));
          Ks[r][c / 2 + e] =
              static_cast<std::uint32_t>(klo) | (static_cast<std::uint32_t>(khi) << 16);
          vout[2 * e] = f32_to_bf16(dev::kv8_dequant(vc[2 * e], sv));
          vout[2 * e + 1] = f32_to_bf16(dev::kv8_dequant(vc[2 * e + 1], sv));
        }
        *reinterpret_cast<uint4*>(&Vs[r][c]) = *reinterpret_cast<const uint4*>(&vout[0]);
        *reinterpret_cast<uint4*>(&Vs[r][c + 8]) = *reinterpret_cast<const uint4*>(&vout[8]);
      }
    }
    __syncthreads();

#pragma unroll
    for (int rr = 0; rr < RB; ++rr) {
      if (rr >= nr) break;
      if (specific && rr != fr) continue;  // row-specific tile: row fr only
      // This row's slice of the split: [t_begin, min(kv_len, split end)).
      const int t_end = min(row_len(rr), t_begin + tps * kTile);
      if (t0 >= t_end) continue;  // row done (warp-uniform)
      const int nrow = min(kTile, t_end - t0);
      // Logits for kv = lane and lane + 32 of this tile, bf16-rounded.
      float s[2];
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int kv = lane + 32 * j;
        if (kv < nrow) {
          float acc = 0.0f;
          const std::uint32_t* const kr = &Ks[kv][0];
          const std::uint32_t* const qr = &qs[rr][warp][0];
#pragma unroll 8
          for (int i = 0; i < kD / 2; ++i) {
            const std::uint32_t w = kr[i];
            const std::uint32_t qw = qr[i];
            acc += dev::bf16_lo(w) * dev::bf16_lo(qw);
            acc += dev::bf16_hi(w) * dev::bf16_hi(qw);
          }
          s[j] = bf16_to_f32(f32_to_bf16(scale * acc));
        } else {
          s[j] = -INFINITY;
        }
      }
      float mt = fmaxf(s[0], s[1]);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, off));
      const float mn = fmaxf(m[rr], mt);
      const float alpha = expf(m[rr] - mn);  // 0 on the first tile
      const float p0 = expf(s[0] - mn), p1 = expf(s[1] - mn);
      float ls = p0 + p1;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
      l[rr] = l[rr] * alpha + ls;
      m[rr] = mn;
      __syncwarp();  // the previous row's PV readers of ps are done
      ps[warp][lane] = p0;
      ps[warp][lane + 32] = p1;
      __syncwarp();
#pragma unroll
      for (int c = 0; c < 4; ++c) o[rr][c] *= alpha;
      for (int kv = 0; kv < nrow; ++kv) {
        const float p = ps[warp][kv];
        const uint2 w = *reinterpret_cast<const uint2*>(&Vs[kv][4 * lane]);
        o[rr][0] += p * dev::bf16_lo(w.x);
        o[rr][1] += p * dev::bf16_hi(w.x);
        o[rr][2] += p * dev::bf16_lo(w.y);
        o[rr][3] += p * dev::bf16_hi(w.y);
      }
    }
    }  // passes
  }

#pragma unroll
  for (int rr = 0; rr < RB; ++rr) {
    if (rr >= nr) break;
    // Rows whose causal length ends before this split never reach it; the
    // combine kernel does not read their slot.
    if (t_begin >= row_len(rr)) continue;
    const std::size_t slot =
        (static_cast<std::size_t>(r0 + rr) * Hq + h) * splits + split;
    if (lane == 0) part_ml[slot] = make_float2(m[rr], l[rr]);
    *reinterpret_cast<float4*>(part_o + slot * kD + 4 * lane) =
        make_float4(o[rr][0], o[rr][1], o[rr][2], o[rr][3]);
  }
}

__global__ void __launch_bounds__(kD)
    attn_combine_kernel(const float2* __restrict__ part_ml, const float* __restrict__ part_o,
                        int Hq, int kv_base, const int* __restrict__ d_pos, int tps,
                        int splits, std::uint16_t* __restrict__ out,
                        const int* __restrict__ row_pos) {
  const int h = blockIdx.x;
  const int row = blockIdx.y;
  const int kv_len = row_pos != nullptr ? row_pos[row] + 1
                                        : (d_pos != nullptr ? *d_pos : kv_base) + row + 1;
  const int used = (kv_len + tps * kTile - 1) / (tps * kTile);
  const std::size_t base = (static_cast<std::size_t>(row) * Hq + h) * splits;
  float M = -INFINITY;
  for (int s = 0; s < used; ++s) M = fmaxf(M, part_ml[base + s].x);
  float L = 0.0f, acc = 0.0f;
  for (int s = 0; s < used; ++s) {
    const float2 ml = part_ml[base + s];
    const float w = expf(ml.x - M);
    L += ml.y * w;
    acc += part_o[(base + s) * kD + threadIdx.x] * w;
  }
  out[(static_cast<std::size_t>(row) * Hq + h) * kD + threadIdx.x] = f32_to_bf16(acc / L);
}

}  // namespace

std::size_t attn_split_scratch_bytes(int Hq, int rows, int stride) {
  const int splits = (((stride + kTile - 1) / kTile) + tiles_per_split(stride) - 1) /
                     tiles_per_split(stride);
  const std::size_t slots = static_cast<std::size_t>(rows) * Hq * splits;
  return slots * (sizeof(float2) + kD * sizeof(float));
}

int attn_split(const std::uint16_t* q, const std::uint16_t* cache_k,
               const std::uint16_t* cache_v, float scale, int Hq, int Hk, int d,
               int kv_base, const int* d_pos, int rows, int stride, void* scratch,
               std::uint16_t* out, const std::int8_t* cache_k8,
               const std::int8_t* cache_v8, const float* cache_k_scale,
               const float* cache_v_scale, bool int8, const std::uint8_t* cache_k4,
               const std::uint8_t* cache_v4, bool int4, const int* d_pt, int max_len,
               const int* row_pos, const int* const* row_pt, const int* row_grp,
               int ngroups, const int* row_tbase, const int* row_tail) {
  // `stride` addresses the cache arena; the split layout follows the longest
  // logical sequence (== stride for a private, contiguous cache).
  const int ml = max_len > 0 ? max_len : stride;
  if (q == nullptr || scratch == nullptr || out == nullptr ||
      !std::isfinite(scale) || d != kD || Hk <= 0 || Hq != kG * Hk ||
      rows <= 0 || rows > kMaxVerifyRows || stride <= 0 || ml > stride ||
      (d_pos == nullptr && row_pos == nullptr && kv_base + rows > ml) ||
      ((row_pos == nullptr) != (row_pt == nullptr)) ||
      (row_grp != nullptr && (row_pos == nullptr || ngroups <= 0 || ngroups > rows)) ||
      ((row_tbase != nullptr) != (row_tail != nullptr)) ||
      (row_tbase != nullptr && row_pos == nullptr) ||
      (int8 && int4) ||
      (int8 && (cache_k8 == nullptr || cache_v8 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (int4 && (cache_k4 == nullptr || cache_v4 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (!int8 && !int4 && (cache_k == nullptr || cache_v == nullptr)))
    return -1;
  const int tps = tiles_per_split(ml);
  const int splits = ((ml + kTile - 1) / kTile + tps - 1) / tps;
  float2* const pml = static_cast<float2*>(scratch);
  float* const po = reinterpret_cast<float*>(pml + static_cast<std::size_t>(rows) * Hq * splits);
  // One block per (kv head, split, group of RB rows). Decode (rows == 1) uses
  // the single-row instance (smallest shared-memory footprint).
  auto launch = [&](auto rbtag, const int* grp, int nz) {
    constexpr int RB = decltype(rbtag)::value;
    const dim3 grid(Hk, splits, nz);
    if (int8)
      attn_split_kernel<true, false, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          cache_k8, cache_v8, cache_k_scale, cache_v_scale, nullptr, nullptr, d_pt, row_pos, row_pt,
          grp, row_tbase, row_tail);
    else if (int4)
      attn_split_kernel<false, true, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos, row_pt,
          grp, row_tbase, row_tail);
    else
      attn_split_kernel<false, false, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_pt, row_pos, row_pt, grp,
          row_tbase, row_tail);
  };
  using One = std::integral_constant<int, 1>;
  using Many = std::integral_constant<int, kRowsPerBlock>;
  // Short contexts (a few tiles per row) have too few blocks to fill the
  // card when rows share a block, and the KV re-read is cheap there, so they
  // keep one row per block; past ~1K positions sharing the tiles wins.
  const bool short_ctx = d_pos == nullptr && kv_base + rows <= kShortCtx;
  // Rows of different sequences never share a tile: one row per block, or
  // one same-sequence row group per block when the caller passes groups.
  if (row_grp != nullptr)
    launch(Many{}, row_grp, ngroups);
  else if (rows == 1 || short_ctx || row_pos != nullptr)
    launch(One{}, nullptr, rows);
  else
    launch(Many{}, nullptr, (rows + kRowsPerBlock - 1) / kRowsPerBlock);
  attn_combine_kernel<<<dim3(Hq, rows), kD>>>(pml, po, Hq, kv_base, d_pos, tps, splits, out,
                                              row_pos);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "granite4: attn_split: %s\n", cudaGetErrorString(e));
    return -1;
  }
  return 0;
}

}  // namespace granite4
