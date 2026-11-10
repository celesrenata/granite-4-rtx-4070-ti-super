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
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
constexpr int kMaxSplitsDefault = 256;
// Splits of the longest sequence (split width = ceil(tiles / max splits)).
// One value per process (G4_ATTN_MAX_SPLITS), so decode, verify and every
// session share one split layout and stay bit-identical to each other.
int max_splits() {
  static const int v = [] {
    const char* e = std::getenv("G4_ATTN_MAX_SPLITS");
    const int x = e != nullptr ? std::atoi(e) : 0;
    return x >= 8 && x <= 1024 ? x : kMaxSplitsDefault;
  }();
  return v;
}
// Verify rows sharing one block's K/V tiles (rows > 1).
constexpr int kRowsPerBlock = 8;
constexpr int kShortCtx = 1024;

int tiles_per_split(int stride) {
  const int tiles = (stride + kTile - 1) / kTile;
  return (tiles + max_splits() - 1) / max_splits();
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
template <bool Int8, bool Int4, int RB, bool K5 = false>
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
                      const int* __restrict__ row_tail, bool attn_pipe_on) {
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
      Int4 ? cache_k4 + static_cast<std::size_t>(hkv) * stride * (K5 ? dev::kK5RowBytes : kD / 2)
           : nullptr;
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
  // KV8 without tree tails (decode, verify, multi rows): the next tile's
  // codes and scales are loaded into registers before the current tile is
  // attended, so the loads overlap the compute (same values, same order).
  // G4_ATTN_PIPE=0 disables.
  constexpr int kItems = (kTile * (kD / 16) + kThreads - 1) / kThreads;
  const bool pipe = Int8 && row_tail == nullptr && attn_pipe_on;
  uint4 pk[kItems], pv[kItems];
  float psk[kItems], psv[kItems];
  auto prefetch = [&](int tn) {
    const int nn = min(kTile, t_end_max - tn);
#pragma unroll
    for (int it = 0; it < kItems; ++it) {
      const int idx = threadIdx.x + it * kThreads;
      if (idx < nn * (kD / 16)) {
        const int r = idx / (kD / 16), c = (idx % (kD / 16)) * 16;
        const int pos = phys(0, tn + r);
        const std::size_t base8 = static_cast<std::size_t>(pos) * kD + c;
        psk[it] = ksc[pos];
        psv[it] = vsc[pos];
        pk[it] = *reinterpret_cast<const uint4*>(k8 + base8);
        pv[it] = *reinterpret_cast<const uint4*>(v8 + base8);
      }
    }
  };
  if (pipe) prefetch(t_begin);
  for (int t0 = t_begin; t0 < t_end_max; t0 += kTile) {
    const int n = min(kTile, t_end_max - t0);
    const bool specific = row_tail != nullptr && t0 + n - 1 > tb;
    const int passes = specific ? nr : 1;
    for (int fr = 0; fr < passes; ++fr) {
    // Later passes of a row-specific tile keep the shared positions (<= tb)
    // from pass 0 and only rewrite the tail positions with row fr's keys.
    const int rs = fr > 0 ? max(0, tb + 1 - t0) : 0;
    __syncthreads();  // previous tile's readers (and the q preload) are done
    if (pipe) {
#pragma unroll
      for (int it = 0; it < kItems; ++it) {
        const int idx = threadIdx.x + it * kThreads;
        if (idx < n * (kD / 16)) {
          const int r = idx / (kD / 16), c = (idx % (kD / 16)) * 16;
          const std::int8_t* const kc = reinterpret_cast<const std::int8_t*>(&pk[it]);
          const std::int8_t* const vc = reinterpret_cast<const std::int8_t*>(&pv[it]);
          const float sk = psk[it], sv = psv[it];
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
      if (t0 + kTile < t_end_max) prefetch(t0 + kTile);
    } else if (!quantized) {
      for (int idx = threadIdx.x; idx < (n - rs) * (kD / 8); idx += kThreads) {
        const int r = rs + idx / (kD / 8), c = (idx % (kD / 8)) * 8;
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
      for (int idx = threadIdx.x; idx < (n - rs) * (kD / 32); idx += kThreads) {
        const int r = rs + idx / (kD / 32), c = (idx % (kD / 32)) * 16;  // code-byte offset
        const int pos = phys(fr, t0 + r);
        const std::size_t cbase = static_cast<std::size_t>(pos) * (kD / 2) + c;
        // K5: the K row is 80 bytes (64 nibble bytes + a 16-byte high-bit plane); this thread's
        // 16 nibble bytes are dims [2c, 2c + 32), whose high bits are 4 plane bytes at 64 + c / 4.
        const std::size_t kbase = K5 ? static_cast<std::size_t>(pos) * dev::kK5RowBytes + c : cbase;
        const float sk = ksc[pos], sv = vsc[pos];
        const uint4 kraw = *reinterpret_cast<const uint4*>(k4 + kbase);
        const uint4 vraw = *reinterpret_cast<const uint4*>(v4 + cbase);
        const unsigned char* const kc = reinterpret_cast<const unsigned char*>(&kraw);
        const unsigned char* const vc = reinterpret_cast<const unsigned char*>(&vraw);
        std::uint32_t khi_bits = 0;
        if (K5)
          khi_bits = *reinterpret_cast<const std::uint32_t*>(
              k4 + static_cast<std::size_t>(pos) * dev::kK5RowBytes + dev::kK5NibbleBytes + c / 4);
        std::uint16_t vout[32];
#pragma unroll
        for (int e = 0; e < 16; ++e) {
          float klo, khi, vlo, vhi;
          if (K5)
            dev::k5_dequant_pair(kc[e], (khi_bits >> (2 * e)) & 1u, (khi_bits >> (2 * e + 1)) & 1u,
                                 sk, klo, khi);
          else
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
      for (int idx = threadIdx.x; idx < (n - rs) * (kD / 16); idx += kThreads) {
        const int r = rs + idx / (kD / 16), c = (idx % (kD / 16)) * 16;  // dim offset
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

// Rows-parallel variant for row groups (up to kRowsPerBlock rows of ONE
// sequence: a batched verify, a multi-session group, a session's draft-tree
// rows). One warp per (row, query head) - kG * kRowsPerBlock warps - so the
// rows attend to each staged K/V tile at the same time instead of one after
// another. Tree rows' own keys (positions past tbase, at most kAttnTailStride
// per row) sit in a per-row tail buffer filled once per block, so a tile that
// reaches past tbase needs no per-row refill. Every warp runs exactly the
// per-row arithmetic of attn_split_kernel (same tiles, same op order, same
// key values), so outputs are bit-identical to one row per block.
constexpr int kRowsWarps = kG * kRowsPerBlock;
constexpr int kTailMax = kAttnTailStride;
constexpr std::size_t kRowsSmem =
    sizeof(std::uint32_t) * kTile * kPitchWords +                     // Ks
    sizeof(std::uint16_t) * kTile * kD +                              // Vs
    sizeof(std::uint32_t) * kRowsWarps * (kD / 2) +                   // qs
    sizeof(float) * kRowsWarps * kTile +                              // ps
    sizeof(std::uint32_t) * kRowsPerBlock * kTailMax * kPitchWords +  // tail K
    sizeof(std::uint16_t) * kRowsPerBlock * kTailMax * kD;            // tail V

// Dequantize / copy logical rows [0, cnt) into (K, V) rows (K as bf16 pairs,
// pitch kPitchWords; V pitch kD); phys(r) is row r's physical arena row.
template <bool Int8, bool Int4, bool K5, class Phys>
__device__ __forceinline__ void fill_kv(std::uint32_t* K, std::uint16_t* V, int cnt, Phys phys,
                                        const std::uint16_t* kb, const std::uint16_t* vb,
                                        const std::int8_t* k8, const std::int8_t* v8,
                                        const std::uint8_t* k4, const std::uint8_t* v4,
                                        const float* ksc, const float* vsc) {
  if (!Int8 && !Int4) {
    for (int idx = threadIdx.x; idx < cnt * (kD / 8); idx += blockDim.x) {
      const int r = idx / (kD / 8), c = (idx % (kD / 8)) * 8;
      const std::size_t off = static_cast<std::size_t>(phys(r)) * kD + c;
      const uint4 kw = *reinterpret_cast<const uint4*>(kb + off);
      std::uint32_t* const kd = K + r * kPitchWords + c / 2;
      kd[0] = kw.x;
      kd[1] = kw.y;
      kd[2] = kw.z;
      kd[3] = kw.w;
      *reinterpret_cast<uint4*>(V + r * kD + c) = *reinterpret_cast<const uint4*>(vb + off);
    }
  } else if (Int4) {
    for (int idx = threadIdx.x; idx < cnt * (kD / 32); idx += blockDim.x) {
      const int r = idx / (kD / 32), c = (idx % (kD / 32)) * 16;
      const int pos = phys(r);
      const std::size_t cbase = static_cast<std::size_t>(pos) * (kD / 2) + c;
      const std::size_t kbase = K5 ? static_cast<std::size_t>(pos) * dev::kK5RowBytes + c : cbase;
      const float sk = ksc[pos], sv = vsc[pos];
      const uint4 kraw = *reinterpret_cast<const uint4*>(k4 + kbase);
      const uint4 vraw = *reinterpret_cast<const uint4*>(v4 + cbase);
      const unsigned char* const kc = reinterpret_cast<const unsigned char*>(&kraw);
      const unsigned char* const vc = reinterpret_cast<const unsigned char*>(&vraw);
      std::uint32_t khi_bits = 0;
      if (K5)
        khi_bits = *reinterpret_cast<const std::uint32_t*>(
            k4 + static_cast<std::size_t>(pos) * dev::kK5RowBytes + dev::kK5NibbleBytes + c / 4);
      std::uint16_t vout[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        float klo, khi, vlo, vhi;
        if (K5)
          dev::k5_dequant_pair(kc[e], (khi_bits >> (2 * e)) & 1u, (khi_bits >> (2 * e + 1)) & 1u,
                               sk, klo, khi);
        else
          dev::q4_dequant_pair(kc[e], sk, klo, khi);
        dev::q4_dequant_pair(vc[e], sv, vlo, vhi);
        K[r * kPitchWords + c + e] = static_cast<std::uint32_t>(f32_to_bf16(klo)) |
                                     (static_cast<std::uint32_t>(f32_to_bf16(khi)) << 16);
        vout[2 * e] = f32_to_bf16(vlo);
        vout[2 * e + 1] = f32_to_bf16(vhi);
      }
#pragma unroll
      for (int e = 0; e < 4; ++e)
        *reinterpret_cast<uint4*>(V + r * kD + 2 * c + 8 * e) =
            *reinterpret_cast<const uint4*>(&vout[8 * e]);
    }
  } else {
    for (int idx = threadIdx.x; idx < cnt * (kD / 16); idx += blockDim.x) {
      const int r = idx / (kD / 16), c = (idx % (kD / 16)) * 16;
      const int pos = phys(r);
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
        K[r * kPitchWords + c / 2 + e] =
            static_cast<std::uint32_t>(klo) | (static_cast<std::uint32_t>(khi) << 16);
        vout[2 * e] = f32_to_bf16(dev::kv8_dequant(vc[2 * e], sv));
        vout[2 * e + 1] = f32_to_bf16(dev::kv8_dequant(vc[2 * e + 1], sv));
      }
      *reinterpret_cast<uint4*>(V + r * kD + c) = *reinterpret_cast<const uint4*>(&vout[0]);
      *reinterpret_cast<uint4*>(V + r * kD + c + 8) = *reinterpret_cast<const uint4*>(&vout[8]);
    }
  }
}

template <bool Int8, bool Int4, bool K5 = false>
__global__ void __launch_bounds__(kRowsWarps * 32)
    attn_rows_kernel(const std::uint16_t* __restrict__ q, const std::uint16_t* __restrict__ k,
                     const std::uint16_t* __restrict__ v, float scale, int Hq, int kv_base,
                     const int* __restrict__ d_pos, int stride, int tps, int splits, int rows,
                     float2* __restrict__ part_ml, float* __restrict__ part_o,
                     const std::int8_t* __restrict__ cache_k8,
                     const std::int8_t* __restrict__ cache_v8,
                     const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                     const std::uint8_t* __restrict__ cache_k4,
                     const std::uint8_t* __restrict__ cache_v4, const int* __restrict__ d_pt,
                     const int* __restrict__ row_pos, const int* const* __restrict__ row_pt,
                     const int* __restrict__ row_grp, const int* __restrict__ row_tbase,
                     const int* __restrict__ row_tail) {
  const int hkv = blockIdx.x;
  const int split = blockIdx.y;
  const int gd = row_grp != nullptr ? row_grp[blockIdx.z] : 0;
  const int r0 = row_grp != nullptr ? (gd & 0xff) : blockIdx.z * kRowsPerBlock;
  const int nr = row_grp != nullptr ? (gd >> 8) : min(kRowsPerBlock, rows - r0);
  const int base = row_pos != nullptr ? row_pos[r0] - r0
                                      : (d_pos != nullptr ? *d_pos : kv_base);
  auto row_len = [&](int rr) {
    return row_pos != nullptr ? row_pos[r0 + rr] + 1 : base + r0 + rr + 1;
  };
  const int t_begin = split * tps * kTile;
  int kv_max = 0;
  for (int rr = 0; rr < nr; ++rr) kv_max = max(kv_max, row_len(rr));
  if (t_begin >= kv_max) return;
  const int t_end_max = min(kv_max, t_begin + tps * kTile);

  extern __shared__ __align__(16) std::uint8_t smem_rows[];
  std::uint32_t* const Ks = reinterpret_cast<std::uint32_t*>(smem_rows);
  std::uint16_t* const Vs = reinterpret_cast<std::uint16_t*>(Ks + kTile * kPitchWords);
  std::uint32_t* const qs = reinterpret_cast<std::uint32_t*>(Vs + kTile * kD);
  float* const ps = reinterpret_cast<float*>(qs + kRowsWarps * (kD / 2));
  std::uint32_t* const TK = reinterpret_cast<std::uint32_t*>(ps + kRowsWarps * kTile);
  std::uint16_t* const TV = reinterpret_cast<std::uint16_t*>(TK + kRowsPerBlock * kTailMax * kPitchWords);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int rr = warp / kG;          // this warp's row in the group
  const int hl = warp % kG;          // query head within the KV head's group
  const int h = hkv * kG + hl;
  const bool live = rr < nr;
  std::uint32_t* const qw_s = qs + warp * (kD / 2);
  float* const pw = ps + warp * kTile;
  if (live) {
    const std::uint32_t* const qrow = reinterpret_cast<const std::uint32_t*>(
        q + (static_cast<std::size_t>(r0 + rr) * Hq + h) * kD);
    for (int i = lane; i < kD / 2; i += 32) qw_s[i] = qrow[i];
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
      Int4 ? cache_k4 + static_cast<std::size_t>(hkv) * stride * (K5 ? dev::kK5RowBytes : kD / 2)
           : nullptr;
  const std::uint8_t* const v4 =
      Int4 ? cache_v4 + static_cast<std::size_t>(hkv) * stride * (kD / 2) : nullptr;
  const float* const ksc =
      quantized ? k_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const float* const vsc =
      quantized ? v_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const int* const pt = row_pt != nullptr ? row_pt[r0] : d_pt;
  auto to_phys = [&](int lp) { return pt != nullptr ? pt[lp] : lp; };
  // Tree groups: every row's keys past tbase, once per block.
  const bool tails = row_tail != nullptr;
  const int tb = tails ? row_tbase[r0] : 0;
  if (tails && t_end_max > tb + 1) {
    for (int t = 0; t < nr; ++t) {
      const int cnt = min(kTailMax, row_len(t) - 1 - tb);
      if (cnt <= 0) continue;
      const int* const tl = row_tail + (r0 + t) * kAttnTailStride;
      fill_kv<Int8, Int4, K5>(TK + t * kTailMax * kPitchWords, TV + t * kTailMax * kD, cnt,
                          [&](int j) { return to_phys(tl[j]); }, kb, vb, k8, v8, k4, v4, ksc,
                          vsc);
    }
  }
  const std::uint32_t* const tkr = TK + (live ? rr : 0) * kTailMax * kPitchWords;
  const std::uint16_t* const tvr = TV + (live ? rr : 0) * kTailMax * kD;

  float m = -INFINITY, l = 0.0f, o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int t_end = live ? min(row_len(rr), t_begin + tps * kTile) : 0;
  for (int t0 = t_begin; t0 < t_end_max; t0 += kTile) {
    const int n = min(kTile, t_end_max - t0);
    // Positions of this tile shared by the group (all of them unless tree
    // tails start inside it).
    const int ns = tails ? max(0, min(n, tb + 1 - t0)) : n;
    __syncthreads();  // previous tile's readers (and the q / tail preloads) are done
    fill_kv<Int8, Int4, K5>(Ks, Vs, ns, [&](int r) { return to_phys(t0 + r); }, kb, vb, k8, v8, k4,
                        v4, ksc, vsc);
    __syncthreads();
    if (t0 >= t_end) continue;  // this warp's row is done (warp-uniform)
    const int nrow = min(kTile, t_end - t0);
    float s[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int kv = lane + 32 * j;
      if (kv < nrow) {
        float acc = 0.0f;
        const std::uint32_t* const kr =
            kv < ns ? Ks + kv * kPitchWords : tkr + (t0 + kv - tb - 1) * kPitchWords;
#pragma unroll 8
        for (int i = 0; i < kD / 2; ++i) {
          const std::uint32_t w = kr[i];
          const std::uint32_t qw = qw_s[i];
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
    const float mn = fmaxf(m, mt);
    const float alpha = expf(m - mn);  // 0 on the first tile
    const float p0 = expf(s[0] - mn), p1 = expf(s[1] - mn);
    float ls = p0 + p1;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, off);
    l = l * alpha + ls;
    m = mn;
    __syncwarp();
    pw[lane] = p0;
    pw[lane + 32] = p1;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < 4; ++c) o[c] *= alpha;
    for (int kv = 0; kv < nrow; ++kv) {
      const float p = pw[kv];
      const std::uint16_t* const vr = kv < ns ? Vs + kv * kD : tvr + (t0 + kv - tb - 1) * kD;
      const uint2 w = *reinterpret_cast<const uint2*>(vr + 4 * lane);
      o[0] += p * dev::bf16_lo(w.x);
      o[1] += p * dev::bf16_hi(w.x);
      o[2] += p * dev::bf16_lo(w.y);
      o[3] += p * dev::bf16_hi(w.y);
    }
  }
  if (!live || t_begin >= row_len(rr)) return;
  const std::size_t slot = (static_cast<std::size_t>(r0 + rr) * Hq + h) * splits + split;
  if (lane == 0) part_ml[slot] = make_float2(m, l);
  *reinterpret_cast<float4*>(part_o + slot * kD + 4 * lane) = make_float4(o[0], o[1], o[2], o[3]);
}

// ---------------------------------------------------------------------------
// Tensor-core split-KV attention (G4_ATTN_MMA, default on): ONE kernel for decode (1 row),
// consecutive-row verify, multi-session rows and draft-tree rows.
//
// Why: the scalar kernels above compute each (row, head) dot product on the CUDA cores out of
// shared memory. A 14-row verify at 16K positions measured ~11% of the K5:V4 bytes-read floor
// and the same time in every KV format, i.e. compute bound (granite4-attn-bench).
//
// Layout (prefill_attention_kernel's flash-attention fragments): a block owns one KV head, one
// split, and a group of up to kMmaRows rows of ONE sequence; its query rows are (row, query head)
// pairs, 4 per row, so up to 32 query rows = two m16 MMA tiles, one per compute warp. Per 64-key
// tile: S = Q K^T (mma.m16n8k16, f32 accumulate), logits rounded to bf16 (the reference rounding),
// causal mask per row, online softmax in f32, O += P V with P as bf16. The tiles, split layout and
// (m, l, o) partials are those of the scalar kernels, so attn_combine_kernel is unchanged.
//
// Exactness: an MMA output row depends only on its own A row and the B tile, and every reduction
// below (row max, row sum) has a fixed per-row shape, so a row's result does not depend on which
// other rows share the block. Decode and grouped verify therefore stay bit-identical to each other
// (the verify == decode contract the speculative paths rely on), checked by the rows / tree tests.
// Tree rows: a tile reaching past a group's shared prefix (tbase) is processed once per row, with
// that row's own keys in the tail positions, and only that row is updated (the others' logits
// are masked, which carries their state exactly), so the per-row tile sequence equals a decode
// of that path. V fragments come from a row-major tile via ldmatrix.trans, and the shared tiles
// are software-pipelined (the next tile's cache loads are in flight during the current tile).
constexpr int kMmaRows = 8;               // rows per block (32 query rows)
constexpr int kMmaThreads = 128;          // 4 warps: 2 compute (one m16 tile each), all 4 fill
constexpr int kMmaP = kD + 8;             // K / V tile pitch (bf16): conflict-free fragment loads
constexpr int kMmaItems = kTile * (kD / 16) / kMmaThreads;  // fill items (position, 16 dims)
constexpr int kMmaPtMax = 512;  // split positions whose page-table entries are staged in smem

__device__ __forceinline__ void mma_16816(float (&c)[4], const std::uint32_t (&a)[4],
                                          std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Four transposed 8x8 bf16 matrices: the m16n8k16 B fragments of two n8 blocks read from a
// row-major [key][dim] tile.
__device__ __forceinline__ void ldsm_x4_trans(std::uint32_t& r0, std::uint32_t& r1,
                                              std::uint32_t& r2, std::uint32_t& r3,
                                              const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ float mma_bf16_round(float x) {
  return __uint_as_float(static_cast<std::uint32_t>(f32_to_bf16(x)) << 16);
}
__device__ __forceinline__ std::uint32_t mma_pack(float lo, float hi) {
  return static_cast<std::uint32_t>(f32_to_bf16(lo)) |
         (static_cast<std::uint32_t>(f32_to_bf16(hi)) << 16);
}

// One fill item: 16 dims [16 * part, 16 * part + 16) of one key position, as loaded from the
// cache (bf16: k0 k1 v0 v1; KV8: k0 v0 + scales; KV4 / K5: k0.xy v0.xy + scales (+ K5 high bits)).
// Loading and storing are separate so a tile's loads can be in flight during the previous
// tile's compute.
struct MmaRaw {
  uint4 k0, k1, v0, v1;
  float sk, sv;
  unsigned hib;
};

template <bool Int8, bool Int4, bool K5>
__device__ __forceinline__ void mma_load_pos(MmaRaw& r, int row, bool valid, int part,
                                             const std::uint16_t* kb, const std::uint16_t* vb,
                                             const std::int8_t* k8, const std::int8_t* v8,
                                             const std::uint8_t* k4, const std::uint8_t* v4,
                                             const float* ksc, const float* vsc) {
  if (!valid) return;
  const int c0 = 16 * part;
  if (!Int8 && !Int4) {
    const uint4* const ks = reinterpret_cast<const uint4*>(kb + static_cast<std::size_t>(row) * kD + c0);
    const uint4* const vs = reinterpret_cast<const uint4*>(vb + static_cast<std::size_t>(row) * kD + c0);
    r.k0 = ks[0];
    r.k1 = ks[1];
    r.v0 = vs[0];
    r.v1 = vs[1];
  } else if (Int8) {
    r.sk = ksc[row];
    r.sv = vsc[row];
    r.k0 = *reinterpret_cast<const uint4*>(k8 + static_cast<std::size_t>(row) * kD + c0);
    r.v0 = *reinterpret_cast<const uint4*>(v8 + static_cast<std::size_t>(row) * kD + c0);
  } else {
    r.sk = ksc[row];
    r.sv = vsc[row];
    // 16 dims = 8 nibble bytes (int4 layout) for both K and V.
    const std::size_t kbase = K5 ? static_cast<std::size_t>(row) * dev::kK5RowBytes + c0 / 2
                                 : static_cast<std::size_t>(row) * (kD / 2) + c0 / 2;
    const uint2 kraw = *reinterpret_cast<const uint2*>(k4 + kbase);
    const uint2 vraw = *reinterpret_cast<const uint2*>(v4 + static_cast<std::size_t>(row) * (kD / 2) + c0 / 2);
    r.k0.x = kraw.x;
    r.k0.y = kraw.y;
    r.v0.x = vraw.x;
    r.v0.y = vraw.y;
    // K5: the 16 high bits of dims [c0, c0 + 16) = plane bytes c0 / 8, c0 / 8 + 1.
    if (K5)
      r.hib = *reinterpret_cast<const std::uint16_t*>(
          k4 + static_cast<std::size_t>(row) * dev::kK5RowBytes + dev::kK5NibbleBytes + c0 / 8);
  }
}

// bf16x2 a * 1 + c (exact here: the results are small integers).
__device__ __forceinline__ std::uint32_t mma_bf16x2_sub(std::uint32_t a, std::uint32_t c) {
  std::uint32_t r;
  asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(0x3f803f80u), "r"(c));
  return r;
}
// Eight 4-bit codes (one word, dim 2e in byte e's low nibble, 2e + 1 in its high nibble) plus
// K5 high bits (bit 2e / 2e + 1 of hib8) -> four bf16 pairs of (code - off), exactly: 0x4300 | n
// is the bf16 128 + n for n < 128, and off_bits holds -(128 + off) in both halves.
template <bool K5>
__device__ __forceinline__ void mma_codes8(std::uint32_t w, unsigned hib8, std::uint32_t off_bits,
                                           std::uint32_t* out) {
  const std::uint32_t lo = w & 0x0f0f0f0fu, hi = (w >> 4) & 0x0f0f0f0fu;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    // Result byte 0 = lo.byte e, byte 2 = hi.byte e (bytes 1 and 3 are masked off).
    const unsigned sel = static_cast<unsigned>(e) | static_cast<unsigned>(4 + e) << 8;
    std::uint32_t x = (__byte_perm(lo, hi, sel) & 0x00ff00ffu) | 0x43004300u;
    if (K5) {
      const unsigned b = hib8 >> (2 * e);
      x |= ((b & 1u) << 4) | ((b & 2u) << 19);
    }
    out[e] = mma_bf16x2_sub(x, off_bits);
  }
}

// Store a loaded item into Ks[kr] / Vs[kr] (row-major). bf16: the cache values. Quantized: the
// integer codes (exact in bf16; KV4 code - 8, K5 code - 16, KV8 the int8), with the position's
// scales in ksc_s / vsc_s - attend applies them to the logits (K) and the probabilities (V), so
// no per-element multiply or rounding here. !valid: zeros (masked logits stay finite and add 0).
template <bool Int8, bool Int4, bool K5>
__device__ __forceinline__ void mma_store_pos(std::uint16_t (*Ks)[kMmaP], std::uint16_t (*Vs)[kMmaP],
                                              float* ksc_s, float* vsc_s, int kr, int part,
                                              const MmaRaw& r, bool valid) {
  const int c0 = 16 * part;
  uint4 ko0, ko1, vo0, vo1;
  if (!valid) {
    ko0 = ko1 = vo0 = vo1 = make_uint4(0u, 0u, 0u, 0u);
  } else if (!Int8 && !Int4) {
    ko0 = r.k0;
    ko1 = r.k1;
    vo0 = r.v0;
    vo1 = r.v1;
  } else if (Int8) {
    std::uint32_t kw[8], vw[8];
    const std::int8_t* const kc = reinterpret_cast<const std::int8_t*>(&r.k0);
    const std::int8_t* const vc = reinterpret_cast<const std::int8_t*>(&r.v0);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      kw[e] = mma_pack(static_cast<float>(kc[2 * e]), static_cast<float>(kc[2 * e + 1]));
      vw[e] = mma_pack(static_cast<float>(vc[2 * e]), static_cast<float>(vc[2 * e + 1]));
    }
    ko0 = make_uint4(kw[0], kw[1], kw[2], kw[3]);
    ko1 = make_uint4(kw[4], kw[5], kw[6], kw[7]);
    vo0 = make_uint4(vw[0], vw[1], vw[2], vw[3]);
    vo1 = make_uint4(vw[4], vw[5], vw[6], vw[7]);
  } else {
    std::uint32_t kw[8], vw[8];
    constexpr std::uint32_t kOff = K5 ? 0xc310c310u : 0xc308c308u;  // -(128 + 16) / -(128 + 8)
    mma_codes8<K5>(r.k0.x, r.hib, kOff, &kw[0]);
    mma_codes8<K5>(r.k0.y, r.hib >> 8, kOff, &kw[4]);
    mma_codes8<false>(r.v0.x, 0u, 0xc308c308u, &vw[0]);
    mma_codes8<false>(r.v0.y, 0u, 0xc308c308u, &vw[4]);
    ko0 = make_uint4(kw[0], kw[1], kw[2], kw[3]);
    ko1 = make_uint4(kw[4], kw[5], kw[6], kw[7]);
    vo0 = make_uint4(vw[0], vw[1], vw[2], vw[3]);
    vo1 = make_uint4(vw[4], vw[5], vw[6], vw[7]);
  }
  *reinterpret_cast<uint4*>(&Ks[kr][c0]) = ko0;
  *reinterpret_cast<uint4*>(&Ks[kr][c0 + 8]) = ko1;
  *reinterpret_cast<uint4*>(&Vs[kr][c0]) = vo0;
  *reinterpret_cast<uint4*>(&Vs[kr][c0 + 8]) = vo1;
  if ((Int8 || Int4) && part == 0) {
    ksc_s[kr] = valid ? r.sk : 0.0f;
    vsc_s[kr] = valid ? r.sv : 0.0f;
  }
}

// Tree = row_tail present (draft-tree rows): a separate instance, so the common paths do not
// carry the per-row save / keep state.
template <bool Int8, bool Int4, bool K5, bool Tree>
__global__ void __launch_bounds__(kMmaThreads, 2)
    attn_mma_kernel(const std::uint16_t* __restrict__ q, const std::uint16_t* __restrict__ k,
                    const std::uint16_t* __restrict__ v, float scale, int Hq, int kv_base,
                    const int* __restrict__ d_pos, int stride, int tps, int splits, int rows,
                    float2* __restrict__ part_ml, float* __restrict__ part_o,
                    const std::int8_t* __restrict__ cache_k8, const std::int8_t* __restrict__ cache_v8,
                    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
                    const std::uint8_t* __restrict__ cache_k4, const std::uint8_t* __restrict__ cache_v4,
                    const int* __restrict__ d_pt, const int* __restrict__ row_pos,
                    const int* const* __restrict__ row_pt, const int* __restrict__ row_grp,
                    const int* __restrict__ row_tbase, const int* __restrict__ row_tail,
                    int group_rows) {
  const int hkv = blockIdx.x;
  const int split = blockIdx.y;
  // Row group: row_grp entry (r0 | nr << 8), else blocks of `group_rows` consecutive rows.
  const int gd = row_grp != nullptr ? row_grp[blockIdx.z] : 0;
  const int r0 = row_grp != nullptr ? (gd & 0xff) : blockIdx.z * group_rows;
  const int nr = row_grp != nullptr ? (gd >> 8) : min(group_rows, rows - r0);
  const int base = row_pos != nullptr ? row_pos[r0] - r0 : (d_pos != nullptr ? *d_pos : kv_base);
  auto row_len = [&](int rr) { return row_pos != nullptr ? row_pos[r0 + rr] + 1 : base + r0 + rr + 1; };
  int kv_max = 0;
  for (int rr = 0; rr < nr; ++rr) kv_max = max(kv_max, row_len(rr));
  const int t_begin = split * tps * kTile;
  if (t_begin >= kv_max) return;
  const int t_end_max = min(kv_max, t_begin + tps * kTile);

  __shared__ __align__(16) std::uint16_t Ks[kTile][kMmaP];
  __shared__ __align__(16) std::uint16_t Vs[kTile][kMmaP];
  __shared__ __align__(8) float ksc_s[kTile], vsc_s[kTile];  // quantized: per-position scales

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  // This compute warp's m16 tile = query rows 16 * warp .. 16 * warp + 15 = rows 4 * warp ..
  // 4 * warp + 3 of the group, 4 query heads each. Fragment rows g and g + 8.
  const bool compute = warp < 2 && 4 * warp < nr;
  int rrow[2], head[2], rlen[2];
  bool ok[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int qr = 16 * warp + g + 8 * h;  // 0..31
    rrow[h] = qr / kG;
    head[h] = hkv * kG + qr % kG;
    ok[h] = compute && rrow[h] < nr;
    rlen[h] = ok[h] ? row_len(rrow[h]) : 0;
  }
  std::uint32_t qf[kD / 16][4];
#pragma unroll
  for (int ks = 0; ks < kD / 16; ++ks) {
    const int kk = ks * 16 + 2 * t4;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const std::uint16_t* const qp = q + (static_cast<std::size_t>(r0 + rrow[h]) * Hq + head[h]) * kD;
      qf[ks][h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(qp + kk) : 0u;
      qf[ks][2 + h] = ok[h] ? *reinterpret_cast<const std::uint32_t*>(qp + kk + 8) : 0u;
    }
  }
  float o[kD / 8][4];
#pragma unroll
  for (int i = 0; i < kD / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

  const bool quantized = Int8 || Int4;
  const std::uint16_t* const kb = quantized ? nullptr : k + static_cast<std::size_t>(hkv) * stride * kD;
  const std::uint16_t* const vb = quantized ? nullptr : v + static_cast<std::size_t>(hkv) * stride * kD;
  const std::int8_t* const k8 = Int8 ? cache_k8 + static_cast<std::size_t>(hkv) * stride * kD : nullptr;
  const std::int8_t* const v8 = Int8 ? cache_v8 + static_cast<std::size_t>(hkv) * stride * kD : nullptr;
  const std::uint8_t* const k4 =
      Int4 ? cache_k4 + static_cast<std::size_t>(hkv) * stride * (K5 ? dev::kK5RowBytes : kD / 2) : nullptr;
  const std::uint8_t* const v4 = Int4 ? cache_v4 + static_cast<std::size_t>(hkv) * stride * (kD / 2) : nullptr;
  const float* const ksc = quantized ? k_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const float* const vsc = quantized ? v_scale + static_cast<std::size_t>(hkv) * stride : nullptr;
  const int* const pt = row_pt != nullptr ? row_pt[r0] : d_pt;

  // Attend the tile in Ks / Vs (positions t0 ..): S = Q K^T, mask, online softmax, O += P V.
  // keep >= 0: only group row `keep` is updated; every other row's logits are masked, so its
  // state is carried exactly (alpha = 1, p = 0: o * 1 + 0, l * 1 + 0), as for a row that has
  // no valid key in a tile.
  auto attend = [&](int t0, int keep) {
    int lim[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) lim[h] = keep < 0 || rrow[h] == keep ? rlen[h] : 0;
    float s[kTile / 8][4];
#pragma unroll
    for (int ni = 0; ni < kTile / 8; ++ni) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.0f;
#pragma unroll
    for (int ks = 0; ks < kD / 16; ++ks)
#pragma unroll
      for (int ni = 0; ni < kTile / 8; ++ni) {
        const std::uint16_t* const kr = &Ks[ni * 8 + g][ks * 16 + 2 * t4];
        mma_16816(s[ni], qf[ks], *reinterpret_cast<const std::uint32_t*>(kr),
                  *reinterpret_cast<const std::uint32_t*>(kr + 8));
      }
    float mx[2] = {m[0], m[1]};
#pragma unroll
    for (int ni = 0; ni < kTile / 8; ++ni)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int h = e >> 1;
        const int j = t0 + ni * 8 + 2 * t4 + (e & 1);
        // Quantized: S holds q . codes; the K scale makes it q . k (one f32 multiply).
        const float dot = Int8 || Int4 ? s[ni][e] * ksc_s[j - t0] : s[ni][e];
        const float val = j < lim[h] ? mma_bf16_round(scale * dot) : -INFINITY;
        s[ni][e] = val;
        mx[h] = fmaxf(mx[h], val);
      }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 1));
      mx[h] = fmaxf(mx[h], __shfl_xor_sync(0xffffffffu, mx[h], 2));
      // A row with no valid key yet keeps m = -inf: alpha = 1 and p = 0 (no NaN).
      alpha[h] = mx[h] == -INFINITY ? 1.0f : expf(m[h] - mx[h]);
      l[h] *= alpha[h];
      m[h] = mx[h];
    }
#pragma unroll
    for (int i = 0; i < kD / 8; ++i) {
      o[i][0] *= alpha[0];
      o[i][1] *= alpha[0];
      o[i][2] *= alpha[1];
      o[i][3] *= alpha[1];
    }
    std::uint32_t pf[kTile / 16][4];
#pragma unroll
    for (int ni = 0; ni < kTile / 8; ++ni) {
      const float e0 = m[0] == -INFINITY ? 0.0f : expf(s[ni][0] - m[0]);
      const float e1 = m[0] == -INFINITY ? 0.0f : expf(s[ni][1] - m[0]);
      const float e2 = m[1] == -INFINITY ? 0.0f : expf(s[ni][2] - m[1]);
      const float e3 = m[1] == -INFINITY ? 0.0f : expf(s[ni][3] - m[1]);
      l[0] += e0 + e1;
      l[1] += e2 + e3;
      // Quantized: V holds codes, so P carries the V scale (l keeps the unscaled sum).
      float sv0 = 1.0f, sv1 = 1.0f;
      if (Int8 || Int4) {
        const float2 sv = *reinterpret_cast<const float2*>(&vsc_s[ni * 8 + 2 * t4]);
        sv0 = sv.x;
        sv1 = sv.y;
      }
      pf[ni / 2][(ni & 1) * 2] = mma_pack(e0 * sv0, e1 * sv1);
      pf[ni / 2][(ni & 1) * 2 + 1] = mma_pack(e2 * sv0, e3 * sv1);
    }
    // B fragments of V via ldmatrix.trans: lanes 8j .. 8j + 7 address matrix j's rows
    // (keys kk * 16 + 8 * (j & 1) + r, dims 8 * (nd + (j >> 1))).
    const int mj = lane >> 3, mr = lane & 7;
#pragma unroll
    for (int kk = 0; kk < kTile / 16; ++kk)
#pragma unroll
      for (int nd = 0; nd < kD / 8; nd += 2) {
        std::uint32_t b0, b1, b2, b3;
        ldsm_x4_trans(b0, b1, b2, b3, &Vs[kk * 16 + 8 * (mj & 1) + mr][8 * (nd + (mj >> 1))]);
        mma_16816(o[nd], pf[kk], b0, b1);
        mma_16816(o[nd + 1], pf[kk], b2, b3);
      }
  };

  // Tree rows: a group shares tbase; a tile whose last position is past it holds row-specific
  // keys. Those are the split's last tiles, so tiles [t_begin, t_sh) are shared by every row.
  const int tb = Tree ? row_tbase[r0] : INT_MAX;
  int t_sh = t_end_max;
  if (Tree) {
    t_sh = t_begin;
    while (t_sh < t_end_max && min(t_sh + kTile, t_end_max) - 1 <= tb) t_sh += kTile;
  }
  if (t_sh > t_begin) {
    // The shared tiles' physical rows, staged once (one global round trip for the whole split
    // instead of a page-table load ahead of every tile's cache loads).
    __shared__ int Ps[kMmaPtMax];
    const bool staged = t_sh - t_begin <= kMmaPtMax;
    if (staged) {
      for (int i = threadIdx.x; i < t_sh - t_begin; i += kMmaThreads)
        Ps[i] = pt != nullptr ? pt[t_begin + i] : t_begin + i;
      __syncthreads();
    }
    // Software pipeline: tile t + 1's cache loads are issued before tile t is attended.
    MmaRaw raw[kMmaItems];
    bool rv[kMmaItems];
    auto load_tile = [&](int t0) {
#pragma unroll
      for (int i = 0; i < kMmaItems; ++i) {
        const int it = threadIdx.x + i * kMmaThreads;
        const int lp = t0 + (it >> 3);
        rv[i] = lp < t_sh;
        const int prow = !rv[i] ? 0 : staged ? Ps[lp - t_begin] : (pt != nullptr ? pt[lp] : lp);
        mma_load_pos<Int8, Int4, K5>(raw[i], prow, rv[i], it & 7, kb, vb, k8, v8, k4, v4, ksc, vsc);
      }
    };
    load_tile(t_begin);
    for (int t0 = t_begin; t0 < t_sh; t0 += kTile) {
      __syncthreads();  // the previous tile's readers of Ks / Vs are done
#pragma unroll
      for (int i = 0; i < kMmaItems; ++i) {
        const int it = threadIdx.x + i * kMmaThreads;
        mma_store_pos<Int8, Int4, K5>(Ks, Vs, ksc_s, vsc_s, it >> 3, it & 7, raw[i], rv[i]);
      }
      __syncthreads();
      if (t0 + kTile < t_sh) load_tile(t0 + kTile);
      if (compute) attend(t0, -1);
    }
  }
  if constexpr (Tree) {
    // Row-specific tiles: one pass per row with that row's own keys past tbase; the pass
    // updates only that row (attend's keep), so every row sees exactly its own sequential
    // tiles. Only the warp holding the row attends.
    for (int t0 = t_sh; t0 < t_end_max; t0 += kTile)
      for (int fr = 0; fr < nr; ++fr) {
        __syncthreads();  // the previous pass's readers of Ks / Vs are done
        for (int it = threadIdx.x; it < kTile * (kD / 16); it += kMmaThreads) {
          const int kr = it >> 3, part = it & 7;
          const int lp = t0 + kr;
          const bool tail = lp > tb;
          if (fr > 0 && !tail) continue;  // later passes rewrite only the tail positions
          int prow = 0;
          bool valid = lp < t_end_max;
          if (tail) {
            const int j = lp - tb - 1;
            valid = valid && j < kAttnTailStride && lp < row_len(fr);
            if (valid) {
              const int lt = row_tail[(r0 + fr) * kAttnTailStride + j];
              prow = pt != nullptr ? pt[lt] : lt;
            }
          } else if (valid) {
            prow = pt != nullptr ? pt[lp] : lp;
          }
          MmaRaw r;
          mma_load_pos<Int8, Int4, K5>(r, prow, valid, part, kb, vb, k8, v8, k4, v4, ksc, vsc);
          mma_store_pos<Int8, Int4, K5>(Ks, Vs, ksc_s, vsc_s, kr, part, r, valid);
        }
        __syncthreads();
        if (compute && (fr >> 2) == warp) attend(t0, fr);
      }
  }
  if (!compute) return;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
  }
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!ok[h] || t_begin >= rlen[h]) continue;
    const std::size_t slot = (static_cast<std::size_t>(r0 + rrow[h]) * Hq + head[h]) * splits + split;
    if (t4 == 0) part_ml[slot] = make_float2(m[h], l[h]);
#pragma unroll
    for (int nd = 0; nd < kD / 8; ++nd) {
      const int col = nd * 8 + 2 * t4;
      *reinterpret_cast<float2*>(part_o + slot * kD + col) = make_float2(o[nd][2 * h], o[nd][2 * h + 1]);
    }
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
               int ngroups, const int* row_tbase, const int* row_tail, int kv_hint, bool k5) {
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
      (k5 && !int4) ||
      (int8 && (cache_k8 == nullptr || cache_v8 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (int4 && (cache_k4 == nullptr || cache_v4 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (!int8 && !int4 && (cache_k == nullptr || cache_v == nullptr)))
    return -1;
  const int tps = tiles_per_split(ml);
  const int splits = ((ml + kTile - 1) / kTile + tps - 1) / tps;
  // Launch only the splits a row can reach (kv_hint: an upper bound on every
  // row's causal length, 0 = unknown). The split width and slot layout stay
  // those of `splits`, so the arithmetic is unchanged; blocks past the rows'
  // ends would only launch and exit.
  const int lsplits =
      kv_hint > 0 ? std::min(splits, (kv_hint + tps * kTile - 1) / (tps * kTile)) : splits;
  float2* const pml = static_cast<float2*>(scratch);
  float* const po = reinterpret_cast<float*>(pml + static_cast<std::size_t>(rows) * Hq * splits);
  // One block per (kv head, split, group of RB rows). Decode (rows == 1) uses
  // the single-row instance (smallest shared-memory footprint).
  static const bool pipe_on = [] {
    const char* e = std::getenv("G4_ATTN_PIPE");
    return e == nullptr || e[0] != '0';
  }();
  auto launch = [&](auto rbtag, const int* grp, int nz) {
    constexpr int RB = decltype(rbtag)::value;
    const dim3 grid(Hk, lsplits, nz);
    if (int8)
      attn_split_kernel<true, false, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          cache_k8, cache_v8, cache_k_scale, cache_v_scale, nullptr, nullptr, d_pt, row_pos, row_pt,
          grp, row_tbase, row_tail, pipe_on);
    else if (int4 && k5)
      attn_split_kernel<false, true, RB, true><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos, row_pt,
          grp, row_tbase, row_tail, pipe_on);
    else if (int4)
      attn_split_kernel<false, true, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos, row_pt,
          grp, row_tbase, row_tail, pipe_on);
    else
      attn_split_kernel<false, false, RB><<<grid, kThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_pt, row_pos, row_pt, grp,
          row_tbase, row_tail, pipe_on);
  };
  using One = std::integral_constant<int, 1>;
  using Many = std::integral_constant<int, kRowsPerBlock>;
  static const bool rows_par = [] {
    const char* e = std::getenv("G4_ATTN_ROWS");
    return e == nullptr || e[0] != '0';
  }();
  auto launch_rows = [&](const int* grp, int nz) {
    static const bool attr = [] {
      return cudaFuncSetAttribute(attn_rows_kernel<false, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kRowsSmem)) == cudaSuccess &&
             cudaFuncSetAttribute(attn_rows_kernel<true, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kRowsSmem)) == cudaSuccess &&
             cudaFuncSetAttribute(attn_rows_kernel<false, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kRowsSmem)) == cudaSuccess &&
             cudaFuncSetAttribute(attn_rows_kernel<false, true, true>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(kRowsSmem)) == cudaSuccess;
    }();
    if (!attr) return false;
    const dim3 grid(Hk, lsplits, nz);
    if (int8)
      attn_rows_kernel<true, false><<<grid, kRowsWarps * 32, kRowsSmem>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          cache_k8, cache_v8, cache_k_scale, cache_v_scale, nullptr, nullptr, d_pt, row_pos,
          row_pt, grp, row_tbase, row_tail);
    else if (int4 && k5)
      attn_rows_kernel<false, true, true><<<grid, kRowsWarps * 32, kRowsSmem>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos,
          row_pt, grp, row_tbase, row_tail);
    else if (int4)
      attn_rows_kernel<false, true><<<grid, kRowsWarps * 32, kRowsSmem>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos,
          row_pt, grp, row_tbase, row_tail);
    else
      attn_rows_kernel<false, false><<<grid, kRowsWarps * 32, kRowsSmem>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_pt, row_pos, row_pt, grp,
          row_tbase, row_tail);
    return true;
  };
  // Short contexts (a few tiles per row) have too few blocks to fill the
  // card when rows share a block, and the KV re-read is cheap there, so they
  // keep one row per block; past ~1K positions sharing the tiles wins.
  const bool short_ctx = d_pos == nullptr && kv_base + rows <= kShortCtx;
  // Rows of different sequences never share a tile: one row per block, or
  // one same-sequence row group per block when the caller passes groups.
  // G4_ATTN_ROWS=0: the row-serial RB=8 blocks instead of rows-parallel ones.
  // G4_ATTN_MMA=0: the scalar kernels below instead of the tensor-core kernel. One kernel
  // serves every case, so decode and verify stay bit-identical under either setting.
  static const bool mma_on = [] {
    const char* e = std::getenv("G4_ATTN_MMA");
    return e == nullptr || e[0] != '0';
  }();
  if (mma_on) {
    // Per-row positions without groups, or a short context (too few blocks to fill the card
    // otherwise): one row per block; else up to kMmaRows rows share the tiles.
    const int group_rows = row_pos != nullptr || short_ctx ? 1 : kMmaRows;
    const int nz = row_grp != nullptr ? ngroups : (rows + group_rows - 1) / group_rows;
    const dim3 grid(Hk, lsplits, nz);
    auto launch_mma = [&](auto tree_tag) {
    constexpr bool Tree = decltype(tree_tag)::value;
    if (int8)
      attn_mma_kernel<true, false, false, Tree><<<grid, kMmaThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          cache_k8, cache_v8, cache_k_scale, cache_v_scale, nullptr, nullptr, d_pt, row_pos,
          row_pt, row_grp, row_tbase, row_tail, group_rows);
    else if (int4 && k5)
      attn_mma_kernel<false, true, true, Tree><<<grid, kMmaThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos,
          row_pt, row_grp, row_tbase, row_tail, group_rows);
    else if (int4)
      attn_mma_kernel<false, true, false, Tree><<<grid, kMmaThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt, row_pos,
          row_pt, row_grp, row_tbase, row_tail, group_rows);
    else
      attn_mma_kernel<false, false, false, Tree><<<grid, kMmaThreads>>>(
          q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, rows, pml, po,
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_pt, row_pos, row_pt, row_grp,
          row_tbase, row_tail, group_rows);
    };
    if (row_tail != nullptr)
      launch_mma(std::true_type{});
    else
      launch_mma(std::false_type{});
  } else if (row_grp != nullptr) {
    if (!(rows_par && launch_rows(row_grp, ngroups))) launch(Many{}, row_grp, ngroups);
  } else if (rows == 1 || short_ctx || row_pos != nullptr) {
    launch(One{}, nullptr, rows);
  } else {
    const int nz = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
    if (!(rows_par && launch_rows(nullptr, nz))) launch(Many{}, nullptr, nz);
  }
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
