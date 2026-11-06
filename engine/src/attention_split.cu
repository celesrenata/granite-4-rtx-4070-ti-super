// Split-KV attention for decode and speculative verify (flash-decoding).
//
// The fused GQA kernel (kernels_phase2.cu) runs one block per query head and
// accumulates the value sum serially over the whole prefix: 32 blocks on a
// 66-SM card, and a dependent chain of kv_len FMAs per output. That is cheap
// at a few hundred positions and dominant at a few thousand (190 us/layer at
// 2 K context). Here the prefix is cut into 64-position tiles:
//
//   attn_split_kernel    grid (Hk, splits, rows), 128 threads. One block owns
//                        one KV head, its G = 4 query heads (one warp each)
//                        and a contiguous run of tiles. K/V tiles are staged
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
#include <cmath>
#include <cstdint>
#include <cstdio>

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

int tiles_per_split(int stride) {
  const int tiles = (stride + kTile - 1) / kTile;
  return (tiles + kMaxSplits - 1) / kMaxSplits;
}

// Int8 (Q8_0): the persistent cache is int8 codes + a per-(head,position)
// float scale. The tile fill dequantizes each row into the same bf16 Ks/Vs
// tiles the bf16 path writes, so everything after the tile fill (logits,
// online softmax, PV) is byte-identical between the two instantiations.
template <bool Int8, bool Int4>
__global__ void __launch_bounds__(kThreads)
    attn_split_kernel(const std::uint16_t* __restrict__ q,
                      const std::uint16_t* __restrict__ k,
                      const std::uint16_t* __restrict__ v, float scale, int Hq,
                      int kv_base, const int* __restrict__ d_pos, int stride, int tps,
                      int splits, float2* __restrict__ part_ml,
                      float* __restrict__ part_o,
                      const std::int8_t* __restrict__ cache_k8,
                      const std::int8_t* __restrict__ cache_v8,
                      const float* __restrict__ k_scale,
                      const float* __restrict__ v_scale,
                      const std::uint8_t* __restrict__ cache_k4,
                      const std::uint8_t* __restrict__ cache_v4,
                      const int* __restrict__ d_pt) {
  const int hkv = blockIdx.x;
  const int split = blockIdx.y;
  const int row = blockIdx.z;
  const int kv_len = (d_pos != nullptr ? *d_pos : kv_base) + row + 1;
  const int t_begin = split * tps * kTile;
  if (t_begin >= kv_len) return;
  const int t_end = min(kv_len, t_begin + tps * kTile);

  __shared__ std::uint32_t Ks[kTile][kPitchWords];
  __shared__ __align__(16) std::uint16_t Vs[kTile][kD];
  __shared__ float qs[kG][kD];
  __shared__ float ps[kG][kTile];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = hkv * kG + warp;
  const std::uint16_t* const qrow = q + (static_cast<std::size_t>(row) * Hq + h) * kD;
  for (int i = lane; i < kD; i += 32) qs[warp][i] = bf16_to_f32(qrow[i]);

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
  float m = -INFINITY, l = 0.0f;
  float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // dims 4 * lane ... 4 * lane + 3

  // Logical position -> physical arena row. The table is the identity
  // (row == position) when d_pt is null, i.e. a contiguous per-head KV
  // layout; a paged arena (shared-prefix pages that are non-contiguous)
  // passes d_pt so each token's row is gathered from the table.
  const int* const pt = d_pt;
  for (int t0 = t_begin; t0 < t_end; t0 += kTile) {
    const int n = min(kTile, t_end - t0);
    __syncthreads();  // previous tile's readers (and the q preload) are done
    if (!quantized) {
      for (int idx = threadIdx.x; idx < n * (kD / 8); idx += kThreads) {
        const int r = idx / (kD / 8), c = (idx % (kD / 8)) * 8;
        const int row = (pt != nullptr) ? pt[t0 + r] : t0 + r;
        const std::size_t off = static_cast<std::size_t>(row) * kD + c;
        const uint4 k4 = *reinterpret_cast<const uint4*>(kb + off);
        std::uint32_t* const kd = &Ks[r][c / 2];
        kd[0] = k4.x;
        kd[1] = k4.y;
        kd[2] = k4.z;
        kd[3] = k4.w;
        *reinterpret_cast<uint4*>(&Vs[r][c]) = *reinterpret_cast<const uint4*>(vb + off);
      }
    } else if (Int4) {
      // KV4: one thread per (row, code-byte) - read the two packed int4 codes,
      // dequantize with the row's scale, and store the packed bf16 word (Ks)
      // and the two bf16 values (Vs) into the same tiles.
      for (int idx = threadIdx.x; idx < n * (kD / 2); idx += kThreads) {
        const int r = idx / (kD / 2), w = idx % (kD / 2);
        const int pos = (pt != nullptr) ? pt[t0 + r] : t0 + r;
        const std::size_t cbase = static_cast<std::size_t>(pos) * (kD / 2) + w;
        const float sk = ksc[pos], sv = vsc[pos];
        float klo, khi, vlo, vhi;
        dev::q4_dequant_pair(k4[cbase], sk, klo, khi);
        dev::q4_dequant_pair(v4[cbase], sv, vlo, vhi);
        Ks[r][w] = static_cast<std::uint32_t>(f32_to_bf16(klo)) |
                   (static_cast<std::uint32_t>(f32_to_bf16(khi)) << 16);
        Vs[r][2 * w] = f32_to_bf16(vlo);
        Vs[r][2 * w + 1] = f32_to_bf16(vhi);
      }
    } else {
      // KV8: one thread per (row, dim-pair) - read the two int8 codes,
      // dequantize with the row's scale, and store the packed bf16 word (Ks)
      // and the two bf16 values (Vs) into the same tiles.
      for (int idx = threadIdx.x; idx < n * (kD / 2); idx += kThreads) {
        const int r = idx / (kD / 2), w = idx % (kD / 2);
        const int pos = (pt != nullptr) ? pt[t0 + r] : t0 + r;
        const std::size_t base = static_cast<std::size_t>(pos) * kD + 2 * w;
        const float sk = ksc[pos], sv = vsc[pos];
        const std::uint16_t klo =
            f32_to_bf16(dev::kv8_dequant(k8[base], sk));
        const std::uint16_t khi =
            f32_to_bf16(dev::kv8_dequant(k8[base + 1], sk));
        Ks[r][w] = static_cast<std::uint32_t>(klo) | (static_cast<std::uint32_t>(khi) << 16);
        Vs[r][2 * w] = f32_to_bf16(dev::kv8_dequant(v8[base], sv));
        Vs[r][2 * w + 1] = f32_to_bf16(dev::kv8_dequant(v8[base + 1], sv));
      }
    }
    __syncthreads();

    // Logits for kv = lane and lane + 32 of this tile, bf16-rounded.
    float s[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int kv = lane + 32 * j;
      if (kv < n) {
        float acc = 0.0f;
        const std::uint32_t* const kr = &Ks[kv][0];
#pragma unroll 8
        for (int i = 0; i < kD / 2; ++i) {
          const std::uint32_t w = kr[i];
          acc += dev::bf16_lo(w) * qs[warp][2 * i];
          acc += dev::bf16_hi(w) * qs[warp][2 * i + 1];
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
    ps[warp][lane] = p0;
    ps[warp][lane + 32] = p1;
    __syncwarp();
#pragma unroll
    for (int c = 0; c < 4; ++c) o[c] *= alpha;
    for (int kv = 0; kv < n; ++kv) {
      const float p = ps[warp][kv];
      const uint2 w = *reinterpret_cast<const uint2*>(&Vs[kv][4 * lane]);
      o[0] += p * dev::bf16_lo(w.x);
      o[1] += p * dev::bf16_hi(w.x);
      o[2] += p * dev::bf16_lo(w.y);
      o[3] += p * dev::bf16_hi(w.y);
    }
    __syncwarp();
  }

  const std::size_t slot = (static_cast<std::size_t>(row) * Hq + h) * splits + split;
  if (lane == 0) part_ml[slot] = make_float2(m, l);
  *reinterpret_cast<float4*>(part_o + slot * kD + 4 * lane) = make_float4(o[0], o[1], o[2], o[3]);
}

__global__ void __launch_bounds__(kD)
    attn_combine_kernel(const float2* __restrict__ part_ml, const float* __restrict__ part_o,
                        int Hq, int kv_base, const int* __restrict__ d_pos, int tps,
                        int splits, std::uint16_t* __restrict__ out) {
  const int h = blockIdx.x;
  const int row = blockIdx.y;
  const int kv_len = (d_pos != nullptr ? *d_pos : kv_base) + row + 1;
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
               const std::uint8_t* cache_v4, bool int4, const int* d_pt) {
  if (q == nullptr || scratch == nullptr || out == nullptr ||
      !std::isfinite(scale) || d != kD || Hk <= 0 || Hq != kG * Hk ||
      rows <= 0 || rows > kMaxVerifyRows || stride <= 0 ||
      (d_pos == nullptr && kv_base + rows > stride) ||
      (int8 && int4) ||
      (int8 && (cache_k8 == nullptr || cache_v8 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (int4 && (cache_k4 == nullptr || cache_v4 == nullptr ||
                cache_k_scale == nullptr || cache_v_scale == nullptr)) ||
      (!int8 && !int4 && (cache_k == nullptr || cache_v == nullptr)))
    return -1;
  const int tps = tiles_per_split(stride);
  const int splits = ((stride + kTile - 1) / kTile + tps - 1) / tps;
  float2* const ml = static_cast<float2*>(scratch);
  float* const po = reinterpret_cast<float*>(ml + static_cast<std::size_t>(rows) * Hq * splits);
  if (int8)
    attn_split_kernel<true, false><<<dim3(Hk, splits, rows), kThreads>>>(
        q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, ml, po,
        cache_k8, cache_v8, cache_k_scale, cache_v_scale, nullptr, nullptr, d_pt);
  else if (int4)
    attn_split_kernel<false, true><<<dim3(Hk, splits, rows), kThreads>>>(
        q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, ml, po,
        nullptr, nullptr, cache_k_scale, cache_v_scale, cache_k4, cache_v4, d_pt);
  else
    attn_split_kernel<false, false><<<dim3(Hk, splits, rows), kThreads>>>(
        q, cache_k, cache_v, scale, Hq, kv_base, d_pos, stride, tps, splits, ml, po,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, d_pt);
  attn_combine_kernel<<<dim3(Hq, rows), kD>>>(ml, po, Hq, kv_base, d_pos, tps, splits, out);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "granite4: attn_split: %s\n", cudaGetErrorString(e));
    return -1;
  }
  return 0;
}

}  // namespace granite4
