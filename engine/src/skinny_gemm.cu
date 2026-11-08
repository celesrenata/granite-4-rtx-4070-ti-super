// Tensor-core skinny GEMM for 1..16 activation rows (decode, batched verify,
// multi-session decode, the EAGLE head): C[t][n] = A[t][:] . W[n][:], A bf16
// [tokens][K], W FP8 e4m3 [N][K] row-major (per-tensor scale).
//
// Why: the per-row GEMVs (verify_gemv.cu) run one block per output row and
// every block re-reads all T activation rows, so their cost grows with T
// (a 14-row verify step costs ~2.3 decode steps). Here a block owns 64 output
// rows (8 warps x one n8 mma tile), stages each 256-wide K chunk of A in
// shared memory once for all 64 rows, and streams its weights exactly once
// with coalesced 16-byte loads; the tensor cores make 16 rows cost about what
// one row costs, so a step stays bound by the weight stream.
//
// mma.m16n8k16 (bf16 x bf16 -> f32). FP8 weights convert exactly to bf16.
// Each lane loads 16 contiguous K bytes of one weight row (n = lane / 4,
// k = 16 * (lane % 4) within a 64-wide K block) and feeds them to four mma
// steps; K is permuted consistently in A and B so those bytes land directly in
// the B fragment slots: in step s, lane c's slots {2c, 2c+1, 2c+8, 2c+9} are
// k = 16c + 4s + {0, 1, 2, 3}. The dot product is unchanged (every k appears
// once in both operands), only the hardware summation order is fixed by this
// layout.
//
// Determinism: row t of the output depends only on row t of A, and the K
// order is fixed by the kernel, never by T. So a T-row call is bit-identical,
// row for row, to T single-row calls - which is what keeps the batched verify
// exact against decode when decode runs this kernel too.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "device_math.cuh"
#include "model_config.h"

namespace {
using granite4::dev::bf16_to_f32;
using granite4::dev::f32_to_bf16;
using granite4::dev::fp8x4_to_f32;

constexpr int kRowsPad = 16;   // mma M
constexpr int kKc = 256;       // K chunk (4 x 64-wide K blocks)
constexpr int kApitch = kKc + 8;  // bf16 elements; +8 breaks bank conflicts
constexpr int kMaxWarps = 8;

enum Mode { kOut = 0, kResid = 1, kSilu = 2, kQkv = 3 };

struct Params {
  const std::uint16_t* a;
  int lda, tokens, n, k;
  const std::uint8_t* w0;  // weights (gate for kSilu, q for kQkv)
  const std::uint8_t* w1;  // up (kSilu) / k (kQkv)
  const std::uint8_t* w2;  // v (kQkv)
  float s0, s1, s2;
  int n1, n2;              // kQkv: rows of k and v (n = rows of q)
  std::uint16_t* out0;     // out (q for kQkv)
  std::uint16_t* out1;     // k (kQkv)
  std::uint16_t* out2;     // v (kQkv)
  const std::uint16_t* residual;
  float rs;
  int ldo;                 // output row pitch (kOut / kResid / kSilu)
};

// Two packed bf16 from two fp8 bytes (exact: e4m3 is a subset of bf16).
__device__ __forceinline__ std::uint32_t fp8x2_to_bf16x2(std::uint32_t w4, int pair) {
  float f0, f1, f2, f3;
  fp8x4_to_f32(w4, f0, f1, f2, f3);
  const float lo = pair == 0 ? f0 : f2;
  const float hi = pair == 0 ? f1 : f3;
  return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const std::uint32_t (&a)[4],
                                         std::uint32_t b0, std::uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// One 64-wide K block (16 fp8 of this lane's weight row held as one uint4)
// against this warp's staged A rows (row pitch kApitch) at local offset kb.
// Rows >= tokens are not staged; their fragments are zero.
__device__ __forceinline__ void kblock(const std::uint16_t* As, int tokens, int kb, int g, int c,
                                       const uint4& wv, float (&acc)[4]) {
  const std::uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
  const bool r0 = g < tokens, r1 = g + 8 < tokens;
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int kk = kb + 16 * c + 4 * s;
    uint2 x0 = make_uint2(0u, 0u), x1 = make_uint2(0u, 0u);
    if (r0) x0 = *reinterpret_cast<const uint2*>(As + g * kApitch + kk);
    if (r1) x1 = *reinterpret_cast<const uint2*>(As + (g + 8) * kApitch + kk);
    const std::uint32_t a[4] = {x0.x, x1.x, x0.y, x1.y};
    mma_bf16(acc, a, fp8x2_to_bf16x2(ww[s], 0), fp8x2_to_bf16x2(ww[s], 1));
  }
}

// Stage this warp's A rows [0, tokens) x [k0, k0 + kKc) (cp.async, 16 B each).
__device__ __forceinline__ void stage_warp(std::uint16_t* As, const Params& p, int k0, int lane) {
  for (int i = lane; i < p.tokens * (kKc / 8); i += 32) {
    const int t = i / (kKc / 8), j = (i % (kKc / 8)) * 8;
    cp_async16(As + t * kApitch + j, p.a + static_cast<std::size_t>(t) * p.lda + k0 + j);
  }
  cp_async_commit();
}

template <bool kTwo>
__device__ __forceinline__ void load_w(uint4 (&w)[4], uint4 (&u)[4], const std::uint8_t* wr,
                                       const std::uint8_t* ur, int k0, int c) {
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    w[b] = __ldg(reinterpret_cast<const uint4*>(wr + k0 + 64 * b + 16 * c));
    if (kTwo) u[b] = __ldg(reinterpret_cast<const uint4*>(ur + k0 + 64 * b + 16 * c));
  }
}

// TILES n8 tiles per block, each split over SPLIT warps along K (warp w:
// tile w / SPLIT, K slice w % SPLIT). Warps are independent until the final
// split-K reduction (fixed order). Dynamic shared memory: per warp two A
// buffers of `tokens` rows, then the reduction area.
template <int MODE, int TILES, int SPLIT>
__global__ void __launch_bounds__(TILES * SPLIT * 32) skinny_kernel(Params p) {
  constexpr bool kTwo = MODE == kSilu;
  constexpr int kNt = 8 * TILES;
  extern __shared__ __align__(16) std::uint8_t smem[];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int tile = warp / SPLIT, ks = warp % SPLIT;
  const int g = lane >> 2, c = lane & 3;
  int n0 = blockIdx.x * kNt;
  const std::uint8_t* w = p.w0;
  float scale = p.s0;
  std::uint16_t* out = p.out0;
  int nrows = p.n, ldo = p.ldo;
  if (MODE == kQkv) {
    ldo = p.n;
    if (n0 >= p.n + p.n1) {
      n0 -= p.n + p.n1;
      w = p.w2, scale = p.s2, out = p.out2, nrows = p.n2, ldo = p.n2;
    } else if (n0 >= p.n) {
      n0 -= p.n;
      w = p.w1, scale = p.s1, out = p.out1, nrows = p.n1, ldo = p.n1;
    }
  }
  const int nw = n0 + 8 * tile;
  const int row = min(nw + g, nrows - 1);
  const bool live = nw < nrows;
  const std::uint8_t* const wr = w + static_cast<std::size_t>(row) * p.k;
  const std::uint8_t* const ur = kTwo ? p.w1 + static_cast<std::size_t>(row) * p.k : nullptr;
  const std::size_t abuf = static_cast<std::size_t>(p.tokens) * kApitch;  // elements
  std::uint16_t* const A0 = reinterpret_cast<std::uint16_t*>(smem) + warp * 2 * abuf;
  std::uint16_t* const A1 = A0 + abuf;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, accu[4] = {0.f, 0.f, 0.f, 0.f};
  // This warp's chunks: [kc0, kc1) of the K / kKc chunks.
  const int nch = p.k / kKc;
  const int kc0 = ks * nch / SPLIT, kc1 = (ks + 1) * nch / SPLIT;
  if (live && kc0 < kc1) {
    uint4 wc[4], uc[4], wn[4], un[4];
    stage_warp(A0, p, kc0 * kKc, lane);
    load_w<kTwo>(wc, uc, wr, ur, kc0 * kKc, c);
    for (int ci = kc0; ci < kc1; ++ci) {
      std::uint16_t* const cur = ((ci - kc0) & 1) ? A1 : A0;
      std::uint16_t* const nxt = ((ci - kc0) & 1) ? A0 : A1;
      const bool more = ci + 1 < kc1;
      if (more) {
        stage_warp(nxt, p, (ci + 1) * kKc, lane);
        load_w<kTwo>(wn, un, wr, ur, (ci + 1) * kKc, c);
        cp_async_wait<1>();
      } else {
        cp_async_wait<0>();
      }
      __syncwarp();  // this warp's chunk ci is in shared memory
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        kblock(cur, p.tokens, 64 * b, g, c, wc[b], acc);
        if (kTwo) kblock(cur, p.tokens, 64 * b, g, c, uc[b], accu);
      }
      __syncwarp();  // done reading cur before it is refilled
      if (more) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          wc[b] = wn[b];
          if (kTwo) uc[b] = un[b];
        }
      }
    }
  }
  // Split-K reduction in fixed order (slice 0, 1, ...): deterministic and
  // independent of tokens.
  if (SPLIT > 1) {
    float* const red = reinterpret_cast<float*>(smem + 2 * abuf * sizeof(std::uint16_t) *
                                                           TILES * SPLIT);
    const int nacc = kTwo ? 8 : 4;
    float* const mine = red + ((tile * SPLIT + ks) * 32 + lane) * nacc;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      mine[e] = acc[e];
      if (kTwo) mine[4 + e] = accu[e];
    }
    __syncthreads();
    if (ks != 0) return;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float sa = 0.f, su = 0.f;
      for (int j = 0; j < SPLIT; ++j) {
        const float* const o = red + ((tile * SPLIT + j) * 32 + lane) * nacc;
        sa += o[e];
        if (kTwo) su += o[4 + e];
      }
      acc[e] = sa;
      if (kTwo) accu[e] = su;
    }
  }
  if (!live) return;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int t = g + (e >> 1) * 8;
    const int n = nw + 2 * c + (e & 1);
    if (t >= p.tokens || n >= nrows) continue;
    const std::size_t o = static_cast<std::size_t>(t) * ldo + n;
    if (MODE == kOut || MODE == kQkv) {
      out[o] = f32_to_bf16(scale * acc[e]);
    } else if (MODE == kResid) {
      const float y = scale * acc[e];
      out[o] = f32_to_bf16(bf16_to_f32(p.residual[o]) + p.rs * y);
    } else {  // kSilu
      const float gs = p.s0 * acc[e];
      const float sig = 1.0f / (1.0f + std::exp(-gs));
      out[o] = f32_to_bf16(gs * sig * (p.s1 * accu[e]));
    }
  }
}


// Block-shared variant for wider batches: WARPS warps (one n8 tile each, full
// K) share one staged copy of A per chunk, so A traffic does not grow with
// the warps. Each row's arithmetic is EXACTLY skinny_kernel<.., 1>'s (same
// chunk / K-block / mma-step order, padding rows are zeros either way), so
// dispatching between the two by token count keeps every row bit-identical.
template <int MODE, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) skinny_shared_kernel(Params p) {
  constexpr bool kTwo = MODE == kSilu;
  constexpr int kNt = 8 * WARPS;
  __shared__ __align__(16) std::uint16_t As[2][kRowsPad][kApitch];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, c = lane & 3;
  int n0 = blockIdx.x * kNt;
  const std::uint8_t* w = p.w0;
  float scale = p.s0;
  std::uint16_t* out = p.out0;
  int nrows = p.n, ldo = p.ldo;
  if (MODE == kQkv) {
    ldo = p.n;
    if (n0 >= p.n + p.n1) {
      n0 -= p.n + p.n1;
      w = p.w2, scale = p.s2, out = p.out2, nrows = p.n2, ldo = p.n2;
    } else if (n0 >= p.n) {
      n0 -= p.n;
      w = p.w1, scale = p.s1, out = p.out1, nrows = p.n1, ldo = p.n1;
    }
  }
  const int nw = n0 + 8 * warp;
  const int row = min(nw + g, nrows - 1);
  const bool live = nw < nrows;
  const std::uint8_t* const wr = w + static_cast<std::size_t>(row) * p.k;
  const std::uint8_t* const ur = kTwo ? p.w1 + static_cast<std::size_t>(row) * p.k : nullptr;
  float acc[4] = {0.f, 0.f, 0.f, 0.f}, accu[4] = {0.f, 0.f, 0.f, 0.f};
  const int nch = p.k / kKc;
  uint4 wc[4], uc[4], wn[4], un[4];
  auto stage = [&](int buf, int k0) {
    for (int i = threadIdx.x; i < p.tokens * (kKc / 8); i += WARPS * 32) {
      const int t = i / (kKc / 8), j = (i % (kKc / 8)) * 8;
      cp_async16(&As[buf][t][j], p.a + static_cast<std::size_t>(t) * p.lda + k0 + j);
    }
    cp_async_commit();
  };
  stage(0, 0);
  load_w<kTwo>(wc, uc, wr, ur, 0, c);
  for (int ci = 0; ci < nch; ++ci) {
    const bool more = ci + 1 < nch;
    if (more) {
      stage((ci + 1) & 1, (ci + 1) * kKc);
      load_w<kTwo>(wn, un, wr, ur, (ci + 1) * kKc, c);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();  // chunk ci staged by every thread
    if (live) {
      const std::uint16_t* const A = &As[ci & 1][0][0];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        kblock(A, p.tokens, 64 * b, g, c, wc[b], acc);
        if (kTwo) kblock(A, p.tokens, 64 * b, g, c, uc[b], accu);
      }
    }
    __syncthreads();  // done reading buffer ci & 1 before it is refilled
    if (more) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        wc[b] = wn[b];
        if (kTwo) uc[b] = un[b];
      }
    }
  }
  if (!live) return;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const int t = g + (e >> 1) * 8;
    const int n = nw + 2 * c + (e & 1);
    if (t >= p.tokens || n >= nrows) continue;
    const std::size_t o = static_cast<std::size_t>(t) * ldo + n;
    if (MODE == kOut || MODE == kQkv) {
      out[o] = f32_to_bf16(scale * acc[e]);
    } else if (MODE == kResid) {
      const float y = scale * acc[e];
      out[o] = f32_to_bf16(bf16_to_f32(p.residual[o]) + p.rs * y);
    } else {  // kSilu
      const float gs = p.s0 * acc[e];
      const float sig = 1.0f / (1.0f + std::exp(-gs));
      out[o] = f32_to_bf16(gs * sig * (p.s1 * accu[e]));
    }
  }
}

bool ok_shape(const void* a, int lda, const void* w, int tokens, int k) {
  return tokens >= 1 && tokens <= kRowsPad && k > 0 && (k % kKc) == 0 && (lda % 8) == 0 &&
         (reinterpret_cast<std::uintptr_t>(a) & 15u) == 0 &&
         (reinterpret_cast<std::uintptr_t>(w) & 15u) == 0;
}
template <int MODE, int TILES, int SPLIT>
int go(const Params& p, int total_rows) {
  const std::size_t smem = 2u * p.tokens * kApitch * sizeof(std::uint16_t) * TILES * SPLIT +
                           (SPLIT > 1 ? std::size_t{TILES} * SPLIT * 32 * 8 * sizeof(float) : 0);
  static const bool attr = [] {
    return cudaFuncSetAttribute(skinny_kernel<MODE, TILES, SPLIT>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                96 * 1024) == cudaSuccess;
  }();
  if (!attr) return -1;
  skinny_kernel<MODE, TILES, SPLIT>
      <<<(total_rows + 8 * TILES - 1) / (8 * TILES), TILES * SPLIT * 32, smem>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
// Up to kPerWarpMaxT tokens each warp stages its own A (independent warps,
// best at decode widths); wider batches share one staged A per block (flat
// cost in the token count). Both sum every row in the same order.
constexpr int kPerWarpMaxT = 4;
constexpr int kSharedWarps = 4;
template <int MODE>
int launch(const Params& p, int total_rows) {
  if (p.tokens <= kPerWarpMaxT) return go<MODE, 1, 1>(p, total_rows);
  skinny_shared_kernel<MODE, kSharedWarps>
      <<<(total_rows + 8 * kSharedWarps - 1) / (8 * kSharedWarps), kSharedWarps * 32>>>(p);
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
}  // namespace

// out[t][n] = bf16(scale * a[t] . W[n])
extern "C" int granite4_skinny_gemv(const std::uint16_t* a, const std::uint8_t* w, float scale,
                                    int tokens, int n, int k, std::uint16_t* out) {
  if (!ok_shape(a, k, w, tokens, k) || n < 1) return -1;
  Params p{};
  p.a = a, p.lda = k, p.tokens = tokens, p.n = n, p.k = k, p.w0 = w, p.s0 = scale, p.out0 = out,
  p.ldo = n;
  return launch<kOut>(p, n);
}
// out[t][n] = bf16(residual[t][n] + rs * scale * a[t] . W[n]); out may alias residual.
extern "C" int granite4_skinny_gemv_resid(const std::uint16_t* a, const std::uint8_t* w,
                                          float scale, const std::uint16_t* residual, float rs,
                                          int tokens, int n, int k, std::uint16_t* out) {
  if (!ok_shape(a, k, w, tokens, k) || n < 1) return -1;
  Params p{};
  p.a = a, p.lda = k, p.tokens = tokens, p.n = n, p.k = k, p.w0 = w, p.s0 = scale, p.out0 = out,
  p.residual = residual, p.rs = rs, p.ldo = n;
  return launch<kResid>(p, n);
}
extern "C" int granite4_skinny_qkv(const std::uint16_t* a, const std::uint8_t* wq,
                                   const std::uint8_t* wk, const std::uint8_t* wv, float sq,
                                   float sk, float sv, int tokens, int nq, int nk, int nv, int k,
                                   std::uint16_t* oq, std::uint16_t* ok, std::uint16_t* ov) {
  if (!ok_shape(a, k, wq, tokens, k) || (reinterpret_cast<std::uintptr_t>(wk) & 15u) != 0 ||
      (reinterpret_cast<std::uintptr_t>(wv) & 15u) != 0 || nq % 64 != 0 || nk % 64 != 0 ||
      nv < 1)
    return -1;
  Params p{};
  p.a = a, p.lda = k, p.tokens = tokens, p.n = nq, p.k = k, p.w0 = wq, p.w1 = wk, p.w2 = wv,
  p.s0 = sq, p.s1 = sk, p.s2 = sv, p.n1 = nk, p.n2 = nv, p.out0 = oq, p.out1 = ok, p.out2 = ov;
  return launch<kQkv>(p, nq + nk + nv);
}
// out[t][n] = silu(sg * Wg[n] . x[t]) * (su * Wu[n] . x[t])
extern "C" int granite4_skinny_silu(const std::uint8_t* wg, const std::uint8_t* wu,
                                    const std::uint16_t* x, float sg, float su, int n, int k,
                                    int tokens, std::uint16_t* out) {
  if (!ok_shape(x, k, wg, tokens, k) || (reinterpret_cast<std::uintptr_t>(wu) & 15u) != 0 ||
      n < 1)
    return -1;
  Params p{};
  p.a = x, p.lda = k, p.tokens = tokens, p.n = n, p.k = k, p.w0 = wg, p.w1 = wu, p.s0 = sg,
  p.s1 = su, p.out0 = out, p.ldo = n;
  return launch<kSilu>(p, n);
}
