// Split-KV attention micro-benchmark at the production serve shapes (Granite 4.2 8B:
// Hq 32, Hk 8, d 128) on a shared arena (rows = the KV pool, max_len = the per-session
// context bound), for every KV format. It times one attention layer and reports:
//   - ms per call (one layer); x40 gives the per-step attention cost
//   - "floor ms": the time to read the K/V bytes the call needs ONCE per row group at the
//     card's measured read bandwidth (--gbps, default 632 from kernels_bench)
//   - efficiency = floor / measured
// Cases: decode (1 row, d_pos) and a batched verify of R rows at consecutive positions,
// grouped like forward_multi_step (groups of up to 8 rows of one sequence) or ungrouped.
//
//   granite4-attn-bench [--pool rows] [--maxlen n] [--rows R] [--iters n] [--gbps x]
//                       [--ctx a,b,c] [--fmt bf16,kv8,kv4,k5v4]
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "attention_gpu.h"

namespace {
constexpr int kHq = 32, kHk = 8, kD = 128;
#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, \
                   __LINE__);                                                         \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

__global__ void fill_bytes(std::uint8_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 2654435761u ^ seed;
    x ^= x >> 13;
    x *= 0x5bd1e995u;
    p[i] = static_cast<std::uint8_t>(x >> 24);
  }
}
// bf16 values in roughly [-2, 2): sign | exponent 126..128 | random mantissa.
__global__ void fill_bf16(std::uint16_t* p, std::size_t n, std::uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    std::uint32_t x = static_cast<std::uint32_t>(i) * 2246822519u ^ seed;
    x ^= x >> 15;
    x *= 0x2c1b3c6du;
    const std::uint16_t sign = static_cast<std::uint16_t>((x >> 31) << 15);
    const std::uint16_t expo = static_cast<std::uint16_t>((126 + (x >> 29) % 3) << 7);
    p[i] = static_cast<std::uint16_t>(sign | expo | ((x >> 8) & 0x7f));
  }
}
__global__ void fill_scales(float* p, std::size_t n) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    p[i] = 0.01f + 0.0001f * static_cast<float>(i % 97);
}

std::vector<int> parse_list(const char* s) {
  std::vector<int> v;
  for (const char* p = s; *p;) {
    v.push_back(std::atoi(p));
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  int pool = 114688, maxlen = 65536, R = 14, iters = 50;
  double gbps = 632.0;
  std::vector<int> ctxs = {512, 4096, 16384, 32768, 55000};
  std::string fmts = "bf16,kv8,kv4,k5v4";
  for (int i = 1; i + 1 < argc; ++i) {
    if (!std::strcmp(argv[i], "--pool")) pool = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--maxlen")) maxlen = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--rows")) R = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gbps")) gbps = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--ctx")) ctxs = parse_list(argv[++i]);
    else if (!std::strcmp(argv[i], "--fmt")) fmts = argv[++i];
  }
  const std::size_t slots = static_cast<std::size_t>(kHk) * pool;
  // One buffer set big enough for any format: K up to d*2 bytes/row (bf16), V the same.
  std::uint8_t *dk, *dv;
  float *dks, *dvs;
  CK(cudaMalloc(&dk, slots * kD * 2));
  CK(cudaMalloc(&dv, slots * kD * 2));
  CK(cudaMalloc(&dks, slots * 4));
  CK(cudaMalloc(&dvs, slots * 4));
  fill_bytes<<<1024, 256>>>(dk, slots * kD * 2, 0x1234u);
  fill_bytes<<<1024, 256>>>(dv, slots * kD * 2, 0x9876u);
  fill_scales<<<256, 256>>>(dks, slots);
  fill_scales<<<256, 256>>>(dvs, slots);
  const bool want_bf16 = fmts.find("bf16") != std::string::npos;
  if (want_bf16) {  // valid bf16 data for the bf16 case (random bytes can be NaN/Inf)
    fill_bf16<<<1024, 256>>>(reinterpret_cast<std::uint16_t*>(dk), slots * kD, 0x55u);
    fill_bf16<<<1024, 256>>>(reinterpret_cast<std::uint16_t*>(dv), slots * kD, 0x77u);
  }
  // Row table: logical position p -> physical arena row (a fixed permutation, like a paged session).
  std::vector<int> pt(maxlen);
  for (int i = 0; i < maxlen; ++i) pt[i] = static_cast<int>((static_cast<long long>(i) * 7919) % pool);
  int* dpt;
  CK(cudaMalloc(&dpt, maxlen * sizeof(int)));
  CK(cudaMemcpy(dpt, pt.data(), maxlen * sizeof(int), cudaMemcpyHostToDevice));
  std::uint16_t *dq, *dout;
  CK(cudaMalloc(&dq, static_cast<std::size_t>(R) * kHq * kD * 2));
  CK(cudaMalloc(&dout, static_cast<std::size_t>(R) * kHq * kD * 2));
  fill_bf16<<<256, 256>>>(dq, static_cast<std::size_t>(R) * kHq * kD, 0x31u);
  void* scratch;
  CK(cudaMalloc(&scratch, granite4::attn_split_scratch_bytes(kHq, R, maxlen)));
  int *dpos1, *drpos, *dgrp;
  const int** drpt;
  CK(cudaMalloc(&dpos1, sizeof(int)));
  CK(cudaMalloc(&drpos, R * sizeof(int)));
  CK(cudaMalloc(&dgrp, R * sizeof(int)));
  CK(cudaMalloc(&drpt, R * sizeof(int*)));
  int *dtb, *dtail;
  CK(cudaMalloc(&dtb, R * sizeof(int)));
  CK(cudaMalloc(&dtail, R * granite4::kAttnTailStride * sizeof(int)));
  // verify_tree: the default solo draft tree (13 nodes + root, infer_main eagle_tree), as rows
  // of one sequence sharing tbase = L; node r's tail = its ancestors' slots then its own.
  const char* const kTree[] = {"0",   "0.0", "0.0.0", "1",     "0.0.0.0", "0.1",  "2",
                               "1.0", "3",   "0.2",   "0.0.1", "0.1.0",   "1.0.0"};
  const int tree_rows = 1 + static_cast<int>(sizeof(kTree) / sizeof(kTree[0]));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaDeviceSynchronize());

  struct Fmt { const char* name; double kbytes, vbytes; bool int8, int4, k5; };
  const Fmt all[] = {{"bf16", 256, 256, false, false, false},
                     {"kv8", 128 + 4, 128 + 4, true, false, false},
                     {"kv4", 64 + 4, 64 + 4, false, true, false},
                     {"k5v4", 80 + 4, 64 + 4, false, true, true}};
  std::printf("{\"pool\":%d,\"maxlen\":%d,\"rows\":%d,\"gbps\":%.1f,\"results\":[\n", pool, maxlen, R,
              gbps);
  bool first = true;
  for (const Fmt& f : all) {
    if (fmts.find(f.name) == std::string::npos) continue;
    if (!strcmp(f.name, "bf16") && !want_bf16) continue;
    auto call = [&](int L, int rows, const int* d_pos, const int* rpos, const int* const* rpt,
                    const int* grp, int ng, const int* tb = nullptr, const int* tl = nullptr) {
      const auto* k16 = reinterpret_cast<const std::uint16_t*>(dk);
      const auto* v16 = reinterpret_cast<const std::uint16_t*>(dv);
      const auto* k8 = reinterpret_cast<const std::int8_t*>(dk);
      const auto* v8 = reinterpret_cast<const std::int8_t*>(dv);
      const bool q = f.int8 || f.int4;
      return granite4::attn_split(dq, q ? nullptr : k16, q ? nullptr : v16, 0.0078125f, kHq, kHk, kD,
                                  0, d_pos, rows, pool, scratch, dout, f.int8 ? k8 : nullptr,
                                  f.int8 ? v8 : nullptr, q ? dks : nullptr, q ? dvs : nullptr, f.int8,
                                  f.int4 ? dk : nullptr, f.int4 ? dv : nullptr, f.int4,
                                  rpos == nullptr ? dpt : nullptr, maxlen, rpos, rpt, grp, ng, tb,
                                  tl, L + rows, f.k5);
    };
    for (const int L : ctxs) {
      if (L + R > maxlen) continue;
      // Rows at positions L .. L + R - 1 of one sequence; groups of up to 8 (forward_multi_step).
      std::vector<int> rpos(R), grp;
      std::vector<const int*> rpt(R, dpt);
      for (int r = 0; r < R; ++r) rpos[r] = L + r;
      for (int r = 0; r < R; r += 8) grp.push_back(r | (std::min(8, R - r) << 8));
      const int pos1 = L;
      CK(cudaMemcpy(dpos1, &pos1, sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(drpos, rpos.data(), R * sizeof(int), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(drpt, rpt.data(), R * sizeof(int*), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dgrp, grp.data(), grp.size() * sizeof(int), cudaMemcpyHostToDevice));
      // Tree rows (only when R matches the tree): positions L + depth, slots L + row.
      std::vector<int> tpos(R, L), ttb(R, L), ttail(R * granite4::kAttnTailStride, 0), tgrp;
      const bool tree_ok = R == tree_rows;
      if (tree_ok) {
        std::vector<std::string> path(R);
        for (int r = 1; r < R; ++r) path[r] = kTree[r - 1];
        for (int r = 1; r < R; ++r) {
          int d = 0;
          for (int a = 0; a < R; ++a) {  // ancestors (proper prefixes) and itself, by depth
            const std::string& pa = path[a];
            if (a == 0) continue;
            const bool prefix = path[r].compare(0, pa.size(), pa) == 0 &&
                                (pa.size() == path[r].size() || path[r][pa.size()] == '.');
            if (!prefix) continue;
            const int depth = 1 + static_cast<int>(std::count(pa.begin(), pa.end(), '.'));
            ttail[r * granite4::kAttnTailStride + depth - 1] = L + a;
            d = std::max(d, depth);
          }
          tpos[r] = L + d;
        }
        for (int r = 0; r < R; r += 8) tgrp.push_back(r | (std::min(8, R - r) << 8));
        CK(cudaMemcpy(dtb, ttb.data(), R * sizeof(int), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dtail, ttail.data(), ttail.size() * sizeof(int), cudaMemcpyHostToDevice));
      }
      struct Case { const char* name; int rows, groups; };
      const Case cases[] = {{"decode", 1, 1}, {"verify_grouped", R, static_cast<int>(grp.size())},
                            {"verify_ungrouped", R, R},
                            {"verify_tree", tree_ok ? R : 0, static_cast<int>(tgrp.size())}};
      for (const Case& c : cases) {
        if (c.rows == 0) continue;
        const bool tree = !strcmp(c.name, "verify_tree");
        if (tree) {
          CK(cudaMemcpy(drpos, tpos.data(), R * sizeof(int), cudaMemcpyHostToDevice));
          CK(cudaMemcpy(dgrp, tgrp.data(), tgrp.size() * sizeof(int), cudaMemcpyHostToDevice));
        }
        auto run = [&]() {
          if (tree) return call(L, c.rows, nullptr, drpos, drpt, dgrp, c.groups, dtb, dtail);
          if (c.rows == 1) return call(L, 1, dpos1, nullptr, nullptr, nullptr, 0);
          if (c.groups < c.rows) return call(L, c.rows, nullptr, drpos, drpt, dgrp, c.groups);
          return call(L, c.rows, nullptr, drpos, drpt, nullptr, 0);
        };
        for (int w = 0; w < 3; ++w)
          if (run() != 0) { std::fprintf(stderr, "launch failed %s %s L=%d\n", f.name, c.name, L); return 1; }
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int it = 0; it < iters; ++it) run();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        ms /= iters;
        // Bytes the call needs: every K/V row of the context once per row group (decode: once).
        const double ctx_rows = static_cast<double>(L + c.rows) * kHk;
        const double bytes = ctx_rows * (f.kbytes + f.vbytes) * c.groups;
        const double floor_ms = bytes / (gbps * 1e9) * 1e3;
        std::printf("%s{\"fmt\":\"%s\",\"case\":\"%s\",\"ctx\":%d,\"ms\":%.4f,\"floor_ms\":%.4f,"
                    "\"efficiency\":%.3f,\"x40_ms\":%.2f}\n",
                    first ? "" : ",", f.name, c.name, L, ms, floor_ms, floor_ms / ms, ms * 40);
        first = false;
      }
    }
  }
  std::printf("]}\n");
  return 0;
}
