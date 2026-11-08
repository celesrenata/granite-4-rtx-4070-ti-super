// EAGLE-style draft head on the GPU (see scripts/eagle/train_eagle.py).
//
// One Granite decoder layer that drafts from the target's own features:
//
//   x_j  = fc([ln_e embed(tok_{j+1}) ; ln_h feat_j]) + b   (2H -> H; the input
//          norms are optional in the head file)
//   h_j  = x_j + O(attn(RoPE(QKV(ln1 x_j))))          causal over head entries
//   h_j += down(silu(gate(ln2 h_j)) * up(ln2 h_j))
//   f^_j = lnf(h_j)                                   the predicted feature j+1
//   draft tok_{j+2} = vocab[argmax(lm_head[vocab] f^_j)]
//
// feat_j is the target's final-normed hidden state (gpu_forward d_hidden) at
// accepted positions and the head's own f^ inside a draft chain. The head
// keeps its own one-layer KV cache, indexed by entry j (= RoPE position j).
// Weights are FP8 e4m3 per tensor (the target's scheme) and run through the
// verify GEMVs; the draft vocabulary rows are gathered from the target's FP8
// lm_head (same bytes and scale). Drafts are only proposals - the target's
// verify forward decides - so this path has no bit-identity requirement.
#include "eagle_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "attention_gpu.h"
#include "prefill_gemm.h"
#include "checkpoint.h"
#include "device_math.cuh"
#include "kernel_math.h"
#include "model_config.h"
#include "safetensors.h"

extern "C" {
int granite4_verify_gemv(const std::uint16_t*, const std::uint8_t*, float, int, int, int,
                         std::uint16_t*);
int granite4_verify_gemv_resid(const std::uint16_t*, const std::uint8_t*, float,
                               const std::uint16_t*, float, int, int, int, std::uint16_t*);
int granite4_verify_qkv(const std::uint16_t*, const std::uint8_t*, const std::uint8_t*,
                        const std::uint8_t*, float, float, float, int, int, int, int, int,
                        std::uint16_t*, std::uint16_t*, std::uint16_t*);
int granite4_verify_silu(const std::uint8_t*, const std::uint8_t*, const std::uint16_t*, float,
                         float, int, int, int, std::uint16_t*);
int granite4_kernel_rmsnorm(const std::uint16_t*, const std::uint16_t*, float, int, int,
                            std::uint16_t*);
int granite4_kernel_verify_qkv_stage(const std::uint16_t*, const std::uint16_t*,
                                     const std::uint16_t*, const float*, const float*, int, int,
                                     int, int, int, int, std::uint16_t*, std::uint16_t*,
                                     std::uint16_t*, std::uint16_t*, const int* d_pt = nullptr);
// Quantized draft KV: rotate K in place at [pos0..pos0+rows) (Q rotation is
// done by the caller on the bf16 q), then append the rotated K + raw V to the
// persistent pool with a per-row scale. int8 (Q8_0) and int4 (Q4_0) forms.
int granite4_kernel_kv_append_int8_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::int8_t* d_cache_k, std::int8_t* d_cache_v,
                                           float* d_cache_k_scale, float* d_cache_v_scale,
                                           int base_pos, int rows, int stride, int Hk, int d,
                                           const int* d_pt);
int granite4_kernel_kv_append_int4_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::uint8_t* d_cache_k, std::uint8_t* d_cache_v,
                                           float* d_cache_k_scale, float* d_cache_v_scale,
                                           int base_pos, int rows, int stride, int Hk, int d,
                                           const int* d_pt);
int granite4_kernel_rope_at_pos_batched(const std::uint16_t* d_x,
                                        const float* d_cos, const float* d_sin, int rows,
                                        int heads, int d, std::uint16_t* d_out);
int granite4_rope_tables_build(double, int, int, float**, float**);
}

namespace granite4 {
namespace {

using dev::bf16_to_f32;
using dev::f32_to_bf16;

// x[r] = [embed(tokens[tok0 + r]) ; feat[r]]  ([rows][2H])
__global__ void concat_kernel(const std::int32_t* tokens, int tok0, const std::uint16_t* emb,
                              const std::uint16_t* feat, int H, std::uint16_t* x) {
  const int r = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= 2 * H) return;
  const std::size_t o = static_cast<std::size_t>(r) * 2 * H + j;
  if (j < H) {
    x[o] = emb[static_cast<std::size_t>(tokens[tok0 + r]) * H + j];
  } else {
    x[o] = feat[static_cast<std::size_t>(r) * H + (j - H)];
  }
}

// In place on the [rows][2H] concat buffer: half 0 (embedding) by ln_e,
// half 1 (feature) by ln_h. One block per (half, row).
constexpr int kNormThreads = 1024;
__global__ void __launch_bounds__(kNormThreads)
    input_norm_kernel(std::uint16_t* x, const std::uint16_t* we, const std::uint16_t* wh, int H,
                      float eps) {
  const int half = blockIdx.x, r = blockIdx.y;
  std::uint16_t* const v = x + static_cast<std::size_t>(r) * 2 * H + static_cast<std::size_t>(half) * H;
  const std::uint16_t* const w = half == 0 ? we : wh;
  float ss = 0.0f;
  for (int j = threadIdx.x; j < H; j += kNormThreads) {
    const float f = bf16_to_f32(v[j]);
    ss += f * f;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  __shared__ float red[kNormThreads / 32];
  __shared__ float scale;
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < kNormThreads / 32; ++i) t += red[i];
    scale = rsqrtf(t / H + eps);
  }
  __syncthreads();
  for (int j = threadIdx.x; j < H; j += kNormThreads)
    v[j] = f32_to_bf16(bf16_to_f32(v[j]) * scale * bf16_to_f32(w[j]));
}

__global__ void bias_kernel(std::uint16_t* x, const float* b, int H) {
  const int r = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  const std::size_t o = static_cast<std::size_t>(r) * H + j;
  x[o] = f32_to_bf16(bf16_to_f32(x[o]) + b[j]);
}

// out = vocab[argmax(logits)] (first maximum), one block.
__global__ void argmax_map_kernel(const std::uint16_t* logits, int n, const std::int32_t* vocab,
                                  std::int32_t* out) {
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = bf16_to_f32(logits[i]);
    if (v > bv) bv = v, bi = i;
  }
  __shared__ float sv[1024];
  __shared__ int si[1024];
  sv[threadIdx.x] = bv;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = sv[threadIdx.x + s];
      const int oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x]))
        sv[threadIdx.x] = ov, si[threadIdx.x] = oi;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = vocab[si[0] == 0x7fffffff ? 0 : si[0]];
}

__global__ void gather_rows_kernel(const std::uint8_t* src, const std::int32_t* rows, int H,
                                   std::uint8_t* dst) {
  const int r = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= H) return;
  dst[static_cast<std::size_t>(r) * H + j] = src[static_cast<std::size_t>(rows[r]) * H + j];
}

bool cuda_ok(cudaError_t e, std::string* err, const char* what) {
  if (e == cudaSuccess) return true;
  *err = std::string(what) + ": " + cudaGetErrorString(e);
  return false;
}

}  // namespace

struct EagleHead {
  int H = 0, I = 0, Hq = 0, Hk = 0, d = 0, V = 0, Vd = 0, max_pos = 0;
  float eps = 1e-5f, att_scale = 0.0f, lm_scale = 0.0f;
  // FP8 weights, per-tensor scales: fc q k v o gate up down
  std::uint8_t *fc = nullptr, *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr,
               *wg = nullptr, *wu = nullptr, *wd = nullptr, *lm = nullptr;
  float sc[8] = {};
  float* bias = nullptr;
  std::uint16_t *ln1 = nullptr, *ln2 = nullptr, *lnf = nullptr;
  std::uint16_t *ln_e = nullptr, *ln_h = nullptr;  // optional input norms
  const std::uint16_t* embed = nullptr;
  std::int32_t* vocab = nullptr;
  std::uint16_t *kv_k = nullptr, *kv_v = nullptr;
  // Quantized persistent draft KV (G4_DRAFT_KV, default 16 = bf16): 8 or 4-bit
  // Q8_0/Q4_0 codes + a per-(head,position) float scale. bf16 kv_k/kv_v hold
  // the 16-bit pool; the *_8/_4 pools hold the quantized ones. kv_bits is the
  // selected width (16/8/4).
  int kv_bits = 16;
  std::int8_t *kv8_k = nullptr, *kv8_v = nullptr;
  float *kv8_k_scale = nullptr, *kv8_v_scale = nullptr;
  std::uint8_t *kv4_k = nullptr, *kv4_v = nullptr;
  float *kv4_k_scale = nullptr, *kv4_v_scale = nullptr;
  float *cos_tbl = nullptr, *sin_tbl = nullptr;
  void* attn_scratch = nullptr;
  std::uint16_t* scratch = nullptr;  // per-row buffers for kMaxVerifyRows rows
  std::uint16_t* logits = nullptr;   // [Vd]
  int written = 0;
  // Shared draft-KV pool (P4). kv_stride is the arena's rows per head: the
  // whole pool for a paged head, max_pos for a private one. A paged head's
  // draft KV lives in the SAME rows as its session's main KV - one row table
  // (the target context's d_pt) addresses both pools - so pages shared or
  // handed back at the main-cache level cover the draft cache too.
  int kv_stride = 0;
  GpuCtx* target = nullptr;  // paged: the session whose row table addresses the KV
  GpuCtx* scratch_ctx = nullptr;  // the session whose prefill scratch the catch-up borrows
  bool paged = false;
  bool owns_tables = true;  // cos/sin: a forked session head borrows the base's
  std::vector<void*> allocs;  // owned allocations (a fork lists only its own)
  // eagle_forward_multi buffers (the lead head's, allocated on first use).
  void* multi_desc = nullptr;
  std::uint16_t *multi_feat = nullptr, *multi_out = nullptr, *multi_last = nullptr,
                *multi_logits = nullptr;

  ~EagleHead() {
    for (void* p : allocs) cudaFree(p);
    if (owns_tables) {
      cudaFree(cos_tbl);
      cudaFree(sin_tbl);
    }
  }
  template <class T>
  bool alloc(T** p, std::size_t bytes, std::string* err) {
    std::string dummy;
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), bytes), err != nullptr ? err : &dummy,
                 "eagle: cudaMalloc"))
      return false;
    allocs.push_back(*p);
    return true;
  }
};

namespace {

struct HostTensor {
  std::vector<float> f;  // values as float
  std::vector<std::int64_t> shape;
};

bool read_tensor(const std::string& path, const TensorManifest& man, const std::string& name,
                 HostTensor* out, std::string* err) {
  const TensorEntry* e = man.find(name);
  if (e == nullptr) {
    *err = "eagle head: missing tensor " + name;
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  in.seekg(8 + man.header_bytes + e->offset_begin);
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(e->nbytes));
  if (!in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()))) {
    *err = "eagle head: short read of " + name;
    return false;
  }
  out->shape = e->shape;
  out->f.resize(static_cast<std::size_t>(e->numel));
  if (e->dtype == "BF16") {
    for (std::size_t i = 0; i < out->f.size(); ++i)
      out->f[i] = bf16_to_f32_host(static_cast<std::uint16_t>(raw[2 * i] | (raw[2 * i + 1] << 8)));
  } else if (e->dtype == "F32") {
    std::memcpy(out->f.data(), raw.data(), raw.size());
  } else if (e->dtype == "I32") {
    for (std::size_t i = 0; i < out->f.size(); ++i) {
      std::int32_t v;
      std::memcpy(&v, raw.data() + 4 * i, 4);
      out->f[i] = static_cast<float>(v);
    }
  } else {
    *err = "eagle head: unsupported dtype " + e->dtype + " for " + name;
    return false;
  }
  return true;
}

bool expect_shape(const HostTensor& t, std::initializer_list<std::int64_t> s, const char* name,
                  std::string* err) {
  if (t.shape == std::vector<std::int64_t>(s)) return true;
  *err = std::string("eagle head: bad shape for ") + name;
  return false;
}

// Per-tensor FP8 e4m3 (scale = max|w| / 448), uploaded; returns the scale.
bool upload_fp8(EagleHead* h, const HostTensor& t, std::uint8_t** dst, float* scale,
                std::string* err) {
  double mx = 0.0;
  for (float v : t.f) mx = std::max(mx, static_cast<double>(std::fabs(v)));
  const double s = mx > 0 ? mx / 448.0 : 1.0;
  std::vector<std::uint8_t> q(t.f.size());
  for (std::size_t i = 0; i < q.size(); ++i) q[i] = fp8_e4m3_encode(t.f[i] / s, nullptr);
  if (!h->alloc(dst, q.size(), err)) return false;
  *scale = static_cast<float>(s);
  return cuda_ok(cudaMemcpy(*dst, q.data(), q.size(), cudaMemcpyHostToDevice), err, "eagle: upload");
}

bool upload_bf16(EagleHead* h, const HostTensor& t, std::uint16_t** dst, std::string* err) {
  std::vector<std::uint16_t> b(t.f.size());
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = f32_to_bf16_host(t.f[i]);
  if (!h->alloc(dst, b.size() * 2, err)) return false;
  return cuda_ok(cudaMemcpy(*dst, b.data(), b.size() * 2, cudaMemcpyHostToDevice), err,
                 "eagle: upload");
}

// Scratch layout per row: xin(2H) x(H) xn(H) q(Hq d) k(Hk d) v(Hk d) qr(Hq d) kr(Hk d)
// att(Hq d) h(H) hn(H) act(I) f(H)
std::size_t row_elems(const EagleHead& h) {
  return 2u * h.H + 5u * h.H + 3u * h.Hq * h.d + 3u * h.Hk * h.d + h.I;
}

}  // namespace

EagleHead* eagle_load(const std::string& path, GpuCtx* target, const ModelConfig& cfg,
                      const ForwardSpec& spec, int max_pos, std::string* err, int kv_rows) {
  if (kv_rows != 0 && kv_rows < max_pos) {
    *err = "eagle head: kv_rows must cover max_pos";
    return nullptr;
  }
  TensorManifest man;
  if (!parse_safetensors_file(path, &man, err)) return nullptr;
  auto h = std::make_unique<EagleHead>();
  h->H = static_cast<int>(cfg.hidden_size);
  h->I = static_cast<int>(cfg.intermediate_size);
  h->Hq = static_cast<int>(cfg.num_attention_heads);
  h->Hk = static_cast<int>(cfg.num_key_value_heads);
  h->d = static_cast<int>(cfg.head_dim);
  h->V = static_cast<int>(cfg.vocab_size);
  h->eps = spec.rms_norm_eps;
  h->att_scale = spec.attention_multiplier;
  h->max_pos = max_pos;
  // kv_rows > 0: the draft KV is one shared arena of kv_rows positions per
  // head, addressed through `target`'s row table (this head serves that
  // session; eagle_fork adds the others). 0: a private [max_pos] cache.
  h->kv_stride = kv_rows > 0 ? kv_rows : max_pos;
  h->paged = kv_rows > 0;
  h->target = h->paged ? target : nullptr;
  h->scratch_ctx = target;
  const std::int64_t H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d;
  HostTensor t;
  struct W {
    const char* name;
    std::initializer_list<std::int64_t> shape;
    std::uint8_t** dst;
    int scale;
  };
  const W ws[] = {{"fc.weight", {H, 2 * H}, &h->fc, 0},  {"q.weight", {nq, H}, &h->wq, 1},
                  {"k.weight", {nk, H}, &h->wk, 2},      {"v.weight", {nk, H}, &h->wv, 3},
                  {"o.weight", {H, nq}, &h->wo, 4},      {"gate.weight", {I, H}, &h->wg, 5},
                  {"up.weight", {I, H}, &h->wu, 6},      {"down.weight", {H, I}, &h->wd, 7}};
  for (const W& w : ws)
    if (!read_tensor(path, man, w.name, &t, err) || !expect_shape(t, w.shape, w.name, err) ||
        !upload_fp8(h.get(), t, w.dst, &h->sc[w.scale], err))
      return nullptr;
  // k and v share one launch with q: verify_qkv takes three pointers and
  // three scales, so they need not be contiguous.
  struct N {
    const char* name;
    std::uint16_t** dst;
  };
  for (const N& n : {N{"ln1.weight", &h->ln1}, N{"ln2.weight", &h->ln2}, N{"lnf.weight", &h->lnf}})
    if (!read_tensor(path, man, n.name, &t, err) || !expect_shape(t, {H}, n.name, err) ||
        !upload_bf16(h.get(), t, n.dst, err))
      return nullptr;
  if (man.find("ln_e.weight") != nullptr || man.find("ln_h.weight") != nullptr) {
    for (const N& n : {N{"ln_e.weight", &h->ln_e}, N{"ln_h.weight", &h->ln_h}})
      if (!read_tensor(path, man, n.name, &t, err) || !expect_shape(t, {H}, n.name, err) ||
          !upload_bf16(h.get(), t, n.dst, err))
        return nullptr;
  }
  if (!read_tensor(path, man, "fc.bias", &t, err) || !expect_shape(t, {H}, "fc.bias", err) ||
      !h->alloc(&h->bias, H * sizeof(float), err) ||
      !cuda_ok(cudaMemcpy(h->bias, t.f.data(), H * sizeof(float), cudaMemcpyHostToDevice), err,
               "eagle: upload"))
    return nullptr;
  // Optional: the head's own attention scale (default: the target's).
  if (man.find("attn_scale") != nullptr) {
    if (!read_tensor(path, man, "attn_scale", &t, err) || t.f.size() != 1 ||
        !std::isfinite(t.f[0]) || t.f[0] <= 0.0f) {
      if (err->empty()) *err = "eagle head: bad attn_scale";
      return nullptr;
    }
    h->att_scale = t.f[0];
  }
  if (!read_tensor(path, man, "draft_vocab", &t, err) || t.shape.size() != 1) {
    if (err->empty()) *err = "eagle head: bad draft_vocab";
    return nullptr;
  }
  h->Vd = static_cast<int>(t.shape[0]);
  std::vector<std::int32_t> vocab(t.f.size());
  for (std::size_t i = 0; i < vocab.size(); ++i) {
    vocab[i] = static_cast<std::int32_t>(t.f[i]);
    if (vocab[i] < 0 || vocab[i] >= h->V) {
      *err = "eagle head: draft_vocab id out of range";
      return nullptr;
    }
  }
  const std::uint8_t* lm_head = nullptr;
  if (gpu_ctx_shared_weights(target, &h->embed, &lm_head, &h->lm_scale) != 0) {
    *err = "eagle head: target weights not loaded";
    return nullptr;
  }
  if (!h->alloc(&h->vocab, vocab.size() * 4, err) ||
      !cuda_ok(cudaMemcpy(h->vocab, vocab.data(), vocab.size() * 4, cudaMemcpyHostToDevice), err,
               "eagle: upload") ||
      !h->alloc(&h->lm, static_cast<std::size_t>(h->Vd) * H, err))
    return nullptr;
  gather_rows_kernel<<<dim3((h->H + 255) / 256, h->Vd), 256>>>(lm_head, h->vocab, h->H, h->lm);
  // G4_DRAFT_KV sets the persistent draft KV width: 16 (bf16, default), 8
  // (Q8_0), or 4 (Q4_0). The quantized pools are 2x (8-bit) and 4x (4-bit)
  // smaller than the bf16 pool, which is what lets the 131K reader pool fit
  // the draft head on a 16 GiB card. Both require the split-KV attention
  // shape (d==128, Hq==4*Hk); anything else falls back to bf16.
  const char* const dkv_env = std::getenv("G4_DRAFT_KV");
  h->kv_bits = 16;
  if (dkv_env != nullptr) {
    const char* end = nullptr;
    const long v = std::strtol(dkv_env, const_cast<char**>(&end), 10);
    if (end != nullptr && *end == '\0' && (v == 8 || v == 4) && h->d == 128 &&
        h->Hq == 4 * h->Hk)
      h->kv_bits = static_cast<int>(v);
  }
  const std::size_t kv = static_cast<std::size_t>(h->Hk) * h->kv_stride * h->d * 2;
  const std::size_t kv_quant = static_cast<std::size_t>(h->Hk) * h->kv_stride *
                               static_cast<std::size_t>(h->d);
  const std::size_t kv4_quant = kv_quant / 2;
  const std::size_t kv_scale = static_cast<std::size_t>(h->Hk) * h->kv_stride * 4;
  bool ok = true;
  switch (h->kv_bits) {
  case 8:
    ok = h->alloc(&h->kv8_k, kv_quant, err) && h->alloc(&h->kv8_v, kv_quant, err) &&
         h->alloc(&h->kv8_k_scale, kv_scale, err) &&
         h->alloc(&h->kv8_v_scale, kv_scale, err);
    break;
  case 4:
    ok = h->alloc(&h->kv4_k, kv4_quant, err) && h->alloc(&h->kv4_v, kv4_quant, err) &&
         h->alloc(&h->kv4_k_scale, kv_scale, err) &&
         h->alloc(&h->kv4_v_scale, kv_scale, err);
    break;
  default:
    ok = h->alloc(&h->kv_k, kv, err) && h->alloc(&h->kv_v, kv, err);
    break;
  }
  if (ok)
    ok = h->alloc(&h->attn_scratch, attn_split_scratch_bytes(h->Hq, kMaxVerifyRows, max_pos),
                  err);
  if (ok) ok = h->alloc(&h->scratch, row_elems(*h) * kMaxVerifyRows * 2, err);
  if (ok) ok = h->alloc(&h->logits, static_cast<std::size_t>(h->Vd) * 2, err);
  if (!ok) return nullptr;
  if (granite4_rope_tables_build(spec.rope_theta, h->d, max_pos, &h->cos_tbl, &h->sin_tbl) != 0) {
    *err = "eagle head: rope tables";
    return nullptr;
  }
  if (!cuda_ok(cudaDeviceSynchronize(), err, "eagle: load")) return nullptr;
  return h.release();
}

void eagle_free(EagleHead* h) { delete h; }

EagleHead* eagle_fork(const EagleHead* base, GpuCtx* target, std::string* err) {
  if (base == nullptr || target == nullptr || !base->paged) {
    *err = "eagle fork: needs a paged base head (eagle_load with kv_rows > 0)";
    return nullptr;
  }
  if (gpu_ctx_kv_stride(target) != base->kv_stride) {
    *err = "eagle fork: the session's KV rows differ from the head's draft pool";
    return nullptr;
  }
  // Share the weights, tables and the draft-KV arena; own the per-session
  // scratch, logits and watermark.
  auto h = std::make_unique<EagleHead>(*base);
  h->allocs.clear();  // the copy must not free what the base owns
  h->multi_desc = nullptr;
  h->multi_feat = h->multi_out = h->multi_last = h->multi_logits = nullptr;
  h->owns_tables = false;
  h->attn_scratch = nullptr;
  h->scratch = nullptr;
  h->logits = nullptr;
  h->written = 0;
  h->target = target;
  h->scratch_ctx = target;
  if (!h->alloc(&h->attn_scratch,
                attn_split_scratch_bytes(h->Hq, kMaxVerifyRows, h->max_pos), err) ||
      !h->alloc(&h->scratch, row_elems(*h) * kMaxVerifyRows * 2, err) ||
      !h->alloc(&h->logits, static_cast<std::size_t>(h->Vd) * 2, err))
    return nullptr;
  return h.release();
}
std::size_t eagle_kv_bytes(const EagleHead* h) {
  if (h == nullptr) return 0;
  const std::size_t rows = static_cast<std::size_t>(h->Hk) * h->kv_stride;
  switch (h->kv_bits) {
    case 8: return 2 * rows * (h->d + 4);
    case 4: return 2 * rows * (h->d / 2 + 4);
    default: return 2 * rows * h->d * 2;
  }
}
void eagle_truncate(EagleHead* h, int written) {
  if (h != nullptr && written >= 0 && written < h->written) h->written = written;
}
int eagle_written(const EagleHead* h) { return h != nullptr ? h->written : -1; }
std::size_t eagle_device_bytes(const EagleHead* h) {
  if (h == nullptr) return 0;
  const std::size_t H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d;
  return 2 * H * H + 2 * nq * H + 2 * nk * H + 3 * I * H + static_cast<std::size_t>(h->Vd) * H;
}

namespace {
// Prompt catch-up on tensor cores: rows [p, p + R) of the head in one pass of
// prefill GEMMs (FP8 x bf16, tensor cores) + prefill attention, in the
// session's prefill scratch (idle between the target's forwards). Same math
// as the per-row verify-GEMV path, different accumulation order: drafts may
// differ in rare near-ties, the output never does (the target verifies).
int eagle_rows_prefill(EagleHead* h, const std::int32_t* d_tokens, int p, int R,
                       const std::uint16_t* d_feat, std::uint16_t* f,
                       const GpuPrefillScratch& s, const int* d_pt) {
  const int H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d;
  const int nqkv = nq + 2 * nk;
  // xin [R][2H] lives in the (not yet used) f32 gate|up buffer.
  std::uint16_t* const xin = reinterpret_cast<std::uint16_t*>(s.gu);
  std::uint16_t* const x = s.hidden;
  std::uint16_t* const xn = s.normed;
  concat_kernel<<<dim3((2 * H + 255) / 256, R), 256>>>(d_tokens, p + 1, h->embed, d_feat, H, xin);
  if (h->ln_e != nullptr)
    input_norm_kernel<<<dim3(2, R), kNormThreads>>>(xin, h->ln_e, h->ln_h, H, h->eps);
  GemmEpilogue ep;
  ep.scale0 = h->sc[0];
  ep.out_bf16 = x;
  ep.ldc = H;
  if (prefill_gemm(GemmOut::kBf16, xin, 2 * H, h->fc, R, H, 2 * H, ep, s.ws, s.ws_bytes) != 0)
    return -1;
  bias_kernel<<<dim3((H + 255) / 256, R), 256>>>(x, h->bias, H);
  if (granite4_kernel_rmsnorm(x, h->ln1, h->eps, R, H, xn) != 0) return -1;
  // q | k | v into one token-major [R][nqkv] row (three tensors, three scales).
  struct P {
    const std::uint8_t* w;
    float sc;
    int off, n;
  };
  for (const P& pj : {P{h->wq, h->sc[1], 0, nq}, P{h->wk, h->sc[2], nq, nk},
                      P{h->wv, h->sc[3], nq + nk, nk}}) {
    GemmEpilogue e;
    e.scale0 = pj.sc;
    e.out_bf16 = s.qkv + pj.off;
    e.ldc = nqkv;
    if (prefill_gemm(GemmOut::kBf16, xn, H, pj.w, R, pj.n, H, e, s.ws, s.ws_bytes) != 0)
      return -1;
  }
  const bool i8 = h->kv_bits == 8, i4 = h->kv_bits == 4;
  int rc;
  if (i8)
    rc = prefill_rope_split_append_int8(s.qkv, h->cos_tbl, h->sin_tbl, p, R, h->Hq, h->Hk, h->d,
                                        h->kv_stride, s.q_rot, h->kv8_k, h->kv8_v,
                                        h->kv8_k_scale, h->kv8_v_scale, d_pt);
  else if (i4)
    rc = prefill_rope_split_append_int4(s.qkv, h->cos_tbl, h->sin_tbl, p, R, h->Hq, h->Hk, h->d,
                                        h->kv_stride, s.q_rot, h->kv4_k, h->kv4_v,
                                        h->kv4_k_scale, h->kv4_v_scale, d_pt);
  else
    rc = prefill_rope_split_append(s.qkv, h->cos_tbl, h->sin_tbl, p, R, h->Hq, h->Hk, h->d,
                                   h->kv_stride, s.q_rot, h->kv_k, h->kv_v, d_pt);
  if (rc != 0 ||
      prefill_attention(s.q_rot, h->kv_k, h->kv_v, h->att_scale, h->Hq, h->Hk, h->d, p, R,
                        h->kv_stride, s.attn, h->kv8_k, h->kv8_v,
                        i8 ? h->kv8_k_scale : h->kv4_k_scale,
                        i8 ? h->kv8_v_scale : h->kv4_v_scale, i8, h->kv4_k, h->kv4_v, i4,
                        d_pt) != 0)
    return -1;
  // hh = x + att Wo  (into x, in place: residual may alias the output)
  GemmEpilogue eo;
  eo.scale0 = h->sc[4];
  eo.out_bf16 = x;
  eo.residual = x;
  eo.ldc = H;
  if (prefill_gemm(GemmOut::kResidual, s.attn, nq, h->wo, R, H, nq, eo, s.ws, s.ws_bytes) != 0 ||
      granite4_kernel_rmsnorm(x, h->ln2, h->eps, R, H, xn) != 0)
    return -1;
  GemmEpilogue eg, eu;
  eg.scale0 = h->sc[5];
  eg.out_f32 = s.gu;
  eg.ldc = 2 * I;
  eu.scale0 = h->sc[6];
  eu.out_f32 = s.gu + I;
  eu.ldc = 2 * I;
  if (prefill_gemm(GemmOut::kF32, xn, H, h->wg, R, I, H, eg, s.ws, s.ws_bytes) != 0 ||
      prefill_gemm(GemmOut::kF32, xn, H, h->wu, R, I, H, eu, s.ws, s.ws_bytes) != 0 ||
      prefill_silu_mul(s.gu, I, R, s.act) != 0)
    return -1;
  GemmEpilogue ed;
  ed.scale0 = h->sc[7];
  ed.out_bf16 = x;
  ed.residual = x;
  ed.ldc = H;
  if (prefill_gemm(GemmOut::kResidual, s.act, I, h->wd, R, H, I, ed, s.ws, s.ws_bytes) != 0 ||
      granite4_kernel_rmsnorm(x, h->lnf, h->eps, R, H, f) != 0)
    return -1;
  return 0;
}
}  // namespace

int eagle_forward(EagleHead* h, const std::int32_t* d_tokens, int pos0, int rows,
                  const std::uint16_t* d_feat, std::uint16_t* d_feat_out,
                  std::int32_t* d_draft_out) {
  if (h == nullptr || rows < 1 || pos0 != h->written || pos0 + rows > h->max_pos) return -1;
  // Paged: the draft KV rows are the session's main-cache rows; never write
  // or read past the positions the host has mapped for it.
  if (h->target != nullptr && pos0 + rows > gpu_ctx_mapped_rows(h->target)) return -1;
  const int* const d_pt = h->target != nullptr ? gpu_ctx_kv_rows_device(h->target) : nullptr;
  const int H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d, half = h->d / 2;
  // Long catch-ups (a prompt, or the rows a resumed turn adds) run in
  // prefill-sized chunks on tensor cores; the per-round steps (<= 16 rows)
  // keep the verify-GEMV path. G4_EAGLE_PREFILL=0 forces the GEMV path.
  constexpr int kPrefillMinRows = 64;
  static const bool prefill_ok = [] {
    const char* e = std::getenv("G4_EAGLE_PREFILL");
    return e == nullptr || e[0] != '0';
  }();
  GpuPrefillScratch ps{};
  const bool use_prefill = prefill_ok && rows >= kPrefillMinRows && h->scratch_ctx != nullptr &&
                           h->d == 128 && h->Hq == 4 * h->Hk &&
                           gpu_ctx_prefill_scratch(h->scratch_ctx, &ps) == 0 && ps.rows >= 64;
  int c0 = 0;
  if (use_prefill) {
    // Whole chunks only; a short tail (< 64 rows) takes the GEMV path below.
    for (; rows - c0 >= kPrefillMinRows; ) {
      const int R = std::min(ps.rows, rows - c0);
      if (eagle_rows_prefill(h, d_tokens, pos0 + c0, R, d_feat + static_cast<std::size_t>(c0) * H,
                             d_feat_out + static_cast<std::size_t>(c0) * H, ps, d_pt) != 0)
        return -1;
      c0 += R;
      h->written = pos0 + c0;
    }
  }
  for (; c0 < rows; c0 += kMaxVerifyRows) {
    const int R = std::min(kMaxVerifyRows, rows - c0);
    const int p = pos0 + c0;
    std::uint16_t* s = h->scratch;
    auto take = [&](std::size_t per_row) {
      std::uint16_t* q = s;
      s += per_row * R;
      return q;
    };
    std::uint16_t *xin = take(2 * H), *x = take(H), *xn = take(H), *q = take(nq), *k = take(nk),
                  *v = take(nk), *qr = take(nq), *kr = take(nk), *att = take(nq), *hh = take(H),
                  *hn = take(H), *act = take(I);
    std::uint16_t* f = d_feat_out + static_cast<std::size_t>(c0) * H;
    concat_kernel<<<dim3((2 * H + 255) / 256, R), 256>>>(
        d_tokens, p + 1, h->embed, d_feat + static_cast<std::size_t>(c0) * H, H, xin);
    if (h->ln_e != nullptr)
      input_norm_kernel<<<dim3(2, R), kNormThreads>>>(xin, h->ln_e, h->ln_h, H, h->eps);
    if (granite4_verify_gemv(xin, h->fc, h->sc[0], R, H, 2 * H, x) != 0) return -1;
    bias_kernel<<<dim3((H + 255) / 256, R), 256>>>(x, h->bias, H);
    if (granite4_kernel_rmsnorm(x, h->ln1, h->eps, R, H, xn) != 0 ||
        granite4_verify_qkv(xn, h->wq, h->wk, h->wv, h->sc[1], h->sc[2], h->sc[3], R, nq, nk, nk,
                            H, q, k, v) != 0)
      return -1;
    if (h->kv_bits == 4 || h->kv_bits == 8) {
      // Quantized draft KV: rotate Q and K into qr/kr at [p, p+R) (in-place
      // would race on the +/-half pairs), append the rotated K + raw V to the
      // persistent pool (Q8_0/Q4_0 codes + per-row scale), and attend over the
      // full prefix with dequant on load.
      const std::size_t cosp = static_cast<std::size_t>(p) * half;
      if (granite4_kernel_rope_at_pos_batched(q, h->cos_tbl + cosp, h->sin_tbl + cosp, R,
                                              h->Hq, h->d, qr) != 0 ||
          granite4_kernel_rope_at_pos_batched(k, h->cos_tbl + cosp, h->sin_tbl + cosp, R,
                                              h->Hk, h->d, kr) != 0)
        return -1;
      if (h->kv_bits == 8) {
        if (granite4_kernel_kv_append_int8_batched(kr, v, h->kv8_k, h->kv8_v, h->kv8_k_scale,
                                                  h->kv8_v_scale, p, R, h->kv_stride, h->Hk,
                                                  h->d, d_pt) != 0 ||
            attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                       h->kv_stride, h->attn_scratch, att, h->kv8_k, h->kv8_v, h->kv8_k_scale,
                       h->kv8_v_scale, true, nullptr, nullptr, false, d_pt, h->max_pos) != 0)
          return -1;
      } else {
        if (granite4_kernel_kv_append_int4_batched(kr, v, h->kv4_k, h->kv4_v, h->kv4_k_scale,
                                                  h->kv4_v_scale, p, R, h->kv_stride, h->Hk,
                                                  h->d, d_pt) != 0 ||
            attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                       h->kv_stride, h->attn_scratch, att, nullptr, nullptr, h->kv4_k_scale,
                       h->kv4_v_scale, false, h->kv4_k, h->kv4_v, true, d_pt,
                       h->max_pos) != 0)
          return -1;
      }
    } else if (granite4_kernel_verify_qkv_stage(
                   q, k, v, h->cos_tbl + static_cast<std::size_t>(p) * half,
                   h->sin_tbl + static_cast<std::size_t>(p) * half, p, R, h->Hq, h->Hk, h->d,
                   h->kv_stride, qr, kr, h->kv_k, h->kv_v, d_pt) != 0 ||
               attn_split(qr, h->kv_k, h->kv_v, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                          h->kv_stride, h->attn_scratch, att, nullptr, nullptr, nullptr,
                          nullptr, false, nullptr, nullptr, false, d_pt, h->max_pos) != 0) {
      return -1;
    }
    if (granite4_verify_gemv_resid(att, h->wo, h->sc[4], x, 1.0f, R, H, nq, hh) != 0 ||
        granite4_kernel_rmsnorm(hh, h->ln2, h->eps, R, H, hn) != 0 ||
        granite4_verify_silu(h->wg, h->wu, hn, h->sc[5], h->sc[6], I, H, R, act) != 0 ||
        granite4_verify_gemv_resid(act, h->wd, h->sc[7], hh, 1.0f, R, H, I, hh) != 0 ||
        granite4_kernel_rmsnorm(hh, h->lnf, h->eps, R, H, f) != 0)
      return -1;
    h->written = p + R;
  }
  if (d_draft_out != nullptr) {
    const std::uint16_t* last = d_feat_out + static_cast<std::size_t>(rows - 1) * H;
    if (granite4_verify_gemv(last, h->lm, h->lm_scale, 1, h->Vd, H, h->logits) != 0) return -1;
    argmax_map_kernel<<<1, 1024>>>(h->logits, h->Vd, h->vocab, d_draft_out);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

namespace {
// concat_kernel with row r's token at tok[r][pos[r] + 1] (multi-session rows).
__global__ void concat_multi_kernel(const GpuMultiRows* __restrict__ mr, const std::uint16_t* emb,
                                    const std::uint16_t* feat, int H, std::uint16_t* x) {
  const int r = blockIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= 2 * H) return;
  const std::size_t o = static_cast<std::size_t>(r) * 2 * H + j;
  if (j < H) {
    x[o] = emb[static_cast<std::size_t>(mr->tok[r][mr->pos[r] + 1]) * H + j];
  } else {
    x[o] = feat[static_cast<std::size_t>(r) * H + (j - H)];
  }
}
}  // namespace

int eagle_forward_multi(EagleHead* const* heads, int n, const EagleRowsReq* req) {
  if (heads == nullptr || req == nullptr || n < 1 || n > kMaxVerifyRows) return -1;
  EagleHead* const h = heads[0];  // lead: scratch, weights (shared by every fork)
  int R = 0;
  for (int i = 0; i < n; ++i) {
    const EagleHead* hi = heads[i];
    if (hi == nullptr || hi->fc != h->fc || hi->kv_stride != h->kv_stride || hi->target == nullptr ||
        req[i].rows < 1 || req[i].pos0 != hi->written || req[i].pos0 + req[i].rows > hi->max_pos ||
        req[i].pos0 + req[i].rows > gpu_ctx_mapped_rows(hi->target))
      return -1;
    R += req[i].rows;
  }
  if (R > kMaxVerifyRows) return -1;
  const int H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d;
  if (h->multi_desc == nullptr &&
      (!h->alloc(&h->multi_desc, sizeof(GpuMultiRows), nullptr) ||
       !h->alloc(&h->multi_feat, static_cast<std::size_t>(kMaxVerifyRows) * H * 2, nullptr) ||
       !h->alloc(&h->multi_out, static_cast<std::size_t>(kMaxVerifyRows) * H * 2, nullptr) ||
       !h->alloc(&h->multi_last, static_cast<std::size_t>(kMaxVerifyRows) * H * 2, nullptr) ||
       !h->alloc(&h->multi_logits, static_cast<std::size_t>(kMaxVerifyRows) * h->Vd * 2, nullptr)))
    return -1;
  GpuMultiRows desc{};
  desc.n = R;
  int r = 0;
  for (int i = 0; i < n; ++i) {
    const int* pt = gpu_ctx_kv_rows_device(heads[i]->target);
    for (int j = 0; j < req[i].rows; ++j, ++r) {
      desc.pos[r] = req[i].pos0 + j;
      desc.pt[r] = pt;
      desc.tok[r] = const_cast<std::int32_t*>(req[i].d_tokens);
    }
    if (cudaMemcpyAsync(h->multi_feat + static_cast<std::size_t>(r - req[i].rows) * H,
                        req[i].d_feat, sizeof(std::uint16_t) * req[i].rows * H,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess)
      return -1;
  }
  if (cudaMemcpyAsync(h->multi_desc, &desc, sizeof desc, cudaMemcpyHostToDevice,
                      cudaStreamPerThread) != cudaSuccess)
    return -1;
  GpuMultiRows* const md = static_cast<GpuMultiRows*>(h->multi_desc);
  std::uint16_t* s = h->scratch;
  auto take = [&](std::size_t per_row) {
    std::uint16_t* q = s;
    s += per_row * R;
    return q;
  };
  std::uint16_t *xin = take(2 * H), *x = take(H), *xn = take(H), *q = take(nq), *k = take(nk),
                *v = take(nk), *qr = take(nq), *kr = take(nk), *att = take(nq), *hh = take(H),
                *hn = take(H), *act = take(I);
  std::uint16_t* const f = h->multi_out;
  concat_multi_kernel<<<dim3((2 * H + 255) / 256, R), 256>>>(md, h->embed, h->multi_feat, H, xin);
  if (h->ln_e != nullptr)
    input_norm_kernel<<<dim3(2, R), kNormThreads>>>(xin, h->ln_e, h->ln_h, H, h->eps);
  if (granite4_verify_gemv(xin, h->fc, h->sc[0], R, H, 2 * H, x) != 0) return -1;
  bias_kernel<<<dim3((H + 255) / 256, R), 256>>>(x, h->bias, H);
  if (granite4_kernel_rmsnorm(x, h->ln1, h->eps, R, H, xn) != 0 ||
      granite4_verify_qkv(xn, h->wq, h->wk, h->wv, h->sc[1], h->sc[2], h->sc[3], R, nq, nk, nk, H,
                          q, k, v) != 0 ||
      gpu_multi_rope(q, h->cos_tbl, h->sin_tbl, md, R, h->Hq, h->d, qr) != 0 ||
      gpu_multi_rope(k, h->cos_tbl, h->sin_tbl, md, R, h->Hk, h->d, kr) != 0)
    return -1;
  const int* const row_pos = md->pos;
  const int* const* const row_pt = md->pt;
  int rc;
  if (h->kv_bits == 8) {
    rc = gpu_multi_kv_append(kr, v, 8, h->kv8_k, h->kv8_v, h->kv8_k_scale, h->kv8_v_scale, md, R,
                             h->kv_stride, h->Hk, h->d);
    if (rc == 0)
      rc = attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, 0, nullptr, R,
                      h->kv_stride, h->attn_scratch, att, h->kv8_k, h->kv8_v, h->kv8_k_scale,
                      h->kv8_v_scale, true, nullptr, nullptr, false, nullptr, h->max_pos,
                      row_pos, row_pt);
  } else if (h->kv_bits == 4) {
    rc = gpu_multi_kv_append(kr, v, 4, h->kv4_k, h->kv4_v, h->kv4_k_scale, h->kv4_v_scale, md, R,
                             h->kv_stride, h->Hk, h->d);
    if (rc == 0)
      rc = attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, 0, nullptr, R,
                      h->kv_stride, h->attn_scratch, att, nullptr, nullptr, h->kv4_k_scale,
                      h->kv4_v_scale, false, h->kv4_k, h->kv4_v, true, nullptr, h->max_pos,
                      row_pos, row_pt);
  } else {
    rc = gpu_multi_kv_append(kr, v, 16, h->kv_k, h->kv_v, nullptr, nullptr, md, R, h->kv_stride,
                             h->Hk, h->d);
    if (rc == 0)
      rc = attn_split(qr, h->kv_k, h->kv_v, h->att_scale, h->Hq, h->Hk, h->d, 0, nullptr, R,
                      h->kv_stride, h->attn_scratch, att, nullptr, nullptr, nullptr, nullptr,
                      false, nullptr, nullptr, false, nullptr, h->max_pos, row_pos, row_pt);
  }
  if (rc != 0 ||
      granite4_verify_gemv_resid(att, h->wo, h->sc[4], x, 1.0f, R, H, nq, hh) != 0 ||
      granite4_kernel_rmsnorm(hh, h->ln2, h->eps, R, H, hn) != 0 ||
      granite4_verify_silu(h->wg, h->wu, hn, h->sc[5], h->sc[6], I, H, R, act) != 0 ||
      granite4_verify_gemv_resid(act, h->wd, h->sc[7], hh, 1.0f, R, H, I, hh) != 0 ||
      granite4_kernel_rmsnorm(hh, h->lnf, h->eps, R, H, f) != 0)
    return -1;
  // Scatter the predicted features; gather each session's last row for its draft.
  r = 0;
  int drafts = 0;
  for (int i = 0; i < n; ++i) {
    if (cudaMemcpyAsync(req[i].d_feat_out, f + static_cast<std::size_t>(r) * H,
                        sizeof(std::uint16_t) * req[i].rows * H, cudaMemcpyDeviceToDevice,
                        cudaStreamPerThread) != cudaSuccess)
      return -1;
    r += req[i].rows;
    if (req[i].d_draft_out != nullptr &&
        cudaMemcpyAsync(h->multi_last + static_cast<std::size_t>(drafts++) * H,
                        f + static_cast<std::size_t>(r - 1) * H, sizeof(std::uint16_t) * H,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess)
      return -1;
    heads[i]->written = req[i].pos0 + req[i].rows;
  }
  if (drafts > 0) {
    if (granite4_verify_gemv(h->multi_last, h->lm, h->lm_scale, drafts, h->Vd, H,
                             h->multi_logits) != 0)
      return -1;
    int d = 0;
    for (int i = 0; i < n; ++i)
      if (req[i].d_draft_out != nullptr)
        argmax_map_kernel<<<1, 1024>>>(h->multi_logits + static_cast<std::size_t>(d++) * h->Vd,
                                       h->Vd, h->vocab, req[i].d_draft_out);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

}  // namespace granite4
