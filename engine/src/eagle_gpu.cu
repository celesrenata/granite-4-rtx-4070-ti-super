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
                                     std::uint16_t*, std::uint16_t*);
// Quantized draft KV: rotate K in place at [pos0..pos0+rows) (Q rotation is
// done by the caller on the bf16 q), then append the rotated K + raw V to the
// persistent pool with a per-row scale. int8 (Q8_0) and int4 (Q4_0) forms.
int granite4_kernel_kv_append_int8_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::int8_t* d_cache_k, std::int8_t* d_cache_v,
                                           float* d_cache_k_scale, float* d_cache_v_scale,
                                           int base_pos, int rows, int stride, int Hk, int d);
int granite4_kernel_kv_append_int4_batched(const std::uint16_t* d_src_k,
                                           const std::uint16_t* d_src_v,
                                           std::uint8_t* d_cache_k, std::uint8_t* d_cache_v,
                                           float* d_cache_k_scale, float* d_cache_v_scale,
                                           int base_pos, int rows, int stride, int Hk, int d);
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
  std::vector<void*> allocs;

  ~EagleHead() {
    for (void* p : allocs) cudaFree(p);
    cudaFree(cos_tbl);
    cudaFree(sin_tbl);
  }
  template <class T>
  bool alloc(T** p, std::size_t bytes, std::string* err) {
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(p), bytes), err, "eagle: cudaMalloc"))
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
                      const ForwardSpec& spec, int max_pos, std::string* err) {
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
  const std::size_t kv = static_cast<std::size_t>(h->Hk) * max_pos * h->d * 2;
  const std::size_t kv_quant = static_cast<std::size_t>(h->Hk) * max_pos *
                               static_cast<std::size_t>(h->d);
  const std::size_t kv4_quant = kv_quant / 2;
  const std::size_t kv_scale = static_cast<std::size_t>(h->Hk) * max_pos * 4;
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
void eagle_truncate(EagleHead* h, int written) {
  if (h != nullptr && written >= 0 && written < h->written) h->written = written;
}
int eagle_written(const EagleHead* h) { return h != nullptr ? h->written : -1; }
std::size_t eagle_device_bytes(const EagleHead* h) {
  if (h == nullptr) return 0;
  const std::size_t H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d;
  return 2 * H * H + 2 * nq * H + 2 * nk * H + 3 * I * H + static_cast<std::size_t>(h->Vd) * H;
}

int eagle_forward(EagleHead* h, const std::int32_t* d_tokens, int pos0, int rows,
                  const std::uint16_t* d_feat, std::uint16_t* d_feat_out,
                  std::int32_t* d_draft_out) {
  if (h == nullptr || rows < 1 || pos0 != h->written || pos0 + rows > h->max_pos) return -1;
  const int H = h->H, I = h->I, nq = h->Hq * h->d, nk = h->Hk * h->d, half = h->d / 2;
  for (int c0 = 0; c0 < rows; c0 += kMaxVerifyRows) {
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
                                                  h->kv8_v_scale, p, R, h->max_pos, h->Hk,
                                                  h->d) != 0 ||
            attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                       h->max_pos, h->attn_scratch, att, h->kv8_k, h->kv8_v, h->kv8_k_scale,
                       h->kv8_v_scale, true, nullptr, nullptr, false) != 0)
          return -1;
      } else {
        if (granite4_kernel_kv_append_int4_batched(kr, v, h->kv4_k, h->kv4_v, h->kv4_k_scale,
                                                  h->kv4_v_scale, p, R, h->max_pos, h->Hk,
                                                  h->d) != 0 ||
            attn_split(qr, nullptr, nullptr, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                       h->max_pos, h->attn_scratch, att, nullptr, nullptr, nullptr, nullptr,
                       false, h->kv4_k, h->kv4_v, true) != 0)
          return -1;
      }
    } else if (granite4_kernel_verify_qkv_stage(
                   q, k, v, h->cos_tbl + static_cast<std::size_t>(p) * half,
                   h->sin_tbl + static_cast<std::size_t>(p) * half, p, R, h->Hq, h->Hk, h->d,
                   h->max_pos, qr, kr, h->kv_k, h->kv_v) != 0 ||
               attn_split(qr, h->kv_k, h->kv_v, h->att_scale, h->Hq, h->Hk, h->d, p, nullptr, R,
                          h->max_pos, h->attn_scratch, att) != 0) {
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

}  // namespace granite4
