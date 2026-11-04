// GPU weight packing (see gpu_pack.h).
#include "gpu_pack.h"

#include "checkpoint.h"
#include "kernel_math.h"
#include "safetensors.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

namespace granite4 {

namespace fs = std::filesystem;

GpuWeightsLayout gpu_layout_for(const ModelConfig& cfg) {
  return GpuWeightsLayout{cfg.hidden_size,        cfg.intermediate_size,
                          cfg.vocab_size,         cfg.num_layers,
                          cfg.num_attention_heads, cfg.num_key_value_heads,
                          cfg.head_dim};
}

namespace {

// ---------------------------------------------------------------------------
// Shared element transforms (the single definition both packers use).
// ---------------------------------------------------------------------------

void store_bf16(std::uint8_t* dst, std::size_t i, float v) {
  const std::uint16_t b = f32_to_bf16_host(v);
  dst[2 * i] = static_cast<std::uint8_t>(b & 0xFF);
  dst[2 * i + 1] = static_cast<std::uint8_t>(b >> 8);
}

float fp8_scale_for(double max_abs) {
  return static_cast<float>(max_abs / kFp8E4M3MaxF);
}

std::uint8_t fp8_encode_scaled(float v, float scale) {
  bool sat = false;
  return fp8_e4m3_encode(static_cast<double>(v) / static_cast<double>(scale), &sat);
}

// Contiguous [begin, end) chunks of [0, n): at most `threads` of them and
// none smaller than `min_per` (so small tensors stay on the calling thread).
std::vector<std::pair<std::size_t, std::size_t>> split(std::size_t n, int threads,
                                                       std::size_t min_per) {
  const std::size_t want = std::max<std::size_t>(1, n / min_per);
  const std::size_t nt =
      std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), want);
  const std::size_t chunk = (n + nt - 1) / nt;
  std::vector<std::pair<std::size_t, std::size_t>> r;
  for (std::size_t b = 0; b < n; b += chunk) r.emplace_back(b, std::min(n, b + chunk));
  return r;
}

// fn(slot, begin, end) for every chunk; chunk 0 runs on the calling thread.
template <typename Fn>
void run_split(std::size_t n, int threads, std::size_t min_per, Fn fn) {
  const auto ranges = split(n, threads, min_per);
  std::vector<std::thread> pool;
  for (std::size_t s = 1; s < ranges.size(); ++s)
    pool.emplace_back([&, s] { fn(s, ranges[s].first, ranges[s].second); });
  if (!ranges.empty()) fn(std::size_t{0}, ranges[0].first, ranges[0].second);
  for (auto& th : pool) th.join();
}

constexpr std::size_t kMinElemsPerThread = 1u << 16;
constexpr std::size_t kMinKeysPerThread = 1u << 11;

template <typename Fn>
void parallel_for(std::size_t n, int threads, Fn fn) {
  run_split(n, threads, kMinElemsPerThread,
            [&](std::size_t, std::size_t b, std::size_t e) { fn(b, e); });
}

// Max |v| (as double). Chunk maxima combine with max, which is
// order-independent, so the result equals the serial scan.
template <typename Get>
double parallel_max_abs(std::size_t n, int threads, const Get& get) {
  std::vector<double> part(split(n, threads, kMinElemsPerThread).size(), 0.0);
  run_split(n, threads, kMinElemsPerThread,
            [&](std::size_t s, std::size_t b, std::size_t e) {
              double mx = 0.0;
              for (std::size_t i = b; i < e; ++i) {
                const double a = std::abs(static_cast<double>(get(i)));
                if (a > mx) mx = a;
              }
              part[s] = mx;
            });
  double mx = 0.0;
  for (double p : part) mx = std::max(mx, p);
  return mx;
}

// ---------------------------------------------------------------------------
// Checkpoint tensor source: raw bytes + the decode load_weights uses.
// ---------------------------------------------------------------------------

struct TensorLoc {
  int shard = -1;
  std::uint64_t file_offset = 0;
  std::uint64_t nbytes = 0;
  std::uint64_t numel = 0;
  std::uint64_t rows = 1;
  std::string dtype;
};

constexpr std::size_t kMinReadBytesPerThread = 4u << 20;

class Shards {
 public:
  ~Shards() {
    for (int fd : fds_)
      if (fd >= 0) ::close(fd);
  }

  bool open(const std::string& dir, std::string* error) {
    std::vector<std::string> names;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
      const std::string n = e.path().filename().string();
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0)
        names.push_back(n);
    }
    if (ec) {
      *error = "cannot list " + dir;
      return false;
    }
    std::sort(names.begin(), names.end());
    if (names.empty()) {
      *error = "no .safetensors shards in " + dir;
      return false;
    }
    for (const auto& n : names) {
      const std::string path = dir + "/" + n;
      TensorManifest m;
      std::string merr;
      if (!parse_safetensors_file(path, &m, &merr)) {
        *error = n + ": " + merr;
        return false;
      }
      const int fd = ::open(path.c_str(), O_RDONLY);
      if (fd < 0) {
        *error = "cannot open " + path;
        return false;
      }
      const int shard = static_cast<int>(fds_.size());
      fds_.push_back(fd);
      paths_.push_back(path);
      for (const auto& t : m.tensors) {
        TensorLoc l;
        l.shard = shard;
        l.file_offset = 8 + static_cast<std::uint64_t>(m.header_bytes) +
                        static_cast<std::uint64_t>(t.offset_begin);
        l.nbytes = static_cast<std::uint64_t>(t.offset_end - t.offset_begin);
        l.rows = t.shape.empty() ? 1 : static_cast<std::uint64_t>(t.shape[0]);
        l.numel = l.rows * (t.shape.size() > 1 ? static_cast<std::uint64_t>(t.shape[1]) : 1);
        l.dtype = t.dtype;
        loc_[t.name] = l;
      }
    }
    return true;
  }

  const TensorLoc* find(const std::string& name) const {
    auto it = loc_.find(name);
    return it == loc_.end() ? nullptr : &it->second;
  }

  // Positional read of a tensor's bytes, split across threads (page-cache
  // copies and NVMe queues both scale with concurrent readers).
  bool read(const TensorLoc& l, std::uint8_t* dst, int threads,
            std::string* error) const {
    const int fd = fds_[static_cast<std::size_t>(l.shard)];
    const auto ranges = split(l.nbytes, threads, kMinReadBytesPerThread);
    std::vector<char> failed(ranges.size(), 0);
    run_split(l.nbytes, threads, kMinReadBytesPerThread,
              [&](std::size_t s, std::size_t b, std::size_t e) {
                while (b < e) {
                  const ssize_t r = ::pread(fd, dst + b, e - b,
                                            static_cast<off_t>(l.file_offset + b));
                  if (r <= 0) {
                    failed[s] = 1;
                    return;
                  }
                  b += static_cast<std::size_t>(r);
                }
              });
    for (char f : failed)
      if (f != 0) {
        *error = "short read from " + paths_[static_cast<std::size_t>(l.shard)];
        return false;
      }
    return true;
  }

 private:
  std::vector<int> fds_;
  std::vector<std::string> paths_;
  std::map<std::string, TensorLoc> loc_;
};

// A checkpoint tensor's raw bytes plus the decode load_weights applies:
// element i as a float. BF16 and per-tensor-scaled FP8 sources have few
// distinct bit patterns (65536 / 256), so every element is a table key and
// the packers transform the table once instead of every element.
struct Decoded {
  enum Kind { kBf16, kFp8Tensor, kFp8Channel };
  Kind kind = kBf16;
  std::vector<std::uint8_t> raw;
  std::vector<float> scale;  // FP8 export: [1] or [rows]
  std::size_t cols = 1;

  bool keyed() const { return kind != kFp8Channel; }
  std::size_t key_count() const { return kind == kBf16 ? 65536u : 256u; }
  std::size_t key(std::size_t i) const {
    return kind == kBf16 ? (static_cast<std::size_t>(raw[2 * i]) |
                            (static_cast<std::size_t>(raw[2 * i + 1]) << 8))
                         : raw[i];
  }
  float value_of_key(std::size_t k) const {
    return kind == kBf16 ? bf16_to_f32_host(static_cast<std::uint16_t>(k))
                         : fp8_to_f32_host(static_cast<std::uint8_t>(k)) * scale[0];
  }
  float operator()(std::size_t i) const {
    if (kind == kFp8Channel) return fp8_to_f32_host(raw[i]) * scale[i / cols];
    return value_of_key(key(i));
  }
};

bool decode_tensor(const Shards& shards, const std::string& name,
                   std::size_t numel, int threads, Decoded* out,
                   std::string* error) {
  const TensorLoc* l = shards.find(name);
  if (l == nullptr) {
    *error = "missing tensor: " + name;
    return false;
  }
  if (l->numel != numel) {
    *error = name + ": unexpected numel " + std::to_string(l->numel) +
             " (expected " + std::to_string(numel) + ")";
    return false;
  }
  if (l->dtype == "BF16") {
    if (l->nbytes != 2 * numel) {
      *error = name + ": BF16 byte size mismatch";
      return false;
    }
    out->kind = Decoded::kBf16;
  } else if (l->dtype == "F8_E4M3") {
    if (l->nbytes != numel) {
      *error = name + ": F8_E4M3 byte size mismatch";
      return false;
    }
    const TensorLoc* s = shards.find(name + "__scale");
    if (s == nullptr) {
      *error = name + ": missing __scale tensor";
      return false;
    }
    std::vector<std::uint8_t> sraw(s->nbytes);
    if (!shards.read(*s, sraw.data(), 1, error)) return false;
    out->scale.resize(sraw.size() / 4);
    std::memcpy(out->scale.data(), sraw.data(), out->scale.size() * 4);
    if (out->scale.empty()) {
      *error = name + "__scale: empty";
      return false;
    }
    out->kind = out->scale.size() > 1 ? Decoded::kFp8Channel : Decoded::kFp8Tensor;
  } else {
    *error = name + ": unsupported dtype " + l->dtype;
    return false;
  }
  out->cols = l->rows > 0 ? numel / l->rows : numel;
  out->raw.resize(l->nbytes);
  return shards.read(*l, out->raw.data(), threads, error);
}

// Packers shared by both producers. `get(i)` yields element i as float.
template <typename Get>
void pack_bf16(std::uint8_t* dst, std::size_t n, int threads, const Get& get) {
  parallel_for(n, threads, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) store_bf16(dst, i, get(i));
  });
}

template <typename Get>
float pack_fp8(std::uint8_t* dst, std::size_t n, int threads, const Get& get) {
  const float scale = fp8_scale_for(parallel_max_abs(n, threads, get));
  if (scale <= 0.0f) {
    std::memset(dst, 0, n);
    return 0.0f;
  }
  parallel_for(n, threads, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) dst[i] = fp8_encode_scaled(get(i), scale);
  });
  return scale;
}

// Keyed forms: identical results (the same per-element function evaluated on
// each distinct key), computed through a key -> output table.
void pack_bf16(std::uint8_t* dst, std::size_t n, int threads, const Decoded& t) {
  if (!t.keyed()) return pack_bf16<Decoded>(dst, n, threads, t);
  std::vector<std::uint16_t> table(t.key_count());
  run_split(table.size(), threads, kMinKeysPerThread,
            [&](std::size_t, std::size_t b, std::size_t e) {
              for (std::size_t k = b; k < e; ++k) table[k] = f32_to_bf16_host(t.value_of_key(k));
            });
  parallel_for(n, threads, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      const std::uint16_t v = table[t.key(i)];
      dst[2 * i] = static_cast<std::uint8_t>(v & 0xFF);
      dst[2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
    }
  });
}

float pack_fp8(std::uint8_t* dst, std::size_t n, int threads, const Decoded& t) {
  if (!t.keyed()) return pack_fp8<Decoded>(dst, n, threads, t);
  // Which keys occur (the max is over occurring values only).
  const std::size_t nk = t.key_count();
  std::vector<std::vector<char>> seen(split(n, threads, kMinElemsPerThread).size(),
                                      std::vector<char>(nk, 0));
  run_split(n, threads, kMinElemsPerThread,
            [&](std::size_t s, std::size_t b, std::size_t e) {
              char* const mark = seen[s].data();
              for (std::size_t i = b; i < e; ++i) mark[t.key(i)] = 1;
            });
  double mx = 0.0;
  for (std::size_t k = 0; k < nk; ++k) {
    bool any = false;
    for (const auto& m : seen) any = any || m[k] != 0;
    if (!any) continue;
    const double a = std::abs(static_cast<double>(t.value_of_key(k)));
    if (a > mx) mx = a;
  }
  const float scale = fp8_scale_for(mx);
  if (scale <= 0.0f) {
    std::memset(dst, 0, n);
    return 0.0f;
  }
  // Encode only the keys that occur (fp8_e4m3_encode is the costly part).
  std::vector<std::uint8_t> table(nk, 0);
  run_split(nk, threads, kMinKeysPerThread, [&](std::size_t, std::size_t b, std::size_t e) {
    for (std::size_t k = b; k < e; ++k) {
      bool any = false;
      for (const auto& m : seen) any = any || m[k] != 0;
      if (any) table[k] = fp8_encode_scaled(t.value_of_key(k), scale);
    }
  });
  parallel_for(n, threads, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) dst[i] = table[t.key(i)];
  });
  return scale;
}

int resolve_threads(int threads) {
  if (threads > 0) return threads;
  const unsigned hc = std::thread::hardware_concurrency();
  return hc == 0 ? 1 : static_cast<int>(hc);
}

}  // namespace

bool pack_gpu_model_from_weights(const ModelWeights& w, GpuModel* out,
                                 std::string* error) {
  const GpuWeightsLayout lay = gpu_layout_for(w.model);
  out->allocate(lay);
  std::uint8_t* const base = out->weights.get();
  const std::size_t H = lay.hidden, I = lay.intermediate, KV = lay.kv_rows();
  auto from = [](const float* p) { return [p](std::size_t i) { return p[i]; }; };

  pack_bf16(base, lay.vocab * H, 1, from(w.embed.data()));
  for (std::size_t layer = 0; layer < lay.num_layers; ++layer) {
    const float* const lb = w.layers.data() + layer * w.layer_bytes;
    std::uint8_t* const dl = base + lay.embed_bytes() + layer * lay.layer_bytes();
    pack_bf16(dl + lay.off_attn_norm(), H, 1, from(lb));
    pack_bf16(dl + lay.off_ffn_norm(), H, 1, from(lb + H));
    out->scales.push_back(pack_fp8(dl + lay.off_q(), H * H, 1, from(lb + w.off_q)));
    out->scales.push_back(pack_fp8(dl + lay.off_k(), KV * H, 1, from(lb + w.off_k)));
    out->scales.push_back(pack_fp8(dl + lay.off_v(), KV * H, 1, from(lb + w.off_v)));
    out->scales.push_back(pack_fp8(dl + lay.off_o(), H * H, 1, from(lb + w.off_o)));
    out->scales.push_back(pack_fp8(dl + lay.off_gate(), I * H, 1, from(lb + w.off_gate)));
    out->scales.push_back(pack_fp8(dl + lay.off_up(), I * H, 1, from(lb + w.off_up)));
    out->scales.push_back(pack_fp8(dl + lay.off_down(), H * I, 1, from(lb + w.off_down)));
  }
  pack_bf16(base + lay.final_norm_off(), H, 1, from(w.final_norm.data()));
  out->scales.push_back(
      pack_fp8(base + lay.lm_head_off(), lay.vocab * H, 1, from(w.lm_head.data())));
  if (out->scales.size() != lay.num_scales()) {
    *error = "gpu pack: scale count mismatch";
    return false;
  }
  return true;
}

bool pack_gpu_model_from_checkpoint(const std::string& dir,
                                    const ModelConfig& cfg, GpuModel* out,
                                    std::string* error, int threads) {
  threads = resolve_threads(threads);
  Shards shards;
  if (!shards.open(dir, error)) return false;
  const GpuWeightsLayout lay = gpu_layout_for(cfg);
  out->allocate(lay);
  std::uint8_t* const base = out->weights.get();
  const std::size_t H = lay.hidden, I = lay.intermediate, KV = lay.kv_rows();

  Decoded t;
  using clock = std::chrono::steady_clock;
  double read_s = 0.0, pack_s = 0.0;
  auto timed = [](double* acc, auto&& fn) {
    const auto t0 = clock::now();
    const bool ok = fn();
    *acc += std::chrono::duration<double>(clock::now() - t0).count();
    return ok;
  };
  auto bf16 = [&](const std::string& name, std::size_t n, std::uint8_t* dst) {
    return timed(&read_s, [&] { return decode_tensor(shards, name, n, threads, &t, error); }) &&
           timed(&pack_s, [&] {
             pack_bf16(dst, n, threads, t);
             return true;
           });
  };
  auto fp8 = [&](const std::string& name, std::size_t n, std::uint8_t* dst) {
    return timed(&read_s, [&] { return decode_tensor(shards, name, n, threads, &t, error); }) &&
           timed(&pack_s, [&] {
             out->scales.push_back(pack_fp8(dst, n, threads, t));
             return true;
           });
  };

  if (!bf16("model.embed_tokens.weight", lay.vocab * H, base)) return false;
  for (std::size_t layer = 0; layer < lay.num_layers; ++layer) {
    const std::string p = "model.layers." + std::to_string(layer) + ".";
    std::uint8_t* const dl = base + lay.embed_bytes() + layer * lay.layer_bytes();
    if (!bf16(p + "input_layernorm.weight", H, dl + lay.off_attn_norm()) ||
        !bf16(p + "post_attention_layernorm.weight", H, dl + lay.off_ffn_norm()) ||
        !fp8(p + "self_attn.q_proj.weight", H * H, dl + lay.off_q()) ||
        !fp8(p + "self_attn.k_proj.weight", KV * H, dl + lay.off_k()) ||
        !fp8(p + "self_attn.v_proj.weight", KV * H, dl + lay.off_v()) ||
        !fp8(p + "self_attn.o_proj.weight", H * H, dl + lay.off_o()) ||
        !fp8(p + "mlp.gate_proj.weight", I * H, dl + lay.off_gate()) ||
        !fp8(p + "mlp.up_proj.weight", I * H, dl + lay.off_up()) ||
        !fp8(p + "mlp.down_proj.weight", H * I, dl + lay.off_down()))
      return false;
  }
  if (!bf16("model.norm.weight", H, base + lay.final_norm_off()) ||
      !fp8("lm_head.weight", lay.vocab * H, base + lay.lm_head_off()))
    return false;
  if (out->scales.size() != lay.num_scales()) {
    *error = "gpu pack: scale count mismatch";
    return false;
  }
  if (std::getenv("G4_LOAD_TIMING") != nullptr)
    std::fprintf(stderr, "G4_LOAD pack_read=%.2fs pack_encode=%.2fs threads=%d\n", read_s,
                 pack_s, threads);
  return true;
}

}  // namespace granite4
