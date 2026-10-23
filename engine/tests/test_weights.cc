// Phase 3 (Integration): weight-loading tests.
//
// Host-only, no CUDA. Builds a tiny synthetic checkpoint (a handful of
// tensors, one shard) in BOTH the original BF16 layout and an FP8-e4m3 export
// layout, loads each with load_weights (enforce_spec=false for the tiny
// dims), and checks the dequantized floats against hand-computed values.
// This pins the loader: BF16 decode, FP8 per-tensor and per-channel
// dequant, and the per-layer block layout that forward_logits_f32 expects.
#include "weights.h"

#include "checkpoint.h"
#include "json.h"
#include "kernel_math.h"

#include <cmath>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace granite4;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

std::atomic<int> g_seq{0};
std::string tmp_dir(const char* tag) {
  std::error_code ec;
  const std::string name = std::string("granite4_w_") + tag + "_" +
                           std::to_string(g_seq.fetch_add(1));
  const fs::path base = fs::temp_directory_path(ec) / name;
  fs::create_directories(base, ec);
  return base.string();
}

// Write a single-tensor or multi-tensor safetensors file with a given header
// + data region (both supplied pre-encoded). Returns the file path.
std::string write_st(const std::string& dir, const std::string& header,
                     const std::string& data, const char* name) {
  const std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary);
  const std::uint64_t n = static_cast<std::uint64_t>(header.size());
  std::uint8_t be[8];
  for (int i = 0; i < 8; ++i) be[i] = static_cast<std::uint8_t>(n >> (8 * i));
  f.write(reinterpret_cast<const char*>(be), 8);
  f.write(header.data(), header.size());
  f.write(data.data(), data.size());
  f.close();
  return path;
}

// A tiny config.json that parses and is structurally consistent (tiny dims).
const char* kTinyConfig =
    "{"
    "\"num_hidden_layers\":1,\"hidden_size\":4,\"intermediate_size\":6,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,\"vocab_size\":5,"
    "\"max_position_embeddings\":16,\"rms_norm_eps\":0.00001,"
    "\"attention_multiplier\":0.0078125,\"residual_multiplier\":1.0,"
    "\"embedding_multiplier\":1.0,\"logits_scaling\":1.0,\"rope_theta\":10000000,"
    "\"rope_scaling\":null,\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":2,"
    "\"model_type\":\"granite\",\"torch_dtype\":\"bfloat16\","
    "\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\",\"attention_bias\":false,"
    "\"mlp_bias\":false,\"use_cache\":true,\"tie_word_embeddings\":false,"
    "\"architectures\":[\"GraniteForCausalLM\"],\"transformers_version\":\"4.47.0\""
    "}";

// BF16 encode (RNE) for a small float set.
std::uint16_t bf16bits(float f) { return f32_to_bf16_host(f); }

// Values that are represented exactly in e4m3 (so fp8 round-trips are exact
// at scale 1.0): 0, +/-1, +/-2, +/-0.5, and the e4m3 max 448.0.
const std::vector<float> kExactFp8 = {0.0f, 1.0f, 2.0f, -1.0f, -2.0f, 0.5f,
                                      448.0f, -448.0f};

std::size_t numel(const std::vector<int64_t>& s) {
  std::size_t ne = 1;
  for (auto d : s) ne *= static_cast<std::size_t>(d);
  return ne;
}

// A tensor to write into the synthetic shard.
struct T {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> vals;   // logical values (float)
  std::vector<float> scale;  // FP8 export scales (unused by the BF16 test)
};

}  // namespace

static void test_bf16_load() {
  // Tiny model: vocab 5, H 4, I 6, L 1, heads 2, d 2 (KV=4). Build the full
  // expected tensor set in BF16, load, and verify a few values.
  const std::string dir = tmp_dir("bf16");
  {
    std::ofstream f(dir + "/config.json");
    f << kTinyConfig;
  }

  // Build tensors. Names + shapes per expected_tensors for the tiny config.
  // We'll place known values and verify they survive the round trip.
  const int H = 4, I = 6, V = 5, KV = 4;  // 2 heads * d 2
  std::vector<T> ts;
  auto mk = [&](const std::string& n, std::vector<int64_t> s, float base) {
    T t;
    t.name = n;
    t.shape = s;
    std::size_t ne = 1;
    for (auto d : s) ne *= static_cast<std::size_t>(d);
    t.vals.resize(ne);
    for (std::size_t i = 0; i < ne; ++i) t.vals[i] = base + static_cast<float>(i);
    ts.push_back(t);
  };
  mk("model.embed_tokens.weight", {V, H}, 0.1f);
  mk("model.layers.0.input_layernorm.weight", {H}, 0.5f);
  mk("model.layers.0.post_attention_layernorm.weight", {H}, 0.6f);
  mk("model.layers.0.self_attn.q_proj.weight", {H, H}, 0.2f);
  mk("model.layers.0.self_attn.k_proj.weight", {KV, H}, 0.3f);
  mk("model.layers.0.self_attn.v_proj.weight", {KV, H}, 0.4f);
  mk("model.layers.0.self_attn.o_proj.weight", {H, H}, 0.7f);
  mk("model.layers.0.mlp.gate_proj.weight", {I, H}, 0.8f);
  mk("model.layers.0.mlp.up_proj.weight", {I, H}, 0.9f);
  mk("model.layers.0.mlp.down_proj.weight", {H, I}, 1.1f);
  mk("model.norm.weight", {H}, 0.55f);
  mk("lm_head.weight", {V, H}, 0.05f);

  // Encode a single shard: header JSON + BF16 data region.
  std::string hdr = "{";
  std::string data;
  int64_t pos = 0;
  bool first = true;
  for (const auto& t : ts) {
    std::size_t ne = 1;
    for (auto d : t.shape) ne *= static_cast<std::size_t>(d);
    if (!first) hdr += ",";
    first = false;
    hdr += "\"";
    hdr += t.name;
    hdr += "\":{\"dtype\":\"BF16\",\"shape\":[";
    for (std::size_t i = 0; i < t.shape.size(); ++i)
      hdr += (i ? "," : "") + std::to_string(t.shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(pos) + "," +
           std::to_string(pos + 2 * ne) + "]}";
    for (float v : t.vals) {
      const std::uint16_t bits = bf16bits(v);
      char b[2] = {static_cast<char>(bits & 0xff), static_cast<char>((bits >> 8) & 0xff)};
      data.append(b, 2);
    }
    pos += 2 * static_cast<int64_t>(ne);
  }
  hdr += "}";
  write_st(dir, hdr, data, "model.safetensors");

  ModelWeights w;
  std::string err;
  expect(load_weights(dir, &w, &err, /*enforce_spec=*/false),
         "bf16 load_weights succeeds");
  // embed[linear 1] == 0.1 + 1 = 1.1 (row-major [V][H]); embed[linear 0] = 0.1.
  expect(std::fabs(w.embed[0] - bf16_to_f32_host(bf16bits(0.1f))) < 1e-6,
         "embed[0] round trip");
  expect(std::fabs(w.embed[1] - bf16_to_f32_host(bf16bits(1.1f))) < 1e-6,
         "embed[1] round trip");
  // lm_head last row last col == 0.05 + (V*H - 1).
  const float lm_last = 0.05f + static_cast<float>(V * H - 1);
  expect(std::fabs(w.lm_head[V * H - 1] - bf16_to_f32_host(bf16bits(lm_last))) < 1e-6,
         "lm_head last element");
  fs::remove_all(dir);
}

static void write_fp8_shard(const std::string& dir,
                            const std::vector<T>& ts, bool per_channel) {
  // Data region: for each tensor, one FP8 byte per element, then the scale
  // tensor (F32, [1] per-tensor or [row] per-channel) - the P1 export layout.
  std::string hdr = "{";
  std::string data;
  int64_t pos = 0;
  bool first = true;
  auto rec = [&](const std::string& name, const std::vector<int64_t>& shape,
                 const char* dtype, int64_t begin, int64_t end) {
    if (!first) hdr += ",";
    first = false;
    hdr += "\"";
    hdr += name;
    hdr += "\":{\"dtype\":\"" + std::string(dtype) + "\",\"shape\":[";
    for (std::size_t i = 0; i < shape.size(); ++i)
      hdr += (i ? "," : "") + std::to_string(shape[i]);
    hdr += "],\"data_offsets\":[" + std::to_string(begin) + "," +
           std::to_string(end) + "]}";
  };
  for (const auto& t : ts) {
    const std::size_t ne = numel(t.shape);
    const int64_t row = t.shape.empty() ? 1 : t.shape[0];
    const int64_t fp8_bytes = static_cast<int64_t>(ne);
    const int64_t w_begin = pos;
    for (float v : t.vals) {
      const std::uint8_t q = fp8_e4m3_encode(v, nullptr);
      data += static_cast<char>(q);
    }
    pos += fp8_bytes;
    rec(t.name, t.shape, "F8_E4M3", w_begin, w_begin + fp8_bytes);
    // scale tensor
    const int64_t scale_elems = per_channel ? row : 1;
    const int64_t s_begin = pos;
    for (int64_t r = 0; r < scale_elems; ++r) {
      const float sc = per_channel ? t.scale[r] : t.scale[0];
      std::uint32_t bits;
      std::memcpy(&bits, &sc, 4);
      for (int b = 0; b < 4; ++b)
        data += static_cast<char>((bits >> (8 * b)) & 0xff);
    }
    pos += 4 * scale_elems;
    rec(t.name + "__scale", {scale_elems}, "F32", s_begin, s_begin + 4 * scale_elems);
  }
  hdr += "}";
  write_st(dir, hdr, data, "model_fp8.safetensors");
}

static void test_fp8_load_per_tensor() {
  // FP8 e4m3 export, per-tensor scale. Every weight value is drawn from the
  // e4m3-exact set and each scale is 1.0, so dequant must round-trip exactly:
  // loaded = fp8_to_f32(encode(value)) * scale == value. This pins the
  // F8_E4M3 + [1]-element __scale path in load_weights.
  const std::string dir = tmp_dir("fp8pt");
  {
    std::ofstream f(dir + "/config.json");
    f << kTinyConfig;
  }
  const int H = 4, I = 6, V = 5, KV = 4;
  std::vector<T> ts;
  auto mk = [&](const std::string& n, std::vector<int64_t> s) {
    T t;
    t.name = n;
    t.shape = s;
    std::size_t ne = numel(s);
    t.vals.resize(ne);
    for (std::size_t i = 0; i < ne; ++i) t.vals[i] = kExactFp8[i % kExactFp8.size()];
    t.scale.assign(s.empty() ? 1 : s[0], 1.0f);  // only scale[0] is used
    ts.push_back(t);
  };
  mk("model.embed_tokens.weight", {V, H});
  mk("model.layers.0.input_layernorm.weight", {H});
  mk("model.layers.0.post_attention_layernorm.weight", {H});
  mk("model.layers.0.self_attn.q_proj.weight", {H, H});
  mk("model.layers.0.self_attn.k_proj.weight", {KV, H});
  mk("model.layers.0.self_attn.v_proj.weight", {KV, H});
  mk("model.layers.0.self_attn.o_proj.weight", {H, H});
  mk("model.layers.0.mlp.gate_proj.weight", {I, H});
  mk("model.layers.0.mlp.up_proj.weight", {I, H});
  mk("model.layers.0.mlp.down_proj.weight", {H, I});
  mk("model.norm.weight", {H});
  mk("lm_head.weight", {V, H});
  write_fp8_shard(dir, ts, /*per_channel=*/false);

  ModelWeights w;
  std::string err;
  expect(load_weights(dir, &w, &err, /*enforce_spec=*/false),
         "fp8 per-tensor load_weights succeeds");
  // lm_head row-major [V][H]; element (r,c) == kExactFp8[(r*H+c) % 8].
  const float* ref = &kExactFp8[0];
  bool ok = true;
  for (std::size_t i = 0; i < (std::size_t)V * H; ++i)
    if (w.lm_head[i] != ref[i % kExactFp8.size()]) ok = false;
  expect(ok, "fp8 per-tensor lm_head round-trips exactly");
  // q_proj [H][H] dequanted row-major.
  for (std::size_t i = 0; i < (std::size_t)H * H; ++i)
    if (w.layers[(std::size_t)0 * w.layer_bytes + w.off_q + i] !=
        ref[i % kExactFp8.size()])
      ok = false;
  expect(ok, "fp8 per-tensor q_proj round-trips exactly");
  fs::remove_all(dir);
}

static void test_fp8_load_per_channel() {
  // FP8 e4m3 export, per-channel scale. Scale is a power of two per row so
  // value/scale stays e4m3-exact, and dequant must apply the per-row scale:
  // loaded[r][c] == value[r][c] * scale[r]. This pins the [row]-element
  // __scale path (the loader's per_channel branch, scale_elems > 1).
  const std::string dir = tmp_dir("fp8pc");
  {
    std::ofstream f(dir + "/config.json");
    f << kTinyConfig;
  }
  const int H = 4, I = 6, V = 5, KV = 4;
  std::vector<T> ts;
  auto mk = [&](const std::string& n, std::vector<int64_t> s) {
    T t;
    t.name = n;
    t.shape = s;
    std::size_t ne = numel(s);
    const int64_t row = s.empty() ? 1 : s[0];
    t.vals.resize(ne);
    t.scale.assign(static_cast<std::size_t>(row), 0.0f);
    for (int64_t r = 0; r < row; ++r)
      t.scale[static_cast<std::size_t>(r)] =
          std::ldexp(1.0f, static_cast<int>(r % 4) - 1);  // 2^-1..2^2
    for (std::size_t i = 0; i < ne; ++i)
      t.vals[i] = kExactFp8[i % kExactFp8.size()];
    ts.push_back(t);
  };
  mk("model.embed_tokens.weight", {V, H});
  mk("model.layers.0.input_layernorm.weight", {H});
  mk("model.layers.0.post_attention_layernorm.weight", {H});
  mk("model.layers.0.self_attn.q_proj.weight", {H, H});
  mk("model.layers.0.self_attn.k_proj.weight", {KV, H});
  mk("model.layers.0.self_attn.v_proj.weight", {KV, H});
  mk("model.layers.0.self_attn.o_proj.weight", {H, H});
  mk("model.layers.0.mlp.gate_proj.weight", {I, H});
  mk("model.layers.0.mlp.up_proj.weight", {I, H});
  mk("model.layers.0.mlp.down_proj.weight", {H, I});
  mk("model.norm.weight", {H});
  mk("lm_head.weight", {V, H});
  write_fp8_shard(dir, ts, /*per_channel=*/true);

  ModelWeights w;
  std::string err;
  expect(load_weights(dir, &w, &err, /*enforce_spec=*/false),
         "fp8 per-channel load_weights succeeds");
  // q_proj [H][H]: row r scale = 2^(r%4 - 1); value = kExactFp8[(r*H+c)%8].
  const float* ref = &kExactFp8[0];
  const float* qp = w.layers.data() + w.off_q;
  bool ok = true;
  for (int r = 0; r < H; ++r)
    for (int c = 0; c < H; ++c) {
      const float scale = std::ldexp(1.0f, static_cast<int>(r % 4) - 1);
      const float want = ref[(r * H + c) % kExactFp8.size()] * scale;
      if (qp[(std::size_t)r * H + c] != want) ok = false;
    }
  expect(ok, "fp8 per-channel q_proj applies per-row scale");
  // embed [V][H]: row-scale dequant.
  for (int r = 0; r < V; ++r)
    for (int c = 0; c < H; ++c) {
      const float scale = std::ldexp(1.0f, static_cast<int>(r % 4) - 1);
      const float want = ref[(r * H + c) % kExactFp8.size()] * scale;
      if (w.embed[(std::size_t)r * H + c] != want) ok = false;
    }
  expect(ok, "fp8 per-channel embed applies per-row scale");
  fs::remove_all(dir);
}

int main() {
  test_bf16_load();
  test_fp8_load_per_tensor();
  test_fp8_load_per_channel();
  if (g_failures) {
    std::printf("weights: %d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("weights: all checks passed\n");
  return 0;
}
