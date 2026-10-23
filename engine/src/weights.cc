// Phase 3 (Integration): model weight loading for inference (impl).
//
// See weights.h. Host-only, stdlib only. Reads a checkpoint directory
// (original BF16 or an FP8 export from `granite4-checkpoint export`) and
// produces the float weight layout `forward_logits_f32` consumes.

#include "weights.h"

#include "checkpoint.h"
#include "config_validation.h"
#include "kernel_math.h"
#include "safetensors.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace granite4 {

namespace fs = std::filesystem;

namespace {

std::vector<std::string> list_shards(const std::string& dir, std::string* error) {
  std::vector<std::string> out;
  const std::string suffix = ".safetensors";
  for (const auto& e : fs::directory_iterator(dir)) {
    const std::string name = e.path().filename().string();
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
      out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  (void)error;
  return out;
}

std::vector<char> read_range(const std::string& path, std::uint64_t begin,
                             std::uint64_t len, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    *error = "cannot open " + path;
    return {};
  }
  f.seekg(static_cast<std::streamoff>(begin), std::ios::beg);
  std::vector<char> buf(len);
  if (len > 0) f.read(buf.data(), static_cast<std::streamsize>(len));
  if (!f) {
    *error = "short read from " + path;
    return {};
  }
  return buf;
}

float bf16_bytes_to_float(const char* b) {
  const std::uint16_t bits =
      static_cast<std::uint16_t>(static_cast<unsigned char>(b[0])) |
      (static_cast<std::uint16_t>(static_cast<unsigned char>(b[1])) << 8);
  return Bf16ToFloat::from_bits(bits);
}

}  // namespace

bool load_weights(const std::string& dir, ModelWeights* out, std::string* error,
                  bool enforce_spec) {
  // 1. Parse + strictly validate config.json (reject before reading weights).
  std::string cfg_bytes;
  {
    std::ifstream f((dir + "/config.json").c_str(), std::ios::binary);
    if (!f) {
      *error = "config.json not found in " + dir;
      return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    cfg_bytes.resize(static_cast<std::size_t>(n));
    f.read(cfg_bytes.data(), n);
    if (!f) {
      *error = "cannot read config.json";
      return false;
    }
  }
  ParsedCheckpointConfig pcfg;
  std::string perr;
  if (!parse_checkpoint_config(cfg_bytes.data(), cfg_bytes.size(), &pcfg, &perr)) {
    *error = "config: " + perr;
    return false;
  }
  if (enforce_spec) {
    if (!config_matches_spec(pcfg)) {
      *error = "config does not match the Granite 4.2 8B spec";
      if (!pcfg.deviations.empty()) *error += ": " + pcfg.deviations.front();
      return false;
    }
  } else if (!is_structurally_consistent(pcfg.model)) {
    *error = "config is not structurally consistent";
    return false;
  }
  const ModelConfig cfg = pcfg.model;

  // 2. Per-layer block layout (shared with forward_logits_f32).
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size,
      L = cfg.num_layers, V = cfg.vocab_size;
  const int64_t KV = cfg.num_key_value_heads * cfg.head_dim;
  const std::size_t q_size = (std::size_t)H * H;
  const std::size_t kv_size = (std::size_t)KV * H;
  const std::size_t io_size = (std::size_t)I * H;
  const std::size_t o_size = (std::size_t)H * H;
  const std::size_t down_size = (std::size_t)H * I;
  out->off_q = 2 * (std::size_t)H;
  out->off_k = out->off_q + q_size;
  out->off_v = out->off_k + kv_size;
  out->off_o = out->off_v + kv_size;
  out->off_gate = out->off_o + o_size;
  out->off_up = out->off_gate + io_size;
  out->off_down = out->off_up + io_size;
  out->layer_bytes = out->off_down + down_size;
  out->embed.assign((std::size_t)V * H, 0.0f);
  out->layers.assign((std::size_t)L * out->layer_bytes, 0.0f);
  out->final_norm.assign((std::size_t)H, 0.0f);
  out->lm_head.assign((std::size_t)V * H, 0.0f);
  out->model = cfg;
  out->bytes_read = 0;

  // 3. Locate every tensor across the shards.
  std::string lerr;
  const std::vector<std::string> shards = list_shards(dir, &lerr);
  if (shards.empty()) {
    *error = "no .safetensors shards in " + dir;
    return false;
  }
  struct Loc {
    std::string shard_path;
    int64_t header_bytes = 0;
    int64_t off_begin = 0, off_end = 0;
    std::string dtype;
    int64_t row = 0, col = 1;
    std::uint64_t numel_elems = 0;  // row * col
    std::uint64_t nbytes = 0;       // offset_end - offset_begin
  };
  std::map<std::string, Loc> loc;
  for (const auto& name : shards) {
    std::string path = dir + "/" + name;
    TensorManifest m;
    std::string merr;
    if (!parse_safetensors_file(path, &m, &merr)) {
      *error = name + ": " + merr;
      return false;
    }
    for (const auto& t : m.tensors) {
      const std::uint64_t row = t.shape.empty() ? 1 : (std::uint64_t)t.shape[0];
      const std::uint64_t col = t.shape.size() > 1 ? (std::uint64_t)t.shape[1] : 1;
      loc[t.name] = {path, m.header_bytes, t.offset_begin, t.offset_end, t.dtype,
                     (int64_t)row, (int64_t)col, row * col,
                     (std::uint64_t)(t.offset_end - t.offset_begin)};
    }
  }

  // 4. Read one tensor (and its FP8 scale) into a contiguous float slot.
  auto load_into = [&](const std::string& name, float* dst, std::size_t nelem) {
    auto it = loc.find(name);
    if (it == loc.end()) {
      *error = "missing tensor: " + name;
      return false;
    }
    const Loc& w = it->second;
    if (w.numel_elems != nelem) {
      *error = name + ": unexpected numel " + std::to_string(w.numel_elems) +
               " (expected " + std::to_string(nelem) + ")";
      return false;
    }
    const std::uint64_t begin =
        8 + (std::uint64_t)w.header_bytes + (std::uint64_t)w.off_begin;
    std::string rerr;
    if (w.dtype == "BF16") {
      std::vector<char> raw = read_range(w.shard_path, begin, w.nbytes, &rerr);
      if (raw.empty()) {
        *error = name + ": " + rerr;
        return false;
      }
      out->bytes_read += w.nbytes;
      for (std::size_t i = 0; i < nelem; ++i) dst[i] = bf16_bytes_to_float(raw.data() + 2 * i);
      return true;
    }
    if (w.dtype == "F8_E4M3") {
      std::vector<char> raw = read_range(w.shard_path, begin, w.nbytes, &rerr);
      if (raw.empty()) {
        *error = name + ": " + rerr;
        return false;
      }
      out->bytes_read += w.nbytes;
      auto sit = loc.find(name + "__scale");
      if (sit == loc.end()) {
        *error = name + ": missing __scale tensor";
        return false;
      }
      const Loc& s = sit->second;
      const std::uint64_t sbegin =
          8 + (std::uint64_t)s.header_bytes + (std::uint64_t)s.off_begin;
      const std::uint64_t sbytes = (std::uint64_t)(s.off_end - s.off_begin);
      std::vector<char> sraw = read_range(s.shard_path, sbegin, sbytes, &rerr);
      if (sraw.empty()) {
        *error = name + "__scale: " + rerr;
        return false;
      }
      out->bytes_read += sbytes;
      const std::size_t scale_elems = sraw.size() / 4;
      std::vector<float> scale(scale_elems);
      for (std::size_t i = 0; i < scale_elems; ++i)
        std::memcpy(&scale[i], sraw.data() + 4 * i, 4);
      const bool per_channel = scale_elems > 1;
      const std::size_t row = (std::size_t)w.row;
      const std::size_t col = row > 0 ? (nelem / row) : nelem;
      for (std::size_t r = 0; r < row; ++r) {
        const float sc = per_channel ? scale[r] : scale[0];
        for (std::size_t c = 0; c < col; ++c)
          dst[r * col + c] =
              fp8_to_f32_host(static_cast<std::uint8_t>(raw[r * col + c])) * sc;
      }
      return true;
    }
    *error = name + ": unsupported dtype " + w.dtype;
    return false;
  };

  auto layer_base = [&](int layer) {
    return out->layers.data() + (std::size_t)layer * out->layer_bytes;
  };

  if (!load_into("model.embed_tokens.weight", out->embed.data(), (std::size_t)V * H))
    return false;
  if (!load_into("model.norm.weight", out->final_norm.data(), (std::size_t)H))
    return false;
  if (!load_into("lm_head.weight", out->lm_head.data(), (std::size_t)V * H))
    return false;

  for (int layer = 0; layer < L; ++layer) {
    float* lb = layer_base(layer);
    const std::string prefix = "model.layers." + std::to_string(layer) + ".";
    if (!load_into(prefix + "input_layernorm.weight", lb + 0, (std::size_t)H)) return false;
    if (!load_into(prefix + "post_attention_layernorm.weight", lb + (std::size_t)H,
                   (std::size_t)H))
      return false;
    if (!load_into(prefix + "self_attn.q_proj.weight", lb + out->off_q,
                   (std::size_t)H * H))
      return false;
    if (!load_into(prefix + "self_attn.k_proj.weight", lb + out->off_k,
                   (std::size_t)KV * H))
      return false;
    if (!load_into(prefix + "self_attn.v_proj.weight", lb + out->off_v,
                   (std::size_t)KV * H))
      return false;
    if (!load_into(prefix + "self_attn.o_proj.weight", lb + out->off_o,
                   (std::size_t)H * H))
      return false;
    if (!load_into(prefix + "mlp.gate_proj.weight", lb + out->off_gate,
                   (std::size_t)I * H))
      return false;
    if (!load_into(prefix + "mlp.up_proj.weight", lb + out->off_up,
                   (std::size_t)I * H))
      return false;
    if (!load_into(prefix + "mlp.down_proj.weight", lb + out->off_down,
                   (std::size_t)H * I))
      return false;
  }
  return true;
}

}  // namespace granite4
