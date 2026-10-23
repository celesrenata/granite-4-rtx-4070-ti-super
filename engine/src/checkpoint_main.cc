// Phase 1 (Weights): strict config validation, expected tensor set,
// checksummed shard verification, and the FP8 e4m3 quantization export.
//
// Host-only; the full checkpoint is streamed in bounded chunks - at no
// point is more than one tensor held in memory at a time.
#include "checkpoint.h"

#include "json.h"
#include "safetensors.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <map>
#include <sstream>

namespace granite4 {

namespace {

bool read_file(const std::string& path, std::string* out, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  if (f.fail()) {
    if (error) *error = "read error on " + path;
    return false;
  }
  *out = ss.str();
  return true;
}

std::string path_join(const std::string& dir, const std::string& name) {
  if (dir.empty() || dir[dir.size() - 1] == '/') return dir + name;
  return dir + "/" + name;
}

// Sorted *.safetensors file names in dir (deterministic load order).
bool list_safetensors(const std::string& dir, std::vector<std::string>* out,
                      std::string* error) {
  out->clear();
  const std::string suffix = ".safetensors";
  DIR* d = ::opendir(dir.c_str());
  if (d == nullptr) {
    if (error) *error = "cannot open directory " + dir;
    return false;
  }
  struct dirent* ent;
  while ((ent = ::readdir(d)) != nullptr) {
    const std::string name = ent->d_name;
    if (name.size() > suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
      out->push_back(name);
    }
  }
  ::closedir(d);
  std::sort(out->begin(), out->end());
  return true;
}

// Stream-hash [begin, begin+len) of a file with bounded memory.
bool hash_file_range(const std::string& path, int64_t begin, int64_t len, Sha256* h,
                     std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + path;
    return false;
  }
  if (!f.seekg(static_cast<std::streamoff>(begin), std::ios::beg)) {
    if (error) *error = "seek failed on " + path;
    return false;
  }
  char buf[16u * 1024u];
  int64_t remaining = len;
  while (remaining > 0) {
    const std::size_t chunk = static_cast<std::size_t>(
        remaining < static_cast<int64_t>(sizeof buf) ? remaining
                                                     : static_cast<int64_t>(sizeof buf));
    f.read(buf, static_cast<std::streamsize>(chunk));
    const std::streamsize got = f.gcount();
    if (got <= 0) {
      if (error) *error = "short read on " + path;
      return false;
    }
    h->update(buf, static_cast<std::size_t>(got));
    remaining -= static_cast<int64_t>(got);
  }
  return f.good();
}

int64_t shape_numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// Decode BF16 elements one at a time while computing the max |value| of a
// row (row-major, `col` elements).
double max_abs_bf16_row(const std::uint8_t* row, int64_t col) {
  double mx = 0.0;
  for (int64_t c = 0; c < col; ++c) {
    const std::uint16_t bits = static_cast<std::uint16_t>(row[2 * c]) |
                               (static_cast<std::uint16_t>(row[2 * c + 1]) << 8);
    const float f = Bf16ToFloat::from_bits(bits);
    const double a = f < 0.0 ? -static_cast<double>(f) : static_cast<double>(f);
    if (a > mx) mx = a;
  }
  return mx;
}

// Encode one scaled element: v/scale rounded to e4m3 (scale > 0).
std::uint8_t encode_scaled(const std::uint16_t bits, float scale, bool* saturated) {
  const float f = Bf16ToFloat::from_bits(bits);
  const double v = static_cast<double>(f) / static_cast<double>(scale);
  return fp8_e4m3_encode(v, saturated);
}

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      case '\r':
        out += "\\r";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned char>(c));
          out += b;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string json_string_array(const std::vector<std::string>& v) {
  std::string out = "[";
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) out += ",";
    out += "\"" + json_escape(v[i]) + "\"";
  }
  out += "]";
  return out;
}

// ---------------------------------------------------------------------------
// config.json field expectations
// ---------------------------------------------------------------------------

bool expect_int_field(const json::Value& v, const char* key, int64_t want,
                      std::vector<std::string>* issues) {
  const json::Value* f = v.find(key);
  if (f == nullptr) {
    issues->push_back(std::string(key) + ": missing (expected " + std::to_string(want) +
                       ")");
    return false;
  }
  if (f->kind != json::Value::Kind::Int) {
    issues->push_back(std::string(key) + ": not an integer (expected " + std::to_string(want) +
                       ")");
    return false;
  }
  if (f->int_value != want) {
    issues->push_back(std::string(key) + ": expected " + std::to_string(want) + ", got " +
                       std::to_string(f->int_value));
    return false;
  }
  return true;
}

bool expect_double_field(const json::Value& v, const char* key, double want,
                         std::vector<std::string>* issues) {
  const json::Value* f = v.find(key);
  if (f == nullptr) {
    issues->push_back(std::string(key) + ": missing (expected " + std::to_string(want) + ")");
    return false;
  }
  double got = 0.0;
  if (f->kind == json::Value::Kind::Double) got = f->double_value;
  else if (f->kind == json::Value::Kind::Int) got = static_cast<double>(f->int_value);
  else {
    issues->push_back(std::string(key) + ": not a number (expected " + std::to_string(want) +
                       ")");
    return false;
  }
  if (got != want) {
    std::ostringstream os;
    os << key << ": expected " << want << ", got " << got;
    issues->push_back(os.str());
    return false;
  }
  return true;
}

bool expect_bool_field(const json::Value& v, const char* key, bool want,
                       std::vector<std::string>* issues) {
  const json::Value* f = v.find(key);
  if (f == nullptr) {
    issues->push_back(std::string(key) + ": missing (expected " +
                       std::string(want ? "true" : "false") + ")");
    return false;
  }
  if (f->kind != json::Value::Kind::Bool || f->bool_value != want) {
    const std::string got = f->kind == json::Value::Kind::Bool
                                ? std::string(f->bool_value ? "true" : "false")
                                : std::string("non-boolean");
    issues->push_back(std::string(key) + ": expected " + std::string(want ? "true" : "false") +
                       ", got " + got);
    return false;
  }
  return true;
}

bool expect_string_field(const json::Value& v, const char* key, const char* want,
                         std::vector<std::string>* issues) {
  const json::Value* f = v.find(key);
  if (f == nullptr) {
    issues->push_back(std::string(key) + ": missing (expected \"" + want + "\")");
    return false;
  }
  if (f->kind != json::Value::Kind::String || f->str != want) {
    const std::string got = f->kind == json::Value::Kind::String ? f->str : std::string("non-string");
    issues->push_back(std::string(key) + ": expected \"" + want + "\", got \"" + got + "\"");
    return false;
  }
  return true;
}

int64_t get_int_or(const json::Value& v, const char* key, int64_t def) {
  const json::Value* f = v.find(key);
  if (f == nullptr) return def;
  if (f->kind == json::Value::Kind::Int) return f->int_value;
  if (f->kind == json::Value::Kind::Double) return static_cast<int64_t>(f->double_value);
  return def;
}

double get_double_or(const json::Value& v, const char* key, double def) {
  const json::Value* f = v.find(key);
  if (f == nullptr) return def;
  if (f->kind == json::Value::Kind::Double) return f->double_value;
  if (f->kind == json::Value::Kind::Int) return static_cast<double>(f->int_value);
  return def;
}

std::string get_string_or(const json::Value& v, const char* key) {
  const json::Value* f = v.find(key);
  return (f != nullptr && f->kind == json::Value::Kind::String) ? f->str : std::string();
}

}  // namespace

// ---------------------------------------------------------------------------
// expected tensor set (the spec, derived from ModelConfig)
// ---------------------------------------------------------------------------

namespace {
// Build a per-layer tensor name. `scope` is the sub-module ("self_attn",
// "mlp") or "" for the layer's own layernorms. `proj` is the full parameter
// suffix *including* its ".weight" (e.g. "q_proj.weight",
// "input_layernorm.weight"), so the result is exactly the safetensors key:
//   ""      -> model.layers.{L}.input_layernorm.weight
//   "mlp"   -> model.layers.{L}.mlp.gate_proj.weight
std::string layer_tensor_name(int layer, const char* scope, const char* proj) {
  std::string s = "model.layers.";
  s += std::to_string(layer);
  s += ".";
  if (scope[0] != '\0') {
    s += scope;
    s += ".";
  }
  s += proj;
  return s;
}
}  // namespace

std::vector<ExpectedTensor> expected_tensors(const ModelConfig& cfg) {
  const int64_t H = static_cast<int64_t>(cfg.hidden_size);
  const int64_t I = static_cast<int64_t>(cfg.intermediate_size);
  const int64_t V = static_cast<int64_t>(cfg.vocab_size);
  const int64_t KV = static_cast<int64_t>(cfg.num_key_value_heads) *
                     static_cast<int64_t>(cfg.head_dim);
  std::vector<ExpectedTensor> out;
  out.push_back({"model.embed_tokens.weight", {V, H}});
  for (int layer = 0; layer < static_cast<int>(cfg.num_layers); ++layer) {
    out.push_back({layer_tensor_name(layer, "", "input_layernorm.weight"), {H}});
    out.push_back({layer_tensor_name(layer, "", "post_attention_layernorm.weight"), {H}});
    out.push_back({layer_tensor_name(layer, "self_attn", "q_proj.weight"), {H, H}});
    out.push_back({layer_tensor_name(layer, "self_attn", "k_proj.weight"), {KV, H}});
    out.push_back({layer_tensor_name(layer, "self_attn", "v_proj.weight"), {KV, H}});
    out.push_back({layer_tensor_name(layer, "self_attn", "o_proj.weight"), {H, H}});
    out.push_back({layer_tensor_name(layer, "mlp", "gate_proj.weight"), {I, H}});
    out.push_back({layer_tensor_name(layer, "mlp", "up_proj.weight"), {I, H}});
    out.push_back({layer_tensor_name(layer, "mlp", "down_proj.weight"), {H, I}});
  }
  out.push_back({"model.norm.weight", {H}});
  out.push_back({"lm_head.weight", {V, H}});
  return out;
}

int64_t expected_bf16_bytes(const ModelConfig& cfg) {
  int64_t total = 0;
  for (const auto& t : expected_tensors(cfg)) total += shape_numel(t.shape) * 2;
  return total;
}

// ---------------------------------------------------------------------------
// strict config validation
// ---------------------------------------------------------------------------

bool parse_checkpoint_config(const char* data, std::size_t nbytes,
                             ParsedCheckpointConfig* out, std::string* error) {
  if (data == nullptr || out == nullptr) {
    if (error) *error = "invalid arguments";
    return false;
  }
  out->deviations.clear();
  json::Value root;
  std::string err;
  if (!json::parse(data, nbytes, &root, &err)) {
    if (error) *error = "config.json is not valid JSON: " + err;
    return false;
  }
  if (root.kind != json::Value::Kind::Object) {
    if (error) *error = "config.json must be a JSON object";
    return false;
  }
  auto& d = out->deviations;
  expect_int_field(root, "num_hidden_layers", 40, &d);
  expect_int_field(root, "hidden_size", 4096, &d);
  expect_int_field(root, "intermediate_size", 12800, &d);
  expect_int_field(root, "num_attention_heads", 32, &d);
  expect_int_field(root, "num_key_value_heads", 8, &d);
  expect_int_field(root, "vocab_size", 100352, &d);
  expect_int_field(root, "max_position_embeddings", 131072, &d);
  // head_dim: not a field in this checkpoint's config.json - derived.
  const int64_t hidden = get_int_or(root, "hidden_size", 0);
  const int64_t heads = get_int_or(root, "num_attention_heads", 0);
  int64_t head_dim = 0;
  if (hidden > 0 && heads > 0) {
    if (hidden % heads == 0) head_dim = hidden / heads;
    else d.push_back("head_dim: hidden_size is not divisible by num_attention_heads");
  }
  if (head_dim != 128) {
    d.push_back("head_dim: expected 128, got " +
               (head_dim == 0 ? std::string("undefined") : std::to_string(head_dim)));
  }
  expect_double_field(root, "rms_norm_eps", 1e-5, &d);
  expect_double_field(root, "attention_multiplier", 1.0 / 128.0, &d);
  expect_double_field(root, "residual_multiplier", 1.0, &d);
  expect_double_field(root, "embedding_multiplier", 1.0, &d);
  expect_double_field(root, "logits_scaling", 1.0, &d);
  expect_int_field(root, "rope_theta", 10000000, &d);
  const json::Value* rs = root.find("rope_scaling");
  if (rs == nullptr)
    d.push_back("rope_scaling: missing (expected null)");
  else if (rs->kind != json::Value::Kind::Null)
    d.push_back("rope_scaling: expected null, got a non-null value");
  expect_int_field(root, "bos_token_id", 100283, &d);
  expect_int_field(root, "eos_token_id", 100257, &d);
  expect_int_field(root, "pad_token_id", 100257, &d);
  expect_string_field(root, "model_type", "granite", &d);
  expect_string_field(root, "torch_dtype", "bfloat16", &d);
  expect_string_field(root, "dtype", "bfloat16", &d);
  expect_string_field(root, "hidden_act", "silu", &d);
  expect_bool_field(root, "attention_bias", false, &d);
  expect_bool_field(root, "mlp_bias", false, &d);
  expect_bool_field(root, "use_cache", true, &d);
  expect_bool_field(root, "tie_word_embeddings", false, &d);
  const json::Value* arch = root.find("architectures");
  if (arch == nullptr || arch->kind != json::Value::Kind::Array || arch->arr.size() != 1 ||
      arch->arr[0].kind != json::Value::Kind::String || arch->arr[0].str != "GraniteForCausalLM") {
    d.push_back("architectures: expected [\"GraniteForCausalLM\"]");
  }
  // Populate the parsed view (values as read, even when off-spec).
  out->model.num_layers = static_cast<std::size_t>(get_int_or(root, "num_hidden_layers", 0));
  out->model.hidden_size = static_cast<std::size_t>(hidden);
  out->model.intermediate_size =
      static_cast<std::size_t>(get_int_or(root, "intermediate_size", 0));
  out->model.num_attention_heads = static_cast<std::size_t>(heads);
  out->model.num_key_value_heads =
      static_cast<std::size_t>(get_int_or(root, "num_key_value_heads", 0));
  out->model.head_dim = static_cast<std::size_t>(head_dim > 0 ? head_dim : 0);
  out->model.vocab_size = static_cast<std::size_t>(get_int_or(root, "vocab_size", 0));
  out->model.max_context_length =
      static_cast<std::size_t>(get_int_or(root, "max_position_embeddings", 0));
  const json::Value* tie = root.find("tie_word_embeddings");
  out->tie_word_embeddings =
      tie != nullptr && tie->kind == json::Value::Kind::Bool && tie->bool_value;
  out->bos_token_id = static_cast<std::uint32_t>(std::max<int64_t>(0, get_int_or(root, "bos_token_id", 0)));
  out->eos_token_id = static_cast<std::uint32_t>(std::max<int64_t>(0, get_int_or(root, "eos_token_id", 0)));
  out->pad_token_id = static_cast<std::uint32_t>(std::max<int64_t>(0, get_int_or(root, "pad_token_id", 0)));
  out->rms_norm_eps = get_double_or(root, "rms_norm_eps", 0.0);
  out->attention_multiplier = get_double_or(root, "attention_multiplier", 0.0);
  out->residual_multiplier = get_double_or(root, "residual_multiplier", 0.0);
  out->embedding_multiplier = get_double_or(root, "embedding_multiplier", 0.0);
  out->logits_scaling = get_double_or(root, "logits_scaling", 0.0);
  out->rope_theta = static_cast<std::uint64_t>(std::max<int64_t>(0, get_int_or(root, "rope_theta", 0)));
  out->model_type = get_string_or(root, "model_type");
  out->torch_dtype = get_string_or(root, "torch_dtype");
  out->dtype = get_string_or(root, "dtype");
  out->hidden_act = get_string_or(root, "hidden_act");
  out->transformers_version = get_string_or(root, "transformers_version");  // recorded, not checked
  out->architectures =
      (arch != nullptr && arch->kind == json::Value::Kind::Array && arch->arr.size() == 1 &&
       arch->arr[0].kind == json::Value::Kind::String)
          ? arch->arr[0].str
          : "";
  return true;
}

bool config_matches_spec(const ParsedCheckpointConfig& cfg) { return cfg.deviations.empty(); }

// ---------------------------------------------------------------------------
// checksummed shard verification
// ---------------------------------------------------------------------------

struct ShardView {
  std::string path;
  std::string name;
  int64_t file_size = 0;
  std::string sha256;
  TensorManifest manifest;
};

bool verify_shards(const char* dir, CheckpointReport* report, std::string* error) {
  if (report == nullptr) {
    if (error) *error = "null report";
    return false;
  }
  std::vector<std::string> shard_names;
  if (!list_safetensors(dir, &shard_names, error)) return false;
  if (shard_names.empty()) {
    report->error = "no *.safetensors files in " + std::string(dir);
    if (error) *error = report->error;
    return false;
  }
  const ModelConfig cfg = report->config_present ? report->config.model : kModelConfig;
  const std::vector<ExpectedTensor> expected = expected_tensors(cfg);

  // name -> expected index (binary search on the sorted name list)
  std::vector<std::pair<std::string, int>> by_name;
  for (int i = 0; i < static_cast<int>(expected.size()); ++i)
    by_name.emplace_back(expected[i].name, i);
  std::sort(by_name.begin(), by_name.end());
  auto find_expected = [&](const std::string& name) -> int {
    auto it = std::lower_bound(
        by_name.begin(), by_name.end(), name,
        [](const auto& p, const std::string& n) { return p.first < n; });
    return (it != by_name.end() && it->first == name) ? it->second : -1;
  };

  std::vector<bool> seen(expected.size(), false);
  for (const auto& name : shard_names) {
    const std::string path = path_join(dir, name);
    {
      Sha256 h;
      std::string rerr;
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      int64_t size = -1;
      if (f) {
        const auto t = f.tellg();
        if (t != std::istream::pos_type(static_cast<std::streamoff>(-1))) {
          size = static_cast<int64_t>(t);
        }
      }
      if (size < 0) {
        report->error = "cannot open shard " + path;
        if (error) *error = report->error;
        return false;
      }
      if (!hash_file_range(path, 0, size, &h, &rerr)) {
        report->error = name + ": " + rerr;
        if (error) *error = report->error;
        return false;
      }
      report->shard_digests.push_back({name, h.hex(), size});
    }
    TensorManifest m;
    std::string merr;
    if (!parse_safetensors_file(path, &m, &merr)) {
      report->error = name + ": " + merr;
      if (error) *error = report->error;
      return false;
    }
    for (const auto& t : m.tensors) {
      Sha256 th;
      std::string rerr;
      if (!hash_file_range(path, 8 + m.header_bytes + t.offset_begin, t.nbytes, &th, &rerr)) {
        report->error = t.name + ": " + rerr;
        if (error) *error = report->error;
        return false;
      }
      report->tensor_digests.push_back({t.name, th.hex()});
      report->tensors += 1;
      report->total_numel += t.numel;
      report->total_bytes += t.nbytes;
      const int idx = find_expected(t.name);
      if (idx < 0) {
        report->issues.push_back("unexpected tensor: " + t.name);
        continue;
      }
      seen[idx] = true;
      const ExpectedTensor& e = expected[idx];
      if (t.shape != e.shape) {
        std::ostringstream os;
        os << t.name << ": shape [";
        for (std::size_t i = 0; i < t.shape.size(); ++i) os << (i ? ", " : "") << t.shape[i];
        os << "] expected [";
        for (std::size_t i = 0; i < e.shape.size(); ++i) os << (i ? ", " : "") << e.shape[i];
        os << "]";
        report->issues.push_back(os.str());
      }
      if (t.dtype != "BF16")
        report->issues.push_back(t.name + ": dtype " + t.dtype + ", expected BF16");
    }
    int64_t covered = 0;
    for (const auto& t : m.tensors) covered += t.nbytes;
    if (covered != m.data_region_bytes) {
      std::ostringstream os;
      os << name << ": tensors cover " << covered << " of " << m.data_region_bytes
         << " data-region bytes";
      report->issues.push_back(os.str());
    }
  }
  for (std::size_t i = 0; i < seen.size(); ++i)
    if (!seen[i]) report->issues.push_back("missing tensor: " + expected[i].name);
  report->embed_tied_to_lm_head = report->config.tie_word_embeddings;
  report->shards = shard_names.size();
  return true;
}

// Cross-check the index file against the parsed shards.
bool index_cross_check(const char* dir, CheckpointReport* report, std::string* error) {
  std::string idx_text;
  if (!read_file(path_join(dir, "model.safetensors.index.json"), &idx_text, error)) {
    report->notes.push_back("model.safetensors.index.json: not present (skipped)");
    return true;
  }
  json::Value root;
  std::string jerr;
  if (!json::parse(idx_text.data(), idx_text.size(), &root, &jerr)) {
    report->issues.push_back("model.safetensors.index.json: " + jerr);
    return true;
  }
  const json::Value* wmap = root.find("weight_map");
  if (wmap == nullptr || wmap->kind != json::Value::Kind::Object) {
    report->issues.push_back("index weight_map: missing or not an object");
    return true;
  }
  const json::Value* meta = root.find("metadata");
  int64_t total_size = 0;
  if (meta != nullptr && meta->kind == json::Value::Kind::Object) {
    const json::Value* ts = meta->find("total_size");
    if (ts != nullptr && ts->kind == json::Value::Kind::Int) total_size = ts->int_value;
  }
  if (total_size != 0 && total_size != report->total_bytes) {
    std::ostringstream os;
    os << "index metadata.total_size=" << total_size << " differs from measured "
       << report->total_bytes << " (difference " << (report->total_bytes - total_size)
       << " bytes: the index omits tensors the shards contain)";
    report->notes.push_back(os.str());
  }
  if (static_cast<std::size_t>(wmap->obj.size()) != report->tensor_digests.size()) {
    std::ostringstream os;
    os << "index weight_map lists " << wmap->obj.size() << " tensors, shards contain "
       << report->tensor_digests.size();
    report->issues.push_back(os.str());
  }
  // Name-set agreement in both directions.
  std::vector<std::string> shard_names;
  for (const auto& td : report->tensor_digests) shard_names.push_back(td.name);
  std::sort(shard_names.begin(), shard_names.end());
  for (const auto& kv : wmap->obj) {
    auto it = std::lower_bound(shard_names.begin(), shard_names.end(), kv.first);
    if (it == shard_names.end() || *it != kv.first)
      report->issues.push_back("index lists tensor absent from shards: " + kv.first);
  }
  std::vector<std::pair<std::string, std::string>> idx_entries;
  for (const auto& kv : wmap->obj) idx_entries.emplace_back(kv.first, kv.second.str);
  std::sort(idx_entries.begin(), idx_entries.end());
  for (const auto& n : shard_names) {
    auto it = std::lower_bound(idx_entries.begin(), idx_entries.end(), n,
                               [](const auto& p, const std::string& s) { return p.first < s; });
    if (it == idx_entries.end() || it->first != n)
      report->issues.push_back("shard tensor absent from index: " + n);
  }
  return true;
}

bool verify_checkpoint(const char* dir, CheckpointReport* report, std::string* error) {
  if (report == nullptr) {
    if (error) *error = "null report";
    return false;
  }
  report->ok = false;
  report->error.clear();
  report->issues.clear();
  report->notes.clear();
  report->tensors = 0;
  report->total_numel = 0;
  report->total_bytes = 0;
  report->shards = 0;
  report->shard_digests.clear();
  report->tensor_digests.clear();
  report->config_present = false;
  bool ok = true;
  if (!load_and_check_config(dir, report, error)) ok = false;
  if (ok && !verify_shards(dir, report, error)) ok = false;
  if (ok) index_cross_check(dir, report, error);
  report->ok = ok && report->issues.empty();
  if (error && !report->ok && report->error.empty()) {
    *error = report->summary();
  }
  return report->ok;
}

bool load_and_check_config(const char* dir, CheckpointReport* report, std::string* error) {
  if (report == nullptr) {
    if (error) *error = "null report";
    return false;
  }
  std::string cfg_text;
  if (!read_file(path_join(dir, "config.json"), &cfg_text, error)) return false;
  std::string parse_error;
  if (!parse_checkpoint_config(cfg_text.data(), cfg_text.size(), &report->config, &parse_error)) {
    report->error = parse_error;
    if (error) *error = parse_error;
    return false;
  }
  report->config_present = true;
  report->issues.insert(report->issues.end(), report->config.deviations.begin(),
                         report->config.deviations.end());
  return true;
}

std::string CheckpointReport::summary() const {
  std::ostringstream os;
  if (ok) {
    os << "OK: " << tensors << " tensors, " << total_bytes << " bytes across "
       << shards << " shard(s)";
  } else {
    os << "FAIL: " << issues.size() << " issue(s)";
    const std::size_t n = std::min<std::size_t>(issues.size(), 5);
    for (std::size_t i = 0; i < n; ++i) os << (i ? "; " : ": ") << issues[i];
    if (issues.size() > n) os << " (+ " << (issues.size() - n) << " more)";
    if (!error.empty()) os << " [error: " << error << "]";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// FP8 quantization
// ---------------------------------------------------------------------------

QuantStats quantize_bf16(const std::uint8_t* data, int64_t row, int64_t col,
                         bool per_channel, std::vector<std::uint8_t>* fp8_out) {
  QuantStats stats;
  const int64_t n = row * col;
  fp8_out->resize(static_cast<std::size_t>(n));
  stats.saturated = 0;
  stats.max_abs = 0.0;
  if (per_channel) {
    stats.channel_scales.resize(static_cast<std::size_t>(row), 0.0f);
    for (int64_t r = 0; r < row; ++r) {
      const std::uint8_t* rowdata = data + 2 * col * r;
      const double m = max_abs_bf16_row(rowdata, col);
      const float scale = static_cast<float>(m / kFp8E4M3MaxF);
      stats.channel_scales[static_cast<std::size_t>(r)] = scale;
      if (m > stats.max_abs) stats.max_abs = m;
      for (int64_t c = 0; c < col; ++c) {
        const std::uint16_t bits = static_cast<std::uint16_t>(rowdata[2 * c]) |
                                   (static_cast<std::uint16_t>(rowdata[2 * c + 1]) << 8);
        bool sat = false;
        const std::uint8_t q =
            scale > 0.0f ? encode_scaled(bits, scale, &sat) : bf16_to_fp8_e4m3(bits, &sat);
        if (sat) stats.saturated++;
        (*fp8_out)[static_cast<std::size_t>(r * col + c)] = q;
      }
    }
    stats.scale = 0.0f;
  } else {
    double mx = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      const std::uint16_t bits = static_cast<std::uint16_t>(data[2 * i]) |
                                 (static_cast<std::uint16_t>(data[2 * i + 1]) << 8);
      const float f = Bf16ToFloat::from_bits(bits);
      const double a = f < 0.0 ? -static_cast<double>(f) : static_cast<double>(f);
      if (a > mx) mx = a;
    }
    stats.max_abs = mx;
    const float scale = static_cast<float>(mx / kFp8E4M3MaxF);
    stats.scale = scale;
    for (int64_t i = 0; i < n; ++i) {
      const std::uint16_t bits = static_cast<std::uint16_t>(data[2 * i]) |
                                 (static_cast<std::uint16_t>(data[2 * i + 1]) << 8);
      bool sat = false;
      const std::uint8_t q =
          scale > 0.0f ? encode_scaled(bits, scale, &sat) : bf16_to_fp8_e4m3(bits, &sat);
      if (sat) stats.saturated++;
      (*fp8_out)[static_cast<std::size_t>(i)] = q;
    }
  }
  return stats;
}

bool quantize_tensor(const char* shard_path, int64_t offset, int64_t row, int64_t col,
                     bool per_channel, std::vector<std::uint8_t>* fp8_out,
                     QuantStats* stats, std::string* error) {
  const int64_t nbytes = row * col * 2;
  std::vector<std::uint8_t> bf16;
  bf16.resize(static_cast<std::size_t>(nbytes));
  std::ifstream f(shard_path, std::ios::binary);
  if (!f) {
    if (error) *error = "cannot open " + std::string(shard_path);
    return false;
  }
  if (!f.seekg(static_cast<std::streamoff>(offset), std::ios::beg)) {
    if (error) *error = "seek failed on " + std::string(shard_path);
    return false;
  }
  f.read(reinterpret_cast<char*>(bf16.data()), static_cast<std::streamsize>(nbytes));
  if (f.gcount() != static_cast<std::streamsize>(nbytes)) {
    if (error) *error = "short read of tensor data in " + std::string(shard_path);
    return false;
  }
  *stats = quantize_bf16(bf16.data(), row, col, per_channel, fp8_out);
  return true;
}

// ---------------------------------------------------------------------------
// quantized export
//
// Streaming exporter: the header (layout is pure size math) is written
// first, then the data region is quantized tensor-by-tensor in plan order,
// appended after the header. At most one tensor (its BF16 source + FP8
// output) is resident at a time.
// ---------------------------------------------------------------------------

bool export_fp8_checkpoint(const char* dir, CheckpointReport* report,
                           const std::string& out_path, bool per_channel,
                           std::string* error) {
  if (report == nullptr || !report->config_present) {
    if (error) *error = "export_fp8_checkpoint requires a verified report";
    return false;
  }
  const std::vector<ExpectedTensor> expected = expected_tensors(report->config.model);

  // Locate every expected tensor in the verified shard manifests.
  // (Re-parse the manifests - they are metadata-only and cheap.)
  std::vector<std::string> shard_names;
  if (!list_safetensors(dir, &shard_names, error)) return false;
  struct Loc {
    std::string shard;
    std::string shard_path;
    int64_t offset = 0;
  };
  std::map<std::string, Loc> loc;
  for (const auto& name : shard_names) {
    const std::string path = path_join(dir, name);
    TensorManifest m;
    std::string merr;
    if (!parse_safetensors_file(path, &m, &merr)) {
      if (error) *error = name + ": " + merr;
      return false;
    }
    for (const auto& t : m.tensors)
      loc[t.name] = {name, path, 8 + m.header_bytes + t.offset_begin};
  }

  // The plan: expected tensor order (embed_tokens, layers..., final norm,
  // lm_head) so the output file has a stable, spec-ordered layout.
  std::vector<const ExpectedTensor*> plan;
  for (const auto& e : expected) plan.push_back(&e);

  struct Rec {
    std::string name;
    std::vector<int64_t> shape;
    int64_t begin = 0;
    int64_t end = 0;
  };
  // Data-region layout first: offsets are cumulative element counts (one FP8
  // byte per element, then the scale tensor), so the header can be serialized
  // without reading any shard data.
  std::vector<Rec> recs;
  int64_t data_pos = 0;
  for (const ExpectedTensor* e : plan) {
    const int64_t row = e->shape[0];
    const int64_t col = e->shape.size() > 1 ? e->shape[1] : 1;
    const int64_t fp8_bytes = row * col;
    const int64_t weight_begin = data_pos;
    data_pos += fp8_bytes;
    const int64_t scale_elems = per_channel ? row : 1;
    recs.push_back({e->name, e->shape, weight_begin, weight_begin + fp8_bytes});
    recs.push_back({e->name + "__scale", {scale_elems}, data_pos,
                    data_pos + 4 * scale_elems});
    data_pos += 4 * scale_elems;
  }

  // Serialize the header (data_offsets are relative to the data region,
  // which starts after the header - see the safetensors layout).
  std::ostringstream hdr;
  hdr << "{";
  bool first = true;
  for (const auto& r : recs) {
    if (!first) hdr << ",";
    first = false;
    hdr << "\"" << json_escape(r.name) << "\":{\"dtype\":"
        << (r.name.size() > 7 && r.name.compare(r.name.size() - 7, 7, "__scale") == 0
                ? "\"F32\""
                : "\"F8_E4M3\"")
        << ",\"shape\":[";
    for (std::size_t i = 0; i < r.shape.size(); ++i)
      hdr << (i ? "," : "") << r.shape[i];
    hdr << "],\"data_offsets\":[" << r.begin << "," << r.end << "]}";
  }
  hdr << ",\"__metadata__\":{\"schema\":\"granite4.fp8.v1\",\"max\":\"448\","
     << "\"total_size\":\"" << data_pos << "\"}}";
  const std::string h = hdr.str();

  // File layout: [8-byte LE header length][header JSON][data region].
  std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
  if (!f) {
    if (error) *error = "cannot open output " + out_path;
    return false;
  }
  char prefix[8];
  for (int i = 0; i < 8; ++i)
    prefix[i] =
        static_cast<char>((static_cast<uint64_t>(h.size()) >> (8 * i)) & 0xFF);
  f.write(prefix, 8);
  f.write(h.data(), static_cast<std::streamsize>(h.size()));
  if (!f.good()) {
    if (error) *error = "header write failed on " + out_path;
    return false;
  }

  // Stream the data region in plan order: quantize each tensor from its
  // source shard, then append the scale tensor; at most one tensor (BF16
  // source + FP8 output) is resident at a time.
  for (std::size_t i = 0; i < plan.size(); ++i) {
    const ExpectedTensor* e = plan[i];
    const auto it = loc.find(e->name);
    const int64_t row = e->shape[0];
    const int64_t col = e->shape.size() > 1 ? e->shape[1] : 1;
    std::vector<std::uint8_t> fp8;
    QuantStats stats;
    if (!quantize_tensor(it->second.shard_path.c_str(), it->second.offset, row, col,
                         per_channel, &fp8, &stats, error)) {
      return false;
    }
    if (fp8.size() != static_cast<std::size_t>(row * col)) {
      if (error) *error = "export: quantized size mismatch for " + e->name;
      return false;
    }
    if (!f.write(reinterpret_cast<const char*>(fp8.data()),
                 static_cast<std::streamsize>(fp8.size()))) {
      if (error) *error = "write error on " + out_path;
      return false;
    }
    const int64_t scale_elems = per_channel ? row : 1;
    std::vector<std::uint32_t> scale_bits(static_cast<std::size_t>(scale_elems));
    if (per_channel) {
      for (int64_t r = 0; r < row; ++r)
        std::memcpy(scale_bits.data() + r,
                    &stats.channel_scales[static_cast<std::size_t>(r)], sizeof(float));
    } else {
      std::memcpy(scale_bits.data(), &stats.scale, sizeof(float));
    }
    if (!f.write(reinterpret_cast<const char*>(scale_bits.data()),
                 static_cast<std::streamsize>(scale_bits.size() * 4))) {
      if (error) *error = "write error on " + out_path;
      return false;
    }
  }
  const auto end_pos = f.tellp();
  f.close();
  if (end_pos !=
      static_cast<std::streamoff>(8 + static_cast<int64_t>(h.size()) + data_pos)) {
    if (error) *error = "export: data region not fully written to " + out_path;
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

namespace {

std::string build_report_json(const CheckpointReport& r) {
  std::ostringstream os;
  os << "{\"schema\":\"granite4.checkpoint-report.v1\",\"ok\":" << (r.ok ? "true" : "false")
     << ",\"issues\":" << json_string_array(r.issues) << ",\"notes\":"
     << json_string_array(r.notes) << ",\"tensors\":" << r.tensors
     << ",\"shards\":" << r.shards << ",\"total_bytes\":" << r.total_bytes
     << ",\"total_numel\":" << r.total_numel << ",\"config_ok\":"
     << (r.config_present && config_matches_spec(r.config) ? "true" : "false")
     << ",\"sha256_shards\":{";
  for (std::size_t i = 0; i < r.shard_digests.size(); ++i) {
    if (i) os << ",";
    os << "\"" << json_escape(r.shard_digests[i].path) << "\":\"" << r.shard_digests[i].sha256
       << "\"";
  }
  os << "},\"sha256_tensors\":{";
  for (std::size_t i = 0; i < r.tensor_digests.size(); ++i) {
    if (i) os << ",";
    os << "\"" << json_escape(r.tensor_digests[i].name) << "\":\""
       << r.tensor_digests[i].sha256 << "\"";
  }
  os << "}}";
  return os.str();
}

}  // namespace

}  // namespace granite4

extern "C" {

int granite4_verify_checkpoint(const char* checkpoint_dir, char* buf, long buf_len,
                               long* used) {
  if (used != nullptr) *used = 0;
  if (checkpoint_dir == nullptr) return 0;
  granite4::CheckpointReport report;
  std::string error;
  granite4::verify_checkpoint(checkpoint_dir, &report, &error);
  const std::string json = granite4::build_report_json(report);
  const long needed = static_cast<long>(json.size()) + 1;
  if (buf == nullptr) {
    if (used != nullptr) *used = needed;
    return 0;
  }
  if (buf_len < needed) {
    if (used != nullptr) *used = needed;
    return 0;
  }
  std::memcpy(buf, json.data(), json.size());
  buf[json.size()] = '\0';
  if (used != nullptr) *used = static_cast<long>(json.size());
  return 1;
}

}  // extern "C"
