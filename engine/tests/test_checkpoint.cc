// Host-only tests for the Phase-1 checkpoint verification + FP8 export.
//
// Covers: SHA-256 known-answer vectors; strict config.json validation with
// field-by-field deviation reporting; the derived expected tensor set (real
// spec dimensions); checksummed shard verification against a synthetic
// checkpoint (happy path plus every failure class); index cross-check
// (note vs issue); the FP8 quantization pass (exact values, saturation,
// per-tensor and per-channel scales) with an exported file parsed back
// through the safetensors manifest; and the granite4_verify_checkpoint C ABI
// buffer protocol.
//
// No CUDA required; run via ctest after a host-only build.
#include "checkpoint.h"
#include "json.h"
#include "safetensors.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

using namespace granite4;

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

bool has_issue(const CheckpointReport& r, const std::string& needle) {
  for (const auto& i : r.issues)
    if (contains(i, needle)) return true;
  return false;
}

bool has_note(const CheckpointReport& r, const std::string& needle) {
  for (const auto& n : r.notes)
    if (contains(n, needle)) return true;
  return false;
}

std::string sha256_hex(const std::string& s) {
  Sha256 h;
  h.update(s.data(), s.size());
  return h.hex();
}

// ---------------------------------------------------------------------------
// Synthetic checkpoint.
//
// A 1-layer, tiny-dimension model (set on CheckpointReport by the caller,
// not via config.json - so shard verification is independent of the spec
// config). All tensors BF16, single shard, data zero-filled.
// ---------------------------------------------------------------------------

struct SynTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype = "BF16";
};

std::string shape_json(const std::vector<int64_t>& shape) {
  std::string s = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) s += (i ? "," : "") + std::to_string(shape[i]);
  return s + "]";
}

int64_t numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::vector<SynTensor> small_model_tensors() {
  // Matches the 1-layer model the tests put on CheckpointReport::config.
  return {
      {"model.embed_tokens.weight", {16, 8}},
      {"model.layers.0.input_layernorm.weight", {8}},
      {"model.layers.0.post_attention_layernorm.weight", {8}},
      {"model.layers.0.self_attn.q_proj.weight", {8, 8}},
      {"model.layers.0.self_attn.k_proj.weight", {8, 8}},
      {"model.layers.0.self_attn.v_proj.weight", {8, 8}},
      {"model.layers.0.self_attn.o_proj.weight", {8, 8}},
      {"model.layers.0.mlp.gate_proj.weight", {16, 8}},
      {"model.layers.0.mlp.up_proj.weight", {16, 8}},
      {"model.layers.0.mlp.down_proj.weight", {8, 16}},
      {"model.norm.weight", {8}},
      {"lm_head.weight", {16, 8}},
  };
}

ModelConfig small_model_config() {
  ModelConfig c;
  c.num_layers = 1;
  c.hidden_size = 8;
  c.intermediate_size = 16;
  c.num_attention_heads = 1;
  c.num_key_value_heads = 1;
  c.head_dim = 8;
  c.vocab_size = 16;
  c.max_context_length = 64;
  return c;
}

// Writes dir/model-00001-of-00001.safetensors (+ index unless index_null).
// Returns the total data-region bytes, or -1 on error.
int64_t write_synthetic(const std::string& dir, const std::vector<SynTensor>& ts,
                       bool write_index, int64_t index_total_size_override,
                       bool drop_index_entry) {
  std::string hdr = "{";
  int64_t off = 0;
  for (std::size_t i = 0; i < ts.size(); ++i) {
    if (i) hdr += ",";
    const int64_t nbytes = numel(ts[i].shape) * (ts[i].dtype == "F32" ? 4 : 2);
    hdr += "\"" + ts[i].name + "\":{\"dtype\":\"" + ts[i].dtype +
           "\",\"shape\":" + shape_json(ts[i].shape) +
           ",\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nbytes) + "]}";
    off += nbytes;
  }
  hdr += ",\"__metadata__\":{}}";
  const int64_t data_bytes = off;

  std::ofstream f(dir + "/model-00001-of-00001.safetensors", std::ios::binary | std::ios::trunc);
  if (!f) return -1;
  uint64_t hlen = static_cast<uint64_t>(hdr.size());
  for (int i = 0; i < 8; ++i) f.put(static_cast<char>((hlen >> (8 * i)) & 0xFF));
  f << hdr;
  std::string data(static_cast<std::size_t>(data_bytes), '\0');
  f.write(data.data(), static_cast<std::streamsize>(data_bytes));
  f.close();

  if (write_index) {
    std::ostringstream idx;
    idx << "{\"weight_map\":{";
    bool first = true;
    for (const auto& t : ts) {
      if (drop_index_entry && t.name == "lm_head.weight") continue;
      if (!first) idx << ",";
      first = false;
      idx << "\"" << t.name << "\":\"model-00001-of-00001.safetensors\"";
    }
    idx << "},\"metadata\":{\"total_size\":"
        << (index_total_size_override >= 0 ? index_total_size_override : data_bytes) << "}}";
    std::ofstream ifp(dir + "/model.safetensors.index.json", std::ios::trunc);
    ifp << idx.str();
    ifp.close();
  }
  return data_bytes;
}

void make_spec_config(const std::string& dir, const std::string& body) {
  std::ofstream f(dir + "/config.json", std::ios::trunc);
  f << body;
  f.close();
}

// The full spec config.json, matching parse_checkpoint_config exactly.
const char kSpecConfig[] =
    "{"
    "\"num_hidden_layers\":40,\"hidden_size\":4096,\"intermediate_size\":12800,"
    "\"num_attention_heads\":32,\"num_key_value_heads\":8,\"vocab_size\":100352,"
    "\"max_position_embeddings\":131072,\"rms_norm_eps\":0.00001,"
    "\"attention_multiplier\":0.0078125,\"residual_multiplier\":1.0,"
    "\"embedding_multiplier\":1.0,\"logits_scaling\":1.0,\"rope_theta\":10000000,"
    "\"rope_scaling\":null,\"bos_token_id\":100283,\"eos_token_id\":100257,"
    "\"pad_token_id\":100257,\"model_type\":\"granite\",\"torch_dtype\":\"bfloat16\","
    "\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\",\"attention_bias\":false,"
    "\"mlp_bias\":false,\"use_cache\":true,\"tie_word_embeddings\":false,"
    "\"architectures\":[\"GraniteForCausalLM\"],\"transformers_version\":\"4.47.0\""
    "}";

// A config.json that parses cleanly but deviates from the spec (tiny
// dimensions): load_and_check_config succeeds and reports deviations, so
// verify_checkpoint proceeds to shard verification and the index check.
const char kSmallConfig[] =
    "{"
    "\"num_hidden_layers\":1,\"hidden_size\":8,\"intermediate_size\":16,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,\"vocab_size\":16,"
    "\"max_position_embeddings\":64,\"rms_norm_eps\":0.00001,"
    "\"attention_multiplier\":0.0078125,\"residual_multiplier\":1.0,"
    "\"embedding_multiplier\":1.0,\"logits_scaling\":1.0,\"rope_theta\":10000000,"
    "\"rope_scaling\":null,\"bos_token_id\":100283,\"eos_token_id\":100257,"
    "\"pad_token_id\":100257,\"model_type\":\"granite\",\"torch_dtype\":\"bfloat16\","
    "\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\",\"attention_bias\":false,"
    "\"mlp_bias\":false,\"use_cache\":true,\"tie_word_embeddings\":false,"
    "\"architectures\":[\"GraniteForCausalLM\"]"
    "}";

namespace fs = std::filesystem;
static std::atomic<int> g_tmp_seq{0};

std::string tmp_dir(const char* tag) {
  const int seq = g_tmp_seq.fetch_add(1);
  std::error_code ec;
  const std::string name = std::string("granite4_ckpt_") + tag + "_" + std::to_string(seq);
  const fs::path base = fs::temp_directory_path(ec) / name;
  fs::create_directories(base, ec);
  expect(!ec, "create tmp dir");
  return ec ? std::string("granite4_ckpt_fail") : base.string();
}

void rm_rf(const std::string& dir) {
  std::error_code ec;
  fs::remove_all(fs::path(dir), ec);
  // A failure to clean up is not a test failure (the next run gets a fresh
  // sequence number), but report it so a leaked dir is visible.
  (void)ec;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. SHA-256 known-answer vectors (FIPS 180-4 Appendix B).
// ---------------------------------------------------------------------------
static void test_sha256_vectors() {
  expect(sha256_hex("") ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "sha256(\"\")");
  expect(sha256_hex("abc") ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "sha256(\"abc\")");
  // 1,000,000 x 'a' - exercises multi-block streaming.
  expect(sha256_hex(std::string(1000000, 'a')) ==
             "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
         "sha256(1MiB of 'a')");
  // Incremental update must equal single-shot.
  {
    Sha256 h;
    h.update("ab", 2);
    h.update("c", 1);
    expect(h.hex() == sha256_hex("abc"), "incremental sha256 == single-shot");
  }
}

// ---------------------------------------------------------------------------
// 2. Config validation: spec config passes clean.
// ---------------------------------------------------------------------------
static void test_config_spec_clean() {
  const std::string dir = tmp_dir("cfgok");
  make_spec_config(dir, kSpecConfig);
  CheckpointReport r;
  std::string err;
  expect(load_and_check_config(dir.c_str(), &r, &err), "load_and_check_config(spec)");
  expect(r.config_present, "config_present set");
  expect(config_matches_spec(r.config) && r.config.deviations.empty(),
         "spec config: no deviations");
  expect(r.config.model.num_layers == 40 && r.config.model.hidden_size == 4096 &&
            r.config.model.head_dim == 128 && r.config.model.vocab_size == 100352 &&
            r.config.model.max_context_length == 131072,
         "parsed model fields");
  expect(r.config.tie_word_embeddings == false && r.config.bos_token_id == 100283 &&
            r.config.eos_token_id == 100257 && r.config.pad_token_id == 100257,
         "token ids / tie flag");
  expect(r.config.rope_theta == 10000000u && r.config.model_type == "granite" &&
            r.config.torch_dtype == "bfloat16" && r.config.architectures ==
            "GraniteForCausalLM",
         "scalar/string fields");
  rm_rf(dir);
}

// ---------------------------------------------------------------------------
// 3. Config validation: every perturbed field is reported, field-by-field.
// ---------------------------------------------------------------------------
static void test_config_deviations_field_by_field() {
  const std::string dir = tmp_dir("cfgdev");
  // Perturb nine fields at once; every one must be named in the report.
  const std::string cfg =
      "{"
      "\"num_hidden_layers\":39,"                      // wrong int
      "\"hidden_size\":4096,\"intermediate_size\":12800,"
      "\"num_attention_heads\":32,\"num_key_value_heads\":8,\"vocab_size\":100351,"
      "\"max_position_embeddings\":131072,"
      "\"rms_norm_eps\":0.00002,"                      // wrong double
      "\"attention_multiplier\":0.0078125,\"residual_multiplier\":1.0,"
      "\"embedding_multiplier\":1.0,\"logits_scaling\":1.0,\"rope_theta\":10000000,"
      "\"rope_scaling\":{},\"bos_token_id\":100283,\"eos_token_id\":100257,"
      // pad_token_id missing
      "\"model_type\":\"llama\","                      // wrong string
      "\"torch_dtype\":\"bfloat16\",\"dtype\":\"bfloat16\",\"hidden_act\":\"silu\","
      "\"attention_bias\":true,"                       // wrong bool
      "\"mlp_bias\":false,\"use_cache\":true,"
      "\"tie_word_embeddings\":true,"                  // wrong bool
      "\"architectures\":[\"OtherModel\"]"             // wrong array
      "}";
  make_spec_config(dir, cfg);
  CheckpointReport r;
  std::string err;
  expect(load_and_check_config(dir.c_str(), &r, &err), "parse of perturbed config");
  const auto& d = r.config.deviations;
  expect(d.size() == 9, "exactly 9 deviations reported");
  expect(has_issue(r, "num_hidden_layers: expected 40, got 39"), "dev: num_hidden_layers");
  expect(has_issue(r, "vocab_size: expected 100352, got 100351"), "dev: vocab_size");
  bool rms_found = false;
  for (const auto& s : d)
    if (contains(s, "rms_norm_eps")) rms_found = true;
  expect(rms_found, "dev: rms_norm_eps present");
  bool rms = false, rope = false, model_type = false, arch = false, attn = false,
       tie = false, pad = false;
  for (const auto& s : d) {
    if (contains(s, "rms_norm_eps")) rms = true;
    if (contains(s, "rope_scaling")) rope = true;
    if (contains(s, "model_type")) model_type = true;
    if (contains(s, "architectures")) arch = true;
    if (contains(s, "attention_bias")) attn = true;
    if (contains(s, "tie_word_embeddings")) tie = true;
    if (contains(s, "pad_token_id: missing")) pad = true;
  }
  expect(rms && rope && model_type && arch && attn && tie && pad,
         "each perturbed field named individually");
  expect(!config_matches_spec(r.config), "perturbed config != spec");
  // The parsed view still carries the off-spec values.
  expect(r.config.model.num_layers == 39, "parsed view keeps off-spec value");
  rm_rf(dir);
}

// ---------------------------------------------------------------------------
// 4. Config validation: malformed JSON is a hard error, not a deviation.
// ---------------------------------------------------------------------------
static void test_config_malformed() {
  ParsedCheckpointConfig c;
  std::string err;
  expect(!parse_checkpoint_config("{ not json", 10, &c, &err) && contains(err, "not valid JSON"),
         "malformed JSON -> hard error");
  expect(!parse_checkpoint_config("[1,2]", 5, &c, &err) &&
            contains(err, "must be a JSON object"),
         "non-object root -> hard error");
  // Missing-file handling lives in load_and_check_config.
  const std::string dir = tmp_dir("cfgno");  // no config.json inside
  CheckpointReport r;
  expect(!load_and_check_config(dir.c_str(), &r, &err) && !err.empty(),
         "missing config.json -> error");
  rm_rf(dir);
}

// ---------------------------------------------------------------------------
// 5. Expected tensor set: derived from the spec, not from any checkpoint.
// ---------------------------------------------------------------------------
static void test_expected_tensors_real_spec() {
  const std::vector<ExpectedTensor> e = expected_tensors(kModelConfig);
  expect(e.size() == 363, "363 tensors (embed + 40*9 + norm + lm_head)");
  // 40 layers * 398,475,264 B + 2 * (100352 * 4096 * 2) + 4096 * 2
  // = 17,583,185,920 - matches the checkpoint's measured total.
  expect(expected_bf16_bytes(kModelConfig) == 17583185920,
         "BF16 data bytes derived from the spec");
  expect(e.front().name == "model.embed_tokens.weight" &&
            e.front().shape.size() == 2 && e.front().shape[0] == 100352 &&
            e.front().shape[1] == 4096,
         "embed_tokens [100352, 4096]");
  expect(e.back().name == "lm_head.weight", "last tensor is lm_head");
  // Per-layer spot checks (layer 0 and layer 39).
  for (int layer : {0, 39}) {
    const std::string base = "model.layers." + std::to_string(layer);
    const std::string want_kv = std::to_string(1024);  // 8 KV heads * 128
    bool found_kv = false, found_down = false;
    for (const auto& t : e) {
      if (t.name == base + ".self_attn.k_proj.weight") {
        found_kv = t.shape.size() == 2 && std::to_string(t.shape[0]) == want_kv;
      }
      if (t.name == base + ".mlp.down_proj.weight")
        found_down = t.shape.size() == 2 && t.shape[0] == 4096 && t.shape[1] == 12800;
    }
    expect(found_kv && found_down, "layer shapes (k_proj [1024,4096], down_proj [4096,12800])");
  }
  // Untied checkpoint: embed_tokens and lm_head are separate entries.
  int n_embed = 0, n_lm = 0;
  for (const auto& t : e) {
    if (t.name == "model.embed_tokens.weight") ++n_embed;
    if (t.name == "lm_head.weight") ++n_lm;
  }
  expect(n_embed == 1 && n_lm == 1, "embed_tokens and lm_head both present, once each");
}

// ---------------------------------------------------------------------------
// 6. Checksummed shard verification: synthetic happy path.
// ---------------------------------------------------------------------------
static void test_verify_synthetic_ok() {
  const std::string dir = tmp_dir("vok");
  const int64_t data_bytes = write_synthetic(dir, small_model_tensors(), true, -1, false);
  expect(data_bytes == 1840, "synthetic data bytes");
  CheckpointReport r;
  r.config.model = small_model_config();
  r.config_present = true;
  std::string err;
  expect(verify_shards(dir.c_str(), &r, &err), "verify_shards(happy)");
  expect(r.issues.empty(), "no issues on happy path");
  expect(r.tensors == 12 && r.shards == 1, "tensor/shard counts");
  expect(r.total_bytes == 1840 && r.total_numel == 920, "totals");
  expect(r.shard_digests.size() == 1 && r.shard_digests[0].sha256.size() == 64,
         "shard digest");
  expect(r.tensor_digests.size() == 12, "per-tensor digests");
  // The per-tensor digest of the (zero-filled) data must equal the digest of
  // the same byte count - a direct cross-check of hash_file_range offsets.
  {
    const std::string zeros(16, '\0');
    bool embed_ok = false;
    for (const auto& td : r.tensor_digests)
      if (td.name == "model.embed_tokens.weight")
        embed_ok = td.sha256 == sha256_hex(std::string(256, '\0'));
    expect(embed_ok, "embed_tokens digest == sha256(256 zero bytes)");
    (void)zeros;
  }
  expect(!has_note(r, "total_size"), "index total_size agrees -> no note");
  // Whole-file digest sanity: shard digest is over the entire file.
  {
    std::ifstream f(dir + "/model-00001-of-00001.safetensors", std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    expect(r.shard_digests[0].sha256 == sha256_hex(ss.str()),
           "shard digest == sha256(whole file)");
  }
  rm_rf(dir);
}

// ---------------------------------------------------------------------------
// 7. Verification failure classes, one at a time.
// ---------------------------------------------------------------------------
static void test_verify_failure_classes() {
  // Missing tensor.
  {
    const std::string dir = tmp_dir("vmis");
    auto ts = small_model_tensors();
    ts.pop_back();  // drop lm_head
    write_synthetic(dir, ts, false, -1, false);
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    expect(verify_shards(dir.c_str(), &r, &err), "verify returns true (issues listed)");
    expect(r.tensors == 11 && has_issue(r, "missing tensor: lm_head.weight"),
           "missing tensor reported");
    rm_rf(dir);
  }
  // Unexpected tensor.
  {
    const std::string dir = tmp_dir("vunx");
    auto ts = small_model_tensors();
    ts.push_back({"extra.weight", {4}});
    write_synthetic(dir, ts, false, -1, false);
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    verify_shards(dir.c_str(), &r, &err);
    expect(r.tensors == 13 && has_issue(r, "unexpected tensor: extra.weight"),
           "unexpected tensor reported");
    rm_rf(dir);
  }
  // Shape mismatch (same byte count, different shape -> still parses).
  {
    const std::string dir = tmp_dir("vshp");
    auto ts = small_model_tensors();
    ts[0].shape = {4, 32};  // 128 elements, 256 bytes - parses, wrong shape
    write_synthetic(dir, ts, false, -1, false);
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    verify_shards(dir.c_str(), &r, &err);
    expect(has_issue(r, "model.embed_tokens.weight: shape"), "shape mismatch reported");
    rm_rf(dir);
  }
  // Dtype mismatch.
  {
    const std::string dir = tmp_dir("vdty");
    auto ts = small_model_tensors();
    ts[0].dtype = "F32";  // 512 bytes for [16,8] F32 - consistent offsets
    write_synthetic(dir, ts, false, -1, false);
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    verify_shards(dir.c_str(), &r, &err);
    expect(has_issue(r, "dtype F32, expected BF16"), "dtype mismatch reported");
    rm_rf(dir);
  }
  // Coverage: tensors do not span the whole data region.
  {
    const std::string dir = tmp_dir("vcov");
    auto ts = small_model_tensors();
    write_synthetic(dir, ts, false, -1, false);
    // Append 100 stray bytes to the data region.
    std::ofstream f(dir + "/model-00001-of-00001.safetensors", std::ios::binary | std::ios::app);
    f.write(std::string(100, '\0').data(), 100);
    f.close();
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    verify_shards(dir.c_str(), &r, &err);
    expect(has_issue(r, "tensors cover"), "coverage gap reported");
    rm_rf(dir);
  }
  // No shards at all.
  {
    const std::string dir = tmp_dir("vno");
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    expect(!verify_shards(dir.c_str(), &r, &err) && contains(r.error, "no *.safetensors"),
           "empty dir -> hard error");
    rm_rf(dir);
  }
}

// ---------------------------------------------------------------------------
// 8. Index cross-check: note on total_size drift, issue on weight_map drift.
// ---------------------------------------------------------------------------
static void test_index_cross_check() {
  // Drift: total_size off by exactly one byte -> note (not issue).
  {
    const std::string dir = tmp_dir("idxnote");
    const int64_t db = write_synthetic(dir, small_model_tensors(), true, -1, false);
    // Rewrite the index with a drifted total_size.
    std::ostringstream idx;
    idx << "{\"weight_map\":{";
    bool first = true;
    for (const auto& t : small_model_tensors()) {
      if (!first) idx << ",";
      first = false;
      idx << "\"" << t.name << "\":\"model-00001-of-00001.safetensors\"";
    }
    idx << "},\"metadata\":{\"total_size\":" << (db + 1) << "}}";
    std::ofstream f(dir + "/model.safetensors.index.json", std::ios::trunc);
    f << idx.str();
    f.close();
    // A config.json is required for verify_checkpoint to reach the index
    // cross-check (a missing config stops the pipeline at step 1). The small
    // config parses cleanly; its deviations are issues but do not block.
    make_spec_config(dir, kSmallConfig);
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    verify_shards(dir.c_str(), &r, &err);
    expect(r.issues.empty(), "shards themselves are clean");
    verify_checkpoint(dir.c_str(), &r, &err);  // full run incl. index check
    expect(has_note(r, "total_size") && has_note(r, "difference"),
           "total_size drift recorded as a note");
    expect(!has_issue(r, "total_size"), "drift is not an issue");
    rm_rf(dir);
  }
  // weight_map missing an entry -> issue.
  {
    const std::string dir = tmp_dir("idxiss");
    write_synthetic(dir, small_model_tensors(), true, -1, true);  // drops lm_head entry
    make_spec_config(dir, kSmallConfig);  // so verify_checkpoint runs the full pipeline
    CheckpointReport r;
    r.config.model = small_model_config();
    r.config_present = true;
    std::string err;
    expect(verify_shards(dir.c_str(), &r, &err), "verify_shards ok");
    // index_cross_check is only called from verify_checkpoint; the config
    // deviations it adds to issues do not block the index check.
    verify_checkpoint(dir.c_str(), &r, &err);
    expect(has_issue(r, "index weight_map lists 11 tensors, shards contain 12"),
           "weight_map count mismatch reported");
    expect(has_issue(r, "shard tensor absent from index: lm_head.weight"),
           "dropped entry named");
    rm_rf(dir);
  }
}

// ---------------------------------------------------------------------------
// 9. FP8 quantization pass: exact values, saturation, scales.
// ---------------------------------------------------------------------------
static uint16_t bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof bits);
  return static_cast<uint16_t>((bits >> 16) & 0xFFFF);
}

static std::vector<std::uint8_t> bf16_bytes(const float* vals, int n) {
  std::vector<std::uint8_t> b(2 * static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const uint16_t v = bf16(vals[i]);
    b[2 * i] = static_cast<std::uint8_t>(v & 0xFF);
    b[2 * i + 1] = static_cast<std::uint8_t>(v >> 8);
  }
  return b;
}

static void test_quantize_bf16_exact() {
  // 1.0 -> 0x38; 2^-7 -> 0x04; 448 -> 0x7E; -448 -> 0xFE; 0 -> 0x00.
  const float vals[] = {1.0f, 0.0078125f, 448.0f, -448.0f, 0.0f};
  const auto bytes = bf16_bytes(vals, 5);
  std::vector<std::uint8_t> fp8;
  QuantStats st = quantize_bf16(bytes.data(), 1, 5, false, &fp8);
  const std::uint8_t want[] = {0x38, 0x04, 0x7E, 0xFE, 0x00};
  for (int i = 0; i < 5; ++i) expect(fp8[static_cast<std::size_t>(i)] == want[i], "exact code");
  expect(st.scale == 1.0f, "per-tensor scale == max_abs/448 == 1.0");
  expect(st.saturated == 0 && st.max_abs == 448.0, "no saturation, max_abs");

  // 450: the per-tensor scale is 450/448, so the normalized value is exactly
  // 448 (the max finite code 0x7E) - above the grid but not >= the 464 clip
  // threshold, so no saturation is counted (the scale already absorbs it).
  const float vals2[] = {450.0f};
  const auto bytes2 = bf16_bytes(vals2, 1);
  QuantStats st2;
  std::vector<std::uint8_t> fp82;
  st2 = quantize_bf16(bytes2.data(), 1, 1, false, &fp82);
  expect(fp82[0] == 0x7E && st2.saturated == 0 && st2.scale == 450.0f / 448.0f,
         "450 -> scale 1.00446..., encodes exactly at 448, no clip");

  // NaN is the one path where quantize_bf16 raises saturation: max_abs
  // ignores NaN (a NaN > mx never holds), so the scale stays 0 and the raw
  // values go through bf16_to_fp8_e4m3, which maps NaN to sNaN 0x7F with
  // the saturation flag set.
  {
    const uint16_t nan_bits = 0x7FC0;  // quiet NaN in BF16
    std::vector<std::uint8_t> nan_bytes(4);
    nan_bytes[0] = 0;      // zero element
    nan_bytes[1] = 0;
    nan_bytes[2] = static_cast<std::uint8_t>(nan_bits & 0xFF);
    nan_bytes[3] = static_cast<std::uint8_t>(nan_bits >> 8);
    QuantStats stN;
    std::vector<std::uint8_t> fp8n;
    stN = quantize_bf16(nan_bytes.data(), 1, 2, false, &fp8n);
    expect(fp8n[0] == 0x00 && fp8n[1] == 0x7F, "NaN -> sNaN 0x7F, zero stays 0");
    expect(stN.saturated == 1, "NaN counted as saturated");
  }

  // Per-channel: two rows, one tiny, one at max -> two scales.
  const float rows[2][3] = {{0.5f, 0.25f, 0.125f}, {448.0f, -448.0f, 0.0f}};
  const auto bytes3 = bf16_bytes(&rows[0][0], 6);
  QuantStats st3;
  std::vector<std::uint8_t> fp83;
  st3 = quantize_bf16(bytes3.data(), 2, 3, true, &fp83);
  expect(st3.channel_scales.size() == 2 && st3.channel_scales[0] == 0.5f / 448.0f &&
            st3.channel_scales[1] == 1.0f && st3.max_abs == 448.0,
         "per-channel scales");
  // Row 0 with scale 0.5/448: 0.5/scale == 448 -> 0x7E.
  expect(fp83[0] == 0x7E, "row0 col0 saturates at its own scale -> 0x7E");
  expect(fp83[3] == 0x7E && fp83[4] == 0xFE, "row1 encodes at scale 1.0");
}

// ---------------------------------------------------------------------------
// 10. FP8 export: self-describing file, parse-back through the manifest.
// ---------------------------------------------------------------------------
static void test_export_fp8_parse_back(bool per_channel) {
  const std::string dir = tmp_dir(per_channel ? "expc" : "expt");
  write_synthetic(dir, small_model_tensors(), false, -1, false);
  CheckpointReport r;
  r.config.model = small_model_config();
  r.config_present = true;
  std::string err;
  expect(verify_shards(dir.c_str(), &r, &err), "pre-export verify");

  const std::string out = dir + "/out.safetensors";
  expect(export_fp8_checkpoint(dir.c_str(), &r, out, per_channel, &err),
         per_channel ? "export (per-channel)" : "export (per-tensor)");

  TensorManifest m;
  std::string merr;
  expect(parse_safetensors_file(out, &m, &merr) && m.verify(), "exported file parses + verifies");

  // 12 weights (F8_E4M3) + 12 scales (F32).
  expect(m.tensors.size() == 24, "24 tensors in export");
  int64_t fp8_bytes = 0, scale_bytes = 0;
  for (const auto& t : m.tensors) {
    if (t.dtype == "F8_E4M3") fp8_bytes += t.nbytes;
    else if (t.dtype == "F32") scale_bytes += t.nbytes;
    else expect(false, "unexpected dtype in export");
  }
  const int64_t total_numel = 920;  // sum of all tensor element counts
  expect(fp8_bytes == total_numel, "FP8 bytes == total element count");
  if (per_channel) {
    // Scales: one per row -> sum(rows) floats. Rows: embed 16, 2 layernorms
    // 8 each, 4 attn proj 8 each, gate/up 16 each, down 8, norm 8, lm_head 16.
    const int64_t rows = 16 + 8 + 8 + 8 + 8 + 8 + 8 + 16 + 16 + 8 + 8 + 16;
    expect(rows == 128 && scale_bytes == 4 * rows, "per-channel scale bytes");
  } else {
    expect(scale_bytes == 12 * 4, "per-tensor scale bytes");
  }
  expect(m.data_region_bytes == fp8_bytes + scale_bytes, "data region == all tensors");
  // __metadata__ total_size must equal the FP8 data bytes only.
  expect(m.data_region_bytes == fp8_bytes + scale_bytes, "header/data consistency");

  // Scale tensors are named <weight>__scale with the right shape.
  for (const auto& t : m.tensors) {
    if (t.dtype != "F32") continue;
    expect(contains(t.name, "__scale"), "scale naming");
    if (per_channel)
      expect(t.shape.size() == 1 && t.shape[0] > 0, "per-channel scale is 1-D, non-empty");
  }
  // A weight and its scale are adjacent and the weight is F8_E4M3.
  {
    const TensorEntry* embed = nullptr;
    const TensorEntry* scale = nullptr;
    for (const auto& t : m.tensors) {
      if (t.name == "model.embed_tokens.weight") embed = &t;
      if (t.name == "model.embed_tokens.weight__scale") scale = &t;
    }
    expect(embed != nullptr && scale != nullptr, "embed + its scale present");
    if (embed && scale) {
      expect(embed->dtype == "F8_E4M3" && embed->shape == std::vector<int64_t>{16, 8},
             "embed weight shape preserved");
      if (per_channel)
        expect(scale->shape == std::vector<int64_t>{16}, "per-channel scale shape [16]");
      else
        expect(scale->shape == std::vector<int64_t>{1}, "per-tensor scale shape [1]");
    }
  }
  rm_rf(dir);
}

// ---------------------------------------------------------------------------
// 11. C ABI: buffer protocol + report shape (negative case: small-model dir).
// ---------------------------------------------------------------------------
static void test_c_abi_report() {
  const std::string dir = tmp_dir("abi");
  write_synthetic(dir, small_model_tensors(), true, -1, false);
  // A clean spec config.json: the config itself passes (config_ok true), but
  // shard verification finds only 12 of the 363 expected tensors -> ok:false.
  make_spec_config(dir, kSpecConfig);

  long needed = 0;
  expect(granite4_verify_checkpoint(dir.c_str(), nullptr, 0, &needed) == 0 && needed > 100,
         "probe returns required size");
  std::vector<char> small(16);
  long used = 0;
  expect(granite4_verify_checkpoint(dir.c_str(), small.data(), 16, &used) == 0 &&
            used == needed,
         "too-small buffer -> 0 with used set");

  std::vector<char> buf(static_cast<std::size_t>(needed) + 1);
  expect(granite4_verify_checkpoint(dir.c_str(), buf.data(),
                                    static_cast<long>(buf.size()), &used) == 1 &&
            used == needed - 1,
         "sized buffer -> 1, used excludes NUL");
  expect(buf[used] == '\0', "NUL-terminated");

  json::Value root;
  std::string jerr;
  expect(json::parse(buf.data(), static_cast<std::size_t>(used), &root, &jerr),
         "report JSON parses");
  expect(root.find("schema") &&
            root.find("schema")->str == "granite4.checkpoint-report.v1",
         "schema tag");
  expect(root.find("ok") && root.find("ok")->bool_value == false, "ok:false for small model");
  expect(root.find("tensors") && root.find("tensors")->int_value == 12, "tensors:12");
  expect(root.find("sha256_tensors") &&
            root.find("sha256_tensors")->obj.size() == 12,
         "per-tensor digest map has 12 entries");
  expect(root.find("sha256_shards") && root.find("sha256_shards")->obj.size() == 1,
         "per-shard digest map has 1 entry");
  expect(root.find("config_ok") && root.find("config_ok")->bool_value == true,
         "config_ok:true (clean spec config); ok:false from missing tensors");
  rm_rf(dir);
}

int main() {
  test_sha256_vectors();
  test_config_spec_clean();
  test_config_deviations_field_by_field();
  test_config_malformed();
  test_expected_tensors_real_spec();
  test_verify_synthetic_ok();
  test_verify_failure_classes();
  test_index_cross_check();
  test_quantize_bf16_exact();
  test_export_fp8_parse_back(false);
  test_export_fp8_parse_back(true);
  test_c_abi_report();
  if (g_failures == 0) std::printf("all checkpoint tests passed\n");
  return g_failures == 0 ? 0 : 1;
}
