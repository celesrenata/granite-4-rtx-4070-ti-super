// Phase 1 (Weights): checkpoint verification and the FP8 quantization pass.
//
// Host-only. stdlib only (plus the in-tree json / safetensors parsers); no
// CUDA, no checkpoint paths hardcoded into the library - a verification run
// is always a function of an explicitly passed directory.
//
// Responsibilities:
//  * SHA-256 (FIPS 180-4) - the checksum primitive for the "loaded and
//    checksummed" exit criterion.
//  * Strict config.json validation: every inference-relevant field of the
//    Granite 4.2 8B checkpoint config is checked against the spec, and a
//    failure reports the exact list of deviating fields (hard fail,
//    field-by-field - no partial acceptance).
//  * The full expected tensor set (363 tensors, derived from the spec
//    constants, not from a checkpoint file): name, shape, BF16 dtype.
//  * Checksummed loading: per-shard and per-tensor SHA-256, offset/coverage
//    invariants, and a bounded sequential read plan.
//  * The FP8 e4m3 quantization pass: per-tensor scale, saturation
//    accounting, and export of a self-describing quantized safetensors
//    file (weights as F8_E4M3, per-tensor scales as F32 tensors).
//
// The quantized export is the Phase 1 storage decision (see
// docs/memory.md): FP8 e4m3 weights with per-tensor scales. Ada (sm_89)
// has no Blackwell-style block-scaled MMA, so block scales are out of
// scope; per-tensor vs per-channel scales is a P2 kernel decision, and
// both paths are implemented here.
#ifndef GRANITE4_CHECKPOINT_H
#define GRANITE4_CHECKPOINT_H

#include "model_config.h"
#include "safetensors.h"

#include <cstdint>
#include <string>
#include <vector>

namespace granite4 {

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4). Incremental; finalize is idempotent-free (one shot).
// ---------------------------------------------------------------------------

class Sha256 {
 public:
  Sha256();
  void update(const void* data, std::size_t nbytes);
  // Lowercase 64-hex-char digest.
  std::string hex() const;

 private:
  void transform(const uint8_t block[64]);
  uint32_t h[8];
  uint8_t buf[64];
  std::size_t buflen = 0;
  std::uint64_t total_bytes = 0;
};

// ---------------------------------------------------------------------------
// FP8 e4m3 (OCP microscaling) and BF16 conversions.
//
// e4m3 layout: s | eeee | mmm, exponent bias 7.
//   max finite     448   (0x7E)
//   min normal     2^-6
//   min subnormal  2^-9  (0x01)
//   sNaN           0x7F   (exponent all-ones, mantissa all-ones)
// Rounding is round-to-nearest-even, in exact integer arithmetic.
// NaN/Inf inputs map to 0x7F (sNaN) and count as saturated; magnitudes that
// round above 448 (|v| >= 464) clip at +-448 and count as saturated.
// ---------------------------------------------------------------------------

constexpr std::uint8_t kFp8E4M3Max = 0x7Eu;     // 448
constexpr float kFp8E4M3MaxF = 448.0f;
constexpr std::uint8_t kFp8E4M3SNaN = 0x7Fu;

// One BF16 element (2 bytes, s|e8|m7).
struct Bf16ToFloat {
  static float from_bits(std::uint16_t bits);
};

// Round an arbitrary double to FP8 e4m3 (round-to-nearest-even, exact
// integer arithmetic on the value's binary representation). *saturated
// (optional) is set when the result is 0x7F or a clipped +-448.
std::uint8_t fp8_e4m3_encode(double value, bool* saturated);

// Round a BF16 element to FP8 e4m3 (wrapper over fp8_e4m3_encode; BF16
// values are exactly representable in double, so nothing is lost).
std::uint8_t bf16_to_fp8_e4m3(std::uint16_t bits, bool* saturated);

// Exact decode of e4m3 to BF16 (every finite e4m3 value is exactly
// representable in BF16; 0x7F maps to the BF16 quiet NaN 0x7F80).
std::uint16_t fp8_e4m3_to_bf16(std::uint8_t x);

// ---------------------------------------------------------------------------
// Strict checkpoint config.json validation.
//
// Every inference-relevant field of the Granite 4.2 8B config is checked
// against the spec. `transformers_version` is recorded but never checked
// (it changes between release lines and does not affect inference).
// head_dim is not a config field in this checkpoint: it is derived as
// hidden_size / num_attention_heads and rejected when the division is not
// exact or does not equal the spec value.
// ---------------------------------------------------------------------------

struct ParsedCheckpointConfig {
  ModelConfig model;      // planning fields (head_dim derived)
  bool tie_word_embeddings = false;
  std::uint32_t bos_token_id = 0;
  std::uint32_t eos_token_id = 0;
  std::uint32_t pad_token_id = 0;
  double rms_norm_eps = 0.0;
  double attention_multiplier = 0.0;
  double residual_multiplier = 0.0;
  double embedding_multiplier = 0.0;
  double logits_scaling = 0.0;
  std::uint64_t rope_theta = 0;
  std::string model_type;
  std::string torch_dtype;
  std::string dtype;
  std::string hidden_act;
  std::string architectures;  // single-entry array expected
  std::string transformers_version;  // recorded, not checked

  // Empty when the config matches the spec exactly; otherwise one entry
  // per deviating field, "field: expected X, got Y".
  std::vector<std::string> deviations;
};

// Parses and strictly validates a checkpoint config.json.
// error carries a machine-readable code on hard input failures
// (missing file is out of scope here: the caller passes the bytes).
bool parse_checkpoint_config(const char* data, std::size_t nbytes,
                             ParsedCheckpointConfig* out, std::string* error);

// True when parse succeeded and deviations is empty.
bool config_matches_spec(const ParsedCheckpointConfig& cfg);

// ---------------------------------------------------------------------------
// Expected tensor set (the Granite 4.2 8B spec, derived from ModelConfig).
// 363 tensors: embed_tokens, lm_head, final norm, and per layer two
// layernorms, four attention projections and three MLP projections.
// ---------------------------------------------------------------------------

struct ExpectedTensor {
  std::string name;
  std::vector<int64_t> shape;  // stored (row-major) shape
};

std::vector<ExpectedTensor> expected_tensors(const ModelConfig& cfg);

// Total BF16 bytes of the expected tensor set (this is the number a
// conforming BF16 checkpoint must contain; the index file's
// metadata.total_size is NOT trusted - it is known to exclude
// lm_head.weight in this checkpoint).
int64_t expected_bf16_bytes(const ModelConfig& cfg);

// ---------------------------------------------------------------------------
// Checksummed loading.
// ---------------------------------------------------------------------------

// One tensor of the load plan, in a shard's data region.
struct WeightEntry {
  std::string name;
  int64_t shard_index = -1;
  std::string shard_path;
  int64_t offset = 0;  // byte offset within the shard's data region
  int64_t row = 1;     // shape[0]
  int64_t col = 1;     // shape[1] for 2-D, 1 for 1-D
  int64_t numel = 0;
  int64_t nbytes = 0;
};

struct CheckpointReport {
  bool ok = false;
  std::string error;  // hard input failure (bad dir, unparseable shard ...)
  std::vector<std::string> issues;  // config deviations, tensor-set deltas,
                                     // dtype/shape mismatches, coverage gaps
  // Notes that do not fail verification (e.g. index total_size mismatch).
  std::vector<std::string> notes;

  std::size_t shards = 0;
  std::size_t tensors = 0;
  int64_t total_numel = 0;
  int64_t total_bytes = 0;  // data-region bytes summed over shards
  bool embed_tied_to_lm_head = false;

  struct ShardDigest {
    std::string path;
    std::string sha256;  // of the whole file, lowercase hex
    int64_t bytes = 0;
  };
  struct TensorDigest {
    std::string name;
    std::string sha256;  // of the tensor's data bytes
  };
  std::vector<ShardDigest> shard_digests;
  std::vector<TensorDigest> tensor_digests;

  ParsedCheckpointConfig config;  // populated when config.json was read
  bool config_present = false;

  // Human-readable summary line.
  std::string summary() const;
};

// Reads config.json from dir and strictly validates it.
// On success fills report.config / config_present and returns true
// (deviations are listed in report.issues, not fatal to this step - the
// caller decides; verify_checkpoint treats them as fatal).
bool load_and_check_config(const char* dir, CheckpointReport* report,
                           std::string* error);

// Parses every *.safetensors shard in dir (file names sorted), checks the
// tensor set against expected_tensors(report.config.model), verifies
// offset coverage per shard, and computes per-shard + per-tensor SHA-256.
// The full 17.5 GB stream is hashed in bounded chunks; nothing larger than
// one tensor is held in memory at a time.
bool verify_shards(const char* dir, CheckpointReport* report,
                   std::string* error);

// Full run: config + shards + index cross-check (the index weight_map must
// agree with the parsed shards shard-by-shard; metadata.total_size is
// recorded as a note when it disagrees with the measured total).
bool verify_checkpoint(const char* dir, CheckpointReport* report,
                       std::string* error);

// ---------------------------------------------------------------------------
// FP8 quantization pass + export.
// ---------------------------------------------------------------------------

struct QuantStats {
  float scale = 0.0;          // per-tensor: max_abs / 448 (0 when max is 0)
  std::vector<float> channel_scales;  // non-empty when per-channel was used
  double max_abs = 0.0;
  std::uint32_t saturated = 0;
};

// Quantize a BF16 buffer (row-major, row x col) to e4m3.
// per_channel == false -> one per-tensor scale; true -> one scale per row.
QuantStats quantize_bf16(const std::uint8_t* data, int64_t row, int64_t col,
                         bool per_channel, std::vector<std::uint8_t>* fp8_out);

// Reads a tensor per the plan and quantizes it (bounded single-tensor
// residency).
bool quantize_tensor(const char* shard_path, int64_t offset, int64_t row,
                     int64_t col, bool per_channel,
                     std::vector<std::uint8_t>* fp8_out, QuantStats* stats,
                     std::string* error);

// Exports a self-describing quantized safetensors file:
//   <name>          : F8_E4M3, same shape as the source tensor
//   <name>__scale   : F32, shape [1] (per-tensor) or [row] (per-channel)
// __metadata__ carries {"schema": "granite4.fp8.v1", "max": 448,
// "total_size": <fp8 data bytes>}
// Returns true on success; error gets a bounded diagnostic otherwise.
// The writer is a streaming exporter: tensors are quantized one at a time
// in plan order and appended; the header is written last (after the data)
// and the 8-byte length prefix patched in place, so the whole file is
// never held in memory.
bool export_fp8_checkpoint(const char* dir, CheckpointReport* report,
                           const std::string& out_path,
                           bool per_channel, std::string* error);

}  // namespace granite4

// ---------------------------------------------------------------------------
// Narrow C ABI (Phase 1 additions).
//
// granite4_verify_checkpoint streams a full verification of checkpoint_dir
// (config + all shards, checksums included) and writes a JSON report into
// the caller-owned buffer. Returns 1 when the buffer was large enough and
// the report fully written (used is set to the written size, excluding any
// trailing NUL); 0 when buf is non-null but too small (used is set to the
// required size, NUL included). The JSON is
// {"schema":"granite4.checkpoint-report.v1","ok":...,"issues":[...],
//  "notes":[...],"tensors":N,"total_bytes":N,"total_numel":N,
//  "config_ok":...,"sha256_shards":{"file":...},"sha256_tensors":{"name":...}}
// ---------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

int granite4_verify_checkpoint(const char* checkpoint_dir, char* buf,
                               long buf_len, long* used);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GRANITE4_CHECKPOINT_H
