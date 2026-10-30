// Phase 3 (Integration): inference CLI.
//
// granite4-infer <subcommand>
//
//   forward --checkpoint DIR --tokens "id,id,..."
//           [--last N] [--gpu 1]
//     Load the model (original BF16 or an FP8 export), run the deterministic
//     forward pass over the given token ids, and write the output logits as
//     JSON. `--last N` restricts output to the final N positions (default:
//     all). This is the primitive the golden parity gate compares against a
//     transformers reference (same checkpoint, same tokens -> same logits,
//     up to float rounding). With `--gpu 1` the forward runs on the device
//     (forward_gpu.cu) instead of the host oracle; the host oracle remains
//     the reference (parity is checked by running both and diffing).
//
//   generate --checkpoint DIR --tokens "id,..." --max-tokens N
//            [--seed S] [--gpu 1] [--draft-lookup K]
//     Greedy (seed=0) or seeded top-p generation, emitting the produced
//     token ids as JSON. (Tokenization/detokenization is done by the caller -
//     the Python harness - so this path is token-level, not text-level.)
//     With `--gpu 1` the per-step forward runs on the device.
//     `--draft-lookup K` (greedy only) proposes up to K repeated spans from
//     the token history, verifies each proposal against the target's greedy
//     argmax one position at a time, and commits the accepted prefix. It
//     prints `G4_DRAFT steps=... proposed=... accepted=...` to stderr so the
//     acceptance rate can be measured before the fused k-token verify path
//     is integrated.
//
// Tokenization is intentionally NOT done here: the checkpoint ships an HF
// `tokenizer.json` that the Python harness loads with the `tokenizers`
// library, keeping token-for-token parity on one side.

#include "checkpoint.h"
#include "draft_lookup.h"
#include "kernel_math.h"
#include "speculative_verify.h"
#include "weights.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
#include "forward_gpu_api.h"
#include "forward_gpu.h"
#include <cuda_runtime.h>
#endif

using namespace granite4;

namespace {

// ---------------------------------------------------------------------------
// GPU forward support (compiled only when GRANITE_CUDA).
//
// The host loads dequantized float weights (ModelWeights, weights.h). To
// feed the device pass we re-pack them into the GpuWeightsLayout
// (forward_gpu.h): the GEMM weight tensors are quantized to FP8 e4m3 with a
// per-tensor scale (max_abs/448 - the P1 storage decision), while the
// embeddings and norm weights stay BF16 (the original checkpoint dtype).
// ---------------------------------------------------------------------------
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1

#define G4_HOST_CUDA_CHECK(expr)                                              \
  do {                                                                        \
    cudaError_t e_ = (expr);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "granite4: host CUDA error at %s:%d: %s\n",        \
                   __FILE__, __LINE__, cudaGetErrorString(e_));               \
      return false;                                                           \
    }                                                                         \
  } while (0)

struct GpuModel {
  GpuWeightsLayout lay;
  std::vector<std::uint8_t> weights;  // packed (GpuWeightsLayout)
  std::vector<float> scales;          // [lay.num_scales()]
};

// Convert a dequantized float row-major matrix to BF16 (2 bytes/element).
// Float values here are exact BF16-decoded originals, so the conversion is
// lossless.
void push_bf16(std::vector<std::uint8_t>* buf, const float* src,
                std::size_t n) {
  buf->resize(buf->size() + 2 * n);
  std::uint8_t* p = buf->data() + buf->size() - 2 * n;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint16_t b = f32_to_bf16_host(src[i]);
    p[2 * i] = static_cast<std::uint8_t>(b & 0xFF);
    p[2 * i + 1] = static_cast<std::uint8_t>(b >> 8);
  }
}

// Quantize a float row-major matrix [rows][cols] to FP8 e4m3 with a
// per-tensor scale (max_abs/448). Returns the scale (0 if the tensor is all
// zero).
float push_fp8_pt(std::vector<std::uint8_t>* buf, const float* src,
                  std::size_t rows, std::size_t cols) {
  double mx = 0.0;
  for (std::size_t i = 0; i < rows * cols; ++i) {
    const double a = std::abs(static_cast<double>(src[i]));
    if (a > mx) mx = a;
  }
  const float scale = static_cast<float>(mx / kFp8E4M3MaxF);
  buf->resize(buf->size() + rows * cols);
  std::uint8_t* p = buf->data() + buf->size() - rows * cols;
  if (scale <= 0.0f) {
    std::memset(p, 0, rows * cols);
    return 0.0f;
  }
  for (std::size_t i = 0; i < rows * cols; ++i) {
    bool sat = false;
    p[i] = fp8_e4m3_encode(static_cast<double>(src[i]) /
                               static_cast<double>(scale),
                           &sat);
  }
  return scale;
}

bool pack_gpu_model(const ModelWeights& w, GpuModel* out, std::string* error) {
  const GpuWeightsLayout lay{w.model.hidden_size, w.model.intermediate_size,
                             w.model.vocab_size, w.model.num_layers,
                             w.model.num_attention_heads,
                             w.model.num_key_value_heads, w.model.head_dim};
  out->lay = lay;
  auto& buf = out->weights;
  auto& scales = out->scales;
  buf.clear();
  scales.clear();
  buf.reserve(lay.total_bytes());
  scales.reserve(lay.num_scales());

  // embed [vocab][hidden] BF16.
  push_bf16(&buf, w.embed.data(), lay.vocab * lay.hidden);

  // Per-layer float offsets (the oracle's safetensors order, kernel_math.h).
  const std::size_t H = lay.hidden, I = lay.intermediate;
  const std::size_t KV = lay.num_kv_heads * lay.head_dim;
  const std::size_t off_post_ln = H;
  const std::size_t off_q = 2 * H;
  const std::size_t off_k = off_q + H * H;
  const std::size_t off_v = off_k + KV * H;
  const std::size_t off_o = off_v + KV * H;
  const std::size_t off_gate = off_o + H * H;
  const std::size_t off_up = off_gate + I * H;
  const std::size_t off_down = off_up + I * H;
  const std::size_t layer_f = off_down + H * I;

  for (std::size_t layer = 0; layer < lay.num_layers; ++layer) {
    const float* const lb =
        w.layers.data() + layer * (w.layer_bytes > 0 ? w.layer_bytes : layer_f);
    // Norm weights stay BF16.
    push_bf16(&buf, lb, H);
    push_bf16(&buf, lb + off_post_ln, H);
    // GEMM weights: FP8 per-tensor (order: q, k, v, o, gate, up, down).
    scales.push_back(push_fp8_pt(&buf, lb + off_q, H, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_k, KV, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_v, KV, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_o, H, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_gate, I, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_up, I, H));
    scales.push_back(push_fp8_pt(&buf, lb + off_down, H, I));
  }

  // final_norm BF16, lm_head FP8.
  push_bf16(&buf, w.final_norm.data(), lay.hidden);
  scales.push_back(
      push_fp8_pt(&buf, w.lm_head.data(), lay.vocab, lay.hidden));

  if (buf.size() != lay.total_bytes()) {
    *error = "gpu pack: buffer size mismatch";
    return false;
  }
  if (scales.size() != lay.num_scales()) {
    *error = "gpu pack: scale count mismatch";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// GPU inference session (P2: persistent weights + incremental decode).
//
// The ~9.5 GB packed weights are uploaded ONCE (gpu_weights_load) and every
// step runs incrementally on the device: gpu_forward_range processes only
// positions [written, end) because the KV history persists in the ctx
// caches. Token/logits buffers are preallocated for max_pos positions; per
// step only the newly computed logits rows are copied back.
// ---------------------------------------------------------------------------
struct GpuSession {
  GpuCtx* ctx = nullptr;
  std::int32_t* d_tokens = nullptr;   // [max_pos]
  std::uint16_t* d_logits = nullptr;  // [max_pos * vocab]
  int max_pos = 0;
  std::int32_t vocab = 0;

  ~GpuSession() { close(); }

  // Open the session: create the ctx and upload the packed weights. `gm`
  // (and `w.model`) must outlive the session.
  bool open(const GpuModel& gm, const ModelConfig& cfg, int max_pos,
            std::string* error) {
    this->max_pos = max_pos;
    this->vocab = static_cast<std::int32_t>(cfg.vocab_size);
    char err[256] = "";
    ctx = gpu_ctx_create(cfg, kForwardSpec, max_pos);
    if (ctx == nullptr) {
      *error = "gpu_ctx_create failed";
      return false;
    }
    if (gpu_weights_load(ctx, gm.weights.data(), gm.weights.size(),
                         gm.scales.data(), gm.scales.size(), err, sizeof err) != 0) {
      *error = std::string("gpu_weights_load: ") + err;
      gpu_ctx_free(ctx);
      ctx = nullptr;
      return false;
    }
    G4_HOST_CUDA_CHECK(
        cudaMalloc(&d_tokens, sizeof(std::int32_t) * static_cast<std::size_t>(max_pos)));
    G4_HOST_CUDA_CHECK(cudaMalloc(
        &d_logits, sizeof(std::uint16_t) *
                       static_cast<std::size_t>(max_pos) *
                       static_cast<std::size_t>(vocab)));
    return true;
  }

  void close() {
    if (d_tokens != nullptr) cudaFree(d_tokens);
    d_tokens = nullptr;
    if (d_logits != nullptr) cudaFree(d_logits);
    d_logits = nullptr;
    if (ctx != nullptr) gpu_ctx_free(ctx);
    ctx = nullptr;
  }

  // Forward for tokens[0, end): incremental (positions below the KV
  // watermark are skipped). Reads the computed rows back into `logits`
  // ([end * vocab] BF16, resized).
  bool step(const std::int32_t* tokens, int end,
            std::vector<std::uint16_t>* logits, std::string* error) {
    const int start = gpu_ctx_written(ctx);
    G4_HOST_CUDA_CHECK(cudaMemcpy(d_tokens, tokens,
                                  sizeof(std::int32_t) * static_cast<std::size_t>(end),
                                  cudaMemcpyHostToDevice));
    char err[256] = "";
    if (gpu_forward_range(ctx, d_tokens, end, d_logits, err, sizeof err) != 0) {
      *error = std::string("gpu_forward_range: ") + err;
      return false;
    }
    const int row_from = (end > start) ? start : 0;
    logits->resize(static_cast<std::size_t>(end) * static_cast<std::size_t>(vocab));
    G4_HOST_CUDA_CHECK(cudaMemcpy(logits->data() +
                                      static_cast<std::size_t>(row_from) *
                                          static_cast<std::size_t>(vocab),
                                  d_logits +
                                      static_cast<std::size_t>(row_from) *
                                          static_cast<std::size_t>(vocab),
                                  sizeof(std::uint16_t) *
                                      static_cast<std::size_t>(end - row_from) *
                                      static_cast<std::size_t>(vocab),
                                  cudaMemcpyDeviceToHost));
    return true;
  }

  // Discard every cached position: the next step replays from position 0.
  // Used to drop KV rows written for rejected speculative drafts.
  void reset() { gpu_ctx_reset(ctx); }

  // Keep the accepted prefix and discard only speculative rows.
  bool truncate(int written) { return gpu_ctx_truncate(ctx, written) == 0; }
};

#endif  // GRANITE_CUDA

// ---------------------------------------------------------------------------

bool parse_args(int argc, char** argv, std::string* sub, std::string* dir,
                std::vector<std::int32_t>* tokens, int* max_tokens, long* seed,
                int* last, int* gpu, int* draft_k) {
  *sub = "";
  *dir = "";
  *max_tokens = 0;
  *seed = 0;
  *last = 0;
  *gpu = 0;
  *draft_k = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "forward" || a == "generate") {
      *sub = a;
    } else if (a == "--checkpoint" && i + 1 < argc) {
      *dir = argv[++i];
    } else if (a == "--tokens" && i + 1 < argc) {
      const std::string t = argv[++i];
      std::stringstream ss(t);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) tokens->push_back(std::stoi(tok));
      }
    } else if (a == "--max-tokens" && i + 1 < argc) {
      *max_tokens = std::atoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      *seed = std::atol(argv[++i]);
    } else if (a == "--last" && i + 1 < argc) {
      *last = std::atoi(argv[++i]);
    } else if (a == "--gpu" && i + 1 < argc) {
      *gpu = std::atoi(argv[++i]);
    } else if (a == "--draft-lookup" && i + 1 < argc) {
      *draft_k = std::atoi(argv[++i]);
    } else {
      return false;
    }
  }
  return !sub->empty() && !dir->empty() && !tokens->empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string sub, dir, err;
  std::vector<std::int32_t> tokens;
  int max_tokens = 0, last = 0, gpu = 0, draft_k = 0;
  long seed = 0;
  if (!parse_args(argc, argv, &sub, &dir, &tokens, &max_tokens, &seed, &last,
                  &gpu, &draft_k)) {
    std::fprintf(
        stderr,
        "usage: granite4-infer forward --checkpoint DIR --tokens id,id,... "
        "[--last N] [--gpu 0|1]\n"
        "       granite4-infer generate --checkpoint DIR --tokens id,id,... "
        "--max-tokens N [--seed S] [--gpu 0|1] [--draft-lookup K]\n");
    return 2;
  }

  ModelWeights w;
  if (!load_weights(dir, &w, &err, /*enforce_spec=*/true)) {
    std::fprintf(stderr, "load_weights: %s\n", err.c_str());
    return 1;
  }

  if (sub == "forward") {
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
    if (gpu != 0) {
      GpuModel gm;
      if (!pack_gpu_model(w, &gm, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
      }
      const int max_pos = static_cast<int>(tokens.size());
      GpuSession session;
      std::vector<std::uint16_t> logits;
      if (!session.open(gm, w.model, max_pos, &err) ||
          !session.step(tokens.data(), max_pos, &logits, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
      }

      // Emit the final N positions (default: all) as JSON.
      const int n = static_cast<int>(tokens.size());
      const int from = last > 0 ? std::max(0, n - last) : 0;
      std::printf("{\"logits\":[");
      for (int t = from; t < n; ++t) {
        if (t > from) std::printf(",");
        std::printf("{\"position\":%d,\"values\":[", t);
        for (std::size_t v = 0; v < gm.lay.vocab; ++v) {
          if (v) std::printf(",");
          std::printf("%.9g",
                      bf16_to_f32_host(
                          logits[static_cast<std::size_t>(t) * gm.lay.vocab +
                                 v]));
        }
        std::printf("]}");
      }
      std::printf("]}\n");
      return 0;
    }
#endif  // GRANITE_CUDA
    if (gpu != 0) {
      std::fprintf(stderr, "granite4-infer was built without CUDA support\n");
      return 1;
    }
    // Host oracle path.
    std::vector<float> logits;
    forward_logits_f32(tokens, w.embed, w.layers, w.final_norm.data(),
                       w.lm_head.data(), w.model, kForwardSpec, &logits);
    const int n = static_cast<int>(tokens.size());
    const int from = last > 0 ? std::max(0, n - last) : 0;
    std::printf("{\"logits\":[");
    for (int t = from; t < n; ++t) {
      if (t > from) std::printf(",");
      std::printf("{\"position\":%d,\"values\":[", t);
      for (std::size_t v = 0; v < w.model.vocab_size; ++v) {
        if (v) std::printf(",");
        std::printf("%.9g",
                    logits[static_cast<std::size_t>(t) * w.model.vocab_size +
                           v]);
      }
      std::printf("]}");
    }
    std::printf("]}\n");
    return 0;
  }

  if (sub == "generate") {
    if (max_tokens <= 0) {
      std::fprintf(stderr, "generate: --max-tokens required\n");
      return 2;
    }
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
    if (gpu != 0) {
      GpuModel gm;
      if (!pack_gpu_model(w, &gm, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
      }
      const int max_pos =
          static_cast<int>(tokens.size()) + max_tokens +
          (draft_k > 0 ? draft_k : 0);
      GpuSession session;
      std::vector<std::uint16_t> logits;
      if (!session.open(gm, w.model, max_pos, &err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
      }
      const int V = static_cast<int>(gm.lay.vocab);

      std::vector<std::int32_t> seq = tokens;
      std::vector<std::int32_t> generated;
      // Sample one token from a logits row: greedy for seed=0, seeded top-p
      // otherwise (the draft path requires greedy so verification is exact).
      auto sample_row = [&](const std::uint16_t* row, int step_index) -> int {
        std::vector<float> p(V);
        for (int i = 0; i < V; ++i) p[i] = bf16_to_f32_host(row[i]);
        if (seed == 0) return sample_greedy_f32(p.data(), V);
        Xorshift64 rng(static_cast<std::uint64_t>(seed) +
                       static_cast<std::uint64_t>(step_index));
        return sample_top_p_f32(p.data(), /*nucleus=*/0.9f, rng.next_float(), V);
      };

      if (draft_k > 0) {
        if (seed != 0) {
          std::fprintf(stderr, "draft-lookup: greedy only (omit --seed)\n");
          return 2;
        }
        long proposed_total = 0, accepted_total = 0;
        int steps = 0;
        std::vector<float> row_f(V);
        std::vector<std::int32_t> target_arg;
        auto argmax_f32 = [](const float* p, int n) {
          int best = 0;
          for (int c = 1; c < n; ++c)
            if (p[c] > p[best]) best = c;
          return best;
        };
        while (static_cast<int>(generated.size()) < max_tokens) {
          // Replay the accepted prefix from position 0; the forwards below
          // advance the KV watermark, and reset discards it.
          session.reset();
          const int len = static_cast<int>(seq.size());

          // Never propose past the output budget or the ctx's position limit.
          const int room =
              std::min(draft_k, max_tokens - static_cast<int>(generated.size()));
          const int pos_room = max_pos - static_cast<int>(seq.size());
          const int k = std::min(room, pos_room);
          const std::vector<std::int32_t> proposal =
              (k > 0) ? draft_lookup(seq, k) : std::vector<std::int32_t>();
          const int nprop = static_cast<int>(proposal.size());
          proposed_total += nprop;

          // One target forward over accepted prefix + draft proposals. The
          // rows at positions [len, len+nprop) verify each proposal; row
          // len+nprop is the target's own next token (correction when the
          // draft is rejected, bonus when it is fully accepted).
          std::vector<std::int32_t> cand(seq);
          cand.insert(cand.end(), proposal.begin(), proposal.end());
          if (!session.step(cand.data(), static_cast<int>(cand.size()), &logits, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
          }
          target_arg.assign(nprop + 1, 0);
          for (int i = 0; i <= nprop; ++i) {
            const std::uint16_t* row =
                logits.data() + static_cast<std::size_t>(len - 1 + i) * V;
            for (int c = 0; c < V; ++c) row_f[c] = bf16_to_f32_host(row[c]);
            target_arg[i] = argmax_f32(row_f.data(), V);
          }
          // The unit-tested gate: accept the longest prefix where the target
          // agrees with the draft, plus one correction (or bonus) token.
          const speculative::VerifyResult v =
              speculative::verify_greedy(target_arg.data(), proposal.data(),
                                         nprop);
          (void)logits;  // rows already consumed via target_arg
          // Commit the accepted prefix, then the gate's tail token: the
          // correction on rejection, the target's own row (bonus) when the
          // draft was fully accepted.
          // Clamp the commit to the output budget (the tail token is the
          // first to drop when the draft over-shoots it).
          const int budget = max_tokens - static_cast<int>(generated.size());
          const int commit = std::min(v.accepted + 1, budget);
          for (int i = 0; i < commit; ++i) {
            const int tok = (i < v.accepted)
                                ? proposal[i]
                                : (v.correction >= 0 ? v.correction
                                                    : target_arg[nprop]);
            seq.push_back(tok);
            generated.push_back(tok);
          }
          accepted_total += v.accepted;
          ++steps;
          session.reset();
        }
        std::fprintf(stderr,
                     "G4_DRAFT steps=%d proposed=%ld accepted=%ld rate=%.3f\n",
                     steps, proposed_total, accepted_total,
                     proposed_total > 0
                         ? static_cast<double>(accepted_total) /
                               static_cast<double>(proposed_total)
                         : 0.0);
        std::printf("{\"generated\":[");
        for (std::size_t i = 0; i < generated.size(); ++i) {
          if (i) std::printf(",");
          std::printf("%d", generated[i]);
        }
        std::printf("]}\n");
        return 0;
      }

      // Prefill: forward the whole prompt once (populates the KV caches and
      // leaves the prompt's final-position row ready to sample from).
      {
        const int len = static_cast<int>(seq.size());
        if (!session.step(seq.data(), len, &logits, &err)) {
          std::fprintf(stderr, "%s\n", err.c_str());
          return 1;
        }
      }
      for (int step = 0; step < max_tokens; ++step) {
        // Sample from the final position's row (computed by the prefill or
        // the previous step's forward).
        const int len = static_cast<int>(seq.size());
        const std::uint16_t* row =
            logits.data() +
            static_cast<std::size_t>(len - 1) * static_cast<std::size_t>(V);
        const int next = sample_row(row, step);
        seq.push_back(next);
        generated.push_back(next);
        // Compute the new position's row (skipped on the last step - it is
        // never sampled).
        if (step + 1 < max_tokens) {
          const int len2 = static_cast<int>(seq.size());
          if (!session.step(seq.data(), len2, &logits, &err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 1;
          }
        }
      }
      std::printf("{\"generated\":[");
      for (std::size_t i = 0; i < generated.size(); ++i) {
        if (i) std::printf(",");
        std::printf("%d", generated[i]);
      }
      std::printf("]}\n");
      return 0;
    }
#endif  // GRANITE_CUDA
    if (gpu != 0) {
      std::fprintf(stderr, "granite4-infer was built without CUDA support\n");
      return 1;
    }

    // Host oracle path.
    std::vector<std::int32_t> seq = tokens;
    std::vector<std::int32_t> generated;
    std::vector<float> logits;
    for (int step = 0; step < max_tokens; ++step) {
      forward_logits_f32(seq, w.embed, w.layers, w.final_norm.data(),
                         w.lm_head.data(), w.model, kForwardSpec, &logits);
      const int len = static_cast<int>(seq.size());
      const float* row =
          logits.data() + static_cast<std::size_t>(len - 1) * w.model.vocab_size;
      const int n = static_cast<int>(w.model.vocab_size);
      int next;
      if (seed == 0) {
        next = sample_greedy_f32(row, n);
      } else {
        Xorshift64 rng(static_cast<std::uint64_t>(seed) +
                       static_cast<std::uint64_t>(step));
        const float u = rng.next_float();
        next = sample_top_p_f32(row, /*nucleus=*/0.9f, u, n);
      }
      seq.push_back(next);
      generated.push_back(next);
    }
    std::printf("{\"generated\":[");
    for (std::size_t i = 0; i < generated.size(); ++i) {
      if (i) std::printf(",");
      std::printf("%d", generated[i]);
    }
    std::printf("]}\n");
    return 0;
  }

  return 2;
}
