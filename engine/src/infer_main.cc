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
//     With `--gpu 1` the forward runs on the device: one prefill call for the
//     prompt, then greedy decode chained on the device (gpu_decode_greedy:
//     one graph launch per token, ids read back once at the end).
//     `--draft-lookup K` (greedy only) proposes up to K repeated spans from
//     the token history, verifies each proposal against the target's greedy
//     argmax in one batched forward, and commits the accepted prefix. It
//     reports `G4_DRAFT steps=... proposed=... accepted=...` to stderr, and
//     echoes the same counters in the stdout JSON as
//     `{"generated":[...],"draft":{"steps":S,"proposed":P,"accepted":A,
//     "accept_rate":R}}` so a benchmark harness can parse one document.
//     `--eos ID` stops generation after the first ID (emitted, then nothing
//     more); without it exactly --max-tokens tokens are produced.
//     Works on the CUDA path (draft-verify against the device logits) and on
//     the host oracle path (draft-verify against the f32 host logits); the
//     host path replays the full accepted prefix each step because the host
//     oracle has no incremental KV watermark.
//
//     `--eagle HEAD.safetensors --eagle-k K` (greedy only) drafts K tokens per
//     round with a trained EAGLE head (eagle_gpu.h) that reads the target's
//     final-normed hidden states, then verifies anchor + drafts in one
//     batched forward (exact greedy gate; output is identical to plain greedy
//     decoding). Reports the same draft counters as --draft-lookup.
//
//   dump-hidden --checkpoint DIR --gpu 1 --in TOKENS.bin --out HIDDEN.bin
//               [--max-context N]
//     Training data for a draft head: TOKENS.bin is a sequence of records
//     (int32 n, then n int32 token ids), each n <= --max-context (default
//     8192). For every record, one forward from an empty cache writes the n
//     final-normed hidden states (the lm_head inputs, n x hidden bf16,
//     little-endian) to HIDDEN.bin, records back to back in input order.
//
//   serve --checkpoint DIR --gpu 1 [--max-context N]
//     Persistent engine: loads the model once (default 8192 positions per
//     request), prints {"ready":true,"max_context":N}, then answers one JSON
//     request per stdin line - {"tokens":[...],"max_tokens":N,"seed":S,
//     "draft_lookup":K,"eos":ID} (all but tokens/max_tokens optional) - with
//     one generate document per stdout line, or {"error":"..."}. Requests
//     run one at a time; EOF on stdin exits. With `--eagle HEAD` the head is
//     loaded once and a request's "eagle":K drafts K tokens per round.
//
// Tokenization is intentionally NOT done here: the checkpoint ships an HF
// `tokenizer.json` that the Python harness loads with the `tokenizers`
// library, keeping token-for-token parity on one side.
//
// G4_LOAD_TIMING=1 prints the load / pack / upload wall times to stderr.

#include "checkpoint.h"
#include "draft_lookup.h"
#include "json.h"
#include "kernel_math.h"
#include "speculative_verify.h"
#include "weights.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
#include "eagle_gpu.h"
#include "forward_gpu.h"
#include "forward_gpu_api.h"
#include "gpu_pack.h"
#include "kv_pool_controller.h"
#include "serve_resume.h"
#include <cuda_runtime.h>
#endif

using namespace granite4;

namespace {

struct Args {
  std::string sub, dir, in, out, eagle;
  std::vector<std::int32_t> tokens;
  int max_tokens = 0, last = 0, gpu = 0, draft_k = 0, eagle_k = 0;
  long seed = 0;
  long eos = -1;  // stop token (-1: none)
  int max_context = 8192;  // serve: positions per request
  int sessions = 1;        // serve: number of resident sessions sharing weights + one KV pool
  int kv_pool_tokens = 0;  // serve: shared KV pool size in positions (0 = sessions x max_context)
  int bench_ctx = 0;       // kv-bench: prefill token count
  int bench_decode = 0;    // kv-bench: decode token count

  bool is_eos(std::int32_t tok) const { return eos >= 0 && tok == eos; }
};

bool parse_args(int argc, char** argv, Args* a) {
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "forward" || s == "generate" || s == "serve" || s == "dump-hidden" ||
        s == "kv-bench") {
      a->sub = s;
    } else if (s == "--checkpoint" && i + 1 < argc) {
      a->dir = argv[++i];
    } else if (s == "--tokens" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      std::string tok;
      while (std::getline(ss, tok, ','))
        if (!tok.empty()) a->tokens.push_back(std::stoi(tok));
    } else if (s == "--max-tokens" && i + 1 < argc) {
      a->max_tokens = std::atoi(argv[++i]);
    } else if (s == "--seed" && i + 1 < argc) {
      a->seed = std::atol(argv[++i]);
    } else if (s == "--last" && i + 1 < argc) {
      a->last = std::atoi(argv[++i]);
    } else if (s == "--gpu" && i + 1 < argc) {
      a->gpu = std::atoi(argv[++i]);
    } else if (s == "--draft-lookup" && i + 1 < argc) {
      a->draft_k = std::atoi(argv[++i]);
    } else if (s == "--eos" && i + 1 < argc) {
      a->eos = std::atol(argv[++i]);
    } else if (s == "--max-context" && i + 1 < argc) {
      a->max_context = std::atoi(argv[++i]);
    } else if (s == "--sessions" && i + 1 < argc) {
      a->sessions = std::atoi(argv[++i]);
    } else if (s == "--kv-pool-tokens" && i + 1 < argc) {
      a->kv_pool_tokens = std::atoi(argv[++i]);
    } else if (s == "--bench-ctx" && i + 1 < argc) {
      a->bench_ctx = std::atoi(argv[++i]);
    } else if (s == "--bench-decode" && i + 1 < argc) {
      a->bench_decode = std::atoi(argv[++i]);
    } else if (s == "--eagle" && i + 1 < argc) {
      a->eagle = argv[++i];
    } else if (s == "--eagle-k" && i + 1 < argc) {
      a->eagle_k = std::atoi(argv[++i]);
    } else if (s == "--in" && i + 1 < argc) {
      a->in = argv[++i];
    } else if (s == "--out" && i + 1 < argc) {
      a->out = argv[++i];
    } else {
      return false;
    }
  }
  if (a->sub == "serve")
    return !a->dir.empty() && a->max_context > 0 && a->sessions > 0;
  if (a->sub == "dump-hidden")
    return !a->dir.empty() && !a->in.empty() && !a->out.empty() && a->max_context > 0;
  if (a->sub == "kv-bench")
    return !a->dir.empty() && a->bench_ctx >= 2 && a->bench_decode > 0;
  return !a->sub.empty() && !a->dir.empty() && !a->tokens.empty();
}

// Emit rows [from, n) of a [n][vocab] logits table as JSON.
template <typename Get>
void print_logits(int from, int n, std::size_t vocab, const Get& get) {
  std::string out = "{\"logits\":[";
  char buf[32];
  for (int t = from; t < n; ++t) {
    if (t > from) out += ',';
    out += "{\"position\":" + std::to_string(t) + ",\"values\":[";
    for (std::size_t v = 0; v < vocab; ++v) {
      if (v) out += ',';
      const int len = std::snprintf(buf, sizeof buf, "%.9g", get(t, v));
      out.append(buf, static_cast<std::size_t>(len));
    }
    out += "]}";
  }
  out += "]}\n";
  std::fwrite(out.data(), 1, out.size(), stdout);
}

int argmax_f32(const float* p, int n) {
  int best = 0;
  for (int c = 1; c < n; ++c)
    if (p[c] > p[best]) best = c;
  return best;
}

void report_draft(const DraftReport& report) {
  std::fprintf(stderr, "G4_DRAFT steps=%ld proposed=%ld accepted=%ld rate=%.3f\n",
               report.steps, report.proposed, report.accepted, draft_accept_rate(report));
}

class LoadTimer {
 public:
  LoadTimer() : on_(std::getenv("G4_LOAD_TIMING") != nullptr) {}
  void lap(const char* what) {
    const auto now = std::chrono::steady_clock::now();
    if (on_)
      std::fprintf(stderr, "G4_LOAD %s=%.2fs\n", what,
                   std::chrono::duration<double>(now - last_).count());
    last_ = now;
  }

 private:
  bool on_;
  std::chrono::steady_clock::time_point last_ = std::chrono::steady_clock::now();
};

// ---------------------------------------------------------------------------
// GPU paths (compiled only when GRANITE_CUDA).
// ---------------------------------------------------------------------------
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1

#define G4_HOST_CUDA_CHECK(expr)                                              \
  do {                                                                        \
    cudaError_t e_ = (expr);                                                  \
    if (e_ != cudaSuccess) {                                                  \
      *error = std::string("cuda: ") + cudaGetErrorString(e_) + " at " +      \
               std::to_string(__LINE__);                                      \
      return false;                                                           \
    }                                                                         \
  } while (0)

// One sequence on the device. The packed weights (streamed straight from the
// checkpoint, gpu_pack.h) are uploaded once; the KV history persists in the
// ctx, so every call forwards only the positions past the watermark. The
// host keeps the token sequence; only new tokens go up, and only argmax ids
// (or the one logits row a sampler needs) come back.
class GpuSession {
 public:
  ~GpuSession() {
    cudaFree(d_feat_);
    cudaFree(d_feat_own_);
    cudaFree(d_tokens_);
    cudaFree(d_logits_);
    cudaFree(d_argmax_);
    if (ctx_ != nullptr) gpu_ctx_free(ctx_);
  }

  // logit_rows: capacity of the logits buffer in rows.
  bool open(const std::string& dir, int max_pos, int logit_rows, std::string* error) {
    LoadTimer timer;
    dir_ = dir;
    ModelConfig cfg;
    if (!load_model_config(dir, &cfg, error, /*enforce_spec=*/true)) return false;
    GpuModel gm;
    if (!pack_gpu_model_from_checkpoint(dir, cfg, &gm, error)) return false;
    timer.lap("pack");
    max_pos_ = max_pos;
    vocab_ = static_cast<int>(cfg.vocab_size);
    hidden_ = static_cast<int>(cfg.hidden_size);
    logit_rows_ = logit_rows;
    ctx_ = gpu_ctx_create(cfg, kForwardSpec, max_pos);
    if (ctx_ == nullptr) {
      *error = "gpu_ctx_create failed";
      return false;
    }
    char err[256] = "";
    if (gpu_weights_load(ctx_, gm.weights.get(), gm.bytes, gm.scales.data(),
                         gm.scales.size(), err, sizeof err) != 0) {
      *error = std::string("gpu_weights_load: ") + err;
      return false;
    }
    timer.lap("upload");
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_tokens_, sizeof(std::int32_t) * max_pos));
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_argmax_, sizeof(std::int32_t) * logit_rows));
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_logits_, sizeof(std::uint16_t) *
                                                  static_cast<std::size_t>(logit_rows) *
                                                  vocab_));
    return true;
  }

  // Open a session that ATTACHES to a shared weight object instead of
  // uploading its own copy. The caller owns `weights` and keeps it alive
  // for the lifetime of every session using it - this is what lets N
  // sessions coexist in VRAM at 9.4 + N x 1.5 GiB instead of N x 9.5 GiB.
  //
  // With `arena`, the session's KV lives in that shared paged arena instead of
  // a private [max_pos] cache; it has no rows until map_rows() gives it some.
  bool open_shared(const std::string& dir, int max_pos, int logit_rows,
                   GpuWeights* weights, std::string* error,
                   GpuKvArena* arena = nullptr) {
    dir_ = dir;
    ModelConfig cfg;
    if (!load_model_config(dir, &cfg, error, /*enforce_spec=*/true)) return false;
    max_pos_ = max_pos;
    vocab_ = static_cast<int>(cfg.vocab_size);
    hidden_ = static_cast<int>(cfg.hidden_size);
    logit_rows_ = logit_rows;
    ctx_ = arena != nullptr ? gpu_ctx_create_paged(cfg, kForwardSpec, max_pos, arena)
                            : gpu_ctx_create(cfg, kForwardSpec, max_pos);
    if (ctx_ == nullptr) {
      *error = "gpu_ctx_create failed";
      return false;
    }
    if (gpu_ctx_use_weights(ctx_, weights) != 0) {
      *error = "gpu_ctx_use_weights failed";
      return false;
    }
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_tokens_, sizeof(std::int32_t) * max_pos));
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_argmax_, sizeof(std::int32_t) * logit_rows));
    G4_HOST_CUDA_CHECK(cudaMalloc(&d_logits_, sizeof(std::uint16_t) *
                                               static_cast<std::size_t>(logit_rows) *
                                               vocab_));
    return true;
  }

  // Borrow `owner`'s prefill scratch and draft-feature buffers (~1 GiB at 32k)
  // instead of allocating a second set. Valid because the serve loop runs one
  // request at a time; call before this session's first forward.
  bool share_scratch_from(GpuSession* owner, std::string* error) {
    if (gpu_ctx_share_prefill_scratch(ctx_, owner->ctx_) != 0) {
      *error = "gpu_ctx_share_prefill_scratch failed";
      return false;
    }
    feat_owner_ = owner;
    return true;
  }
  // Publish the first n arena rows of this session's KV (see
  // gpu_ctx_set_kv_rows_n); positions past n cannot be forwarded.
  bool map_rows(const int* rows, int n) { return gpu_ctx_set_kv_rows_n(ctx_, rows, n) == 0; }
  int mapped() const { return gpu_ctx_mapped_rows(ctx_); }
  int vocab() const { return vocab_; }
  int hidden() const { return hidden_; }
  int max_pos() const { return max_pos_; }
  int logit_rows() const { return logit_rows_; }
  int written() const { return gpu_ctx_written(ctx_); }
  bool truncate(int written) { return gpu_ctx_truncate(ctx_, written) == 0; }
  // Resolved KV-cache mode (after the split-attention capability check in
  // gpu_ctx_create) and active attention path - so callers can report ground
  // truth instead of trusting G4_KV8.
  bool kv8() const { return gpu_ctx_is_kv8(ctx_); }
  bool kv4() const { return gpu_ctx_is_kv4(ctx_); }
  bool split_attn() const { return gpu_ctx_is_split_attn(ctx_); }


  bool upload(const std::vector<std::int32_t>& seq, int from, int to, std::string* error) {
    if (to <= from) return true;
    G4_HOST_CUDA_CHECK(cudaMemcpy(d_tokens_ + from, seq.data() + from,
                                  sizeof(std::int32_t) * (to - from),
                                  cudaMemcpyHostToDevice));
    return true;
  }

  // Forward [written, end); logits (and greedy ids) for rows >= logits_from,
  // and (with d_hidden) final-normed hidden states for rows >= hidden_from.
  bool forward(int end, int logits_from, std::string* error, std::uint16_t* d_hidden = nullptr,
               int hidden_from = -1) {
    if (end - logits_from > logit_rows_) {
      *error = "logits buffer too small";
      return false;
    }
    GpuForwardOut out;
    out.d_logits = d_logits_;
    out.logits_from = logits_from;
    out.d_argmax = d_argmax_;
    out.d_hidden = d_hidden;
    out.hidden_from = hidden_from;
    char err[256] = "";
    if (gpu_forward(ctx_, d_tokens_, end, out, err, sizeof err) != 0) {
      *error = std::string("gpu_forward: ") + err;
      return false;
    }
    return true;
  }

  // Forward [written, end) and write the final-normed hidden state of rows
  // [from, end) to d_hidden ([end - from][hidden], device); no lm_head.
  bool forward_hidden(int end, int from, std::uint16_t* d_hidden, std::string* error) {
    GpuForwardOut out;
    out.logits_from = from;
    out.d_hidden = d_hidden;
    char err[256] = "";
    if (gpu_forward(ctx_, d_tokens_, end, out, err, sizeof err) != 0) {
      *error = std::string("gpu_forward: ") + err;
      return false;
    }
    return true;
  }

  bool argmax(int rows, std::vector<std::int32_t>* ids, std::string* error) {
    ids->resize(static_cast<std::size_t>(rows));
    G4_HOST_CUDA_CHECK(cudaMemcpy(ids->data(), d_argmax_, sizeof(std::int32_t) * rows,
                                  cudaMemcpyDeviceToHost));
    return true;
  }

  bool logits(int rows, std::vector<std::uint16_t>* out, std::string* error) {
    out->resize(static_cast<std::size_t>(rows) * vocab_);
    G4_HOST_CUDA_CHECK(cudaMemcpy(out->data(), d_logits_, sizeof(std::uint16_t) * out->size(),
                                  cudaMemcpyDeviceToHost));
    return true;
  }

  // Greedy chain of `steps` positions from the watermark (see
  // gpu_decode_greedy); the ids land in d_tokens[written + 1 ...].
  bool decode_greedy(int steps, std::string* error) {
    char err[256] = "";
    if (gpu_decode_greedy(ctx_, d_tokens_, steps, d_logits_, err, sizeof err) != 0) {
      *error = std::string("gpu_decode_greedy: ") + err;
      return false;
    }
    return true;
  }

  // Device token ids [from, to) back to the host.
  bool tokens(int from, int to, std::vector<std::int32_t>* out, std::string* error) {
    out->resize(static_cast<std::size_t>(to - from));
    G4_HOST_CUDA_CHECK(cudaMemcpy(out->data(), d_tokens_ + from,
                                  sizeof(std::int32_t) * (to - from),
                                  cudaMemcpyDeviceToHost));
    return true;
  }

  // Enqueue a device -> host copy of ids [from, to) into `host` (pinned) and
  // record `done` after it; nothing waits.
  bool tokens_async(int from, int to, std::int32_t* host, cudaEvent_t done,
                    std::string* error) {
    G4_HOST_CUDA_CHECK(cudaMemcpyAsync(host, d_tokens_ + from,
                                       sizeof(std::int32_t) * (to - from),
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
    G4_HOST_CUDA_CHECK(cudaEventRecord(done, cudaStreamPerThread));
    return true;
  }

  GpuCtx* ctx() { return ctx_; }
  std::uint16_t* logits_buffer() { return d_logits_; }
  std::int32_t* argmax_buffer() { return d_argmax_; }
  std::int32_t* d_tokens() { return d_tokens_; }
  const std::string& dir() const { return dir_; }
  // Device scratch for draft-head features: [max_pos][hidden] (prompt) and
  // [kMaxVerifyRows][hidden] (verify rows, chain); allocated on first use.
  bool feature_buffers(std::uint16_t** prompt, std::uint16_t** verify, std::uint16_t** chain,
                       std::string* error) {
    if (feat_owner_ != nullptr) {
      // The [max_pos] prompt-feature buffer is shared (EagleRun uses it only
      // inside start()); the small verify / chain buffers are per session, so
      // two sessions can draft in the same batched round.
      std::uint16_t *p, *v, *c;
      if (!feat_owner_->feature_buffers(&p, &v, &c, error)) return false;
      if (d_feat_own_ == nullptr) {
        const std::size_t H = static_cast<std::size_t>(hidden_);
        G4_HOST_CUDA_CHECK(cudaMalloc(&d_feat_own_, 2 * H * 2 * kMaxVerifyRows));
      }
      *prompt = p;
      *verify = d_feat_own_;
      *chain = d_feat_own_ + static_cast<std::size_t>(kMaxVerifyRows) * hidden_;
      return true;
    }
    if (d_feat_ == nullptr) {
      const std::size_t H = static_cast<std::size_t>(hidden_);
      G4_HOST_CUDA_CHECK(cudaMalloc(&d_feat_, 2 * H * (max_pos_ + 2 * kMaxVerifyRows)));
    }
    *prompt = d_feat_;
    *verify = d_feat_ + static_cast<std::size_t>(max_pos_) * hidden_;
    *chain = *verify + static_cast<std::size_t>(kMaxVerifyRows) * hidden_;
    return true;
  }

  // Copy the first argmax id to d_tokens[at] on the device.
  bool argmax_to_token(int at, std::string* error) {
    G4_HOST_CUDA_CHECK(cudaMemcpy(d_tokens_ + at, d_argmax_, sizeof(std::int32_t),
                                  cudaMemcpyDeviceToDevice));
    return true;
  }

 private:
  GpuCtx* ctx_ = nullptr;
  std::int32_t* d_tokens_ = nullptr;   // [max_pos]
  std::uint16_t* d_logits_ = nullptr;  // [logit_rows][vocab]
  std::int32_t* d_argmax_ = nullptr;   // [logit_rows]
  std::uint16_t* d_feat_ = nullptr;    // draft-head features (feature_buffers)
  GpuSession* feat_owner_ = nullptr;   // borrowed feature buffers (share_scratch_from)
  std::uint16_t* d_feat_own_ = nullptr;  // per-session verify/chain rows when borrowing
  std::string dir_;
  int max_pos_ = 0, vocab_ = 0, hidden_ = 0, logit_rows_ = 0;
};

int fail(const std::string& err) {
  std::fprintf(stderr, "%s\n", err.c_str());
  return 1;
}

int gpu_forward_cmd(const Args& a) {
  std::string err;
  const int n = static_cast<int>(a.tokens.size());
  GpuSession s;
  std::vector<std::uint16_t> logits;
  if (!s.open(a.dir, n, n, &err) || !s.upload(a.tokens, 0, n, &err) ||
      !s.forward(n, 0, &err) || !s.logits(n, &logits, &err))
    return fail(err);
  const std::size_t V = static_cast<std::size_t>(s.vocab());
  print_logits(a.last > 0 ? std::max(0, n - a.last) : 0, n, V, [&](int t, std::size_t v) {
    return bf16_to_f32_host(logits[static_cast<std::size_t>(t) * V + v]);
  });
  return 0;
}

// Pinned host ids plus one event per decode chunk (EOS polling).
class TokenStream {
 public:
  ~TokenStream() {
    for (auto& e : events_) cudaEventDestroy(e);
    cudaFreeHost(host_);
  }
  bool init(int n, int chunks, std::string* error) {
    G4_HOST_CUDA_CHECK(cudaHostAlloc(&host_, sizeof(std::int32_t) * n, cudaHostAllocDefault));
    events_.resize(static_cast<std::size_t>(chunks), nullptr);
    for (auto& e : events_)
      G4_HOST_CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    return true;
  }
  std::int32_t* host() { return host_; }
  cudaEvent_t event(int i) { return events_[static_cast<std::size_t>(i)]; }

 private:
  std::int32_t* host_ = nullptr;
  std::vector<cudaEvent_t> events_;
};

// The generate modes run on an already open session (one-shot CLI or the
// persistent `serve` loop). When prefix >= 0, the session's KV watermark
// starts at `prefix` (the first `prefix` positions of a.tokens are already
// cached; the host uploads only the tail and forwards [prefix, plen)).
// When prefix < 0 (the CLI one-shot path and the first turn of a serve
// session), the cache is reset to empty and the full prompt is prefilled.
// Preconditions (checked by check_request): the final watermark
// plen + max_tokens (+ draft_k) fits the session's positions and
// draft_k + 1 fits its logits rows.
struct Generation {
  std::vector<std::int32_t> ids;
  DraftReport report;
  bool drafted = false;
};

bool check_request(const GpuSession& s, const Args& a, int prefix, std::string* error) {
  const long plen = static_cast<long>(a.tokens.size());
  // The KV watermark ends at the absolute position plen + max_tokens (+
  // draft_k), independent of `prefix` — the prefix saves prefill compute
  // (only [prefix, plen) is forwarded) but not KV capacity, since the cache
  // is indexed by absolute position. Bound on the final watermark.
  const long need = plen + a.max_tokens + a.draft_k +
                    (a.eagle_k > 0 ? a.eagle_k + 1 : 0);
  if (a.tokens.empty() || a.max_tokens <= 0) {
    *error = "empty prompt or max_tokens <= 0";
    return false;
  }
  if (prefix > static_cast<int>(plen)) {
    *error = "resume prefix exceeds prompt length";
    return false;
  }
  if (need > s.max_pos()) {
    *error = "prompt + max_tokens exceeds the context (" + std::to_string(s.max_pos()) + ")";
    return false;
  }
  if (a.draft_k < 0 || a.draft_k + 1 > s.logit_rows()) {
    *error = "draft_lookup out of range";
    return false;
  }
  if (a.draft_k > 0 && a.seed != 0) {
    *error = "draft-lookup: greedy only (omit --seed)";
    return false;
  }
  if (a.eagle_k < 0 || a.eagle_k + 1 > std::min(s.logit_rows(), kMaxVerifyRows) ||
      (a.eagle_k > 0 && (a.seed != 0 || a.draft_k > 0))) {
    *error = "eagle: K out of range, or combined with --seed / --draft-lookup";
    return false;
  }
  for (std::int32_t t : a.tokens)
    if (t < 0 || t >= s.vocab()) {
      *error = "token id out of range";
      return false;
    }
  return true;
}

bool generate_greedy(GpuSession& s, const Args& a, Generation* out, std::string* error,
                     int prefix = -1) {
  const int plen = static_cast<int>(a.tokens.size());
  // Prefill: the prompt's KV plus the last row's greedy token, written on the
  // device straight into d_tokens[plen]; the decode chain continues from it.
  // With prefix >= 0 the watermark already holds the first `prefix` positions,
  // so only the tail is uploaded and only [prefix, plen) is forwarded.
  const int p = prefix > 0 ? prefix : 0;
  if (!s.truncate(prefix >= 0 ? p : 0) || !s.upload(a.tokens, p, plen, error) ||
      !s.forward(plen, plen - 1, error) || !s.argmax_to_token(plen, error))
    return false;
  if (a.eos < 0) {
    // No stop token: the whole chain is enqueued at once and read back once.
    return (a.max_tokens == 1 || s.decode_greedy(a.max_tokens - 1, error)) &&
           s.tokens(plen, plen + a.max_tokens, &out->ids, error);
  }
  // Chunks of kChunk tokens, each followed by an async readback; chunk c + 1
  // is enqueued before chunk c is inspected, so the GPU never waits on the
  // EOS check. At most one chunk past the stop is computed and discarded.
  constexpr int kChunk = 8;
  const int n = a.max_tokens;
  const int chunks = (n + kChunk - 1) / kChunk;
  TokenStream ts;
  if (!ts.init(n, chunks, error)) return false;
  // Chunk c covers ids [c * kChunk, min(n, (c + 1) * kChunk)); id 0 exists
  // after the prefill, id i > 0 after decode step i - 1.
  auto enqueue = [&](int c) {
    const int lo = c * kChunk, hi = std::min(n, lo + kChunk);
    const int steps = hi - std::max(lo, 1);
    if (steps > 0 && !s.decode_greedy(steps, error)) return false;
    return s.tokens_async(plen + lo, plen + hi, ts.host() + lo, ts.event(c), error);
  };
  bool ok = enqueue(0);
  bool stopped = false;
  for (int c = 0; ok && c < chunks && !stopped; ++c) {
    if (c + 1 < chunks) ok = enqueue(c + 1);
    if (cudaEventSynchronize(ts.event(c)) != cudaSuccess) {
      *error = "event sync failed";
      ok = false;
    }
    for (int i = c * kChunk; ok && i < std::min(n, (c + 1) * kChunk); ++i) {
      out->ids.push_back(ts.host()[i]);
      if (a.is_eos(ts.host()[i])) {
        stopped = true;
        break;
      }
    }
  }
  cudaStreamSynchronize(cudaStreamPerThread);
  return ok;
}

bool generate_sampled(GpuSession& s, const Args& a, Generation* out, std::string* error,
                      int prefix = -1) {
  std::vector<std::int32_t> seq = a.tokens;
  const int plen = static_cast<int>(seq.size());
  const int pre = prefix > 0 ? prefix : 0;
  if (!s.truncate(prefix >= 0 ? pre : 0) || !s.upload(seq, pre, plen, error)) return false;
  const int V = s.vocab();
  std::vector<std::uint16_t> row;
  std::vector<float> p(static_cast<std::size_t>(V));
  for (int step = 0; step < a.max_tokens; ++step) {
    const int len = static_cast<int>(seq.size());
    if (!s.forward(len, len - 1, error) || !s.logits(1, &row, error)) return false;
    for (int i = 0; i < V; ++i) p[i] = bf16_to_f32_host(row[i]);
    Xorshift64 rng(static_cast<std::uint64_t>(a.seed) + static_cast<std::uint64_t>(step));
    const int next = sample_top_p_f32(p.data(), /*nucleus=*/0.9f, rng.next_float(), V);
    seq.push_back(next);
    out->ids.push_back(next);
    if (a.is_eos(next)) break;
    if (!s.upload(seq, len, len + 1, error)) return false;
  }
  return true;
}

bool generate_draft(GpuSession& s, const Args& a, Generation* out, std::string* error,
                    int prefix = -1) {
  std::vector<std::int32_t> seq = a.tokens;
  const int plen = static_cast<int>(seq.size());
  const int max_pos = plen + a.max_tokens + a.draft_k;
  std::vector<std::int32_t>& generated = out->ids;
  DraftReport& report = out->report;
  out->drafted = true;
  std::vector<std::int32_t> ids;
  // Prefill once: the KV watermark then advances by only the committed
  // tokens, so a step costs O(draft_k) instead of replaying the prefix.
  // `anchor` is the target's greedy token after the last committed position.
  // With prefix >= 0 the watermark already holds the first `prefix` positions,
  // so only the tail is uploaded and only [prefix, plen) is forwarded; the
  // loop's invariant (len == seq.size() == watermark) is unchanged.
  const int pre = prefix > 0 ? prefix : 0;
  if (!s.truncate(prefix >= 0 ? pre : 0) || !s.upload(seq, pre, plen, error) ||
      !s.forward(plen, plen - 1, error) || !s.argmax(1, &ids, error))
    return false;
  std::int32_t anchor = ids[0];

  std::vector<std::int32_t> target_arg;
  while (static_cast<int>(generated.size()) < a.max_tokens) {
    const int len = static_cast<int>(seq.size());  // == KV watermark
    // Never propose past the output budget or the request's position bound;
    // reserve one slot for the committed tail token so its KV always fits.
    const int room = std::min(a.draft_k, a.max_tokens - static_cast<int>(generated.size()));
    const int k = std::min(room, max_pos - len - 1);
    const std::vector<std::int32_t> proposal =
        (k > 0) ? draft_lookup(seq, k) : std::vector<std::int32_t>();
    const int nprop = static_cast<int>(proposal.size());
    report.proposed += nprop;

    // target_arg[0] = anchor (row len-1); target_arg[i] = greedy token of
    // row len-1+i, from one batched forward over the draft rows only.
    target_arg.assign(1, anchor);
    if (nprop > 0) {
      std::vector<std::int32_t> cand(seq);
      cand.insert(cand.end(), proposal.begin(), proposal.end());
      if (!s.upload(cand, len, len + nprop, error) || !s.forward(len + nprop, len, error) ||
          !s.argmax(nprop, &ids, error))
        return false;
      target_arg.insert(target_arg.end(), ids.begin(), ids.end());
    }
    // The unit-tested gate: the longest agreeing prefix plus one correction
    // (or bonus) token, clamped to the output budget and the position bound.
    const VerifyResult v = verify_greedy(target_arg.data(), proposal.data(), nprop);
    const int budget =
        std::min(a.max_tokens - static_cast<int>(generated.size()), max_pos - len);
    const CommitPlan plan =
        commit_speculative(v, proposal.data(), nprop, target_arg[nprop], budget);
    bool stop = false;
    for (std::int32_t tok : plan.tokens) {
      seq.push_back(tok);
      generated.push_back(tok);
      if (a.is_eos(tok)) {
        stop = true;
        break;
      }
    }
    report.accepted += v.accepted;
    ++report.steps;
    if (stop || static_cast<int>(generated.size()) >= a.max_tokens) break;

    // The accepted prefix has valid KV; the tail token (correction or bonus)
    // does not, so truncate to the valid rows and forward the tail (one row:
    // a chained greedy step whose id lands in d_tokens[newlen]).
    const int valid_kv = len + v.accepted;
    const int newlen = static_cast<int>(seq.size());
    if (newlen > valid_kv) {
      if (!s.truncate(valid_kv) || !s.upload(seq, valid_kv, newlen, error)) return false;
      if (newlen - valid_kv == 1 && valid_kv > 0) {
        if (!s.decode_greedy(1, error) || !s.tokens(newlen, newlen + 1, &ids, error))
          return false;
      } else if (!s.forward(newlen, newlen - 1, error) || !s.argmax(1, &ids, error)) {
        return false;
      }
      anchor = ids[0];
    } else {
      if (!s.truncate(newlen)) {
        *error = "kv truncate failed";
        return false;
      }
      anchor = target_arg[static_cast<std::size_t>(newlen - len)];
    }
  }
  return true;
}

// Adaptive drafting. EAGLE pays only while its rounds produce tokens faster
// than plain chained greedy decode, and both sides of that move with the
// context: a round is the head steps plus a (k + 1)-row verify (~1.5 decode
// steps at short context, ~2.2 at 13k on the RTX 4070 Ti SUPER), and the
// head's acceptance depends on the text (75-80% on the serving styles it was
// trained on, ~8% on novel prose). So the policy measures instead of guessing:
// after kEagleProbeRounds timed rounds (the first round, which carries the
// prompt catch-up, is not timed), a request
//   - keeps drafting if it averages >= kEagleKeepAccept accepted tokens/round,
//   - stops if it averages < G4_EAGLE_MIN_ACCEPT (default 0.35),
//   - otherwise decodes a timed plain burst of 8 tokens and keeps whichever is
//     faster in tokens per millisecond (host-timed; both paths sync on their
//     readback). Winning EAGLE re-syncs the head with one hidden-only forward
//     over the burst rows - verify rows are bit-identical to decode rows, so
//     the KV and the output are unchanged.
// Stopping finishes the request on the chained greedy decode and the
// conversation remembers it: later turns skip the head (and its catch-up).
// Output is identical in every case (the verify is exact greedy).
// G4_EAGLE_MIN_ACCEPT=0 disables the policy (always draft).
struct EaglePolicy {
  bool off = false;  // drafting disabled for this conversation
};
double eagle_min_accept() {
  static const double v = [] {
    const char* e = std::getenv("G4_EAGLE_MIN_ACCEPT");
    return e != nullptr ? std::atof(e) : 0.35;
  }();
  return v;
}
constexpr int kEagleProbeRounds = 4;
constexpr double kEagleKeepAccept = 2.0;
constexpr int kEagleBurst = 8;

// EAGLE draft-verify loop. Invariants at the top of a round: the target KV
// holds positions [0, L); the anchor (the target's greedy token at L, already
// emitted) is in d_tokens[L]; head entries [0, hq) are valid and `feat`
// holds the target features of positions [hq, L). One round: head catch-up
// over [hq, L) (its last row drafts d_1), k - 1 chained head steps (d_2..d_k
// into d_tokens), one target forward over [L, L + k] (anchor + drafts) that
// yields the greedy ids t_0..t_k and the features of those rows. The exact
// gate accepts the prefix d_i == t_{i-1}; t_a is the new anchor. Everything
// up to the read-back of t and d is enqueued without a host round trip.
// One EAGLE generation as a resumable object, so the serve scheduler can run
// it round by round next to other sessions. start() prefills and emits the
// first token; round() runs one draft + verify round; after any round the
// target KV holds [0, L) and the anchor (last emitted token) sits at
// d_tokens[L], which is also exactly the plain-decode invariant, so
// yield_plain() can hand the request to the (batched) greedy decode at any
// round boundary.
class EagleRun {
 public:
  EagleRun(GpuSession& s, EagleHead* head, const Args& a, Generation* out,
           EaglePolicy* policy)
      : s_(s), head_(head), a_(a), out_(out), gen_(out->ids), report_(out->report),
        policy_(policy) {}
  ~EagleRun() {
    // Drop head entries past the last true-feature row (chain entries came
    // from predicted features) so the next turn resumes from truth.
    if (head_ != nullptr && started_) eagle_truncate(head_, head_valid_);
    cudaFree(d_last_);
  }
  EagleRun(const EagleRun&) = delete;
  EagleRun& operator=(const EagleRun&) = delete;

  // Prefill (resuming from `prefix` when > 0), emit the first token and run
  // the head's catch-up over the prompt (its draft d_1 lands at
  // d_tokens[L + 1]). The shared prompt-feature buffer is used only here.
  // *done is set when the request already finished. false = error.
  bool start(int prefix, bool* done, std::string* error) {
    *done = false;
    if (head_ == nullptr) {
      *error = "eagle: no head loaded (start with --eagle HEAD)";
      return false;
    }
    seq_ = a_.tokens;
    const int plen = static_cast<int>(seq_.size());
    out_->drafted = true;
    std::uint16_t* f_prompt = nullptr;
    if (!s_.feature_buffers(&f_prompt, &f_verify_, &f_chain_, error)) return false;
    // Resume from the session's cached conversation. The target KV holds
    // [0, prefix); the head holds entries [0, eagle_written) (built from true
    // target features, so valid whenever they lie inside the cached prefix).
    // The head's catch-up needs the target's features for [hq0, plen), so the
    // target re-forwards from hq0 = min(prefix, head entries): a handful of
    // positions after a normal turn (the head trails the target by one round),
    // not the whole prompt. prefix < 0 (or no head history): from scratch.
    // At least one row is always forwarded - it yields the first token.
    int hq0 = 0;
    if (prefix > 0) hq0 = std::max(0, std::min({prefix, eagle_written(head_), plen - 1}));
    std::vector<std::int32_t> ids;
    if (!s_.truncate(hq0) || !s_.upload(seq_, 0, plen, error) ||
        !s_.forward(plen, plen - 1, error, f_prompt, hq0) || !s_.argmax(1, &ids, error))
      return false;
    eagle_truncate(head_, hq0);
    started_ = true;
    head_valid_ = hq0;
    L_ = plen;
    hq_ = hq0;
    if (emit(ids[0])) {
      *done = true;
      return true;
    }
    if (!s_.upload(seq_, L_, L_ + 1, error)) return false;
    const char* const eam = std::getenv("G4_EAGLE_TIMING");
    et_ = eam != nullptr && eam[0] == '1';
    min_accept_ = eagle_min_accept();
    decided_ = min_accept_ <= 0.0;
    t0_ = std::chrono::steady_clock::now();
    // Head catch-up over [hq0, L): writes head entries and drafts d_1.
    const std::size_t H = static_cast<std::size_t>(s_.hidden());
    if (d_last_ == nullptr && cudaMalloc(&d_last_, H * 2) != cudaSuccess) {
      *error = "eagle: cudaMalloc failed";
      return false;
    }
    const int rows = L_ - hq_;
    std::int32_t* const d_tok = s_.d_tokens();
    if (eagle_forward(head_, d_tok, hq_, rows, f_prompt, f_prompt, d_tok + L_ + 1) != 0 ||
        cudaMemcpyAsync(d_last_, f_prompt + static_cast<std::size_t>(rows - 1) * H, H * 2,
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread) != cudaSuccess) {
      *error = "eagle: head forward failed";
      return false;
    }
    head_valid_ = L_;
    step1_done_ = true;
    return true;
  }

  // ---- one round, in phases (so a scheduler can batch sessions) ----------
  // plan_k: drafts this round (0 = verify the anchor alone).
  int plan_k() {
    const int remaining = a_.max_tokens - static_cast<int>(gen_.size());
    k_ = std::max(0, std::min({a_.eagle_k, remaining - 1, s_.max_pos() - L_ - 2}));
    r_t0_ = std::chrono::steady_clock::now();
    return k_;
  }
  // Head step 1 (catch-up over the last round's verified rows), unless start()
  // already ran it. Valid after plan_k() with k > 0.
  bool needs_step1() const { return k_ > 0 && !step1_done_; }
  EagleRowsReq step1_req() {
    std::int32_t* const d_tok = s_.d_tokens();
    return EagleRowsReq{d_tok, hq_, L_ - hq_, f_verify_, f_verify_, d_tok + L_ + 1};
  }
  // After step 1: its last predicted feature feeds chain step 2.
  void step1_finished() {
    step1_done_ = true;
    head_valid_ = L_;  // entries [hq, L) came from true target features
    last_ = f_verify_ + static_cast<std::size_t>(L_ - hq_ - 1) * s_.hidden();
  }
  // Chain step i (2..k): one row at position L + i - 2.
  EagleRowsReq chain_req(int i) {
    std::int32_t* const d_tok = s_.d_tokens();
    const std::uint16_t* in = i == 2 ? (last_ != nullptr ? last_ : d_last_) : f_chain_;
    return EagleRowsReq{d_tok, L_ + i - 2, 1, in, f_chain_, d_tok + L_ + i};
  }
  int verify_from() const { return L_; }
  int k() const { return k_; }
  std::uint16_t* verify_features() { return f_verify_; }

  // Commit after the verify forward over [L, L + k] (target greedy ids
  // ids[0..k], features in verify_features()). Returns true when the loop
  // should stop: done(), plain() or an error. `probe` = this session's round
  // is timed for the adaptive policy (solo rounds).
  bool commit(const std::vector<std::int32_t>& ids, bool probe, std::string* error) {
    const int k = k_;
    if (!s_.tokens(L_ + 1, L_ + 1 + k, &drafts_, error)) return true;
    int acc = 0;
    while (acc < k && drafts_[static_cast<std::size_t>(acc)] == ids[static_cast<std::size_t>(acc)])
      ++acc;
    report_.proposed += k;
    report_.accepted += acc;
    ++report_.steps;
    if (et_) {
      et_rounds_++;
      et_acc_ += acc;
      et_prop_ += k;
    }
    recent_acc_ += acc;
    ++recent_rounds_;
    if (!decided_ && k > 0 && probe) {
      if (rounds_seen_ > 0) {  // round 0 carries the head's prompt catch-up
        probe_acc_ += acc;
        probe_tok_ += acc + 1;
        probe_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - r_t0_).count();
      }
      ++rounds_seen_;
    }
    for (int i = 0; i < acc && !done_; ++i) done_ = emit(drafts_[static_cast<std::size_t>(i)]);
    if (!done_) done_ = emit(ids[static_cast<std::size_t>(acc)]);
    if (done_) return true;
    // Target KV is valid for the anchor and the accepted drafts; the head's
    // chain entries (built on predicted features) are dropped.
    const int nl = L_ + acc + 1;
    if (!s_.truncate(nl) || !s_.upload(seq_, nl, nl + 1, error)) {
      if (error->empty()) *error = "kv truncate failed";
      return true;
    }
    eagle_truncate(head_, L_);
    head_valid_ = std::min(head_valid_, L_);
    hq_ = L_;
    L_ = nl;
    step1_done_ = false;
    last_ = nullptr;
    if (probe && !decided_ && rounds_seen_ > kEagleProbeRounds) {
      decided_ = true;
      const double avg = static_cast<double>(probe_acc_) / kEagleProbeRounds;
      const int left = a_.max_tokens - static_cast<int>(gen_.size());
      // The burst plus the rows already pending for the head ([hq, L)) must
      // fit one verify-sized hidden forward for the re-sync.
      const int burst = std::min({kEagleBurst, left, kMaxVerifyRows - (L_ - hq_)});
      if (avg < min_accept_) {
        plain_ = true;
      } else if (avg < kEagleKeepAccept && burst >= 2) {
        const auto b0 = std::chrono::steady_clock::now();
        std::vector<std::int32_t> bt;
        if (!s_.decode_greedy(burst, error) || !s_.tokens(L_ + 1, L_ + 1 + burst, &bt, error))
          return true;
        const double plain_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0)
                .count();
        const double plain_rate = burst / std::max(plain_ms, 1e-3);
        const double eagle_rate = probe_tok_ / std::max(probe_ms_, 1e-3);
        if (std::getenv("G4_SERVE_DEBUG") != nullptr || et_)
          std::fprintf(stderr, "G4_EAGLE probe: %.2f accepted/round, eagle %.1f tok/s, "
                       "plain %.1f tok/s -> %s\n", avg, 1000.0 * eagle_rate,
                       1000.0 * plain_rate, plain_rate >= eagle_rate ? "plain" : "eagle");
        for (std::int32_t t : bt)
          if ((done_ = emit(t))) break;
        if (done_) return true;
        if (plain_rate >= eagle_rate) {
          plain_ = true;  // the anchor already sits at d_tokens[written]
        } else if (!resync(burst, error)) {
          return true;
        }
      }
      if (plain_ && policy_ != nullptr) policy_->off = true;  // measured: not paying
      if (plain_ && (std::getenv("G4_SERVE_DEBUG") != nullptr || et_))
        std::fprintf(stderr, "G4_EAGLE drafting off after %d rounds (%.2f accepted/round)\n",
                     rounds_seen_, avg);
    }
    return plain_;
  }
  // After `burst` plainly decoded tokens (already emitted): features of
  // [hq, L + burst) for the head's next catch-up, recomputed bit-identically
  // by a verify-shaped hidden forward (rows [hq, L) verified, rows
  // [L, L + burst) decoded).
  bool resync(int burst, std::string* error) {
    if (!s_.truncate(hq_) || !s_.forward_hidden(L_ + burst, hq_, f_verify_, error)) return false;
    L_ += burst;
    return true;
  }
  // Accepted drafts per round since the last call (batched policy).
  double take_recent_accept() {
    const double v = recent_rounds_ ? static_cast<double>(recent_acc_) / recent_rounds_ : 0.0;
    recent_acc_ = recent_rounds_ = 0;
    return v;
  }
  int pending_rows() const { return L_ - hq_; }

  // One solo round (head steps, verify, commit); see commit() for the result.
  bool round(std::string* error) {
    const int k = plan_k();
    std::int32_t* const d_tok = s_.d_tokens();
    Ev h0{}, v0{};
    if (k > 0) {
      if (et_) cudaEventCreate(&h0.a), cudaEventCreate(&h0.b),
          cudaEventRecord(h0.a, cudaStreamPerThread);
      bool ok = true;
      if (needs_step1()) {
        const EagleRowsReq q = step1_req();
        ok = eagle_forward(head_, q.d_tokens, q.pos0, q.rows, q.d_feat, q.d_feat_out,
                           q.d_draft_out) == 0;
        step1_finished();
      }
      for (int i = 2; ok && i <= k; ++i) {
        const EagleRowsReq q = chain_req(i);
        ok = eagle_forward(head_, d_tok, q.pos0, 1, q.d_feat, q.d_feat_out, q.d_draft_out) == 0;
      }
      if (et_) cudaEventRecord(h0.b, cudaStreamPerThread), h_ev_.push_back(h0);
      if (!ok) {
        *error = "eagle: head forward failed";
        return true;
      }
    }
    // Target verify over anchor + drafts: greedy ids t_0..t_k and features.
    if (et_) cudaEventCreate(&v0.a), cudaEventCreate(&v0.b),
        cudaEventRecord(v0.a, cudaStreamPerThread);
    std::vector<std::int32_t> ids;
    if (!s_.forward(L_ + k + 1, L_, error, f_verify_, L_) || !s_.argmax(k + 1, &ids, error))
      return true;
    if (et_) cudaEventRecord(v0.b, cudaStreamPerThread), v_ev_.push_back(v0);
    return commit(ids, true, error);
  }

  // Leave drafting at a round boundary (another session needs the batch);
  // the conversation's policy is untouched.
  void yield_plain() { plain_ = true; }
  bool done() const { return done_; }
  bool plain() const { return plain_; }
  // Emit a token decoded outside the rounds (plain continuation); true = stop.
  bool emit(std::int32_t tok) {
    seq_.push_back(tok);
    gen_.push_back(tok);
    return a_.is_eos(tok) || static_cast<int>(gen_.size()) >= a_.max_tokens;
  }

  void report_timing() {
    if (!et_) return;
    const double wall =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    double hms = 0, vms = 0;
    float dms = 0;
    for (auto& e : h_ev_)
      if (cudaEventSynchronize(e.b) == cudaSuccess)
        dms = 0, cudaEventElapsedTime(&dms, e.a, e.b), hms += dms, cudaEventDestroy(e.a),
        cudaEventDestroy(e.b);
    for (auto& e : v_ev_)
      if (cudaEventSynchronize(e.b) == cudaSuccess)
        dms = 0, cudaEventElapsedTime(&dms, e.a, e.b), vms += dms, cudaEventDestroy(e.a),
        cudaEventDestroy(e.b);
    h_ev_.clear();
    v_ev_.clear();
    const double R = et_rounds_ ? static_cast<double>(et_rounds_) : 1.0;
    const double tok = static_cast<double>(gen_.size());
    std::fprintf(stderr, "G4_EAGLE head_ms=%.1f (%.3f/rnd) verify_ms=%.1f (%.3f/rnd) "
                 "read_ms=%.1f wall_ms=%.1f rounds=%ld acc=%ld prop=%ld tok/rnd=%.2f "
                 "head%%=%.0f verify%%=%.0f tok/s=%.1f\n",
                 hms, hms / R, vms, vms / R, wall - hms - vms, wall, et_rounds_, et_acc_,
                 et_prop_, (et_acc_ + et_rounds_) / R, wall ? 100 * hms / wall : 0.0,
                 wall ? 100 * vms / wall : 0.0, wall ? tok / (wall / 1000.0) : 0.0);
  }

 private:
  struct Ev {
    cudaEvent_t a, b;
  };
  GpuSession& s_;
  EagleHead* head_;
  const Args& a_;
  Generation* out_;
  std::vector<std::int32_t>& gen_;
  DraftReport& report_;
  EaglePolicy* policy_;
  std::vector<std::int32_t> seq_, drafts_;
  std::uint16_t *f_verify_ = nullptr, *f_chain_ = nullptr;
  std::uint16_t* d_last_ = nullptr;      // start()'s last catch-up feature (own copy)
  const std::uint16_t* last_ = nullptr;  // step-2 input after an in-round step 1
  int L_ = 0, hq_ = 0, head_valid_ = 0, k_ = 0;
  bool started_ = false, done_ = false, plain_ = false, decided_ = false, et_ = false;
  bool step1_done_ = false;
  double min_accept_ = 0.0, probe_ms_ = 0.0;
  int rounds_seen_ = 0, probe_acc_ = 0, probe_tok_ = 0;
  int recent_acc_ = 0, recent_rounds_ = 0;
  long et_rounds_ = 0, et_acc_ = 0, et_prop_ = 0;
  std::vector<Ev> h_ev_, v_ev_;
  std::chrono::steady_clock::time_point t0_, r_t0_;
};

// Plain chained greedy decode from the current watermark (anchor at
// d_tokens[written]) until `emit` says stop; 8 tokens per readback.
template <class Emit>
bool decode_tail(GpuSession& s, int remaining, Emit emit, std::string* error) {
  constexpr int kChunk = 8;
  std::vector<std::int32_t> ids;
  while (remaining > 0) {
    const int pos = s.written();
    const int steps = std::min(kChunk, remaining);
    if (!s.decode_greedy(steps, error) || !s.tokens(pos + 1, pos + 1 + steps, &ids, error))
      return false;
    for (std::int32_t t : ids) {
      --remaining;
      if (emit(t)) return true;
    }
  }
  return true;
}

bool generate_eagle(GpuSession& s, EagleHead* head, const Args& a, Generation* out,
                    std::string* error, int prefix = -1, EaglePolicy* policy = nullptr) {
  EagleRun run(s, head, a, out, policy);
  bool done = false;
  if (!run.start(prefix, &done, error)) return false;
  while (!done && !run.round(error)) {
  }
  if (!error->empty()) return false;
  if (!run.done() && run.plain() &&
      !decode_tail(s, a.max_tokens - static_cast<int>(out->ids.size()),
                   [&](std::int32_t t) { return run.emit(t); }, error))
    return false;
  run.report_timing();
  return true;
}

bool gpu_generate(GpuSession& s, const Args& a, Generation* out, std::string* error,
                  EagleHead* head = nullptr, int prefix = -1, EaglePolicy* policy = nullptr) {
  if (!check_request(s, a, prefix, error)) return false;
  // A conversation whose drafts were not paying decodes plainly (same tokens).
  if (a.eagle_k > 0 && policy != nullptr && policy->off)
    return generate_greedy(s, a, out, error, prefix);
  if (a.eagle_k > 0) return generate_eagle(s, head, a, out, error, prefix, policy);
  if (a.draft_k > 0) return generate_draft(s, a, out, error, prefix);
  return a.seed == 0 ? generate_greedy(s, a, out, error, prefix)
                     : generate_sampled(s, a, out, error, prefix);
}

// Load the --eagle head for an open session (nullptr when none requested).
bool open_eagle(GpuSession& s, const std::string& path, EagleHead** head, std::string* err) {
  *head = nullptr;
  if (path.empty()) return true;
  ModelConfig cfg;
  if (!load_model_config(s.dir(), &cfg, err, /*enforce_spec=*/true)) return false;
  *head = eagle_load(path, s.ctx(), cfg, kForwardSpec, s.max_pos(), err);
  return *head != nullptr;
}

int gpu_generate_cmd(const Args& a) {
  std::string err;
  GpuSession s;
  Generation g;
  const int max_pos = static_cast<int>(a.tokens.size()) + a.max_tokens + a.draft_k +
                      (a.eagle_k > 0 ? a.eagle_k + 1 : 0);
  const int rows = std::max(a.draft_k, a.eagle_k) + 1;
  EagleHead* head = nullptr;
  if (!s.open(a.dir, max_pos, rows, &err) || !open_eagle(s, a.eagle, &head, &err))
    return fail(err);
  const bool ok = gpu_generate(s, a, &g, &err, head);
  eagle_free(head);
  if (!ok) return fail(err);
  if (g.drafted) report_draft(g.report);
  std::printf("%s", generate_document(g.ids, g.drafted ? &g.report : nullptr).c_str());
  return 0;
}

// `kv-bench`: load, time prefill (a.bench_ctx tokens from 0 to N), then time
// decode (a.bench_decode greedy tokens from N to N + D). Emits one JSON
// object per run on stdout:
//   {"prefill_ms":..., "decode_ms":..., "kv_free_before":...,
//    "kv_free_after":..., "tokens_generated":..., "decode_tps":...}
//
// Memory is measured with cudaMemGetInfo around the GPU forward calls, so
// "kv_free_before/after" reports the *free* device memory at each phase —
// the difference between runs (bf16 vs int8) is the persistent KV pool
// delta on the same GPU with weights already resident.
int gpu_kv_bench_cmd(const Args& a) {
  std::string err;
  const int N = a.bench_ctx;
  const int D = a.bench_decode;
  if (N < 2 || D <= 0) return fail("kv-bench: need --bench-ctx N>=2 --bench-decode D");
  // max_pos must hold the prompt plus every generated token plus one (the
  // decode chain writes d_tokens[written..written+steps], written = N after
  // prefill, steps = D; gpu_decode_greedy requires written + steps + 1 <= max_pos).
  const int max_pos = N + D + 1;
  GpuSession s;
  if (!s.open(a.dir, max_pos, 1, &err)) return fail(err);

  // Deterministic pseudo-tokens so bf16 and int8 runs process identical ids.
  const int V = s.vocab();
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(N));
  std::uint64_t r = a.seed != 0 ? static_cast<std::uint64_t>(a.seed)
                                : 0x9E3779B97F4A7C15ull;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    r = r * 6364136223846793005ull + 1442695040888963407ull;
    tokens[i] = static_cast<std::int32_t>(
        (r >> 16) % static_cast<std::uint64_t>(V > 0 ? V : 1));
  }
  if (!s.upload(tokens, 0, N, &err)) return fail(err);

  const char* kv8_env = std::getenv("G4_KV8");
  const bool kv8_requested = kv8_env != nullptr && kv8_env[0] == '1';
  const bool kv4_requested = kv8_env != nullptr && kv8_env[0] == '4';
  const bool kv8 = s.kv8();  // resolved (after the split-attention check)
  const bool kv4 = s.kv4();
  if (kv8_requested && !kv8)
    std::fprintf(stderr,
                 "kv-bench: WARNING G4_KV8=1 requested but the session fell "
                 "back to bf16 (config not split-attention capable: needs "
                 "d==128 and Hq==4*Hk)\n");
  if (kv4_requested && !kv4)
    std::fprintf(stderr,
                 "kv-bench: WARNING G4_KV8=4 requested but the session fell "
                 "back to bf16 (config not split-attention capable: needs "
                 "d==128 and Hq==4*Hk)\n");

  // Free memory right after the session is fully open (weights + KV pool +
  // fixed scratch are resident; the KV pool is the only thing that differs
  // between bf16 and int8). Measuring here — not around the forward calls,
  // which allocate transient activation scratch that is identical in both
  // modes — isolates the persistent KV cache footprint.
  long long free_after_load = 0;
  {
    std::size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
      free_after_load = static_cast<long long>(free_b);
  }

  // Prefill: forward all N prompt positions; the argmax for the last prompt
  // row (position N-1) is placed at d_tokens[N] via argmax_to_token.
  // Same convention as generate_greedy. cudaDeviceSynchronize makes this GPU
  // time, not the host time to enqueue the kernels.
  const auto t0 = std::chrono::steady_clock::now();
  if (!s.forward(N, N - 1, &err)) return fail(err);
  if (!s.argmax_to_token(N, &err)) return fail(err);
  cudaDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();

  // Decode: device-side greedy chain — same path as generate_greedy.
  // Reads d_tokens[N] as the first token; writes tokens N+1..N+D. Sync on the
  // stream before stopping the clock so this is GPU time, not the host time
  // to enqueue D async graph replays.
  const auto t2 = std::chrono::steady_clock::now();
  if (!s.decode_greedy(D, &err)) return fail(err);
  cudaDeviceSynchronize();
  const auto t3 = std::chrono::steady_clock::now();

  const double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
  const double tps = D > 0 ? 1000.0 * static_cast<double>(D) / decode_ms : 0.0;

  // Report the persistent KV-pool footprint as the free-memory difference
  // between this run (bf16 / int8 / int4 pool) measured right after load. The
  // cross-run delta is the KV cache size difference at the same max_pos.
  std::fprintf(stdout,
      "{\"kv8\":%s,\"kv4\":%s,\"split_attn\":%s,\"prefill_ms\":%.3f,\"decode_ms\":%.3f,"
      "\"decode_tps\":%.3f,\"free_after_load\":%lld,\"max_pos\":%d,"
      "\"ctx\":%d,\"decode\":%d}\n",
      kv8 ? "true" : "false",
      kv4 ? "true" : "false",
      s.split_attn() ? "true" : "false",
      prefill_ms, decode_ms, tps, free_after_load, max_pos, N, D);
  std::fflush(stdout);
  return 0;
}
int gpu_dump_hidden_cmd(const Args& a) {
  std::string err;
  GpuSession s;
  if (!s.open(a.dir, a.max_context, 1, &err)) return fail(err);
  const std::size_t H = static_cast<std::size_t>(s.hidden());
  std::FILE* in = std::fopen(a.in.c_str(), "rb");
  std::FILE* out = std::fopen(a.out.c_str(), "wb");
  if (in == nullptr || out == nullptr) return fail("dump-hidden: cannot open --in / --out");
  std::uint16_t* d_hidden = nullptr;
  std::uint16_t* h_hidden = nullptr;
  const std::size_t cap = static_cast<std::size_t>(a.max_context) * H;
  if (cudaMalloc(&d_hidden, cap * 2) != cudaSuccess ||
      cudaHostAlloc(&h_hidden, cap * 2, cudaHostAllocDefault) != cudaSuccess)
    return fail("dump-hidden: allocation failed");
  std::vector<std::int32_t> seq;
  long records = 0, tokens = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::int32_t n; std::fread(&n, sizeof n, 1, in) == 1;) {
    if (n <= 0 || n > a.max_context) return fail("dump-hidden: record length out of range");
    seq.resize(static_cast<std::size_t>(n));
    if (std::fread(seq.data(), sizeof(std::int32_t), seq.size(), in) != seq.size())
      return fail("dump-hidden: truncated record");
    for (std::int32_t t : seq)
      if (t < 0 || t >= s.vocab()) return fail("dump-hidden: token id out of range");
    if (!s.truncate(0) || !s.upload(seq, 0, n, &err) || !s.forward_hidden(n, 0, d_hidden, &err))
      return fail(err);
    const std::size_t bytes = static_cast<std::size_t>(n) * H * 2;
    if (cudaMemcpy(h_hidden, d_hidden, bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
      return fail("dump-hidden: copy back failed");
    if (std::fwrite(h_hidden, 1, bytes, out) != bytes) return fail("dump-hidden: write failed");
    ++records;
    tokens += n;
    if (records % 200 == 0) {
      const double sec =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::fprintf(stderr, "dump-hidden: %ld records, %ld tokens, %.0f tok/s\n", records, tokens,
                   tokens / sec);
    }
  }
  cudaFree(d_hidden);
  cudaFreeHost(h_hidden);
  std::fclose(in);
  if (std::fclose(out) != 0) return fail("dump-hidden: close failed");
  std::fprintf(stderr, "dump-hidden: done, %ld records, %ld tokens\n", records, tokens);
  return 0;
}

// `serve`: load once, then answer one JSON request per stdin line with one
// JSON line on stdout (see the file header). Errors are reported per request
// ({"error": "..."}) and do not end the loop; EOF on stdin does.
int gpu_serve_cmd(const Args& cli) {
  constexpr int kServeLogitRows = 16;  // draft_lookup <= 15
  constexpr int kPoolPage = 64;        // rows per KV page (the attention tile)
  const int n_sessions = std::max(1, cli.sessions);
  std::string err;
  // 1. Load and upload the weight object ONCE (the ~9.5 GiB blob lives
  //    once per device); every session attaches to it via
  //    gpu_ctx_use_weights.
  ModelConfig cfg;
  if (!load_model_config(cli.dir, &cfg, &err, /*enforce_spec=*/true))
    return fail(err);
  GpuModel gm;
  if (!pack_gpu_model_from_checkpoint(cli.dir, cfg, &gm, &err))
    return fail(err);
  GpuWeights* weights = gpu_weights_create();
  if (weights == nullptr) return fail("gpu_weights_create failed");
  {
    char werr[256] = "";
    if (gpu_weights_load(weights, gm.weights.get(), gm.bytes,
                         gm.scales.data(), gm.scales.size(), werr,
                         sizeof werr) != 0)
      return fail(std::string("gpu_weights_load: ") + werr);
  }
  struct WtGuard {
    GpuWeights* w;
    ~WtGuard() { gpu_weights_free_obj(w); }
  } wt_guard{weights};
  constexpr int kMaxSessions = 16;
  if (n_sessions > kMaxSessions)
    return fail("--sessions exceeds " + std::to_string(kMaxSessions));
  // 2. ONE shared KV pool for every session (qwen3-27b-5090's plan-once,
  //    allocate-once pool): a single arena of pool_rows positions per
  //    (layer, kv head), cut into 64-row pages that the KvPoolController
  //    hands to sessions on demand. A session's context addresses its pages
  //    through a position -> row table, so a conversation costs only the
  //    pages it actually holds and an idle one gives the rest back. The
  //    default pool lets every session fill --max-context at once.
  const long want_tokens = cli.kv_pool_tokens > 0
                               ? static_cast<long>(cli.kv_pool_tokens)
                               : static_cast<long>(n_sessions) * cli.max_context;
  const int pool_pages = static_cast<int>((want_tokens + kPoolPage - 1) / kPoolPage);
  const int pool_rows = pool_pages * kPoolPage;
  if (pool_rows < cli.max_context)
    return fail("--kv-pool-tokens (" + std::to_string(pool_rows) +
                ") is smaller than --max-context (" + std::to_string(cli.max_context) + ")");
  char aerr[256] = "";
  GpuKvArena* arena = gpu_kv_arena_create(cfg, pool_rows, aerr, sizeof aerr);
  if (arena == nullptr) return fail(std::string("kv arena: ") + aerr);
  struct ArenaGuard {
    GpuKvArena* a;
    ~ArenaGuard() { gpu_kv_arena_free(a); }
  } arena_guard{arena};
  KvPoolConfig pool_cfg;
  pool_cfg.layout = KvLayout::from_config(cfg);
  pool_cfg.tokens_per_page = kPoolPage;
  pool_cfg.page_bytes = pool_cfg.tokens_per_page * pool_cfg.layout.bytes_per_token;
  pool_cfg.num_pages = static_cast<std::size_t>(pool_pages);
  pool_cfg.budget_bytes = pool_cfg.num_pages * pool_cfg.page_bytes;
  KvPoolController pool(pool_cfg);
  // 3. N sessions on the shared weights and the shared pool (a stack array:
  //    GpuSession must never be copied/moved - it owns CUDA pointers).
  //    Requests run one at a time, so sessions 1.. borrow session 0's prefill
  //    scratch and feature buffers instead of each holding ~1 GiB of its own.
  GpuSession sessions[kMaxSessions];
  int slot[kMaxSessions];
  for (int i = 0; i < n_sessions; ++i) {
    if (!sessions[i].open_shared(cli.dir, cli.max_context, kServeLogitRows,
                                 weights, &err, arena))
      return fail("open_shared[" + std::to_string(i) + "]: " + err);
    if (i > 0 && !sessions[i].share_scratch_from(&sessions[0], &err))
      return fail("share_scratch[" + std::to_string(i) + "]: " + err);
    slot[i] = pool.open_slot();
    if (slot[i] < 0) return fail("kv pool: no slot for session " + std::to_string(i));
  }
  // 4. EAGLE: ONE draft-KV pool in the same row space as the main pool (the
  //    head's draft KV for a position sits at the same arena row as the main
  //    KV, so one row table per session addresses both), the head weights
  //    loaded once, and a per-session head (scratch + watermark) forked from
  //    the base so every session speculates.
  EagleHead* heads[kMaxSessions] = {};
  struct HeadGuard {
    EagleHead** h;
    int n;
    ~HeadGuard() {
      for (int i = n - 1; i >= 0; --i) eagle_free(h[i]);  // forks before the base
    }
  } head_guard{heads, n_sessions};
  if (!cli.eagle.empty()) {
    heads[0] = eagle_load(cli.eagle, sessions[0].ctx(), cfg, kForwardSpec, cli.max_context,
                          &err, pool_rows);
    if (heads[0] == nullptr) return fail(err);
    for (int i = 1; i < n_sessions; ++i) {
      heads[i] = eagle_fork(heads[0], sessions[i].ctx(), &err);
      if (heads[i] == nullptr) return fail("eagle_fork[" + std::to_string(i) + "]: " + err);
    }
  }
  const bool have_eagle = heads[0] != nullptr;
  // Per-session conversation state: the tokens whose KV the session holds,
  // and when it last served a request (LRU for pool eviction).
  std::vector<std::vector<std::int32_t>> hist(n_sessions);
  std::vector<long> last_used(n_sessions, 0);
  std::vector<EaglePolicy> policy(n_sessions);  // per conversation
  long tick = 0;
  // Give session i enough pool pages for `need` positions and publish its row
  // table. Its resident pages never move, so a resumed conversation's KV stays
  // valid. If the pool is short, the least-recently-used OTHER session's
  // resident KV is dropped (it just re-prefills next time) until it fits.
  auto prepare = [&](int i, long need, std::string* e) -> bool {
    const std::size_t want =
        ((static_cast<std::size_t>(need) + kPoolPage - 1) / kPoolPage) * kPoolPage;
    while (pool.pages_missing(slot[i], want) > pool.free_pages()) {
      int victim = -1;
      for (int j = 0; j < n_sessions; ++j)
        if (j != i && pool.slot_pages(slot[j]) > 0 &&
            (victim < 0 || last_used[j] < last_used[victim]))
          victim = j;
      if (victim < 0) {
        *e = "kv pool exhausted: need " + std::to_string(want) + " positions, pool holds " +
             std::to_string(pool_rows);
        return false;
      }
      pool.drop(slot[victim]);
      policy[victim] = EaglePolicy{};
      hist[victim].clear();
      sessions[victim].truncate(0);
      eagle_truncate(heads[victim], 0);
    }
    if (!pool.ensure(slot[i], want)) {
      *e = "kv pool: could not reserve " + std::to_string(want) + " positions";
      return false;
    }
    std::vector<int> rows(want);
    const int n_map = static_cast<int>(std::min<std::size_t>(want, cli.max_context));
    if (!pool.row_table(slot[i], want, rows.data()) || !sessions[i].map_rows(rows.data(), n_map)) {
      *e = "kv pool: row table rejected";
      return false;
    }
    return true;
  };
  // 5. Warm-up: load the lazily loaded kernels and capture the decode
  //    graph on EVERY session (the graph references that session's
  //    device pointers, so a warm-up on session 0 does not help the
  //    others). Uses an 8-token prompt, three greedy steps.
  for (int i = 0; i < n_sessions; ++i) {
    Args w;
    w.tokens = {791, 6864, 315, 9822, 374, 13, 578, 7754};
    w.max_tokens = 4;
    Generation g;
    if (cli.max_context >= 16 &&
        (!prepare(i, 64, &err) || !gpu_generate(sessions[i], w, &g, &err, heads[i])))
      return fail("warm-up[" + std::to_string(i) + "]: " + err);
  }
  std::fprintf(stderr,
               "granite4: serve ready: %d session(s) x %d positions, KV pool %d positions "
               "(%d pages), main KV %.2f GiB, draft KV %.2f GiB\n",
               n_sessions, cli.max_context, pool_rows, pool_pages,
               static_cast<double>(gpu_kv_arena_bytes(arena)) / (1024.0 * 1024.0 * 1024.0),
               static_cast<double>(eagle_kv_bytes(heads[0])) / (1024.0 * 1024.0 * 1024.0));
  std::printf("{\"ready\":true,\"max_context\":%d,\"sessions\":%d,\"kv_pool_tokens\":%d,"
              "\"eagle\":%s}\n",
              cli.max_context, n_sessions, pool_rows, have_eagle ? "true" : "false");
  std::fflush(stdout);
  // 5b. Warm-up left each session's watermark advanced past the synthetic
  //     8-token prompt. Reset so the first serve request starts from an
  //     empty KV (matching the empty hist); give the warm-up pages back.
  for (int i = 0; i < n_sessions; ++i) {
    sessions[i].truncate(0);
    eagle_truncate(heads[i], 0);
    pool.drop(slot[i]);
  }
  // 6. Requests dispatch to the session whose KV prefix is the longest
  //    full prefix of the new prompt; that session resumes from its
  //    watermark (only the tail is uploaded and prefilled). When no session
  //    matches (or the pool is empty), the request goes to the next
  //    session in round-robin and gets a full re-prefill (truncate(0)).
  //    Each session's `hist` holds the token sequence whose KV is cached
  //    in that session; after a successful generate, hist extends by the
  //    generated ids. A new prompt that shares a prefix with an existing
  //    hist reuses the cached KV positions instead of re-prefilling them,
  //    which is the primary cost of a long context in a chat conversation.
  // 6. Concurrent scheduler. Requests (one JSON line each, optional integer
  //    "id" echoed in the reply) are read on a separate thread, so a request
  //    can arrive while others generate. Each request becomes a job bound to
  //    an idle session: the session sharing the longest prefix with its
  //    prompt (resume from there) or an empty / round-robin one. Every loop
  //    iteration advances the active jobs by one unit:
  //      - one job alone, EAGLE: one draft + verify round;
  //      - plain greedy jobs: one chunk of up to 8 tokens, ALL of them in one
  //        batched multi-session step (gpu_decode_multi: the weights stream
  //        once for every session; each session's tokens are bit-identical to
  //        decoding it alone).
  //    A second job arriving while an EAGLE job runs moves that job to plain
  //    decode at its round boundary (exact either way), so both batch.
  //    Seeded sampling and draft-lookup requests run to completion on their
  //    session (the other jobs pause meanwhile).
  struct LineQueue {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
  } inq;
  std::thread reader([&inq] {
    std::string ln;
    for (int c; (c = std::getchar()) != EOF;) {
      if (c != '\n') {
        ln.push_back(static_cast<char>(c));
        continue;
      }
      {
        std::lock_guard<std::mutex> lk(inq.m);
        inq.q.push_back(std::move(ln));
      }
      inq.cv.notify_one();
      ln.clear();
    }
    std::lock_guard<std::mutex> lk(inq.m);
    inq.eof = true;
    inq.cv.notify_one();
  });
  struct Job {
    long long id = 0;
    bool has_id = false;
    Args a;
    int sess = -1;
    int resume = 0;
    int wm_before = 0;
    Generation g;
    std::unique_ptr<EagleRun> eagle;  // non-null while drafting
    bool plain = false;               // plain greedy decode (batchable)
    bool done = false;
    std::string err;
  };
  std::list<std::unique_ptr<Job>> jobs;  // stable addresses (EagleRun keeps refs)
  std::vector<bool> busy(n_sessions, false);
  const bool debug = std::getenv("G4_SERVE_DEBUG") != nullptr;
  int next = 0;
  auto send = [](const std::string& s) {
    std::fwrite(s.data(), 1, s.size(), stdout);
    std::fflush(stdout);
  };
  auto id_prefix = [](bool has, long long id) {
    return has ? "{\"id\":" + std::to_string(id) + "," : std::string("{");
  };
  auto send_error = [&](bool has, long long id, const std::string& e) {
    std::string esc;
    for (char ch : e) {
      if (ch == '"' || ch == '\\') esc.push_back('\\');
      if (static_cast<unsigned char>(ch) >= 0x20) esc.push_back(ch);
    }
    send(id_prefix(has, id) + "\"error\":\"" + esc + "\"}\n");
  };
  // Record what the session's KV holds after a job, exactly as the
  // sequential loop did (see the comments there in history).
  auto finish = [&](Job& j) {
    const int picked = j.sess;
    GpuSession& s = sessions[picked];
    j.eagle.reset();  // its destructor trims the head to its true-feature rows
    const bool ok = j.err.empty();
    if (ok) {
      std::string doc = generate_document(j.g.ids, j.g.drafted ? &j.g.report : nullptr);
      send(id_prefix(j.has_id, j.id) + doc.substr(1));
      auto& h = hist[picked];
      h.assign(j.a.tokens.begin(), j.a.tokens.end());
      for (std::int32_t t : j.g.ids) h.push_back(t);
      // The watermark can overshoot the committed tokens (a decode chunk past
      // the stop is discarded) or undershoot by one; positions
      // [0, min(written, h.size())) hold the committed prefix's KV.
      const int valid = std::min(s.written(), static_cast<int>(h.size()));
      if (!s.truncate(valid) || valid > cli.max_context) {
        h.clear();
        s.truncate(0);
        eagle_truncate(heads[picked], 0);
        pool.drop(slot[picked]);
      } else {
        h.resize(static_cast<std::size_t>(valid));
        eagle_truncate(heads[picked], std::min(eagle_written(heads[picked]), valid));
        pool.trim(slot[picked], static_cast<std::size_t>(valid));
      }
    } else {
      send_error(j.has_id, j.id, j.err);
      if (s.written() != j.wm_before) {
        hist[picked].clear();
        s.truncate(0);
        eagle_truncate(heads[picked], 0);
        pool.drop(slot[picked]);
      }
    }
    busy[picked] = false;
    j.done = true;
  };
  // Plain greedy start: prefill (from the resume point) and the first token;
  // afterwards d_tokens[written] holds the last emitted token.
  auto start_plain = [&](Job& j) -> bool {
    GpuSession& s = sessions[j.sess];
    const int plen = static_cast<int>(j.a.tokens.size());
    const int p = j.resume > 0 ? j.resume : 0;
    std::vector<std::int32_t> first;
    if (!s.truncate(p) || !s.upload(j.a.tokens, p, plen, &j.err) ||
        !s.forward(plen, plen - 1, &j.err) || !s.argmax_to_token(plen, &j.err) ||
        !s.tokens(plen, plen + 1, &first, &j.err))
      return false;
    j.g.ids.push_back(first[0]);
    j.plain = true;
    if (j.a.is_eos(first[0]) || static_cast<int>(j.g.ids.size()) >= j.a.max_tokens)
      finish(j);
    return true;
  };
  auto active_count = [&] {
    int n = 0;
    for (auto& j : jobs)
      if (!j->done) ++n;
    return n;
  };
  // Parse one request line and start it on an idle session. false = no
  // idle session (the line stays queued).
  auto admit = [&](const std::string& line) -> bool {
    int idle = -1;
    for (int i = 0; i < n_sessions; ++i)
      if (!busy[i]) idle = i;
    if (idle < 0) return false;
    json::Value req;
    std::string perr;
    auto j = std::make_unique<Job>();
    if (!json::parse(line.data(), line.size(), &req, &perr) ||
        req.kind != json::Value::Kind::Object) {
      send_error(false, 0, "bad request: " + perr);
      return true;
    }
    if (req.find("id") != nullptr) {
      j->has_id = true;
      j->id = req.get_int("id", 0);
    }
    Args& a = j->a;
    const json::Value* toks = req.find("tokens");
    if (toks != nullptr && toks->kind == json::Value::Kind::Array)
      for (const auto& t : toks->arr)
        a.tokens.push_back(t.kind == json::Value::Kind::Int
                               ? static_cast<std::int32_t>(t.int_value)
                               : -1);
    a.max_tokens = static_cast<int>(req.get_int("max_tokens", 0));
    a.seed = static_cast<long>(req.get_int("seed", 0));
    a.draft_k = static_cast<int>(req.get_int("draft_lookup", 0));
    a.eos = static_cast<long>(req.get_int("eos", -1));
    a.eagle_k = static_cast<int>(req.get_int("eagle", 0));
    // Session pick among the idle sessions, starting at the round-robin
    // pointer: the longest shared prefix (>= one page), else an empty one.
    std::vector<int> order;
    for (int k = 0; k < n_sessions; ++k) {
      const int i = (next + k) % n_sessions;
      if (!busy[i]) order.push_back(i);
    }
    std::vector<ServeSessionState> state(order.size());
    for (std::size_t k = 0; k < order.size(); ++k) {
      state[k].history = hist[order[k]];
      state[k].watermark = sessions[order[k]].written();
    }
    const ServeDecision dec = serve_decide(state, 0, a.tokens, kPoolPage);
    j->sess = order[static_cast<std::size_t>(dec.picked)];
    j->resume = std::getenv("G4_FORCE_REPREFILL") != nullptr ? 0 : dec.resume;
    next = (j->sess + 1) % n_sessions;
    GpuSession& s = sessions[j->sess];
    j->wm_before = s.written();
    if (debug)
      std::fprintf(stderr, "G4_SERVE id=%lld session=%d resume=%d prompt=%zu wm=%d "
                   "free_pages=%zu active=%d\n", j->id, j->sess, j->resume, a.tokens.size(),
                   s.written(), pool.free_pages(), active_count());
    if (!check_request(s, a, j->resume, &j->err)) {
      send_error(j->has_id, j->id, j->err);
      return true;  // session untouched
    }
    // Pool pages for everything this job can write (+ one decode chunk of
    // overshoot past the stop).
    const long need = static_cast<long>(a.tokens.size()) + a.max_tokens + a.draft_k +
                      (a.eagle_k > 0 ? a.eagle_k + 1 : 0) + 8;
    if (!prepare(j->sess, std::min<long>(need, cli.max_context), &j->err)) {
      send_error(j->has_id, j->id, j->err);
      return true;
    }
    busy[j->sess] = true;
    last_used[j->sess] = ++tick;
    if (j->resume == 0) {
      eagle_truncate(heads[j->sess], 0);
      policy[j->sess] = EaglePolicy{};  // a new conversation probes again
    }
    Job& jr = *j;
    jobs.push_back(std::move(j));
    const bool greedy = a.seed == 0 && a.draft_k == 0;
    bool ok;
    if (greedy && a.eagle_k > 0 && heads[jr.sess] != nullptr && !policy[jr.sess].off) {
      jr.eagle = std::make_unique<EagleRun>(s, heads[jr.sess], jr.a, &jr.g, &policy[jr.sess]);
      bool done = false;
      ok = jr.eagle->start(jr.resume, &done, &jr.err);
      if (ok && done) finish(jr);
    } else if (greedy) {
      ok = start_plain(jr);
    } else {
      // Sampled / draft-lookup: monolithic on this session.
      ok = gpu_generate(s, jr.a, &jr.g, &jr.err, nullptr, jr.resume);
      if (ok) finish(jr);
    }
    if (!ok) {
      if (jr.err.empty()) jr.err = "generation failed";
      finish(jr);
    }
    return true;
  };
  // ---- batched EAGLE round over several sessions ------------------------
  // Head steps (step 1 + chain) for all runs in eagle_forward_multi calls,
  // one verify forward over every run's anchor + drafts (gpu_forward_multi),
  // then each run commits. Policy: after kEagleProbeRounds timed rounds the
  // batch decodes a timed plain burst (gpu_decode_multi, all sessions) and
  // keeps whichever path produced tokens faster; a winning EAGLE re-syncs
  // every head (exact). The decision is per batch composition.
  std::uint16_t* d_multi_hidden = nullptr;
  if (cudaMalloc(&d_multi_hidden, sizeof(std::uint16_t) * kMaxVerifyRows *
                                      static_cast<std::size_t>(cfg.hidden_size)) != cudaSuccess)
    return fail("cudaMalloc failed (batched EAGLE features)");
  struct BatchPolicy {
    std::vector<Job*> members;
    int rounds = 0;
    long tokens = 0;
    double ms = 0.0;
    bool decided = false;
  } bp;
  auto fail_job = [&](Job& j, const std::string& e) {
    if (j.err.empty()) j.err = e;
    j.eagle->report_timing();
    finish(j);
  };
  auto batched_round = [&](std::vector<Job*>& ej) {
    if (bp.members != ej) bp = BatchPolicy{ej};
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t gen_before = 0;
    for (Job* j : ej) gen_before += j->g.ids.size();
    int kmax = 0;
    std::vector<int> ks;
    for (Job* j : ej) {
      ks.push_back(j->eagle->plan_k());
      kmax = std::max(kmax, ks.back());
    }
    // Head steps, batched across sessions (falls back to per-session calls
    // when the rows do not fit one batch).
    auto head_step = [&](std::vector<std::pair<Job*, EagleRowsReq>>& reqs) -> bool {
      if (reqs.empty()) return true;
      std::vector<EagleHead*> hs;
      std::vector<EagleRowsReq> rq;
      for (auto& [j, q] : reqs) {
        hs.push_back(heads[j->sess]);
        rq.push_back(q);
      }
      if (eagle_forward_multi(hs.data(), static_cast<int>(hs.size()), rq.data()) == 0)
        return true;
      for (std::size_t i = 0; i < hs.size(); ++i)
        if (eagle_forward(hs[i], rq[i].d_tokens, rq[i].pos0, rq[i].rows, rq[i].d_feat,
                          rq[i].d_feat_out, rq[i].d_draft_out) != 0)
          return false;
      return true;
    };
    std::vector<std::pair<Job*, EagleRowsReq>> reqs;
    for (std::size_t i = 0; i < ej.size(); ++i)
      if (ej[i]->eagle->needs_step1()) reqs.emplace_back(ej[i], ej[i]->eagle->step1_req());
    bool ok = head_step(reqs);
    for (auto& [j, q] : reqs) j->eagle->step1_finished();
    for (int step = 2; ok && step <= kmax; ++step) {
      reqs.clear();
      for (std::size_t i = 0; i < ej.size(); ++i)
        if (ks[i] >= step) reqs.emplace_back(ej[i], ej[i]->eagle->chain_req(step));
      ok = head_step(reqs);
    }
    if (!ok) {
      for (Job* j : ej) fail_job(*j, "eagle: head forward failed");
      return;
    }
    // One verify forward over every run's anchor + drafts.
    std::vector<GpuCtx*> rc;
    std::vector<std::int32_t*> rt;
    std::vector<int> rp;
    for (std::size_t i = 0; i < ej.size(); ++i)
      for (int r = 0; r <= ks[i]; ++r) {
        rc.push_back(sessions[ej[i]->sess].ctx());
        rt.push_back(sessions[ej[i]->sess].d_tokens());
        rp.push_back(ej[i]->eagle->verify_from() + r);
      }
    const int nrows = static_cast<int>(rc.size());
    GpuSession& lead = sessions[ej[0]->sess];
    std::vector<std::int32_t> ids;
    char cerr[256] = "";
    if (nrows > kMaxVerifyRows ||
        gpu_forward_multi(rc.data(), rt.data(), rp.data(), nrows, lead.logits_buffer(),
                          lead.argmax_buffer(), d_multi_hidden, cerr, sizeof cerr) != 0 ||
        !lead.argmax(nrows, &ids, &err)) {
      for (Job* j : ej) fail_job(*j, std::string("eagle batched verify: ") + cerr);
      return;
    }
    const std::size_t H = static_cast<std::size_t>(cfg.hidden_size);
    int off = 0;
    for (std::size_t i = 0; i < ej.size(); ++i) {
      Job& j = *ej[i];
      const int n = ks[i] + 1;
      cudaMemcpyAsync(j.eagle->verify_features(), d_multi_hidden + off * H,
                      sizeof(std::uint16_t) * n * H, cudaMemcpyDeviceToDevice,
                      cudaStreamPerThread);
      std::vector<std::int32_t> mine(ids.begin() + off, ids.begin() + off + n);
      off += n;
      if (j.eagle->commit(mine, false, &j.err)) {
        if (!j.err.empty() || j.eagle->done()) {
          j.eagle->report_timing();
          finish(j);
        } else {
          j.plain = true;
        }
      }
    }
    // Policy (per batch): time rounds 2..5, then one timed plain burst.
    std::size_t gen_after = 0;
    for (Job* j : ej) gen_after += j->g.ids.size();
    ++bp.rounds;
    if (bp.rounds >= 2 && !bp.decided) {
      bp.tokens += static_cast<long>(gen_after - gen_before);
      bp.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                   .count();
    }
    if (bp.decided || bp.rounds < kEagleProbeRounds + 1 || eagle_min_accept() <= 0.0) return;
    bp.decided = true;
    std::vector<Job*> live;
    for (Job* j : ej)
      if (!j->done && !j->plain) live.push_back(j);
    if (live.size() < 2) return;
    int burst = kEagleBurst;
    for (Job* j : live) {
      GpuSession& s = sessions[j->sess];
      burst = std::min({burst, j->a.max_tokens - static_cast<int>(j->g.ids.size()),
                        kMaxVerifyRows - j->eagle->pending_rows(), s.mapped() - s.written(),
                        s.max_pos() - 1 - s.written()});
    }
    if (burst < 2) return;
    std::vector<GpuCtx*> cs;
    std::vector<std::int32_t*> ts;
    std::vector<int> pos;
    for (Job* j : live) {
      cs.push_back(sessions[j->sess].ctx());
      ts.push_back(sessions[j->sess].d_tokens());
      pos.push_back(sessions[j->sess].written());
    }
    const auto b0 = std::chrono::steady_clock::now();
    if (gpu_decode_multi(cs.data(), ts.data(), static_cast<int>(live.size()), burst,
                         lead.logits_buffer(), cerr, sizeof cerr) != 0) {
      for (Job* j : live) fail_job(*j, std::string("eagle burst: ") + cerr);
      return;
    }
    std::vector<std::vector<std::int32_t>> got(live.size());
    for (std::size_t i = 0; i < live.size(); ++i)
      if (!sessions[live[i]->sess].tokens(pos[i] + 1, pos[i] + 1 + burst, &got[i], &err)) {
        fail_job(*live[i], "eagle burst readback failed");
        return;
      }
    const double plain_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b0).count();
    const double plain_rate = static_cast<double>(burst * live.size()) / std::max(plain_ms, 1e-3);
    const double eagle_rate = static_cast<double>(bp.tokens) / std::max(bp.ms, 1e-3);
    const bool to_plain = plain_rate >= eagle_rate;
    if (debug)
      std::fprintf(stderr, "G4_EAGLE batch probe (%zu sessions): eagle %.1f tok/s, plain %.1f "
                   "tok/s -> %s\n", live.size(), 1000.0 * eagle_rate, 1000.0 * plain_rate,
                   to_plain ? "plain" : "eagle");
    for (std::size_t i = 0; i < live.size(); ++i) {
      Job& j = *live[i];
      bool stop = false;
      for (std::int32_t t : got[i])
        if ((stop = j.eagle->emit(t))) break;
      if (stop) {
        j.eagle->report_timing();
        finish(j);
      } else if (to_plain) {
        j.eagle->yield_plain();
        j.plain = true;
      } else if (!j.eagle->resync(burst, &j.err)) {
        fail_job(j, "eagle resync failed");
      }
    }
  };
  std::deque<std::string> pending;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(inq.m);
      if (active_count() == 0 && pending.empty())
        inq.cv.wait(lk, [&] { return !inq.q.empty() || inq.eof; });
      while (!inq.q.empty()) {
        pending.push_back(std::move(inq.q.front()));
        inq.q.pop_front();
      }
      if (inq.eof && inq.q.empty() && pending.empty() && active_count() == 0) break;
    }
    while (!pending.empty() && admit(pending.front())) pending.pop_front();
    jobs.remove_if([](const std::unique_ptr<Job>& j) { return j->done; });
    if (jobs.empty()) continue;
    // EAGLE jobs: a solo round when alone, else one batched round for all.
    {
      std::vector<Job*> ej;
      for (auto& jp : jobs)
        if (!jp->done && jp->eagle != nullptr && !jp->plain) ej.push_back(jp.get());
      if (ej.size() == 1) {
        Job& j = *ej[0];
        if (j.eagle->round(&j.err)) {
          if (!j.err.empty() || j.eagle->done()) {
            j.eagle->report_timing();
            finish(j);
          } else {
            j.plain = true;  // drafting stopped paying: plain decode from here
          }
        }
      } else if (ej.size() >= 2) {
        batched_round(ej);
      }
    }
    // Plain jobs: one batched chunk for all of them.
    std::vector<Job*> pj;
    for (auto& jp : jobs)
      if (!jp->done && jp->plain) pj.push_back(jp.get());
    if (pj.empty()) continue;
    int steps = 8;
    for (Job* j : pj) {
      GpuSession& s = sessions[j->sess];
      steps = std::min({steps, s.mapped() - s.written(), s.max_pos() - 1 - s.written(),
                        j->a.max_tokens - static_cast<int>(j->g.ids.size())});
    }
    if (steps < 1) {
      // A job at the end of its rows (only reachable at the context bound).
      for (Job* j : pj) finish(*j);
      continue;
    }
    std::vector<int> pos(pj.size());
    for (std::size_t i = 0; i < pj.size(); ++i) pos[i] = sessions[pj[i]->sess].written();
    bool ok = true;
    std::string e;
    if (pj.size() == 1) {
      ok = sessions[pj[0]->sess].decode_greedy(steps, &e);
    } else {
      std::vector<GpuCtx*> cs;
      std::vector<std::int32_t*> ts;
      for (Job* j : pj) {
        cs.push_back(sessions[j->sess].ctx());
        ts.push_back(sessions[j->sess].d_tokens());
      }
      char cerr[256] = "";
      ok = gpu_decode_multi(cs.data(), ts.data(), static_cast<int>(pj.size()), steps,
                            sessions[pj[0]->sess].logits_buffer(), cerr, sizeof cerr) == 0;
      if (!ok) e = std::string("gpu_decode_multi: ") + cerr;
    }
    for (std::size_t i = 0; i < pj.size(); ++i) {
      Job& j = *pj[i];
      std::vector<std::int32_t> ids;
      if (!ok || !sessions[j.sess].tokens(pos[i] + 1, pos[i] + 1 + steps, &ids, &e)) {
        j.err = e.empty() ? "decode failed" : e;
        finish(j);
        continue;
      }
      for (std::int32_t t : ids) {
        if (j.eagle != nullptr) {
          if (j.eagle->emit(t)) {
            j.eagle->report_timing();
            finish(j);
            break;
          }
        } else {
          j.g.ids.push_back(t);
          if (j.a.is_eos(t) || static_cast<int>(j.g.ids.size()) >= j.a.max_tokens) {
            finish(j);
            break;
          }
        }
      }
    }
  }
  reader.join();
  cudaFree(d_multi_hidden);
  return 0;
}
#endif  // GRANITE_CUDA

// ---------------------------------------------------------------------------
// Host oracle paths.
// ---------------------------------------------------------------------------

int host_forward_cmd(const Args& a, const ModelWeights& w) {
  std::vector<float> logits;
  forward_logits_f32(a.tokens, w.embed, w.layers, w.final_norm.data(), w.lm_head.data(),
                     w.model, kForwardSpec, &logits);
  const int n = static_cast<int>(a.tokens.size());
  const std::size_t V = w.model.vocab_size;
  print_logits(a.last > 0 ? std::max(0, n - a.last) : 0, n, V,
               [&](int t, std::size_t v) { return logits[static_cast<std::size_t>(t) * V + v]; });
  return 0;
}

int host_generate(const Args& a, const ModelWeights& w) {
  std::vector<std::int32_t> seq = a.tokens;
  std::vector<std::int32_t> generated;
  std::vector<float> logits;
  const int V = static_cast<int>(w.model.vocab_size);
  auto forward = [&](const std::vector<std::int32_t>& toks) {
    forward_logits_f32(toks, w.embed, w.layers, w.final_norm.data(), w.lm_head.data(),
                       w.model, kForwardSpec, &logits);
  };
  if (a.draft_k > 0) {
    // Host mirror of the CUDA draft-verify-commit loop. forward_logits_f32
    // returns f32 logits for every position and has no KV watermark, so
    // each step replays the whole accepted prefix from position 0.
    DraftReport report;
    while (static_cast<int>(generated.size()) < a.max_tokens) {
      const int len = static_cast<int>(seq.size());
      const int room = std::min(a.draft_k, a.max_tokens - static_cast<int>(generated.size()));
      const std::vector<std::int32_t> proposal =
          (room > 0) ? draft_lookup(seq, room) : std::vector<std::int32_t>();
      const int nprop = static_cast<int>(proposal.size());
      report.proposed += nprop;
      std::vector<std::int32_t> cand(seq);
      cand.insert(cand.end(), proposal.begin(), proposal.end());
      forward(cand);
      std::vector<std::int32_t> target_arg(nprop + 1, 0);
      for (int i = 0; i <= nprop; ++i)
        target_arg[i] = argmax_f32(logits.data() + static_cast<std::size_t>(len - 1 + i) * V, V);
      const VerifyResult v = verify_greedy(target_arg.data(), proposal.data(), nprop);
      const int budget = a.max_tokens - static_cast<int>(generated.size());
      const CommitPlan plan =
          commit_speculative(v, proposal.data(), nprop, target_arg[nprop], budget);
      bool stop = false;
      if (plan.commit > 0) {
        for (std::int32_t tok : plan.tokens) {
          seq.push_back(tok);
          generated.push_back(tok);
          if (a.is_eos(tok)) {
            stop = true;
            break;
          }
        }
        report.accepted += v.accepted;
      }
      ++report.steps;
      if (stop) break;
    }
    report_draft(report);
    std::printf("%s", generate_document(generated, &report).c_str());
    return 0;
  }
  for (int step = 0; step < a.max_tokens; ++step) {
    forward(seq);
    const float* row =
        logits.data() + static_cast<std::size_t>(seq.size() - 1) * w.model.vocab_size;
    int next;
    if (a.seed == 0) {
      next = sample_greedy_f32(row, V);
    } else {
      Xorshift64 rng(static_cast<std::uint64_t>(a.seed) + static_cast<std::uint64_t>(step));
      next = sample_top_p_f32(row, /*nucleus=*/0.9f, rng.next_float(), V);
    }
    seq.push_back(next);
    generated.push_back(next);
    if (a.is_eos(next)) break;
  }
  std::printf("%s", generate_document(generated, nullptr).c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, &a)) {
    std::fprintf(
        stderr,
        "usage: granite4-infer forward --checkpoint DIR --tokens id,id,... "
        "[--last N] [--gpu 0|1]\n"
        "       granite4-infer generate --checkpoint DIR --tokens id,id,... "
        "--max-tokens N [--seed S] [--gpu 0|1] [--draft-lookup K] [--eos ID]\n"
        "         [--eagle HEAD.safetensors --eagle-k K]\n"
        "       granite4-infer serve --checkpoint DIR --gpu 1 [--max-context N] [--eagle HEAD]\n"
        "       granite4-infer dump-hidden --checkpoint DIR --gpu 1 --in TOKENS.bin "
        "--out HIDDEN.bin [--max-context N]\n"
        "       granite4-infer kv-bench --checkpoint DIR --gpu 1 --bench-ctx N "
        "--bench-decode D [--seed S]   (G4_KV8=1 int8 KV, G4_KV8=4 int4 KV)\n");
    return 2;
  }
  if (a.sub == "generate" && a.max_tokens <= 0) {
    std::fprintf(stderr, "generate: --max-tokens required\n");
    return 2;
  }
  if (a.sub == "kv-bench" && (a.bench_ctx < 2 || a.bench_decode <= 0)) {
    std::fprintf(stderr, "kv-bench: --bench-ctx N (>=2) and --bench-decode D required\n");
    return 2;
  }
  if (a.sub == "generate" && a.draft_k > 0 && a.seed != 0) {
    std::fprintf(stderr, "draft-lookup: greedy only (omit --seed)\n");
    return 2;
  }

  if ((a.sub == "serve" || a.sub == "dump-hidden") && a.gpu == 0) {
    std::fprintf(stderr, "%s: requires --gpu 1\n", a.sub.c_str());
    return 2;
  }
  if (a.gpu != 0) {
#if defined(GRANITE_CUDA) && GRANITE_CUDA == 1
    if (a.sub == "forward") return gpu_forward_cmd(a);
    if (a.sub == "serve") return gpu_serve_cmd(a);
    if (a.sub == "dump-hidden") return gpu_dump_hidden_cmd(a);
    if (a.sub == "kv-bench") return gpu_kv_bench_cmd(a);
    return gpu_generate_cmd(a);
#else
    std::fprintf(stderr, "granite4-infer was built without CUDA support\n");
    return 1;
#endif
  }

  ModelWeights w;
  std::string err;
  if (!load_weights(a.dir, &w, &err, /*enforce_spec=*/true)) {
    std::fprintf(stderr, "load_weights: %s\n", err.c_str());
    return 1;
  }
  return a.sub == "forward" ? host_forward_cmd(a, w) : host_generate(a, w);
}
