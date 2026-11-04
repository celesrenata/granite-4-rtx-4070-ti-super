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

  bool is_eos(std::int32_t tok) const { return eos >= 0 && tok == eos; }
};

bool parse_args(int argc, char** argv, Args* a) {
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    if (s == "forward" || s == "generate" || s == "serve" || s == "dump-hidden") {
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
  if (a->sub == "serve") return !a->dir.empty() && a->max_context > 0;
  if (a->sub == "dump-hidden")
    return !a->dir.empty() && !a->in.empty() && !a->out.empty() && a->max_context > 0;
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

  int vocab() const { return vocab_; }
  int hidden() const { return hidden_; }
  int max_pos() const { return max_pos_; }
  int logit_rows() const { return logit_rows_; }
  int written() const { return gpu_ctx_written(ctx_); }
  bool truncate(int written) { return gpu_ctx_truncate(ctx_, written) == 0; }


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
  std::int32_t* d_tokens() { return d_tokens_; }
  const std::string& dir() const { return dir_; }
  // Device scratch for draft-head features: [max_pos][hidden] (prompt) and
  // [kMaxVerifyRows][hidden] (verify rows, chain); allocated on first use.
  bool feature_buffers(std::uint16_t** prompt, std::uint16_t** verify, std::uint16_t** chain,
                       std::string* error) {
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
// persistent `serve` loop) and start from an empty KV cache. Preconditions
// (checked by check_request): prompt + max_tokens (+ draft_k) fits the
// session's positions and draft_k + 1 fits its logits rows.
struct Generation {
  std::vector<std::int32_t> ids;
  DraftReport report;
  bool drafted = false;
};

bool check_request(const GpuSession& s, const Args& a, std::string* error) {
  const long need = static_cast<long>(a.tokens.size()) + a.max_tokens + a.draft_k +
                    (a.eagle_k > 0 ? a.eagle_k + 1 : 0);
  if (a.tokens.empty() || a.max_tokens <= 0) {
    *error = "empty prompt or max_tokens <= 0";
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

bool generate_greedy(GpuSession& s, const Args& a, Generation* out, std::string* error) {
  const int plen = static_cast<int>(a.tokens.size());
  // Prefill: the prompt's KV plus the last row's greedy token, written on the
  // device straight into d_tokens[plen]; the decode chain continues from it.
  if (!s.truncate(0) || !s.upload(a.tokens, 0, plen, error) ||
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

bool generate_sampled(GpuSession& s, const Args& a, Generation* out, std::string* error) {
  std::vector<std::int32_t> seq = a.tokens;
  const int plen = static_cast<int>(seq.size());
  if (!s.truncate(0) || !s.upload(seq, 0, plen, error)) return false;
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

bool generate_draft(GpuSession& s, const Args& a, Generation* out, std::string* error) {
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
  if (!s.truncate(0) || !s.upload(seq, 0, plen, error) ||
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

// EAGLE draft-verify loop. Invariants at the top of a round: the target KV
// holds positions [0, L); the anchor (the target's greedy token at L, already
// emitted) is in d_tokens[L]; head entries [0, hq) are valid and `feat`
// holds the target features of positions [hq, L). One round: head catch-up
// over [hq, L) (its last row drafts d_1), k - 1 chained head steps (d_2..d_k
// into d_tokens), one target forward over [L, L + k] (anchor + drafts) that
// yields the greedy ids t_0..t_k and the features of those rows. The exact
// gate accepts the prefix d_i == t_{i-1}; t_a is the new anchor. Everything
// up to the read-back of t and d is enqueued without a host round trip.
bool generate_eagle(GpuSession& s, EagleHead* head, const Args& a, Generation* out,
                    std::string* error) {
  if (head == nullptr) {
    *error = "eagle: no head loaded (start with --eagle HEAD)";
    return false;
  }
  std::vector<std::int32_t> seq = a.tokens;
  const int plen = static_cast<int>(seq.size());
  std::vector<std::int32_t>& generated = out->ids;
  DraftReport& report = out->report;
  out->drafted = true;
  std::uint16_t *f_prompt, *f_verify, *f_chain;
  if (!s.feature_buffers(&f_prompt, &f_verify, &f_chain, error)) return false;
  std::vector<std::int32_t> ids;
  if (!s.truncate(0) || !s.upload(seq, 0, plen, error) ||
      !s.forward(plen, plen - 1, error, f_prompt, 0) || !s.argmax(1, &ids, error))
    return false;
  eagle_truncate(head, 0);
  auto emit = [&](std::int32_t tok) {
    seq.push_back(tok);
    generated.push_back(tok);
    return a.is_eos(tok) || static_cast<int>(generated.size()) >= a.max_tokens;
  };
  if (emit(ids[0])) return true;
  int L = plen, hq = 0;
  std::uint16_t* feat = f_prompt;  // features of positions [hq, L); outputs overwrite them
  std::int32_t* const d_tok = s.d_tokens();
  const std::size_t H = static_cast<std::size_t>(s.hidden());
  std::vector<std::int32_t> drafts;
  if (!s.upload(seq, L, L + 1, error)) return false;
  // G4_EAGLE_TIMING=1: per-round device time split into head-draft and target
  // verify (CUDA events around each; readback = wall - head - verify). Zero cost
  // when unset.
  const char* const eam = std::getenv("G4_EAGLE_TIMING");
  const bool et = eam != nullptr && eam[0] == '1';
  struct Ev { cudaEvent_t a, b; };
  std::vector<Ev> h_ev, v_ev;
  long et_rounds = 0, et_acc = 0, et_prop = 0;
  const auto et_t0 = std::chrono::steady_clock::now();
  auto round = [&](std::string* out_err) -> bool {
    const int remaining = a.max_tokens - static_cast<int>(generated.size());
    const int k = std::max(0, std::min({a.eagle_k, remaining - 1, s.max_pos() - L - 2}));
    Ev h0, v0;
    if (et && k > 0) cudaEventCreate(&h0.a), cudaEventCreate(&h0.b);
    if (k > 0) {
      if (et) cudaEventRecord(h0.a, cudaStreamPerThread);
      const int rows = L - hq;
      std::uint16_t* const last = feat + static_cast<std::size_t>(rows - 1) * H;
      bool ok = eagle_forward(head, d_tok, hq, rows, feat, feat, d_tok + L + 1) == 0;
      for (int i = 2; ok && i <= k; ++i)
        ok = eagle_forward(head, d_tok, L + i - 2, 1, i == 2 ? last : f_chain, f_chain,
                           d_tok + L + i) == 0;
      if (et) cudaEventRecord(h0.b, cudaStreamPerThread), h_ev.push_back(h0);
      if (!ok) {
        *out_err = "eagle: head forward failed";
        return true;
      }
    }
    // Target verify over anchor + drafts: greedy ids t_0..t_k and features.
    if (et) cudaEventCreate(&v0.a), cudaEventCreate(&v0.b);
    if (et) cudaEventRecord(v0.a, cudaStreamPerThread);
    if (!s.forward(L + k + 1, L, error, f_verify, L) || !s.argmax(k + 1, &ids, error) ||
        !s.tokens(L + 1, L + 1 + k, &drafts, error))
      return true;
    if (et) cudaEventRecord(v0.b, cudaStreamPerThread), v_ev.push_back(v0);
    int acc = 0;
    while (acc < k && drafts[static_cast<std::size_t>(acc)] == ids[static_cast<std::size_t>(acc)])
      ++acc;
    report.proposed += k;
    report.accepted += acc;
    ++report.steps;
    if (et) { et_rounds++; et_acc += acc; et_prop += k; }
    bool stop = false;
    for (int i = 0; i < acc && !stop; ++i) stop = emit(drafts[static_cast<std::size_t>(i)]);
    if (!stop) stop = emit(ids[static_cast<std::size_t>(acc)]);
    if (stop) return true;
    // Target KV is valid for the anchor and the accepted drafts; the head's
    // chain entries (built on predicted features) are dropped.
    const int nl = L + acc + 1;
    if (!s.truncate(nl) || !s.upload(seq, nl, nl + 1, error)) {
      if (error->empty()) *error = "kv truncate failed";
      return true;
    }
    eagle_truncate(head, L);
    hq = L;
    feat = f_verify;
    L = nl;
    return false;
  };
  bool stop = false;
  do stop = round(error); while (!stop && error->empty());
  if (!error->empty()) return false;
  const double et_wall =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - et_t0).count();
  if (et) {
    double hms = 0, vms = 0;
    float dms = 0;
    for (auto& e : h_ev) if (cudaEventSynchronize(e.b) == cudaSuccess)
      dms = 0, cudaEventElapsedTime(&dms, e.a, e.b), hms += dms,
      cudaEventDestroy(e.a), cudaEventDestroy(e.b);
    for (auto& e : v_ev) if (cudaEventSynchronize(e.b) == cudaSuccess)
      dms = 0, cudaEventElapsedTime(&dms, e.a, e.b), vms += dms,
      cudaEventDestroy(e.a), cudaEventDestroy(e.b);
    const double R = et_rounds ? static_cast<double>(et_rounds) : 1.0;
    const double tok = static_cast<double>(generated.size());
    std::fprintf(stderr, "G4_EAGLE head_ms=%.1f (%.3f/rnd) verify_ms=%.1f (%.3f/rnd) "
                   "read_ms=%.1f wall_ms=%.1f rounds=%ld acc=%ld prop=%ld tok/rnd=%.2f "
                   "head%%=%.0f verify%%=%.0f tok/s=%.1f\n",
                hms, hms / R, vms, vms / R, et_wall - hms - vms, et_wall, et_rounds, et_acc,
                et_prop, (et_acc + et_rounds) / R, et_wall ? 100 * hms / et_wall : 0.0,
                et_wall ? 100 * vms / et_wall : 0.0, et_wall ? tok / (et_wall / 1000.0) : 0.0);
  }
  return true;
}

bool gpu_generate(GpuSession& s, const Args& a, Generation* out, std::string* error,
                  EagleHead* head = nullptr) {
  if (!check_request(s, a, error)) return false;
  if (a.eagle_k > 0) return generate_eagle(s, head, a, out, error);
  if (a.draft_k > 0) return generate_draft(s, a, out, error);
  return a.seed == 0 ? generate_greedy(s, a, out, error) : generate_sampled(s, a, out, error);
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

// `dump-hidden`: final-normed hidden states for every position of every
// input record (see the file header).
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
  std::string err;
  GpuSession s;
  constexpr int kServeLogitRows = 16;  // draft_lookup <= 15
  EagleHead* head = nullptr;
  if (!s.open(cli.dir, cli.max_context, kServeLogitRows, &err) ||
      !open_eagle(s, cli.eagle, &head, &err))
    return fail(err);
  struct HeadGuard {
    EagleHead* h;
    ~HeadGuard() { eagle_free(h); }
  } head_guard{head};
  {
    // Warm-up: load the lazily loaded kernels and capture the decode graph
    // before reporting ready, so the first real request pays neither (an
    // 8-token prompt runs the prefill path, then three chained greedy steps).
    Args w;
    w.tokens = {791, 6864, 315, 9822, 374, 13, 578, 7754};
    w.max_tokens = 4;
    Generation g;
    if (cli.max_context >= 16 && !gpu_generate(s, w, &g, &err)) return fail("warm-up: " + err);
  }
  std::printf("{\"ready\":true,\"max_context\":%d,\"eagle\":%s}\n", cli.max_context,
              head != nullptr ? "true" : "false");
  std::fflush(stdout);
  std::string line;
  for (int c; (c = std::getchar()) != EOF;) {
    if (c != '\n') {
      line.push_back(static_cast<char>(c));
      continue;
    }
    json::Value req;
    std::string perr;
    Args a;
    Generation g;
    std::string reply;
    if (!json::parse(line.data(), line.size(), &req, &perr) ||
        req.kind != json::Value::Kind::Object) {
      err = "bad request: " + perr;
    } else {
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
      err.clear();
      if (gpu_generate(s, a, &g, &err, head)) reply = generate_document(g.ids, g.drafted ? &g.report : nullptr);
    }
    if (reply.empty()) {
      std::string esc;
      for (char ch : err) {
        if (ch == '"' || ch == '\\') esc.push_back('\\');
        if (static_cast<unsigned char>(ch) >= 0x20) esc.push_back(ch);
      }
      reply = "{\"error\":\"" + esc + "\"}\n";
    }
    std::fwrite(reply.data(), 1, reply.size(), stdout);
    std::fflush(stdout);
    line.clear();
  }
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
        "--out HIDDEN.bin [--max-context N]\n");
    return 2;
  }
  if (a.sub == "generate" && a.max_tokens <= 0) {
    std::fprintf(stderr, "generate: --max-tokens required\n");
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
