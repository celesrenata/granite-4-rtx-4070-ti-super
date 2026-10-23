// Phase 3 (Integration): inference CLI.
//
// granite4-infer <subcommand>
//
//   forward --checkpoint DIR --tokens "id,id,..."
//           [--last N]
//     Load the model (original BF16 or an FP8 export), run the deterministic
//     forward pass (forward_logits_f32) over the given token ids, and write
//     the output logits as JSON. `--last N` restricts output to the final N
//     positions (default: all). This is the primitive the golden parity gate
//     compares against a transformers reference (same checkpoint, same
//     tokens -> same logits, up to float rounding).
//
//   generate --checkpoint DIR --tokens "id,..." --max-tokens N
//            [--seed S]
//     Greedy (seed=0) or seeded top-p generation, emitting the produced token
//     ids as JSON. (Tokenization/detokenization is done by the caller - the
//     Python harness - so this path is token-level, not text-level.)
//
// Host-only, no CUDA. Tokenization is intentionally NOT done here: the
// checkpoint ships a HF `tokenizer.json` that the Python harness loads with
// the `tokenizers` library, keeping token-for-token parity on one side.

#include "kernel_math.h"
#include "weights.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace granite4;

namespace {

bool parse_args(int argc, char** argv, std::string* sub, std::string* dir,
                std::vector<std::int32_t>* tokens, int* max_tokens,
                long* seed, int* last) {
  *sub = "";
  *dir = "";
  *max_tokens = 0;
  *seed = 0;
  *last = 0;
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
      *max_tokens = std::stoi(argv[++i]);
    } else if (a == "--seed" && i + 1 < argc) {
      *seed = std::stol(argv[++i]);
    } else if (a == "--last" && i + 1 < argc) {
      *last = std::stoi(argv[++i]);
    }
  }
  return !sub->empty() && !dir->empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string sub, dir;
  std::vector<std::int32_t> tokens;
  int max_tokens = 0;
  long seed = 0;
  int last = 0;
  if (!parse_args(argc, argv, &sub, &dir, &tokens, &max_tokens, &seed, &last)) {
    std::fprintf(stderr,
                 "usage: granite4-infer forward --checkpoint DIR --tokens "
                 "id,... [--last N]\n"
                 "       granite4-infer generate --checkpoint DIR --tokens id,..."
                 " --max-tokens N [--seed S]\n");
    return 2;
  }

  ModelWeights w;
  std::string err;
  if (!load_weights(dir, &w, &err)) {
    std::fprintf(stderr, "granite4-infer: load failed: %s\n", err.c_str());
    return 1;
  }
  const ForwardSpec spec = kForwardSpec;

  if (sub == "forward") {
    std::vector<float> logits;
    forward_logits_f32(tokens, w.embed, w.layers, w.final_norm.data(),
                       w.lm_head.data(), w.model, spec, &logits);
    const int n = static_cast<int>(tokens.size());
    const int V = static_cast<int>(w.model.vocab_size);
    const int start = (last > 0 && last < n) ? n - last : 0;
    std::printf("{\"tokens\":[");
    for (int t = 0; t < n; ++t) std::printf("%d%s", tokens[t], t + 1 < n ? "," : "");
    std::printf("],\"positions\":%d,", n);
    std::printf("\"logits\":[");
    for (int t = start; t < n; ++t) {
      if (t > start) std::printf(",");
      std::printf("[");
      for (int i = 0; i < V; ++i)
        std::printf("%.6g%s", logits[(std::size_t)t * V + i], i + 1 < V ? "," : "");
      std::printf("]");
    }
    std::printf("]}\n");
    return 0;
  }

  if (sub == "generate") {
    if (max_tokens <= 0) {
      std::fprintf(stderr, "generate: --max-tokens required\n");
      return 2;
    }
    std::vector<std::int32_t> seq = tokens;
    std::vector<std::int32_t> produced;
    for (int step = 0; step < max_tokens; ++step) {
      std::vector<float> logits;
      forward_logits_f32(seq, w.embed, w.layers, w.final_norm.data(),
                         w.lm_head.data(), w.model, spec, &logits);
      const int V = static_cast<int>(w.model.vocab_size);
      const float* last_logits = logits.data() + (seq.size() - 1) * V;
      int next = 0;
      if (seed == 0) {
        next = sample_greedy_f32(last_logits, V);
      } else {
        Xorshift64 rng(seed + step);
        const float u = rng.next_float();
        next = sample_top_p_f32(last_logits, 0.9f, u, V);
      }
      seq.push_back(next);
      produced.push_back(next);
    }
    std::printf("{\"generated\":[");
    for (int i = 0; i < (int)produced.size(); ++i)
      std::printf("%d%s", produced[i], i + 1 < (int)produced.size() ? "," : "");
    std::printf("]}\n");
    return 0;
  }
  return 2;
}
