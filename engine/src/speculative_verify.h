// Greedy speculative-decoding accept/reject gate.
//
// Pure host function: no CUDA, no model weights, no allocation. Unit-testable
// on any platform. The CUDA fused forward pass (verify_batched.cu) produces
// the K target-argmax tokens in one shot; this gate then decides which
// prefix of the drafter's proposals to keep.

#ifndef GRANITE_SPECULATIVE_VERIFY_H
#define GRANITE_SPECULATIVE_VERIFY_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace granite4 {

// Result of a greedy speculative-verify gate.
struct VerifyResult {
  int accepted;         // Number of draft tokens accepted (0..K).
  std::int32_t correction; // Token to append after the accepted prefix.
                           // -1 if the full draft was accepted (no mismatch).
};

// Decide which prefix of `draft` to accept, given the target model's own
// greedy argmax at each draft position.
//
// Semantics (matches the inline loop in infer_main.cc):
//   For j in [0, K):
//     If target[j] != draft[j]: accept j tokens, correction = target[j], stop.
//     Otherwise: accept j+1 tokens.
//   If all K match: accepted = K, correction = -1.
//
// The caller appends `draft[0..accepted-1]` to the sequence, then (if
// correction != -1) appends `correction`.
VerifyResult verify_greedy(const std::int32_t* target_argmax,
                           const std::int32_t* draft, int K);

// Which tokens the speculative step commits, given the gate's verdict and the
// remaining output budget.
//
// `tokens` holds `commit` entries, in append order: the accepted draft prefix,
// then the gate's tail token — the correction on a rejection, or the target's
// own final row (the bonus) when the draft was fully accepted.
struct CommitPlan {
  int commit;  // Number of tokens to commit (0..min(accepted+1, budget)).
  std::vector<std::int32_t> tokens;
};

// Derive the commit plan from `v` and the tokens still allowed this step
// (`budget = max_tokens - generated.size()`).
//
// Semantics (matches the inline block in infer_main.cc):
//   commit = 0                          if budget <= 0
//   commit = min(v.accepted + 1, budget) otherwise
//
// The tail token is always emitted: on a rejection it is the correction, on a
// full acceptance it is `bonus` (the target's own next token). When budget is
// exactly 1 the accepted prefix is dropped and only the tail is kept — on a
// rejection that tail is the correction, which is what greedy semantics
// dictate.
CommitPlan commit_speculative(const VerifyResult& v,
                              const std::int32_t* draft, int K,
                              std::int32_t bonus, int budget);

}  // namespace granite4

#endif  // GRANITE_SPECULATIVE_VERIFY_H
