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
#include <string>
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

// Cumulative draft-lookup counters over a `generate --draft-lookup` run.
// `proposed` counts every token the drafter proposed; `accepted` counts the
// ones the gate kept; `steps` counts verification rounds.
struct DraftReport {
  long steps = 0;
  long proposed = 0;
  long accepted = 0;
};

// `accepted / proposed` (0.0 when nothing was proposed).
inline double draft_accept_rate(const DraftReport& r) {
  return r.proposed > 0
             ? static_cast<double>(r.accepted) / static_cast<double>(r.proposed)
             : 0.0;
}

// Assemble the `generate` stdout JSON document: the produced token ids
// followed, when `report` is non-null, by the draft-verify counters
// (`{"generated":[...],"draft":{"steps":S,"proposed":P,"accepted":A,
// "accept_rate":R}}`). One canonical producer so the CUDA and host emit sites
// stay in lockstep; the exact shape is pinned by a unit test.
std::string generate_document(const std::vector<std::int32_t>& generated,
                              const DraftReport* report);

}  // namespace granite4

#endif  // GRANITE_SPECULATIVE_VERIFY_H
