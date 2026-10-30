#include "speculative_verify.h"

#include <algorithm>

namespace granite4 {

VerifyResult verify_greedy(const std::int32_t* target_argmax,
                           const std::int32_t* draft, int K) {
  for (int j = 0; j < K; ++j) {
    if (target_argmax[j] != draft[j]) {
      return VerifyResult{.accepted = j, .correction = target_argmax[j]};
    }
  }
  return VerifyResult{.accepted = K, .correction = -1};
}

CommitPlan commit_speculative(const VerifyResult& v,
                              const std::int32_t* draft, int K,
                              std::int32_t bonus, int budget) {
  (void)K;
  const int commit = (budget > 0) ? std::min(v.accepted + 1, budget) : 0;
  CommitPlan plan;
  plan.commit = commit;
  if (commit <= 0) return plan;

  // The accepted prefix, then the gate's tail token. On a rejection the tail
  // is the correction; on a full acceptance it is the bonus (the target's own
  // next token). When the budget clamps commit below accepted + 1 the tail
  // is dropped, so only the prefix up to `commit` is kept.
  const int prefix = std::min(v.accepted, commit);
  plan.tokens.reserve(static_cast<std::size_t>(commit));
  for (int i = 0; i < commit; ++i) {
    const int tok =
        (i < prefix) ? draft[i]
                     : (v.correction >= 0 ? v.correction : bonus);
    plan.tokens.push_back(static_cast<std::int32_t>(tok));
  }
  return plan;
}

}  // namespace granite4
