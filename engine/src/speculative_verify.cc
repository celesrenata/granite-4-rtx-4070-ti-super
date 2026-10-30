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

}  // namespace granite4
