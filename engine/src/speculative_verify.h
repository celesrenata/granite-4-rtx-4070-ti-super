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

}  // namespace granite4

#endif  // GRANITE_SPECULATIVE_VERIFY_H
