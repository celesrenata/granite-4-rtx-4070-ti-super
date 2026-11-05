// Session selection for the persistent serve loop. Pure host logic, no CUDA:
// picks the session whose KV watermark is a full prefix of the new prompt,
// so only the tail is uploaded and prefilled; on a miss, falls back to the
// round-robin pointer's next session and resets its KV.
//
// Host-only because the serve loop is the only caller and the decision is
// a pure function of (per-session history, per-session watermark, new
// prompt, round-robin pointer). Extracting it lets the algorithm be
// unit-tested on a box without a GPU, which is where it's hardest to
// get right (every session's KV must stay consistent with its history).

#ifndef GRANITE_SERVE_RESUME_H
#define GRANITE_SERVE_RESUME_H

#include <cstdint>
#include <cstddef>
#include <vector>

namespace granite4 {

// Per-session state the serve loop maintains alongside its GpuSession
// pool. Invariant after each turn (maintained by the loop, not by this
// function): `watermark == history.size()`, i.e. the KV cache holds
// exactly the positions of `history[0 .. history.size())`.
struct ServeSessionState {
  std::vector<std::int32_t> history;  // tokens whose KV is cached
  int watermark = 0;                  // == history.size() when valid
};

// The decision a serve request makes: which session to run on and how
// many leading prompt tokens to skip in the prefill (they're already in
// that session's KV). `resume == 0` means a full re-prefill
// (truncate(0)); `resume > 0` means the session's watermark starts at
// `resume` and only the tail of the prompt is uploaded and forwarded.
//
// Picking rule: the session with the longest `history[i]` that is a
// full prefix of `prompt` and whose `watermark[i] == history[i].size()`
// (i.e. a "clean" session whose KV really is `history`). Ties break by
// session index (lower index wins). On a miss, the round-robin pointer's
// next session is used with `resume == 0`.
//
// `next` is the round-robin pointer the serve loop will advance to after
// this request, so the next miss lands on the next session in order.
struct ServeDecision {
  int picked = 0;       // session to run on
  int resume = 0;       // leading prompt tokens already cached
  int next = 0;         // updated round-robin pointer for the next miss
};

ServeDecision serve_decide(const std::vector<ServeSessionState>& sessions,
                           int next,
                           const std::vector<std::int32_t>& prompt);

}  // namespace granite4

#endif  // GRANITE_SERVE_RESUME_H
