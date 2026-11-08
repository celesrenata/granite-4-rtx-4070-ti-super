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
// Picking rule: for every clean session (watermark == history.size(), so its
// KV really is `history`), the reusable length is the longest common prefix of
// `history` and `prompt`, capped at prompt.size() - 1 (at least one prompt
// token is always forwarded: it yields the first generated token). KV rows
// [0, lcp) are valid whatever follows them, so a session whose history merely
// *diverges* from the prompt late - a chat template that re-renders the last
// assistant turn differently, an edited or regenerated last message - still
// resumes from the divergence point instead of re-prefilling everything. The
// session with the longest such prefix wins (ties: lower index), provided it
// is at least `min_resume` positions; a shorter overlap (e.g. a shared system
// prompt) is a miss, so it never evicts another conversation's cache for a
// negligible saving. On a miss, the first empty session at or after the
// round-robin pointer is used, else the pointer's session, with resume == 0.
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
                           const std::vector<std::int32_t>& prompt,
                           int min_resume = 1);

}  // namespace granite4

#endif  // GRANITE_SERVE_RESUME_H
