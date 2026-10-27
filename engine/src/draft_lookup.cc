#include "draft_lookup.h"

#include <algorithm>
#include <utility>

namespace granite4 {
namespace {
// Short matches (a single repeated token) are too noisy to be worth a
// verification slot; two-token suffixes are the smallest useful anchor.
constexpr int kMinMatch = 2;
constexpr int kMaxMatch = 32;

// Longest suffix of `history` that occurs earlier, non-overlapping. Returns
// all start indices of that suffix, most recent first.
std::vector<int> match_starts(const std::vector<std::int32_t>& history, int len) {
  const int n = static_cast<int>(history.size());
  const int suffix = n - len;
  std::vector<int> starts;
  for (int start = suffix - 1; start >= 0; --start)
    if (std::equal(history.begin() + start, history.begin() + start + len,
                   history.begin() + suffix))
      starts.push_back(start);
  return starts;
}
}  // namespace

std::vector<std::int32_t> draft_lookup(const std::vector<std::int32_t>& history,
                                        int max_tokens) {
  if (max_tokens <= 0 || history.size() < 2) return {};
  const int n = static_cast<int>(history.size());

  int len = 0;
  std::vector<int> active;
  for (int candidate = std::min(n - 1, kMaxMatch); candidate >= kMinMatch; --candidate) {
    active = match_starts(history, candidate);
    if (!active.empty()) {
      len = candidate;
      break;
    }
  }
  if (len == 0) return {};

  // Consensus extension: at each position after the match, take the token
  // most occurrences agree on; ties go to the most recent occurrence. This
  // beats copying one arbitrary occurrence when the same n-gram recurs with
  // different continuations.
  std::vector<std::int32_t> out;
  for (int m = 0; m < max_tokens && !active.empty(); ++m) {
    int best_tok = -1, best_count = 0, best_start = -1;
    for (int start : active) {
      const int idx = start + len + m;
      if (idx >= n) continue;
      const std::int32_t tok = history[idx];
      int count = 0;
      for (int other : active) {
        const int oi = other + len + m;
        if (oi < n && history[oi] == tok) ++count;
      }
      if (count > best_count || (count == best_count && start > best_start)) {
        best_count = count;
        best_tok = tok;
        best_start = start;
      }
    }
    if (best_tok < 0) break;
    out.push_back(best_tok);
    std::vector<int> next;
    for (int start : active) {
      const int idx = start + len + m;
      if (idx < n && history[idx] == best_tok) next.push_back(start);
    }
    active.swap(next);
  }
  return out;
}

}  // namespace granite4
