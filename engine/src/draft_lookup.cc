#include "draft_lookup.h"

#include <algorithm>

namespace granite4 {
namespace {
// Short matches (a single repeated token) are too noisy to be worth a
// verification slot; two-token suffixes are the smallest useful anchor.
constexpr int kMinMatch = 2;
constexpr int kMaxMatch = 32;
}  // namespace

std::vector<std::int32_t> draft_lookup(const std::vector<std::int32_t>& history,
                                        int max_tokens) {
  if (max_tokens <= 0 || history.size() < 2) return {};
  const int n = static_cast<int>(history.size());
  const int max_len = std::min(n - 1, kMaxMatch);
  for (int len = max_len; len >= kMinMatch; --len) {
    const int suffix = n - len;
    // Most recent non-overlapping occurrence first: its continuation is the
    // closest copy in the history and the likeliest to repeat.
    for (int start = suffix - 1; start >= 0; --start) {
      if (!std::equal(history.begin() + start, history.begin() + start + len,
                      history.begin() + suffix))
        continue;
      const int available = std::min(max_tokens, n - (start + len));
      if (available <= 0) continue;
      return std::vector<std::int32_t>(history.begin() + start + len,
                                       history.begin() + start + len + available);
    }
  }
  return {};
}

}  // namespace granite4
