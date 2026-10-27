#include "draft_lookup.h"

#include <algorithm>

namespace granite4 {

std::vector<std::int32_t> draft_lookup(const std::vector<std::int32_t>& history,
                                        int max_tokens) {
  if (max_tokens <= 0 || history.size() < 2) return {};
  const int n = static_cast<int>(history.size());
  const int newest = n - 1;
  const int max_match = std::min(n - 1, 32);  // bounded CPU search per step
  for (int len = max_match; len >= 1; --len) {
    const int suffix = n - len;
    for (int start = 0; start + len < suffix; ++start) {
      if (!std::equal(history.begin() + start, history.begin() + start + len,
                      history.begin() + suffix))
        continue;
      const int available = std::min(max_tokens, n - (start + len));
      if (available <= 0) continue;
      return std::vector<std::int32_t>(history.begin() + start + len,
                                       history.begin() + start + len + available);
    }
  }
  (void)newest;
  return {};
}

}  // namespace granite4
