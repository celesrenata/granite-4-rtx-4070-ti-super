#include "draft_lookup.h"

#include <cstdio>
#include <vector>

using granite4::draft_lookup;

int main() {
  const std::vector<std::int32_t> h = {10, 20, 30, 40, 10, 20, 30};
  const auto got = draft_lookup(h, 4);
  const std::vector<std::int32_t> want = {40, 10, 20, 30};
  if (got != want) {
    std::fprintf(stderr, "draft lookup mismatch\n");
    return 1;
  }
  // The most recent non-overlapping occurrence wins: both occurrences of
  // {10,20,30} have a continuation, but the later one (50) is chosen.
  const std::vector<std::int32_t> tie = {10, 20, 30, 40, 10, 20, 30, 50,
                                         99, 10, 20, 30};
  const auto tie_pick = draft_lookup(tie, 4);
  const std::vector<std::int32_t> want_tie = {50, 99, 10, 20};
  if (tie_pick != want_tie) {
    std::fprintf(stderr, "draft recent-occurrence mismatch\n");
    return 1;
  }
  // No proposal for trivial histories or a zero budget; single repeated
  // tokens are below the two-token minimum match.
  if (!draft_lookup({1}, 4).empty() || !draft_lookup(h, 0).empty()) return 1;
  if (!draft_lookup({1, 2, 1}, 2).empty()) return 1;
  std::puts("draft_lookup: PASS");
  return 0;
}
