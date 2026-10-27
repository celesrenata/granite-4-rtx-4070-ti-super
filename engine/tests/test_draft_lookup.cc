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
  if (!draft_lookup({1}, 4).empty() || !draft_lookup(h, 0).empty()) return 1;
  std::puts("draft_lookup: PASS");
  return 0;
}
