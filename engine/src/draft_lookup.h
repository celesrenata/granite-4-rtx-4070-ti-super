#pragma once

#include <cstdint>
#include <vector>

namespace granite4 {

// Propose up to max_tokens by finding the longest suffix of `history` that
// occurred earlier and copying the tokens following that occurrence. Returns
// an empty vector when no usable repetition exists. This is a zero-weight,
// tokenizer-agnostic speculative drafter; the target model still verifies all
// proposals before emission.
std::vector<std::int32_t> draft_lookup(const std::vector<std::int32_t>& history,
                                        int max_tokens);

}  // namespace granite4
