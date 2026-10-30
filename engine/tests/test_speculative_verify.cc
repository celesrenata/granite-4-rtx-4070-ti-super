// Unit tests for the greedy speculative-verify gate.
//
// The gate is the pure host logic that decides which prefix of the drafter's
// proposals to keep once the target model's greedy argmax for each draft
// position is known. The CUDA fused forward pass (verify_batched.cu) produces
// those K argmax tokens; this file exercises only the accept/reject decision.

#include "speculative_verify.h"

#include <cstdio>

namespace gv = granite4;

namespace {

struct TestCase {
  const char* name;
  int K;
  int target[16];  // Target model's greedy argmax at each draft position.
  int draft[16];   // Drafter's proposed tokens.
  int want_accepted;
  std::int32_t want_correction;  // -1 = full draft accepted, no correction.
};

const TestCase kCases[] = {
    // Mismatch at the very first position: nothing accepted, target's own
    // token is the correction.
    {"mismatch at j=0", 3, {5, 5, 5}, {9, 9, 9}, 0, 5},
    // Full agreement: all K tokens accepted, no correction.
    {"all accepted", 3, {1, 2, 3}, {1, 2, 3}, 3, -1},
    // Longest-common-prefix semantics: accept the prefix, correct on the
    // first divergence.
    {"accept prefix, correct at j=1", 4, {7, 8, 9, 10}, {7, 3, 8, 9}, 1, 8},
    // Mismatch on the last position: the whole prefix is kept.
    {"correct at last position", 3, {4, 4, 4}, {4, 4, 2}, 2, 4},
    // Two proposals, first matches.
    {"K=2 accept one", 2, {11, 12}, {11, 13}, 1, 12},
    // Single-token draft that matches.
    {"K=1 accept", 1, {99}, {99}, 1, -1},
    // Single-token draft that does not match.
    {"K=1 reject", 1, {99}, {98}, 0, 99},
    // Mismatch at the second of three, with different values throughout.
    {"mismatch mid with distinct values", 3, {10, 20, 30}, {10, 21, 30}, 1, 20},
};

bool run_case(const TestCase& c) {
  gv::VerifyResult r = gv::verify_greedy(c.target, c.draft, c.K);
  if (r.accepted != c.want_accepted || r.correction != c.want_correction) {
    std::fprintf(stderr, "FAIL %s: accepted=%d correction=%d (want %d / %d)\n",
                 c.name, r.accepted, (int)r.correction, c.want_accepted,
                 (int)c.want_correction);
    return false;
  }
  return true;
}

// Exercises commit_speculative: given a gate verdict plus the remaining
// output budget, which tokens actually get appended. `target`/`draft` are
// the K sequences (verify_greedy decides the verdict), `bonus` is the
// target's own next token (used only on full acceptance), and
// want_tokens[0..want_commit-1] is the expected committed list in append
// order.
struct CommitCase {
  const char* name;
  int K;
  int target[16];
  int draft[16];
  int bonus;
  int budget;
  int want_commit;
  int want_tokens[16];
};

const CommitCase kCommitCases[] = {
    // Full acceptance, generous budget: all K draft tokens + the bonus.
    {"full accept, budget>need", 3, {10, 20, 30}, {10, 20, 30}, 99, 5, 4,
     {10, 20, 30, 99}},
    // Full acceptance, budget exactly == accepted: no room for the bonus.
    {"full accept, budget==accepted", 3, {10, 20, 30}, {10, 20, 30}, 99, 3, 3,
     {10, 20, 30}},
    // Full acceptance, budget<accepted: only the prefix up to budget.
    {"full accept, budget<accepted", 3, {10, 20, 30}, {10, 20, 30}, 99, 2, 2,
     {10, 20}},
    // Full acceptance, budget=1: first draft token only.
    {"full accept, budget=1", 3, {10, 20, 30}, {10, 20, 30}, 99, 1, 1, {10}},
    // Full acceptance, budget=0: nothing committed.
    {"full accept, budget=0", 3, {10, 20, 30}, {10, 20, 30}, 99, 0, 0, {}},
    // Rejection, generous budget: accepted prefix + correction.
    {"reject, budget>need", 4, {10, 20, 77, 88}, {10, 20, 30, 40}, 99, 10, 3,
     {10, 20, 77}},
    // Rejection, budget==accepted: correction is clamped away.
    {"reject, budget==accepted", 4, {10, 20, 77, 88}, {10, 20, 30, 40}, 99, 2,
     2, {10, 20}},
    // Rejection, budget=1: first prefix token only.
    {"reject, budget=1", 4, {10, 20, 77, 88}, {10, 20, 30, 40}, 99, 1, 1,
     {10}},
    // Rejection at j=0, budget=1: only the correction (prefix empty).
    {"reject j=0, budget=1", 3, {9, 1, 2}, {5, 6, 7}, 99, 1, 1, {9}},
    // Rejection at j=0, budget=0: nothing committed.
    {"reject j=0, budget=0", 3, {9, 1, 2}, {5, 6, 7}, 99, 0, 0, {}},
};

bool run_commit_case(const CommitCase& c) {
  gv::VerifyResult v = gv::verify_greedy(c.target, c.draft, c.K);
  gv::CommitPlan p =
      gv::commit_speculative(v, c.draft, c.K, c.bonus, c.budget);
  if (p.commit != c.want_commit) {
    std::fprintf(stderr, "FAIL(commit) %s: commit=%d (want %d)\n", c.name,
                 p.commit, c.want_commit);
    return false;
  }
  if (static_cast<int>(p.tokens.size()) != c.want_commit) {
    std::fprintf(stderr, "FAIL(commit) %s: size=%zu (want %d)\n", c.name,
                 p.tokens.size(), c.want_commit);
    return false;
  }
  for (int i = 0; i < c.want_commit; ++i) {
    if (p.tokens[i] != c.want_tokens[i]) {
      std::fprintf(stderr, "FAIL(commit) %s: tokens[%d]=%d (want %d)\n",
                   c.name, i, (int)p.tokens[i], c.want_tokens[i]);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;
  for (const auto& c : kCases) ok = run_case(c) && ok;
  for (const auto& c : kCommitCases) ok = run_commit_case(c) && ok;

  // K=0 degenerate case: nothing to verify, no correction.
  {
    int dummy = 0;
    gv::VerifyResult r = gv::verify_greedy(&dummy, &dummy, 0);
    if (r.accepted != 0 || r.correction != -1) {
      std::fprintf(stderr, "FAIL K=0: accepted=%d correction=%d\n", r.accepted,
                   (int)r.correction);
      ok = false;
    }
  }

  if (!ok) return 1;
  std::puts("test_speculative_verify: all tests passed.");
  return 0;
}
