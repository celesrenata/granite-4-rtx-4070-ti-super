// Unit tests for serve_decide: the session-selection decision made by the
// persistent serve loop. Picks the session whose KV prefix is the longest
// full prefix of the new prompt (so only the tail is prefilled), with a
// round-robin fallback on a miss. Pure host logic, no CUDA.
//
// The serve loop (CUDA-only, in infer_main.cc) delegates to serve_decide
// and is the only caller, so the invariants that are hardest to get right
// on a GPU box — longest-prefix match, stale-watermark skip, round-robin
// fallback, pointer advance — are exercised here on any platform.

#include "serve_resume.h"

#include <cstdio>
#include <string>
#include <vector>

namespace gv = granite4;

namespace {

using Dec = gv::ServeDecision;

Dec decide(const std::vector<gv::ServeSessionState>& sessions, int next,
           std::vector<std::int32_t> prompt) {
  return gv::serve_decide(sessions, next, prompt);
}

// Check a single decision: picked session, resume count, and the advanced
// round-robin pointer.
bool run_case(const char* name, const std::vector<gv::ServeSessionState>& sessions,
              int next, std::vector<std::int32_t> prompt, int want_picked,
              int want_resume, int want_next) {
  const Dec d = decide(sessions, next, std::move(prompt));
  if (d.picked != want_picked || d.resume != want_resume || d.next != want_next) {
    std::fprintf(stderr, "FAIL(%s): picked=%d resume=%d next=%d; want %d %d %d\n",
                 name, d.picked, d.resume, d.next, want_picked, want_resume, want_next);
    return false;
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;

  // --- Empty pool / no history: round-robin fallback, no resume ---------
  ok = run_case("empty pool", {}, 0, {1, 2, 3}, /*picked*/ 0, 0, 0) && ok;

  {
    std::vector<gv::ServeSessionState> s(2);
    ok = run_case("no history -> rrobin", s, 0, {1, 2, 3}, 0, 0, 1) && ok;
    ok = run_case("no history -> rrobin (next=1)", s, 1, {1, 2, 3}, 1, 0, 0) && ok;
  }

  // --- Single session, prefix match -------------------------------------
  {
    gv::ServeSessionState a;
    a.history = {1, 2, 3};
    a.watermark = 3;
    std::vector<gv::ServeSessionState> s{a};
    // Exact prefix: resume from 3, only the tail is prefilled.
    ok = run_case("single hit", s, 0, {1, 2, 3, 4, 5}, 0, 3, 0) && ok;
    // History shorter than the prompt (prompt is a superset of the prefix):
    // still a hit, resume from the full history.
    ok = run_case("single hit short hist", s, 0, {1, 2, 3, 4}, 0, 3, 0) && ok;
    // History longer than the prompt: cannot resume (would need to rewind).
    ok = run_case("single miss (hist longer)", s, 0, {1, 2}, 0, 0, 0) && ok;
    // Mismatch at j=0: miss, full re-prefill.
    ok = run_case("single miss (mismatch)", s, 0, {9, 2, 3}, 0, 0, 0) && ok;
    // Mismatch at j=2 (prefix [1,2] matches but [1,2,3] doesn't): miss,
    // because the session's cached KV is all of [1,2,3], not [1,2].
    ok = run_case("single miss (mismatch j=2)", s, 0, {1, 2, 7}, 0, 0, 0) && ok;
  }

  // --- Two sessions, longest-prefix wins --------------------------------
  {
    gv::ServeSessionState a, b;
    a.history = {1, 2};
    a.watermark = 2;
    b.history = {1, 2, 3, 4};
    b.watermark = 4;
    std::vector<gv::ServeSessionState> s{a, b};
    // Both match; the longer one (b, 4) wins.
    ok = run_case("two hit, longest wins", s, 0, {1, 2, 3, 4, 5}, 1, 4, 0) && ok;
    // Prompt shorter than b but longer than a: a matches (2), b does not
    // (hist longer than prompt). a wins.
    ok = run_case("two hit, shorter hist wins", s, 0, {1, 2, 7}, 0, 2, 1) && ok;
    // Both miss: round-robin pointer picks.
    ok = run_case("two miss -> rrobin(0)", s, 0, {9, 9, 9, 9, 9}, 0, 0, 1) && ok;
    ok = run_case("two miss -> rrobin(1)", s, 1, {9, 9, 9, 9, 9}, 1, 0, 0) && ok;
  }

  // --- Stale watermark: session is a miss even if the prefix matches ----
  {
    gv::ServeSessionState a;
    a.history = {1, 2, 3};
    a.watermark = 5;  // watermark != history.size() -> stale (last turn failed)
    std::vector<gv::ServeSessionState> s{a};
    // Even though history is a prefix of the prompt, the KV is not what
    // the history claims, so the session is a miss.
    ok = run_case("stale watermark -> miss", s, 0, {1, 2, 3, 4}, 0, 0, 0) && ok;
  }

  // --- Stale watermark with a clean session available -------------------
  {
    gv::ServeSessionState stale, clean;
    stale.history = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    stale.watermark = 99;  // stale
    clean.history = {1, 2};
    clean.watermark = 2;
    std::vector<gv::ServeSessionState> s{stale, clean};
    // Stale session skipped; clean session (shorter prefix) wins.
    ok = run_case("stale skipped, clean wins", s, 0, {1, 2, 3, 4, 5}, 1, 2, 0) && ok;
  }

  // --- Round-robin pointer advances past the hit ------------------------
  {
    gv::ServeSessionState a, b;
    a.history = {1, 2};
    a.watermark = 2;
    b.history = {3, 4};
    b.watermark = 2;
    std::vector<gv::ServeSessionState> s{a, b};
    // Hit on a (prefix [1,2]); pointer advances to 1 so the next miss
    // lands on b, not on a again.
    ok = run_case("hit advances rr pointer", s, 0, {1, 2, 9}, 0, 2, 1) && ok;
    // Hit on b (prefix [3,4]); pointer wraps past 1 to 0.
    ok = run_case("hit on b wraps rr pointer", s, 1, {3, 4, 9}, 1, 2, 0) && ok;
    // Miss on both: rrobin(1) is b; pointer wraps to 0.
    ok = run_case("miss wraps rr pointer", s, 1, {9, 9, 9}, 1, 0, 0) && ok;
  }

  // --- Ties break by session index (lower wins) -------------------------
  {
    gv::ServeSessionState a, b;
    a.history = {1, 2};
    a.watermark = 2;
    b.history = {1, 2};
    b.watermark = 2;
    std::vector<gv::ServeSessionState> s{a, b};
    // Equal-length matching prefixes: lower index (a) wins.
    ok = run_case("tie -> lower index", s, 0, {1, 2, 7}, 0, 2, 1) && ok;
  }

  // --- Watermark == 0 with empty history is a no-op miss ---------------
  {
    gv::ServeSessionState a, b;
    a.history = {};
    a.watermark = 0;
    b.history = {};
    b.watermark = 0;
    std::vector<gv::ServeSessionState> s{a, b};
    ok = run_case("all empty -> rrobin", s, 1, {1, 2}, 1, 0, 0) && ok;
  }

  if (!ok) {
    std::fprintf(stderr, "serve_resume: FAIL\n");
    return 1;
  }
  std::fprintf(stdout, "serve_resume: all tests passed\n");
  return 0;
}
