// Unit tests for serve_decide: the session-selection decision made by the
// persistent serve loop. Picks the session sharing the longest prefix with
// the new prompt (so only the rest is prefilled), with an empty-session /
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

// Check a single decision: picked session, resume count, and the advanced
// round-robin pointer.
bool run_case(const char* name, const std::vector<gv::ServeSessionState>& sessions,
              int next, std::vector<std::int32_t> prompt, int want_picked,
              int want_resume, int want_next, int min_resume = 1) {
  const Dec d = gv::serve_decide(sessions, next, prompt, min_resume);
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
    // History longer than the prompt (an edited / regenerated last turn):
    // rewind to the shared prefix, capped at plen - 1 so one token is
    // forwarded.
    ok = run_case("rewind (hist longer)", s, 0, {1, 2}, 0, 1, 0) && ok;
    // Prompt == history: still forward the last token (resume plen - 1).
    ok = run_case("prompt == history", s, 0, {1, 2, 3}, 0, 2, 0) && ok;
    // Mismatch at j=0: miss, full re-prefill.
    ok = run_case("single miss (mismatch)", s, 0, {9, 2, 3}, 0, 0, 0) && ok;
    // Mismatch at j=2: KV rows [0, 2) are still valid, so resume from 2.
    ok = run_case("late mismatch resumes at the divergence", s, 0, {1, 2, 7}, 0, 2, 0) && ok;
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

  // --- Chat-template divergence: history = turn N prompt + generated ids,
  // but turn N+1 re-renders the end of turn N differently (e.g. the
  // generation prompt's `<think></think>` is dropped from the history).
  // The session must resume from the divergence, not re-prefill. ---------
  {
    gv::ServeSessionState a, b;
    for (int i = 0; i < 1000; ++i) a.history.push_back(10 + i % 50);
    a.history.push_back(7);  // "<think></think>" of the generation prompt
    for (int i = 0; i < 40; ++i) a.history.push_back(500 + i);  // generated
    a.watermark = static_cast<int>(a.history.size());
    for (int i = 0; i < 800; ++i) b.history.push_back(900 + i);
    b.watermark = 800;
    std::vector<std::int32_t> p(a.history.begin(), a.history.begin() + 1000);
    for (int i = 0; i < 40; ++i) p.push_back(500 + i);  // reply, no think tag
    for (int i = 0; i < 30; ++i) p.push_back(3);        // next user turn
    std::vector<gv::ServeSessionState> s{b, a};
    ok = run_case("template divergence resumes near the end", s, 0, p, 1, 1000, 0, 64) && ok;
  }

  // --- min_resume: a short overlap (shared system prompt) is a miss and
  // goes to an empty session instead of evicting a live conversation. -----
  {
    gv::ServeSessionState a, empty;
    a.history = {1, 2, 3, 4, 5, 6, 7, 8};
    a.watermark = 8;
    std::vector<gv::ServeSessionState> s{a, empty};
    ok = run_case("short overlap below min_resume -> empty session", s, 0,
                  {1, 2, 3, 9, 9, 9}, 1, 0, 0, /*min_resume=*/4) && ok;
    ok = run_case("overlap at min_resume is a hit", s, 0, {1, 2, 3, 4, 9}, 0, 4, 1,
                  /*min_resume=*/4) && ok;
  }

  // --- Miss with no empty session: the round-robin pointer's session. ----
  {
    gv::ServeSessionState a, b, c;
    a.history = {1};
    a.watermark = 1;
    b.history = {2};
    b.watermark = 1;
    std::vector<gv::ServeSessionState> s{a, b, c};
    ok = run_case("miss prefers empty after pointer", s, 0, {9, 9}, 2, 0, 0) && ok;
    c.history = {3};
    c.watermark = 1;
    s = {a, b, c};
    ok = run_case("miss, none empty -> pointer", s, 1, {9, 9}, 1, 0, 2) && ok;
  }

  if (!ok) {
    std::fprintf(stderr, "serve_resume: FAIL\n");
    return 1;
  }
  std::fprintf(stdout, "serve_resume: all tests passed\n");
  return 0;
}
