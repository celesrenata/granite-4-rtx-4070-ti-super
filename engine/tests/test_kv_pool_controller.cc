// Phase 4 (Integration): host KV pool controller tests.
//
// Host-only, no CUDA. Exercises the session-slot lifecycle on top of the
// Phase-3 pool: prefill (prefix reuse), commit (incremental page marking), the
// device row table, and the two reuse wins (concurrent + sequential). Built on
// the same small pool as test_kv_pool.cc (4 tokens/page, 4 pages, 16-token
// capacity) so the arithmetic is exact.
//
// Note on the row table: page ids come from the free list (popped from the
// back), so even a single session's pages are *not* laid out in logical order.
// The invariant is a valid, stable permutation of arena rows - not the
// identity. The pre-pool contiguous kernels are the identity only in the
// single-session, page-0..n case, which this suite does not rely on.
#include "kv_pool_controller.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace granite4;

namespace {

int g_failures = 0;
void expect(bool cond, const char* what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

// 4 tokens/page, 4 pages -> 16-token capacity.
KvPoolConfig small_pool() {
  ModelConfig mc;
  mc.num_layers = 2;
  mc.num_key_value_heads = 2;
  mc.head_dim = 4;
  return KvPoolConfig::from_budget(mc, /*tokens_per_page=*/4, /*budget=*/512);
}

// Prompt of `n` tokens starting at `base` (contiguous, so page-aligned prefixes
// are distinct per base).
PagedKV::Tokens prompt(std::uint32_t base, std::size_t n) {
  PagedKV::Tokens t(n);
  for (std::size_t i = 0; i < n; ++i) t[i] = base + i;
  return t;
}

// Verify `rows` is a valid device row table for `n` logical positions over a
// pool of `pages` pages of `B` rows each: every row in range, rows within a
// page are a contiguous increasing run (page*B .. page*B+B-1), and no two
// positions share a row. This is the invariant the paged kernels need.
bool rows_valid(const int* rows, std::size_t n, std::size_t pages,
                std::size_t B) {
  const std::size_t cap = pages * B;
  std::vector<bool> seen(cap, false);
  for (std::size_t p = 0; p < n; ++p) {
    const int r = rows[p];
    if (r < 0 || static_cast<std::size_t>(r) >= cap) return false;
    if (seen[r]) return false;  // duplicate row
    seen[r] = true;
    // Row's page-local index must equal the position's page-local index
    // (token order within a page is preserved by the pool).
    const std::size_t row_local = static_cast<std::size_t>(r) % B;
    if (row_local != p % B) return false;
  }
  return true;
}

}  // namespace

static void test_slot_capacity() {
  KvPoolController c(small_pool());
  expect(c.max_slots() == 4, "max_slots == num_pages");
  expect(c.new_slot() == 0, "slot 0");
  expect(c.new_slot() == 1, "slot 1");
  expect(c.new_slot() == 2, "slot 2");
  expect(c.new_slot() == 3, "slot 3");
  expect(c.new_slot() == -1, "5th slot rejected at capacity (4 pages)");
  c.release_slot(0);
  expect(c.new_slot() == 0, "freed slot id is recycled");
}

static void test_prefill_fresh_and_rows() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  auto p = c.prefill(a, prompt(100, 10));  // 10 tokens -> 3 pages
  expect(p.ok, "prefill ok");
  expect(p.reused == 0, "no reuse on first prefill");
  expect(c.committed(a) == 10, "committed 10 tokens");
  expect(c.free_pages() == 1, "3 of 4 pages used");
  int rows[10] = {0};
  expect(c.row_table(a, 10, rows), "row_table ok");
  expect(rows_valid(rows, 10, c.pool_pages(), c.tokens_per_page()),
         "row table is a valid permutation");
}

static void test_concurrent_prefix_reuse() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  const int b = c.new_slot();
  const PagedKV::Tokens shared = prompt(100, 16);  // 4 full pages
  auto pa = c.prefill(a, shared);
  expect(pa.ok && pa.reused == 0, "A: fresh prefill");
  auto pb = c.prefill(b, shared);
  expect(pb.ok, "B prefill ok");
  expect(pb.reused == 16, "B reuses all 4 cached pages");
  expect(c.committed(b) == 16, "B committed 16 (no new pages)");
  expect(c.shared_pages() == 4, "4 pages shared (ref 2)");
  expect(c.used_pages() == 4, "still only 4 pages total (no duplicates)");
  expect(c.free_pages() == 0, "pool full, no extra allocation");
}

static void test_sequential_prefix_reuse() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  auto pa = c.prefill(a, prompt(100, 16));
  expect(pa.ok, "A prefill ok");
  c.release_slot(a);
  // A's 4 pages are now ref-0 but retained in the cache.
  expect(c.retained_pages() == 4, "4 pages retained after release");
  const int b = c.new_slot();
  auto pb = c.prefill(b, prompt(100, 16));
  expect(pb.ok, "B prefill ok");
  expect(pb.reused == 16, "B reuses A's retained pages (sequential win)");
  expect(c.retained_pages() == 0, "retention consumed by B's refs");
}

static void test_partial_prefix_reuse() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  const int b = c.new_slot();
  // A fills 3 pages (12 tokens) so one page stays free for B's tail.
  c.prefill(a, prompt(100, 12));
  // B shares the first 12 tokens (3 pages) then diverges.
  PagedKV::Tokens tb = prompt(100, 12);
  tb.push_back(999);  // divergence
  tb.push_back(998);
  auto pb = c.prefill(b, tb);
  expect(pb.ok, "B prefill ok");
  expect(pb.reused == 12, "B reuses the 3 shared pages only");
  expect(c.committed(b) == 14, "B committed 14 (12 reused + 2 tail)");
  expect(c.free_pages() == 0, "B's tail used the last free page");
}

static void test_commit_extends_stably() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  c.prefill(a, prompt(100, 6));  // 2 pages, page 1 partial (2 tokens)
  expect(c.committed(a) == 6, "committed 6 after prefill");
  expect(c.cached_pages() == 1, "only the full first page cached");
  int rows_before[6] = {0};
  expect(c.row_table(a, 6, rows_before), "row_table(6) ok");
  // Generate 2 more tokens -> page 1 becomes full.
  PagedKV::Tokens all = prompt(100, 6);
  all.push_back(900);
  all.push_back(901);
  c.commit(a, all);
  expect(c.committed(a) == 8, "committed 8 after commit");
  expect(c.cached_pages() == 2, "second page now full + cached");
  int rows_after[8] = {0};
  expect(c.row_table(a, 8, rows_after), "row_table(8) ok");
  expect(rows_valid(rows_after, 8, c.pool_pages(), c.tokens_per_page()),
         "extended row table is valid");
  // Stability: extending must not reorder already-committed rows.
  bool stable = true;
  for (int i = 0; i < 6; ++i) stable = (rows_before[i] == rows_after[i]);
  expect(stable, "row table is stable under growth");
}

static void test_commit_noop_within_page() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  c.prefill(a, prompt(100, 4));  // exactly 1 full page
  expect(c.committed(a) == 4, "committed 4");
  // Commit the same 4 tokens: nothing new.
  c.commit(a, prompt(100, 4));
  expect(c.committed(a) == 4, "commit no-op keeps 4");
  expect(c.free_pages() == 3, "no page added by no-op commit");
}

static void test_prefill_over_capacity() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  auto p = c.prefill(a, prompt(100, 20));  // 20 > 16 capacity
  expect(!p.ok, "over-capacity prefill fails");
  expect(c.committed(a) == 0, "nothing committed on failure");
  expect(c.free_pages() == 4, "no pages allocated on failure");
}

static void test_eviction_under_pressure() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  c.prefill(a, prompt(100, 16));  // fills all 4 pages (cached, A live)
  c.release_slot(a);              // 4 pages retained (ref 0, cached)
  const int b = c.new_slot();
  auto pb = c.prefill(b, prompt(500, 16));  // different prefix, needs all 4
  expect(pb.ok, "B prefill ok after eviction");
  expect(pb.reused == 0, "B does not share A's different prefix");
  expect(c.evicted_pages() == 4, "4 pages evicted to make room");
  expect(c.used_pages() == 4, "pool fully used by B");
}

// The device contract the paged serve path relies on: ONE shared arena of
// pool_pages()*B rows serves every live slot simultaneously. The pool's
// per-slot row tables must therefore map each logical position to a global
// arena row such that no arena row is ever claimed by two *different* logical
// positions (that is a foreign-KV read). Slots that share a prefix must agree
// on the same physical rows (their pages are shared), so a shared prefix
// dedups in the arena instead of being stored twice.
static void test_shared_arena_device_contract() {
  KvPoolController c(small_pool());
  const std::size_t B = c.tokens_per_page();
  const std::size_t arena = c.pool_pages() * B;
  const int a = c.new_slot();
  const int b = c.new_slot();
  c.prefill(a, prompt(100, 12));  // 3 full pages, fresh
  PagedKV::Tokens tb = prompt(100, 12);
  tb.push_back(999);
  tb.push_back(998);
  auto pb = c.prefill(b, tb);  // shares the 12-token prefix, adds a 2-token tail
  expect(pb.ok, "B prefill ok");

  const std::size_t nA = c.committed(a);  // 12
  const std::size_t nB = c.committed(b);  // 14
  std::vector<int> ra(nA), rb(nB);
  expect(c.row_table(a, nA, ra.data()), "A row table ok");
  expect(c.row_table(b, nB, rb.data()), "B row table ok");

  // (a) every row lands inside the shared arena.
  for (int r : ra) expect(r >= 0 && static_cast<std::size_t>(r) < arena,
                          "A row in range");
  for (int r : rb) expect(r >= 0 && static_cast<std::size_t>(r) < arena,
                          "B row in range");

  // (b) the shared prefix maps to identical global rows in both slots (they
  // physically share the same pages).
  for (std::size_t i = 0; i < nA; ++i)
    expect(ra[i] == rb[i], "shared prefix maps to identical arena rows");

  // (c) no arena row is claimed by two different logical positions - the
  // foreign-KV-read invariant. owner_pos[r] records the logical position first
  // seen at row r; a second claimant must be the same position.
  std::vector<int> owner_pos(arena, -1);
  bool no_foreign = true;
  auto claim = [&](int r, int pos) {
    if (owner_pos[r] >= 0 && owner_pos[r] != pos) no_foreign = false;
    owner_pos[r] = pos;
  };
  for (std::size_t i = 0; i < nA; ++i) claim(ra[i], static_cast<int>(i));
  for (std::size_t i = 0; i < nB; ++i) claim(rb[i], static_cast<int>(i));
  expect(no_foreign, "no arena row is claimed by two logical positions");

  // (d) the union of A's and B's rows is exactly B's table: A's prefix is a
  // prefix of B's, so shared pages are stored once (dedup) and B's tail adds
  // only its own rows.
  std::vector<bool> seen(arena, false);
  std::size_t distinct = 0;
  for (int r : ra)
    if (!seen[r]) {
      seen[r] = true;
      ++distinct;
    }
  for (int r : rb)
    if (!seen[r]) {
      seen[r] = true;
      ++distinct;
    }
  expect(distinct == nB, "union of slots == B's rows (shared prefix deduped)");
  expect(distinct < nA + nB, "sharing actually reduces arena traffic");
}

static void test_release_slot_retains_cache() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  c.prefill(a, prompt(100, 8));  // 2 full pages -> both cached
  c.release_slot(a);
  expect(c.committed(a) == 0, "slot empty after release");
  expect(c.retained_pages() == 2, "full pages retained for reuse");
  // Reuse the same slot id for the SAME prompt: the pages come straight back.
  auto p = c.prefill(a, prompt(100, 8));
  expect(p.ok, "re-prefill on same slot ok");
  expect(p.reused == 8, "re-prefill reuses its own retained pages");
  expect(c.committed(a) == 8, "re-prefill committed 8");
}

static void test_shutdown() {
  KvPoolController c(small_pool());
  const int a = c.new_slot();
  c.prefill(a, prompt(100, 8));
  c.shutdown();
  expect(c.pool_pages() == 4, "pool intact after shutdown");
  expect(c.free_pages() == 4, "all pages free after shutdown");
  expect(c.committed(a) == 0, "slot reports 0 after shutdown");
  expect(c.cached_pages() == 0, "cache emptied");
}

// Resident-session lifecycle: two sessions grow inside one pool, never share a
// row, keep their rows stable as they grow, and give pages back when trimmed.
static void test_resident_two_sessions_one_pool() {
  KvPoolController c(small_pool());  // 4 pages x 4 rows
  const int a = c.open_slot();
  const int b = c.open_slot();
  expect(a >= 0 && b >= 0 && a != b, "two resident slots");
  expect(c.ensure(a, 8), "A grows to 2 pages");
  expect(c.ensure(b, 8), "B grows to 2 pages (pool now full)");
  expect(c.free_pages() == 0, "pool full");
  int ra[8], rb[8];
  expect(c.row_table(a, 8, ra) && c.row_table(b, 8, rb), "row tables");
  std::vector<bool> seen(16, false);
  bool disjoint = true;
  for (int i = 0; i < 8; ++i) {
    if (seen[ra[i]]) disjoint = false;
    seen[ra[i]] = true;
  }
  for (int i = 0; i < 8; ++i) {
    if (seen[rb[i]]) disjoint = false;
    seen[rb[i]] = true;
  }
  expect(disjoint, "A and B never share an arena row");
  // The pool is exhausted: neither can grow, and a failed grow changes nothing.
  expect(!c.ensure(a, 12), "A cannot grow past the pool");
  expect(c.committed(a) == 8, "failed grow leaves A untouched");
  int ra2[8];
  expect(c.row_table(a, 8, ra2), "A row table after failed grow");
  bool same = true;
  for (int i = 0; i < 8; ++i) same = same && ra[i] == ra2[i];
  expect(same, "A rows unchanged by a failed grow");
  // B hands back its second page; A can then grow, and A's old rows hold still.
  c.trim(b, 4);
  expect(c.free_pages() == 1, "trim returned B's tail page");
  expect(c.ensure(a, 12), "A grows into the returned page");
  int ra3[12];
  expect(c.row_table(a, 12, ra3), "A 12-row table");
  same = true;
  for (int i = 0; i < 8; ++i) same = same && ra[i] == ra3[i];
  expect(same, "growing A never moves its committed rows");
  expect(rows_valid(ra3, 12, c.pool_pages(), c.tokens_per_page()),
         "A table is a valid arena table");
  int rb2[4];
  expect(c.row_table(b, 4, rb2), "B keeps its first page");
  same = true;
  for (int i = 0; i < 4; ++i) same = same && rb[i] == rb2[i];
  expect(same, "trimming B never moves its kept rows");
}
static void test_resident_drop_and_missing() {
  KvPoolController c(small_pool());
  const int a = c.open_slot();
  expect(c.pages_missing(a, 10) == 3, "10 tokens need 3 pages");
  expect(c.ensure(a, 10), "grow to 10");
  expect(c.slot_pages(a) == 3, "3 pages held");
  expect(c.pages_missing(a, 12) == 0, "12 tokens fit the 3 pages");
  expect(c.pages_missing(a, 13) == 1, "13 tokens need one more");
  c.trim(a, 5);
  expect(c.slot_pages(a) == 2, "trim to 5 tokens keeps 2 pages");
  expect(c.committed(a) == 5, "trim lowers the filled count");
  expect(c.ensure(a, 9), "regrow after trim");
  c.drop(a);
  expect(c.slot_pages(a) == 0, "drop returns every page");
  expect(c.free_pages() == 4, "pool whole after drop");
  expect(c.cached_pages() == 0, "resident pages are never cached");
}
int main() {
  test_resident_two_sessions_one_pool();
  test_resident_drop_and_missing();
  test_slot_capacity();
  test_prefill_fresh_and_rows();
  test_concurrent_prefix_reuse();
  test_sequential_prefix_reuse();
  test_partial_prefix_reuse();
  test_commit_extends_stably();
  test_commit_noop_within_page();
  test_prefill_over_capacity();
  test_eviction_under_pressure();
  test_release_slot_retains_cache();
  test_shutdown();
  if (g_failures == 0) std::printf("kv_pool_controller: all tests passed\n");
  else std::printf("kv_pool_controller: %d FAILED\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
