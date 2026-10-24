// Phase 3 (Integration): paged KV memory manager + prefix cache tests.
//
// Host-only, no CUDA. The pool is sized with a small synthetic layout so the
// arithmetic is exact and the pool is a handful of pages; a couple of checks
// also run against the real Granite 4.2 8B layout to pin the docs/memory.md
// byte budget.
#include "kv_pool.h"

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

// A small layout the whole suite is built on.
//   bytes_per_token = 2 * layers(2) * kv_heads(2) * head_dim(4) = 32
//   page (4 tokens) = 128 bytes; budget 512 B -> 4 pages, 16-token capacity.
KvPoolConfig small_pool() {
  ModelConfig mc;
  mc.num_layers = 2;
  mc.num_key_value_heads = 2;
  mc.head_dim = 4;
  return KvPoolConfig::from_budget(mc, /*tokens_per_page=*/4, /*budget=*/512);
}

std::vector<std::uint32_t> seq_tokens(std::uint32_t base, std::size_t n) {
  std::vector<std::uint32_t> t(n);
  for (std::size_t i = 0; i < n; ++i) t[i] = base + i;
  return t;
}

}  // namespace

static void test_layout() {
  const KvLayout l = small_pool().layout;
  expect(l.bytes_per_token == 32, "layout bytes_per_token == 32");
  expect(l.token_bytes() == 32, "token_bytes() == bytes_per_token");
  // token-major: [token][layer][k|v][head][dim]. Per-token = 2*2*2*4 = 32.
  // token 0, layer 0, K, head 0, dim 0 -> 0
  expect(l.offset(4, 0, 0, 0, 0, 0) == 0, "offset base");
  // advance one dim -> +1
  expect(l.offset(4, 0, 0, 0, 0, 1) == 1, "offset dim stride");
  // one head -> +head_dim (4)
  expect(l.offset(4, 0, 0, 0, 1, 0) == 4, "offset head stride");
  // K -> V is +num_kv_heads*head_dim (8)
  expect(l.offset(4, 0, 0, 1, 0, 0) == 8, "offset K->V stride");
  // next layer: +2*num_kv_heads*head_dim (16)
  expect(l.offset(4, 0, 1, 0, 0, 0) == 16, "offset layer stride");
  // next token: +bytes_per_token (32)
  expect(l.offset(4, 1, 0, 0, 0, 0) == 32, "offset token stride");
  // a specific interior element
  expect(l.offset(4, 2, 1, 1, 1, 3) == 2 * 32 + 1 * 16 + 8 + 4 + 3,
         "offset interior");
  // out of range -> SIZE_MAX
  expect(l.offset(4, 4, 0, 0, 0, 0) == static_cast<std::size_t>(-1),
         "offset token_local out of range");
  expect(l.offset(4, 0, 2, 0, 0, 0) == static_cast<std::size_t>(-1),
         "offset layer out of range");
  expect(l.offset(4, 0, 0, 2, 0, 0) == static_cast<std::size_t>(-1),
         "offset kv out of range");
}

static void test_pool_sizing() {
  const KvPoolConfig c = small_pool();
  expect(c.page_bytes == 128, "page_bytes == 128");
  expect(c.num_pages == 4, "num_pages == floor(512/128)");
  expect(c.capacity_tokens() == 16, "capacity_tokens == 4*4");
  expect(c.total_bytes() == 512, "total_bytes == num_pages*page_bytes");
  PagedKV pool(c);
  expect(pool.pool_pages() == 4, "pool_pages == 4");
  expect(pool.free_pages() == 4, "all pages free at start");
  expect(pool.used_pages() == 0, "nothing used at start");
  expect(pool.free_pages() + pool.used_pages() == 4, "free+used==pool");
}

// Pin the docs/memory.md budget with the real Granite 4.2 8B layout.
static void test_real_layout_budget() {
  const ModelConfig mc = kModelConfig;
  const KvLayout l = KvLayout::from_config(mc);
  // 2 * 40 layers * 8 KV heads * 128 head dim = 81,920 elements/token.
  expect(l.bytes_per_token == 81920, "real layout bytes_per_token == 81920");
  const std::size_t B = 16;
  const std::size_t budget = 5767168000ull;  // 5.4 GiB (docs/memory.md)
  const KvPoolConfig c = KvPoolConfig::from_budget(mc, B, budget);
  expect(c.page_bytes == 16ull * 81920ull, "real page_bytes");
  expect(c.num_pages == budget / c.page_bytes, "real num_pages == floor");
  expect(c.total_bytes() <= budget, "pool fits the budget");
  expect(c.capacity_tokens() == c.num_pages * B, "real capacity tokens");
}

static void test_alloc_and_locate() {
  PagedKV pool(small_pool());
  const int s = pool.create_sequence();
  expect(s == 0, "first sequence id is 0");
  const auto toks = seq_tokens(100, 10);  // 10 tokens -> 3 pages (4,4,2)
  expect(pool.reserve(s, 10, toks), "reserve 10 tokens");
  expect(pool.sequence_tokens(s) == 10, "sequence_tokens == 10");
  // 10 tokens over B=4 -> pages hold tokens 0-3, 4-7, 8-9 (3 pages used).
  expect(pool.free_pages() == 1, "3 of 4 pages used");
  // locate() maps each token to its physical page; tokens within the same
  // page share a page id, and the within-page offset is the token's
  // page-local position. (Page ids come from the free list, so assert the
  // structure - same-page grouping and local offset - not exact ids.)
  for (std::size_t t = 0; t < 10; ++t) {
    const auto [page, local] = pool.locate(s, t);
    expect(local == t % 4, "locate token-local");
    if (t % 4 != 0) {
      const auto first = pool.locate(s, t - (t % 4));
      expect(page == first[0], "same page as the page's first token");
    }
    if (t + 1 < 10 && (t + 1) % 4 != 0) {
      const auto nxt = pool.locate(s, t + 1);
      expect(page == nxt[0], "same page as the next token");
    }
  }
  expect(pool.locate(s, 10) == (std::array<std::size_t, 2>{
                                   static_cast<std::size_t>(-1),
                                   static_cast<std::size_t>(-1)}),
         "locate out-of-range token");
  // capacity: a 17th token needs a 4th page that is free (still fits).
  auto toks11 = seq_tokens(100, 11);
  expect(pool.reserve(s, 11, toks11), "grow to 11 tokens (uses last page)");
  expect(pool.free_pages() == 0, "all 4 pages now used");
  auto toks17 = seq_tokens(100, 17);  // would need a 5th page
  expect(!pool.reserve(s, 17, toks17), "reserve beyond capacity fails");
  expect(pool.sequence_tokens(s) == 11, "sequence size unchanged on failure");
  pool.release(s);
}

static void test_prefix_reuse() {
  PagedKV pool(small_pool());
  const auto shared = seq_tokens(500, 16);  // 4 full pages
  const int a = pool.create_sequence();
  expect(pool.reserve(a, 16, shared), "A reserve 16");
  for (std::size_t p = 0; p < 4; ++p) expect(pool.mark_written(a, p), "A cache page");
  expect(pool.cached_pages() == 4, "4 pages cached");
  expect(pool.free_pages() == 0, "all pages used by A");

  const int b = pool.create_sequence();
  const std::size_t matched = pool.match_prefix(b, shared);
  expect(matched == 16, "B reuses the full 16-token prefix");
  expect(pool.sequence_tokens(b) == 16, "B filled == 16");
  expect(pool.free_pages() == 0, "reuse allocates nothing new");
  expect(pool.shared_pages() == 4, "all 4 pages shared (ref 2)");
  // locate() resolves B's tokens into A's pages (ids allocated from the
  // free-list back: A's pages are 3,2,1,0).
  {
    const auto [page, local] = pool.locate(b, 5);
    expect(page == 2 && local == 1, "B locate maps into A's 2nd page (id 2)");
  }
  // A different prefix must NOT match.
  const int c = pool.create_sequence();
  const auto other = seq_tokens(999, 16);
  expect(pool.match_prefix(c, other) == 0, "unrelated prefix does not match");
  expect(pool.sequence_tokens(c) == 0, "C filled == 0");
  pool.release(a);
  pool.release(b);
  pool.release(c);
  expect(pool.free_pages() == 0, "retained pages stay allocated after release");
  expect(pool.retained_pages() == 4, "4 pages retained in cache");
}

static void test_lru_eviction() {
  PagedKV pool(small_pool());
  const auto prompt = seq_tokens(700, 16);  // fills all 4 pages
  const int a = pool.create_sequence();
  pool.reserve(a, 16, prompt);
  for (std::size_t p = 0; p < 4; ++p) pool.mark_written(a, p);
  pool.release(a);  // 4 pages retained (ref 0), LRU-ordered by mark order
  expect(pool.retained_pages() == 4, "4 retained");

  const int b = pool.create_sequence();
  const auto other = seq_tokens(900, 4);  // needs 1 page -> evict 1 LRU
  expect(pool.reserve(b, 4, other), "B reserve evicts a retained page");
  expect(pool.evicted_pages() == 1, "exactly one eviction");
  expect(pool.free_pages() == 0, "evicted page is reused for B");
  expect(pool.used_pages() == 4, "pool still full");
  expect(pool.retained_pages() == 3, "3 pages still retained");
  expect(pool.cached_pages() == 3, "3 cache entries remain");
  expect(pool.locate(b, 3) == (std::array<std::size_t, 2>{
                                   3, static_cast<std::size_t>(3)}),
         "B token 3 lands in the evicted page 3");
  pool.release(b);
}

// A page a live sequence references must never be evicted; when the only
// cached pages are referenced, the allocation fails instead of clobbering.
static void test_eviction_respects_live_refs() {
  // 2-page pool: B=4, budget 256 B.
  ModelConfig mc;
  mc.num_layers = 2;
  mc.num_key_value_heads = 2;
  mc.head_dim = 4;
  PagedKV pool(KvPoolConfig::from_budget(mc, 4, 256));
  expect(pool.pool_pages() == 2, "2-page pool");

  const int a = pool.create_sequence();
  const auto a_toks = seq_tokens(10, 4);
  pool.reserve(a, 4, a_toks);
  pool.mark_written(a, 0);

  const int b = pool.create_sequence();
  const auto b_toks = seq_tokens(50, 4);
  pool.reserve(b, 4, b_toks);  // fills the 2nd page
  expect(pool.free_pages() == 0, "both pages used by live sequences");

  const int c = pool.create_sequence();
  const auto c_toks = seq_tokens(90, 4);
  expect(!pool.reserve(c, 4, c_toks), "cannot evict live-referenced pages");
  expect(pool.free_pages() == 0, "no page was evicted");
  expect(pool.ref_count(pool.locate(a, 0)[0]) == 1, "A's page still ref 1");
  expect(pool.ref_count(pool.locate(b, 0)[0]) == 1, "B's page still ref 1");
  expect(pool.sequence_tokens(c) == 0, "C allocated nothing");
}

static void test_block_table() {
  PagedKV pool(small_pool());
  const int s = pool.create_sequence();
  pool.reserve(s, 8, seq_tokens(100, 8));  // 2 pages
  pool.mark_written(s, 0);
  pool.mark_written(s, 1);
  const auto& bt = pool.block_table(s);
  expect(bt.size() == 2, "block_table has 2 pages");
  expect(pool.locate(s, 0)[0] == bt[0], "token 0 on first page from table");
  expect(pool.locate(s, 4)[0] == bt[1], "token 4 on second page from table");
  expect(pool.locate(s, 7)[1] == 3, "token 7 offset = 3 on its page");
  // Released sequence has empty block_table.
  pool.release(s);
  const auto& empty = pool.block_table(s);
  expect(empty.empty(), "released sequence block_table empty");
  // Out-of-range sequence has empty block_table.
  expect(pool.block_table(-1).empty(), "negative seq block_table empty");
  expect(pool.block_table(999).empty(), "invalid seq block_table empty");
}

static void test_release_frees() {
  PagedKV pool(small_pool());
  const int a = pool.create_sequence();
  pool.reserve(a, 8, seq_tokens(0, 8));
  pool.mark_written(a, 0);
  pool.mark_written(a, 1);
  expect(pool.cached_pages() == 2, "2 cached before release");
  expect(pool.free_pages() == 2, "2 pages free while A is live");
  pool.release(a);
  // A's 2 full pages were cached -> retained on release, not freed.
  expect(pool.retained_pages() == 2, "2 retained");
  expect(pool.free_pages() == 2, "release frees nothing for cached pages");

  // A sequence that never caches a page frees fully on release.
  const int b = pool.create_sequence();
  pool.reserve(b, 4, seq_tokens(1, 4));
  expect(pool.free_pages() == 1, "B uses the last free page");
  pool.release(b);
  expect(pool.free_pages() == 2, "uncached page is freed on release");
  expect(pool.retained_pages() == 2, "retained count unaffected");
}

static void test_clear_and_bounds() {
  PagedKV pool(small_pool());
  const int a = pool.create_sequence();
  pool.reserve(a, 4, seq_tokens(1, 4));
  pool.mark_written(a, 0);
  pool.clear();
  expect(pool.free_pages() == 4, "clear frees all pages");
  expect(pool.cached_pages() == 0, "clear empties the cache");
  expect(pool.live_sequences() == 0, "clear drops sequences");
  expect(pool.evicted_pages() == 0, "clear resets eviction counter");

  // Live-sequence bound: a 4-page pool allows at most 4 live sequences.
  const int s0 = pool.create_sequence();
  const int s1 = pool.create_sequence();
  const int s2 = pool.create_sequence();
  const int s3 = pool.create_sequence();
  const int s4 = pool.create_sequence();
  expect(s0 >= 0 && s1 >= 0 && s2 >= 0 && s3 >= 0, "4 live sequences OK");
  expect(s4 == -1, "5th live sequence refused on a 4-page pool");
  pool.release(s0);
  const int s5 = pool.create_sequence();
  expect(s5 >= 0, "a slot frees after release");
}

int main() {
  test_layout();
  test_pool_sizing();
  test_real_layout_budget();
  test_alloc_and_locate();
  test_prefix_reuse();
  test_lru_eviction();
  test_eviction_respects_live_refs();
  test_release_frees();
  test_clear_and_bounds();
  test_block_table();
  if (g_failures) {
    std::printf("kv_pool: %d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("kv_pool: all checks passed\n");
  return 0;
}
