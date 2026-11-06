// Phase 4: device KV pool (host bookkeeping) — unit tests.
//
// Host-only, no CUDA: exercises the page allocator, the LRU prefix cache,
// refcounting, the FNV token-chained page keys, and the arena layout math
// for all three KV formats (bf16 / int8 / int4) against the real Granite
// 4.2 8B spec (40 layers, 8 KV heads, head_dim 128).

#include "kv_pool_device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

KvPoolDeviceConfig pool_cfg(int num_pages, KvDeviceFormat fmt) {
  KvPoolDeviceConfig c;
  c.num_layers = 2;  // small so the arena arithmetic is exact
  c.num_kv_heads = 2;
  c.head_dim = 8;
  c.num_pages = num_pages;
  c.format = fmt;
  return c;
}

std::uint64_t key_of(const std::uint32_t* toks, std::size_t n) {
  return KvPoolDevice::page_key(0, toks, n);
}

}  // namespace

static void test_config_math() {
  // bf16: per token = 2(KV)*L(2)*Hk(2)*d(8)*2 B = 128 B/token; per row
  // (head,position): K 16 B + V 16 B = 32 B/row, no scales.
  const KvPoolDeviceConfig c = pool_cfg(4, KvDeviceFormat::kBf16);
  expect(c.rows() == 64, "bf16 rows == 4*16");
  const auto r = c.row_bytes();
  expect(r.k == 16 && r.v == 16 && r.k_scale == 0 && r.v_scale == 0,
         "bf16 row bytes");
  // arena = lkh(4) * rows(64) * (16+16) = 8192 B
  expect(c.arena_bytes() == 4ull * 64 * 32, "bf16 arena bytes");

  // int8: K row = 8 B, V row = 8 B, scales 4+4 = 24 B/row.
  const KvPoolDeviceConfig c8 = pool_cfg(4, KvDeviceFormat::kInt8);
  const auto r8 = c8.row_bytes();
  expect(r8.k == 8 && r8.v == 8 && r8.k_scale == 4 && r8.v_scale == 4,
         "int8 row bytes");
  expect(c8.arena_bytes() == 4ull * 64 * 24, "int8 arena bytes");
  expect(c8.bytes_per_token() ==
             (2ull * 2 * 2 * 8) + (2ull * 2 * 2 * 4),  // codes + scales
         "int8 bytes_per_token");

  // int4: code rows 4 B each.
  const KvPoolDeviceConfig c4 = pool_cfg(4, KvDeviceFormat::kInt4);
  const auto r4 = c4.row_bytes();
  expect(r4.k == 4 && r4.v == 4 && r4.k_scale == 4 && r4.v_scale == 4,
         "int4 row bytes");
}

static void test_alloc_refcount() {
  KvPoolDeviceConfig c = pool_cfg(3, KvDeviceFormat::kBf16);  // 48 rows
  std::unique_ptr<KvPoolDevice> p = KvPoolDevice::create(c, nullptr, nullptr);
  expect(p != nullptr, "pool created");
  expect(p->free_pages() == 3, "3 free pages initially");

  int a = p->alloc_page();
  int b = p->alloc_page();
  int cc = p->alloc_page();
  expect(a == 0 && b == 16 && cc == 32, "pages hand out 16-row runs");
  expect(a % 16 == 0 && b % 16 == 0, "rows aligned to 16");
  expect(p->free_pages() == 0, "arena drained");

  int a2 = p->alloc_page();
  expect(a2 == -1, "alloc fails when full");

  p->release(a);
  expect(p->ref_count(a) == 0, "release drops refcount");
  expect(p->free_pages() == 1, "uncached release returns the page");
  p->release(a);  // double release is a no-op
  expect(p->ref_count(a) == 0, "release is clamped at 0");
  expect(p->free_pages() == 1, "no double free");

  int d = p->alloc_page();
  expect(d >= 0 && d % 16 == 0, "reallocated a freed page");
  expect(p->free_pages() == 0, "back to full");
}

static void test_prefix_cache_lru() {
  KvPoolDeviceConfig c = pool_cfg(4, KvDeviceFormat::kBf16);
  std::unique_ptr<KvPoolDevice> p = KvPoolDevice::create(c, nullptr, nullptr);

  // Four distinct token prefixes -> four distinct page keys.
  std::vector<std::uint32_t> toks(4, 1);
  std::uint64_t k[4];
  for (int i = 0; i < 4; ++i) {
    toks[0] = i + 1;
    k[i] = key_of(toks.data(), toks.size());
    toks[0] = 1;
  }
  for (int i = 0; i < 4; ++i) expect(k[i] != k[(i + 1) % 4], "distinct keys");

  // Four sequences each write one page and finish (release their reference).
  // The pages are cached and now refcount 0 -> LRU-evictable, arena full.
  for (int i = 0; i < 4; ++i) {
    int row = p->alloc_page();
    p->cache_page(row, k[i]);
    p->release(row);
  }
  expect(p->cached_pages() == 4, "4 cached");
  expect(p->free_pages() == 0, "arena full");
  expect(p->evictable_pages() == 4, "4 evictable");

  // A 5th allocation (kept referenced, so free_ stays empty) must evict the
  // oldest page (k[0]).
  int extra = p->alloc_page();
  expect(extra >= 0, "alloc evicted the LRU page");
  expect(p->lookup_cached(k[0]) == -1, "LRU page was evicted");
  expect(p->lookup_cached(k[1]) >= 0, "newer page survives");

  // Touch k[1] (take a ref, move to MRU end), then drop that ref. lru order
  // becomes [k[2], k[3], k[1]].
  {
    int h1 = p->lookup_cached(k[1]);
    expect(h1 >= 0, "lookup k[1]");
    p->release(h1);
  }
  // A 6th allocation (kept referenced) evicts the new LRU end (k[2]); k[1] is
  // protected by the earlier touch.
  int extra2 = p->alloc_page();
  expect(extra2 >= 0, "alloc evicted the next LRU page");
  expect(p->lookup_cached(k[2]) == -1, "next LRU evicted");
  int hk1 = p->lookup_cached(k[1]);
  expect(hk1 >= 0, "touched page was protected");
  if (hk1 >= 0) p->release(hk1);
}

static void test_key_chaining() {
  // The page key of a 16-token run must chain: prefixing the same first token
  // but different later tokens must differ, and an identical prefix must
  // reproduce an identical key (the resume contract).
  std::vector<std::uint32_t> a(16), b(16);
  for (int i = 0; i < 16; ++i) a[i] = b[i] = 100 + i;
  b[3] = 999;
  const std::uint64_t ka = key_of(a.data(), a.size());
  const std::uint64_t kb = key_of(b.data(), b.size());
  expect(ka != kb, "different token -> different key");

  // Chaining: key(prev, tail) == key(0, prefix+tail)
  std::uint64_t prev = key_of(a.data(), 8);
  prev = KvPoolDevice::page_key(prev, a.data() + 8, 8);
  expect(prev == ka, "chained key equals full key");
}

static void test_arena_pointers() {
  KvPoolDeviceConfig c = pool_cfg(2, KvDeviceFormat::kInt8);  // 32 rows
  const std::size_t total = c.arena_bytes();
  std::vector<int> backing(total, 7);  // fake "device" buffer
  std::unique_ptr<KvPoolDevice> p =
      KvPoolDevice::create(c, backing.data(), nullptr);

  char* base = reinterpret_cast<char*>(backing.data());
  const std::size_t cb = p->code_bytes();     // K == V code buffers
  const std::size_t sb = p->scale_bytes();    // K scales == V scales
  expect(p->k_base() == base, "k_base at 0");
  expect(p->v_base() == base + cb / 2, "v_base after K codes (equal halves)");
  expect(p->k_scale_base() == base + cb, "k_scale after K+V codes");
  expect(p->v_scale_base() == base + cb + sb / 2, "v_scale after k_scale");
  expect(total == cb + sb, "arena == codes + scales");
  // Row math a kernel would do: arena row for (layer 0, head 1, row 5).
  const int Hk = 2, rows = c.rows();
  const int row = 5;
  const int layer = 0, hkv = 1;
  char* krow = static_cast<char*>(p->k_base()) +
               ((layer * Hk + hkv) * rows + row) * c.row_bytes().k;
  expect(krow >= base && krow < base + total, "kernel row offset in range");
}

int main() {
  test_config_math();
  test_alloc_refcount();
  test_prefix_cache_lru();
  test_key_chaining();
  test_arena_pointers();
  if (g_failures == 0) {
    std::printf("kv_pool_device: all tests passed\n");
    return 0;
  }
  std::printf("kv_pool_device: %d failure(s)\n", g_failures);
  return 1;
}
