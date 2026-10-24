// Phase 3 (Integration): paged KV memory manager + prefix cache.
//
// Host-only. stdlib only (plus model_config.h); no CUDA. This is the
// allocation / sharing bookkeeping that the GPU attention path builds on
// (see docs/memory.md for the byte budget it is wired to).
//
// The design is vLLM-style paged attention at the *management* level:
//
//   * The KV pool is a fixed set of equal-size **pages**. A page holds
//     `tokens_per_page` consecutive tokens of KV for *all* layers and KV
//     heads (FP8 e4m3, one byte per element).
//   * A sequence's logical KV is an ordered **block table** of page ids.
//   * Full pages are **content-addressable**: the key is a hash chained over
//     the page-aligned token prefix the page represents. A new sequence whose
//     prompt shares an already-cached page-aligned prefix reuses those
//     physical pages (ref-counted) instead of allocating new ones.
//   * Only *complete* pages are shared. Each sequence's tail (partially
//     filled) page is private; it is registered in the cache only once its
//     last token has been written.
//
// Eviction policy (this milestone): a page is freed the moment its last
// sequence releases it (ref count -> 0). This is the memory-safe baseline —
// the pool never leaks, and concurrent sequences with a shared prompt reuse
// the same physical KV (the scheduler's primary win). Retaining pages after
// their last user leaves, to serve *sequential* reuse, would be an LRU cache
// over a bounded pool: a Phase-4 tuning decision, out of scope here.
//
// The GPU writes KV into a (page, token) slot and reads it back at attention
// time; the *byte layout* inside a page is fixed here (kv_byte_offset) so the
// compaction kernel, the P2 attention kernel's input, and the host tests all
// agree on the physical placement.

#ifndef GRANITE4_KV_POOL_H
#define GRANITE4_KV_POOL_H

#include "model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace granite4 {

// ---------------------------------------------------------------------------
// KV byte layout (one page).
//
// Within a page the bytes are laid out token-major so that the KV for one
// token is contiguous and a full page is a single contiguous region:
//
//   [token_local][layer][k|v][kv_head][head_dim]
//
// A single KV element (one head-dim slot for one layer, one of {K,V}, one
// head, one token) is one FP8 byte.
// ---------------------------------------------------------------------------

struct KvLayout {
  std::size_t num_layers;
  std::size_t num_kv_heads;
  std::size_t head_dim;
  std::size_t bytes_per_token;  // 2 * layers * kv_heads * head_dim

  static KvLayout from_config(const ModelConfig& cfg);

  // Byte offset *within one page* of the KV element for the token at
  // `token_local`, layer, K(0)/V(1), kv head, head-dim index. In range
  // [0, tokens_per_page * bytes_per_token).
  std::size_t offset(std::size_t tokens_per_page, std::size_t token_local,
                     std::size_t layer, int kv, std::size_t head,
                     std::size_t dim) const;

  // Bytes a single token's KV occupies (the page-independent part).
  std::size_t token_bytes() const { return bytes_per_token; }
};

// ---------------------------------------------------------------------------
// Pool configuration, sized from a byte budget (docs/memory.md).
// ---------------------------------------------------------------------------

struct KvPoolConfig {
  KvLayout layout;
  std::size_t tokens_per_page = 16;
  std::size_t budget_bytes = 0;    // the byte budget the pool is sized from
  std::size_t num_pages = 0;       // pool_pages = floor(budget / page_bytes)
  std::size_t page_bytes = 0;      // tokens_per_page * bytes_per_token

  // Size the pool from the KV byte budget. num_pages is a whole-page count;
  // the last (partial) bytes of the budget are unused, which is the same
  // rounding a physical page pool performs.
  static KvPoolConfig from_budget(const ModelConfig& model,
                                  std::size_t tokens_per_page,
                                  std::size_t budget_bytes);

  std::size_t capacity_tokens() const { return num_pages * tokens_per_page; }
  std::size_t total_bytes() const { return num_pages * page_bytes; }
};

// ---------------------------------------------------------------------------
// Paged KV manager.
//
// Owns a pool of `num_pages` pages. Pages are free until a sequence
// references them; a page referenced by >1 sequence is shared (read-only).
// The prefix cache is a map from the chained page-prefix hash to the page id;
// entries are verified against the stored token ids on lookup, so a hash
// collision can never wrongly share pages.
// ---------------------------------------------------------------------------

class PagedKV {
 public:
  using Tokens = std::vector<std::uint32_t>;
  explicit PagedKV(KvPoolConfig config);

  // Allocate a new (empty) sequence. Returns a sequence id, or -1 when there
  // are no sequences left to create (bounded by the pool).
  int create_sequence();
  // Share cached pages for the longest page-aligned prefix of `tokens`.
  // Returns the number of tokens satisfied from the cache (a multiple of
  // tokens_per_page) and sets the sequence's filled count accordingly. The
  // matched pages are ref-counted; the caller still reserves the tail.
  std::size_t match_prefix(int seq, const Tokens& tokens);

  // Ensure `seq` has capacity for at least `token_count` tokens, allocating
  // private pages for the non-shared tail. Fills each new page with the
  // corresponding token ids (from `tokens`) so the page can later be cached.
  // Returns false when the pool cannot hold the requested tokens.
  bool reserve(int seq, std::size_t token_count, const Tokens& tokens);

  // Register the sequence's page at `page_index` in the prefix cache (call
  // once the page's last token has been written). No-op if the page is
  // already shared/cached. Returns true if it was newly cached.
  bool mark_written(int seq, std::size_t page_index);

  // Physical location of token `token`'s KV: {page_id, token_local}.
  std::array<std::size_t, 2> locate(int seq, std::size_t token) const;

  // Decrement every page the sequence references; free the pages that reach
  // ref count 0. The sequence becomes unusable.
  void release(int seq);

  // Release every sequence and empty the cache (back to a full free pool).
  void clear();

  // Introspection (capacity / residency wiring + tests).
  // The sequence's block table (page id per page), in allocation order.
  // This is the input the paged_kv_compact kernel consumes (device mirror).
  const std::vector<std::size_t>& block_table(int seq) const;

  const KvPoolConfig& config() const { return config_; }
  std::size_t pool_pages() const { return config_.num_pages; }
  std::size_t capacity_tokens() const { return config_.capacity_tokens(); }
  std::size_t free_pages() const;
  std::size_t used_pages() const;
  std::size_t shared_pages() const;   // pages with ref_count > 1
  std::size_t cached_pages() const;   // pages present in the prefix cache
  std::size_t evicted_pages() const;   // pages evicted from the cache so far
  std::size_t retained_pages() const;  // cached pages with ref_count == 0
  std::size_t live_sequences() const;  // sequences currently live
  std::size_t sequence_tokens(int seq) const;  // filled count for a live seq
  std::size_t ref_count(std::size_t page_id) const;
  bool page_in_cache(std::size_t page_id) const;

 private:
  struct Page {
    std::vector<std::uint32_t> tokens;  // token ids this page holds (<= B)
    std::uint64_t key = 0;               // chained prefix hash (0 = unset)
    int ref_count = 0;
    bool in_cache = false;
    std::uint64_t last_used = 0;         // LRU tick (for eviction)
  };

  struct Sequence {
    std::vector<std::size_t> block_table;  // page ids, in order
    std::size_t filled = 0;                // tokens stored
    bool live = false;
  };

  static std::uint64_t fnv1a(const std::uint32_t* data, std::size_t n,
                             std::uint64_t seed);

  void add_ref(int seq, std::size_t page_id);
  void drop_ref(std::size_t page_id);
  std::uint64_t prefix_key(int seq, std::size_t page_index) const;
  // Evict the least-recently-used cached page that no live sequence
  // references (ref_count == 0). Returns the freed page id, or the largest
  // page id + 1 sentinel when there is nothing evictable.
  std::size_t evict_one();

  KvPoolConfig config_;
  std::vector<Page> pages_;
  std::vector<Sequence> seqs_;
  std::vector<std::size_t> free_list_;
  std::uint64_t lru_tick_ = 0;      // monotonically increasing LRU clock
  std::size_t evicted_pages_ = 0;   // total evictions (integrity/telemetry)
  // prefix key (chained FNV over a page-aligned token prefix) -> page id.
  // Only complete pages are ever registered; the entry's token ids are
  // re-checked on lookup, so a collision never wrongly shares pages.
  std::map<std::uint64_t, std::size_t> cache_;
};

}  // namespace granite4

#endif  // GRANITE4_KV_POOL_H
