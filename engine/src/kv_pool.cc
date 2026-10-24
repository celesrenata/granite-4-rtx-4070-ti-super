// Phase 3 (Integration): paged KV memory manager + prefix cache (impl).
//
// See kv_pool.h for the design. Host-only, stdlib only.
//
// Allocation invariant:
//   a page is ALLOCATED  iff (ref_count > 0) OR (in_cache)
//   a page is FREE       iff (ref_count == 0) AND NOT in_cache
// A page that a sequence no longer references but that is still in the
// prefix cache is *retained* (ref_count == 0, in_cache) and evictable;
// reserve() evicts from the LRU end of that set when the free pool is empty.

#include "kv_pool.h"

#include <algorithm>
#include <climits>

namespace granite4 {

namespace {
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
}  // namespace

// ---------------------------------------------------------------------------
// Layout / config.
// ---------------------------------------------------------------------------

KvLayout KvLayout::from_config(const ModelConfig& cfg) {
  KvLayout l;
  l.num_layers = cfg.num_layers;
  l.num_kv_heads = cfg.num_key_value_heads;
  l.head_dim = cfg.head_dim;
  // K and V, per layer, per KV head, head_dim elements; FP8 e4m3 = 1 byte each.
  l.bytes_per_token =
      2ull * cfg.num_layers * cfg.num_key_value_heads * cfg.head_dim;
  return l;
}

std::size_t KvLayout::offset(std::size_t tokens_per_page,
                             std::size_t token_local, std::size_t layer,
                             int kv, std::size_t head, std::size_t dim) const {
  if (layer >= num_layers || head >= num_kv_heads || dim >= head_dim ||
      token_local >= tokens_per_page || (kv != 0 && kv != 1)) {
    return static_cast<std::size_t>(-1);  // out of range: no offset
  }
  const std::size_t per_token = num_layers * 2ull * num_kv_heads * head_dim;
  return token_local * per_token +
         (layer * 2ull + static_cast<std::size_t>(kv)) * num_kv_heads *
             head_dim +
         head * head_dim + dim;
}

KvPoolConfig KvPoolConfig::from_budget(const ModelConfig& model,
                                       std::size_t tokens_per_page,
                                       std::size_t budget_bytes) {
  KvPoolConfig c;
  c.layout = KvLayout::from_config(model);
  c.tokens_per_page = tokens_per_page;
  c.budget_bytes = budget_bytes;
  c.page_bytes = tokens_per_page * c.layout.bytes_per_token;
  c.num_pages = c.page_bytes == 0 ? 0 : budget_bytes / c.page_bytes;
  return c;
}

// ---------------------------------------------------------------------------
// PagedKV.
// ---------------------------------------------------------------------------

std::uint64_t PagedKV::fnv1a(const std::uint32_t* data, std::size_t n,
                             std::uint64_t seed) {
  std::uint64_t h = seed;
  for (std::size_t i = 0; i < n; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= kFnvPrime;
  }
  return h;
}

PagedKV::PagedKV(KvPoolConfig config)
    : config_(std::move(config)), pages_(config_.num_pages) {
  free_list_.resize(config_.num_pages);
  for (std::size_t i = 0; i < config_.num_pages; ++i) free_list_[i] = i;
}

void PagedKV::add_ref(int seq, std::size_t page_id) {
  Page& p = pages_[page_id];
  p.ref_count += 1;
  seqs_[seq].block_table.push_back(page_id);
}

void PagedKV::drop_ref(std::size_t page_id) {
  Page& p = pages_[page_id];
  if (p.ref_count > 0) p.ref_count -= 1;
  if (p.ref_count == 0) {
    if (p.in_cache) {
      // Retained: still cached for later reuse, now evictable. The page
      // stays allocated and in the cache; reserve() may evict it.
      return;
    }
    // Not cached: free it back to the pool.
    p.key = 0;
    p.tokens.clear();
    p.last_used = 0;
    free_list_.push_back(page_id);
  }
}

std::size_t PagedKV::evict_one() {
  // LRU: evict the cached page no live sequence references (ref_count == 0)
  // with the lowest last_used tick.
  std::size_t best = pages_.size();  // sentinel: nothing evictable
  std::uint64_t best_tick = UINT64_MAX;
  for (std::size_t i = 0; i < pages_.size(); ++i) {
    const Page& p = pages_[i];
    if (p.in_cache && p.ref_count == 0 && p.last_used < best_tick) {
      best_tick = p.last_used;
      best = i;
    }
  }
  if (best == pages_.size()) return best;
  Page& p = pages_[best];
  const std::uint64_t key = p.key;
  const auto it = cache_.find(key);
  if (it != cache_.end() && it->second == best) cache_.erase(it);
  p.in_cache = false;
  p.key = 0;
  p.tokens.clear();
  p.last_used = 0;
  free_list_.push_back(best);
  ++evicted_pages_;
  return best;
}

int PagedKV::create_sequence() {
  // Bound live sequences by the pool: a sequence without a page is useless.
  if (live_sequences() >= std::max<std::size_t>(1, config_.num_pages)) {
    return -1;
  }
  seqs_.push_back(Sequence{std::vector<std::size_t>{}, 0, true});
  return static_cast<int>(seqs_.size()) - 1;
}

std::uint64_t PagedKV::prefix_key(int seq, std::size_t page_index) const {
  const Sequence& s = seqs_[seq];
  std::uint64_t h = kFnvOffset;
  for (std::size_t i = 0; i < page_index; ++i) {
    const Page& p = pages_[s.block_table[i]];
    h = fnv1a(p.tokens.data(), p.tokens.size(), h);
  }
  return h;
}

std::size_t PagedKV::match_prefix(int seq, const Tokens& tokens) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live ||
      seqs_[seq].filled != 0) {
    return 0;  // only a brand-new sequence can adopt a prefix
  }
  Sequence& s = seqs_[seq];
  const std::size_t B = config_.tokens_per_page;
  std::size_t pos = 0;
  while (pos + B <= tokens.size()) {
    // Key for the page covering tokens[pos, pos+B): the page's tokens
    // hashed with the chained prefix key of the pages before it.
    const std::uint64_t key =
        fnv1a(tokens.data() + pos, B, prefix_key(seq, pos / B));
    const auto it = cache_.find(key);
    if (it == cache_.end()) break;
    const Page& cand = pages_[it->second];
    if (cand.tokens.size() != B) break;  // stale/cleared page: stop
    bool ok = true;
    for (std::size_t j = 0; j < B; ++j) {
      if (cand.tokens[j] != tokens[pos + j]) {
        ok = false;
        break;
      }
    }
    if (!ok) break;  // hash collision: stop at the longest verified prefix
    add_ref(seq, it->second);
    pages_[it->second].last_used = ++lru_tick_;  // reuse: refresh LRU
    s.filled += B;
    pos += B;
  }
  return pos;
}

bool PagedKV::reserve(int seq, std::size_t token_count, const Tokens& tokens) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return false;
  }
  if (tokens.size() < token_count) return false;  // contract: full list
  Sequence& s = seqs_[seq];
  if (token_count <= s.filled) return true;
  const std::size_t B = config_.tokens_per_page;
  const std::size_t new_tokens = token_count - s.filled;
  const std::size_t pages_needed = (new_tokens + B - 1) / B;

  // Evict retained (cached, unreferenced) pages until the free pool can hold
  // the requested pages. Fail before allocating anything if we cannot.
  if (free_list_.size() < pages_needed) {
    for (std::size_t e = free_list_.size(); e < pages_needed; ++e) {
      if (evict_one() == pages_.size()) return false;
    }
  }

  const std::size_t start = s.filled;  // capture: s.filled moves per page
  for (std::size_t k = 0; k < pages_needed; ++k) {
    const std::size_t page_id = free_list_.back();
    free_list_.pop_back();
    Page& p = pages_[page_id];
    const std::size_t lo = start + k * B;
    const std::size_t hi = std::min(lo + B, token_count);
    p.tokens.assign(tokens.begin() + lo, tokens.begin() + hi);
    p.key = 0;
    p.in_cache = false;
    p.ref_count = 0;
    p.last_used = 0;
    add_ref(seq, page_id);
    s.filled = hi;
    if (hi < start + (k + 1) * B) break;  // partial last page
  }
  return true;
}

bool PagedKV::mark_written(int seq, std::size_t page_index) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return false;
  }
  Sequence& s = seqs_[seq];
  if (page_index >= s.block_table.size()) return false;
  const std::size_t page_id = s.block_table[page_index];
  Page& p = pages_[page_id];
  if (p.ref_count != 1) return false;  // shared pages are already cached
  if (p.tokens.size() != config_.tokens_per_page) return false;  // incomplete
  // The cache key is the chained prefix key extended with this page's own
  // tokens - exactly what match_prefix recomputes, so two sequences with the
  // same token prefix share pages.
  const std::uint64_t key =
      fnv1a(p.tokens.data(), p.tokens.size(), prefix_key(seq, page_index));
  p.key = key;
  // A key already mapped to another page is a hash collision: that older
  // entry is no longer addressable (its page stays allocated, just uncached).
  const auto it = cache_.find(key);
  if (it != cache_.end() && it->second != page_id) {
    pages_[it->second].in_cache = false;
  }
  cache_[key] = page_id;
  p.in_cache = true;
  p.last_used = ++lru_tick_;
  return true;
}

std::array<std::size_t, 2> PagedKV::locate(int seq, std::size_t token) const {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return {static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)};
  }
  const Sequence& s = seqs_[seq];
  if (token >= s.filled) {
    return {static_cast<std::size_t>(-1), static_cast<std::size_t>(-1)};
  }
  const std::size_t B = config_.tokens_per_page;
  return {s.block_table[token / B], token % B};
}

void PagedKV::release(int seq) {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return;
  }
  Sequence& s = seqs_[seq];
  for (const std::size_t page_id : s.block_table) drop_ref(page_id);
  s.block_table.clear();
  s.filled = 0;
  s.live = false;
}

void PagedKV::clear() {
  for (std::size_t i = 0; i < seqs_.size(); ++i) {
    if (seqs_[i].live) release(static_cast<int>(i));
  }
  seqs_.clear();
  cache_.clear();
  for (Page& p : pages_) p = Page{};
  free_list_.resize(config_.num_pages);
  for (std::size_t i = 0; i < config_.num_pages; ++i) free_list_[i] = i;
  lru_tick_ = 0;
  evicted_pages_ = 0;
}

std::size_t PagedKV::free_pages() const { return free_list_.size(); }

std::size_t PagedKV::used_pages() const {
  return config_.num_pages - free_list_.size();
}

std::size_t PagedKV::shared_pages() const {
  std::size_t n = 0;
  for (const Page& p : pages_)
    if (p.ref_count > 1) ++n;
  return n;
}

std::size_t PagedKV::cached_pages() const { return cache_.size(); }

std::size_t PagedKV::evicted_pages() const { return evicted_pages_; }

std::size_t PagedKV::retained_pages() const {
  std::size_t n = 0;
  for (const Page& p : pages_)
    if (p.in_cache && p.ref_count == 0) ++n;
  return n;
}

std::size_t PagedKV::live_sequences() const {
  std::size_t n = 0;
  for (const Sequence& s : seqs_)
    if (s.live) ++n;
  return n;
}

std::size_t PagedKV::sequence_tokens(int seq) const {
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return 0;
  }
  return seqs_[seq].filled;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------
const std::vector<std::size_t>& PagedKV::block_table(int seq) const {
  static const std::vector<std::size_t> empty;
  if (seq < 0 || seq >= static_cast<int>(seqs_.size()) || !seqs_[seq].live) {
    return empty;
  }
  return seqs_[seq].block_table;
}

std::size_t PagedKV::ref_count(std::size_t page_id) const {
  if (page_id >= pages_.size()) return 0;
  return pages_[page_id].ref_count;
}

bool PagedKV::page_in_cache(std::size_t page_id) const {
  if (page_id >= pages_.size()) return false;
  return pages_[page_id].in_cache;
}

}  // namespace granite4
