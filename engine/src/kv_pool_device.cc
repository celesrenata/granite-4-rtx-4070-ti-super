// Phase 4: device KV pool (impl). See kv_pool_device.h for the design.
//
// Host-only bookkeeping over a caller-allocated device arena. The arena is
// laid out so that each buffer (K codes, V codes, K scales, V scales) is
// exactly the static [L][Hk][rows][...] cache the kernels expect today, with
// rows == num_pages * 16:
//
//   offset(buffer, layer, hkv, row, in_row)
//     = ((layer * Hk + hkv) * rows + row) * row_bytes + in_row
//
// and the per-sequence page table replaces `row` with
// `table[row/16] + row%16`.
//
// Invariant per page p (a page == a 16-token run of KV for all layers/heads):
//   FREE   iff ref_[p] == 0 && !cached_[p]   (p is in free_)
//   CACHED iff cached_[p]                    (p is in lru_)
// A page can be both referenced and cached (a sequence is mid-write and the
// prefix is also cached). ref_[p] counts references from sequences;
// cached_[p] is an independent retention by the prefix cache.

#include "kv_pool_device.h"

#include <algorithm>

namespace granite4 {

namespace {
constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr int kMapInit = 1024;
}  // namespace

// ---------------------------------------------------------------------------
// Config.
// ---------------------------------------------------------------------------

std::size_t KvPoolDeviceConfig::bytes_per_token() const {
  const std::size_t n =
      2ull * num_layers * num_kv_heads;  // K+V elements per layer set per token
  switch (format) {
    case KvDeviceFormat::kInt8:
      // codes: 1 B/element; scales: one f32 per (head,position) row, x2 (K,V).
      return n * head_dim + 2ull * num_layers * num_kv_heads * sizeof(float);
    case KvDeviceFormat::kInt4:
      return n * (head_dim / 2) + 2ull * num_layers * num_kv_heads * sizeof(float);
    case KvDeviceFormat::kBf16:
      return n * head_dim * 2ull;
  }
  return 0;
}

KvPoolDeviceConfig::RowBytes KvPoolDeviceConfig::row_bytes() const {
  RowBytes r;
  switch (format) {
    case KvDeviceFormat::kInt8:
      r.k = head_dim;
      r.v = head_dim;
      r.k_scale = sizeof(float);
      r.v_scale = sizeof(float);
      break;
    case KvDeviceFormat::kInt4:
      r.k = head_dim / 2;
      r.v = head_dim / 2;
      r.k_scale = sizeof(float);
      r.v_scale = sizeof(float);
      break;
    case KvDeviceFormat::kBf16:
      r.k = head_dim * 2;
      r.v = head_dim * 2;
      r.k_scale = 0;
      r.v_scale = 0;
      break;
  }
  return r;
}

std::size_t KvPoolDeviceConfig::arena_bytes() const {
  const RowBytes r = row_bytes();
  const std::size_t lkh =
      static_cast<std::size_t>(num_layers) * num_kv_heads;
  // K buffer + V buffer + K scale buffer + V scale buffer.
  return lkh * rows() *
         (r.k + r.v + r.k_scale + r.v_scale);
}

// ---------------------------------------------------------------------------
// FNV-1a (token chaining identical to the host manager in kv_pool.cc).
// ---------------------------------------------------------------------------

std::uint64_t KvPoolDevice::page_key(std::uint64_t prev,
                                     const std::uint32_t* tokens,
                                     std::size_t n) {
  if (prev == 0) prev = kFnvOffset;
  for (std::size_t i = 0; i < n; ++i) {
    const std::uint32_t t = tokens[i];
    for (int b = 0; b < 4; ++b) {
      prev ^= (t >> (8 * b)) & 0xffULL;
      prev *= kFnvPrime;
    }
  }
  return prev;
}

// ---------------------------------------------------------------------------
// Pool.
// ---------------------------------------------------------------------------

KvPoolDevice::KvPoolDevice(const KvPoolDeviceConfig& cfg, void* arena)
    : cfg_(cfg),
      arena_(arena),
      ref_(new int[cfg_.num_pages]()),
      key_(new std::uint64_t[cfg_.num_pages]()) {
  const KvPoolDeviceConfig::RowBytes r = cfg.row_bytes();
  const std::size_t lkh =
      static_cast<std::size_t>(cfg_.num_layers) * cfg_.num_kv_heads;
  code_bytes_ = lkh * cfg_.rows() * (r.k + r.v);
  scale_bytes_ = lkh * cfg_.rows() * (r.k_scale + r.v_scale);
  free_.resize(cfg_.num_pages);
  cached_.assign(cfg_.num_pages, false);
  // Hand out page 0 first: free_ is a stack popped from the back, so store in
  // descending order (back == page 0).
  for (int p = 0; p < cfg_.num_pages; ++p) free_[p] = cfg_.num_pages - 1 - p;
  map_keys_.assign(kMapInit, 0);
  map_rows_.assign(kMapInit, -1);
  map_mask_ = kMapInit - 1;
}

KvPoolDevice::~KvPoolDevice() = default;

std::unique_ptr<KvPoolDevice> KvPoolDevice::create(
    const KvPoolDeviceConfig& cfg, void* arena, std::string* error) {
  if (cfg.num_layers <= 0 || cfg.num_kv_heads <= 0 || cfg.head_dim <= 0 ||
      cfg.num_pages <= 0) {
    if (error) *error = "kv pool: invalid config";
    return nullptr;
  }
  if (cfg.num_layers > 4096 || cfg.num_kv_heads > 4096 ||
      cfg.head_dim > 4096 || cfg.num_pages > (1 << 28)) {
    if (error) *error = "kv pool: config out of range";
    return nullptr;
  }
  return std::unique_ptr<KvPoolDevice>(new KvPoolDevice(cfg, arena));
}

void* KvPoolDevice::k_base() const { return arena_; }
void* KvPoolDevice::v_base() const {
  return static_cast<char*>(arena_) + code_bytes_ / 2;  // K and V code buffers are equal
}
void* KvPoolDevice::k_scale_base() const {
  return static_cast<char*>(arena_) + code_bytes_;
}
void* KvPoolDevice::v_scale_base() const {
  return static_cast<char*>(arena_) + code_bytes_ + scale_bytes_ / 2;
}

int KvPoolDevice::alloc_page() {
  int p = -1;
  if (!free_.empty()) {
    p = free_.back();
    free_.pop_back();
  } else {
    // Evict the oldest cached, unreferenced page. A cached page that a live
    // sequence still references (mid-write) is not evictable.
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
      if (ref_[*it] == 0) {
        p = *it;
        lru_.erase(it);
        break;
      }
    }
    if (p < 0) return -1;  // arena full, nothing evictable
    map_erase(key_[p]);
    cached_[p] = false;
  }
  ref_[p] = 1;
  return p * kKvPageTokens;
}

void KvPoolDevice::release(int row) {
  const int p = row / kKvPageTokens;
  if (ref_[p] > 0) {
    --ref_[p];
    if (ref_[p] == 0 && !cached_[p]) free_.push_back(p);  // fully free
  }
}

int KvPoolDevice::ref_count(int row) const { return ref_[row / kKvPageTokens]; }

void KvPoolDevice::cache_page(int row, std::uint64_t key) {
  const int p = row / kKvPageTokens;
  if (cached_[p]) return;  // already cached; equal keys == equal prefixes, so
                           // the existing key is fine
  cached_[p] = true;
  key_[p] = key;
  lru_touch(p);
  map_insert(key, p);
}

int KvPoolDevice::lookup_cached(std::uint64_t key) {
  const int p = find_key(key);
  if (p < 0) return -1;
  ++ref_[p];
  // touch LRU
  for (auto it = lru_.begin(); it != lru_.end(); ++it)
    if (*it == p) {
      lru_.erase(it);
      break;
    }
  lru_.push_back(p);
  return p * kKvPageTokens;
}

int KvPoolDevice::evict_lru(int n) {
  int ev = 0;
  while (ev < n) {
    int p = -1;
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
      if (ref_[*it] == 0) {
        p = *it;
        lru_.erase(it);
        break;
      }
    }
    if (p < 0) break;  // nothing evictable
    map_erase(key_[p]);
    cached_[p] = false;
    free_.push_back(p);
    ++ev;
  }
  return ev;
}

// ---- key map (open addressing) ----

int KvPoolDevice::find_key(std::uint64_t key) const {
  std::size_t i = (key & 0xffffffffULL) & map_mask_;
  for (int probe = 0; probe <= map_mask_; ++probe, i = (i + 1) & map_mask_) {
    if (map_keys_[i] == 0) return -1;
    if (map_keys_[i] == key) return map_rows_[i];
  }
  return -1;
}

void KvPoolDevice::map_insert(std::uint64_t key, int row) {
  if ((map_count_ + 1) * 2 > static_cast<int>(map_keys_.size()))
    key_map_resize();
  std::size_t i = (key & 0xffffffffULL) & map_mask_;
  for (int probe = 0;; ++probe, i = (i + 1) & map_mask_) {
    if (map_keys_[i] == 0 || map_keys_[i] == key) {
      if (map_keys_[i] == 0) ++map_count_;
      map_keys_[i] = key;
      map_rows_[i] = row;
      return;
    }
  }
}

void KvPoolDevice::map_erase(std::uint64_t key) {
  std::size_t i = (key & 0xffffffffULL) & map_mask_;
  for (int probe = 0; probe <= map_mask_; ++probe, i = (i + 1) & map_mask_) {
    if (map_keys_[i] == 0) return;
    if (map_keys_[i] == key) {
      // Rehash the rest of the cluster to close the hole (no tombstones).
      std::size_t j = i;
      map_keys_[j] = 0;
      map_rows_[j] = -1;
      --map_count_;
      j = (j + 1) & map_mask_;
      while (map_keys_[j] != 0) {
        const std::uint64_t k2 = map_keys_[j];
        const int r2 = map_rows_[j];
        map_keys_[j] = 0;
        map_rows_[j] = -1;
        map_insert(k2, r2);
        j = (j + 1) & map_mask_;
      }
      return;
    }
  }
}

void KvPoolDevice::key_map_resize() {
  const auto ok = map_keys_;
  const auto orow = map_rows_;
  const int old_mask = map_mask_;
  map_keys_.resize(map_keys_.size() * 2);
  map_rows_.resize(map_rows_.size() * 2);
  std::fill(map_keys_.begin(), map_keys_.end(), 0ULL);
  std::fill(map_rows_.begin(), map_rows_.end(), -1);
  map_mask_ = static_cast<int>(map_keys_.size()) - 1;
  map_count_ = 0;
  for (int i = 0; i <= old_mask; ++i)
    if (ok[i] != 0) map_insert(ok[i], orow[i]);
}

void KvPoolDevice::lru_touch(int p) {
  for (int q : lru_) if (q == p) return;
  lru_.push_back(p);
}

}  // namespace granite4
