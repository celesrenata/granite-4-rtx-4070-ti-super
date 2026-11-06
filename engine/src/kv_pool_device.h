// Phase 4: device KV pool — one shared arena, 16-token pages, prefix cache.
//
// The static per-session cache the GPU path uses today is laid out
//   K codes     [L][Hk][max_pos][d]      (int8, bf16, or int4 codes)
//   V codes     [L][Hk][max_pos][d]
//   K scales    [L][Hk][max_pos]         (int8/int4 only, one f32 per row)
//   V scales    [L][Hk][max_pos]
// and the attention / append kernels address row t of KV head hkv inside
// the layer-l slice with the flat expression (hkv * stride + t) * d,
// stride == max_pos.
//
// This pool keeps exactly that layout but with stride := rows of a shared
// arena (rows == num_pages * kPageTokens), and replaces the flat row
// identity with a per-sequence page table on device:
//
//   arena_row(t) = table[t / kPageTokens] + (t % kPageTokens)
//
// so the kernel's row expression becomes
//   (hkv * rows + arena_row(t)) * d
// (and hkv * rows + arena_row(t) for scales). A "page" is one 16-token run
// of KV for ALL layers and heads in K, V and both scale buffers; a sequence
// owns a reference to each page it has written, so pages are shared across
// sequences and across requests through the prefix cache below.
//
// Bookkeeping is host-only (this file is host code; the arena itself is a
// caller-allocated device buffer, one cudaMalloc). The FNV-chained page keys
// are the same chaining the host manager (kv_pool.h) uses, so a page key
// identifies the token prefix 0 .. 16*(p+1).

#ifndef GRANITE_KV_POOL_DEVICE_H
#define GRANITE_KV_POOL_DEVICE_H

#include "model_config.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace granite4 {

constexpr int kKvPageTokens = 16;

enum class KvDeviceFormat {
  kInt8,  // Q8_0 codes + per-row f32 scale   (G4_KV8=1)
  kInt4,  // Q4_0 codes + per-row f32 scale   (G4_KV8=4)
  kBf16,  // bf16 codes, no scales
};

struct KvPoolDeviceConfig {
  int num_layers = 0;
  int num_kv_heads = 0;
  int head_dim = 0;
  int num_pages = 0;  // arena capacity in pages (16 tokens each)
  KvDeviceFormat format = KvDeviceFormat::kInt8;

  int rows() const { return num_pages * kKvPageTokens; }
  // Bytes of one (layer, kv head, token) across K + V + scales.
  std::size_t bytes_per_token() const;
  // Arena byte size for this config (K, V and both scale buffers).
  std::size_t arena_bytes() const;
  // Bytes of one row (head, position) of one buffer, per buffer kind.
  struct RowBytes {
    std::size_t k;  // K codes
    std::size_t v;  // V codes
    std::size_t k_scale;
    std::size_t v_scale;
    std::size_t all() const { return k + v + k_scale + v_scale; }
  };
  RowBytes row_bytes() const;
};

class KvPoolDevice {
 public:
  // `arena` is a caller-allocated device buffer of at least
  // cfg.arena_bytes(). It may be nullptr (bookkeeping-only; used by the host
  // tests — base() then returns nullptr and offsets are still computed).
  static std::unique_ptr<KvPoolDevice> create(const KvPoolDeviceConfig& cfg,
                                              void* arena, std::string* error);

  ~KvPoolDevice();
  KvPoolDevice(const KvPoolDevice&) = delete;
  KvPoolDevice& operator=(const KvPoolDevice&) = delete;

  const KvPoolDeviceConfig& config() const { return cfg_; }
  void* arena() const { return arena_; }

  // Arena buffer bases (K codes, V codes, K scales, V scales). Scales are
  // nullptr for bf16.
  void* k_base() const;
  void* v_base() const;
  void* k_scale_base() const;
  void* v_scale_base() const;
  // Bytes of the K and V code buffers (== the scale buffers for int8/int4,
  // since one f32 scale per row is 4 bytes vs d/d2 code bytes — NOT equal
  // for bf16; callers size per buffer).
  std::size_t code_bytes() const { return code_bytes_; }
  std::size_t scale_bytes() const { return scale_bytes_; }

  // Allocate one page; returns the page's arena row (row 0 of its 16-row run),
  // or -1 if the arena is full and no cached page is evictable. The caller
  // holds the returned reference until release().
  int alloc_page();
  // Release a reference. When the last reference drops the page is retained
  // in the prefix cache (if cached) and becomes LRU-evictable.
  void release(int row);
  // Page reference count (0 when free or cached-without-refs).
  int ref_count(int row) const;

  // ---- prefix cache ------------------------------------------------------
  // `key` identifies the token prefix 0 .. 16*(p+1) of the sequence that
  // wrote the page; it must chain all earlier page keys (page_key()).
  // Marks the page cached under `key` (no-op if it is already cached under
  // this key; a page that is cached under a different key keeps that one —
  // equal keys are equal prefixes, so the choice is irrelevant).
  void cache_page(int row, std::uint64_t key);
  // Find the cached page under `key`; takes a reference (ref_count + 1) and
  // returns its arena row, or -1. The caller releases it when done.
  int lookup_cached(std::uint64_t key);
  // Number of cached pages (ref_count 0, in_cache).
  int cached_pages() const { return static_cast<int>(lru_.size()); }

  // Evict up to n LRU cached pages. Returns pages evicted.
  int evict_lru(int n);

  // Pages currently unallocated (refcount 0, not cached).
  int free_pages() const { return static_cast<int>(free_.size()); }
  // Evictable pages: cached and unreferenced (the LRU end of lru_ is what
  // alloc_page() reuses).
  int evictable_pages() const {
    int n = 0;
    for (int p : lru_) if (ref_[p] == 0) ++n;
    return n;
  }

  // FNV-1a over a token slice, chained from `prev` (0 for the first page).
  // The key of page p chains all tokens 0 .. 16*(p+1).
  static std::uint64_t page_key(std::uint64_t prev, const std::uint32_t* tokens,
                                std::size_t n);

 private:
  KvPoolDevice(const KvPoolDeviceConfig& cfg, void* arena);
  int find_key(std::uint64_t key) const;  // cached row or -1
  void key_map_resize();
  void map_insert(std::uint64_t key, int row);
  void map_erase(std::uint64_t key);
  void lru_touch(int row);

  KvPoolDeviceConfig cfg_;
  void* arena_ = nullptr;
  std::size_t code_bytes_ = 0;  // K buffer == V buffer
  std::size_t scale_bytes_ = 0;  // 0 for bf16
  std::unique_ptr<int[]> ref_;  // per row: references held
  std::unique_ptr<std::uint64_t[]> key_;  // cached key (when cached_[row])
  int used_pages_ = 0;
  std::vector<int> free_;  // free page rows, LIFO
  std::vector<bool> cached_;  // per row: in prefix cache
  // LRU list (int rows, oldest first). A row is in the list iff cached_.
  std::vector<int> lru_;
  // Open-addressing key map: key -> row.
  std::vector<std::uint64_t> map_keys_;  // 0 == empty
  std::vector<int> map_rows_;  // -1 empty
  int map_mask_ = 0;
  int map_count_ = 0;
};

}  // namespace granite4

#endif  // GRANITE_KV_POOL_DEVICE_H
