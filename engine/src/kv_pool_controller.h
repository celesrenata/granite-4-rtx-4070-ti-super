// Phase 4 (Integration): host controller that drives the Phase-3 paged KV
// pool for the live forward / serve loop.
//
// The pool (kv_pool.h) does allocation + prefix-cache bookkeeping but knows
// nothing about *sessions*. The serve loop owns a fixed set of session slots
// (one per concurrent request), each of which needs:
//   * a new request -> adopt any cached page-aligned prefix, reserve the tail
//   * each generated token -> extend, mark full pages writable-for-cache
//   * a device row table (position -> arena row) for the paged kernels
//
// This controller owns one PagedKV and maps session slots to pool sequences.
// It is the missing Phase-3 wiring: nothing in the live path was calling the
// pool before. Layout-agnostic (it reasons in pages and token ids, never
// bytes), so it unit-tests on a host without CUDA.
//
// Two reuse wins fall out of the pool's semantics for free:
//   * concurrent: overlapping sessions with a shared prompt hold the same
//     ref-counted pages; the later one skips re-prefilling them.
//   * sequential: a session's full pages stay cached (retained at ref 0) after
//     it ends, so a later session with the same prefix reuses them until LRU
//     eviction under pool pressure.

#pragma once

#include <cstddef>
#include <vector>

#include "kv_pool.h"

namespace granite4 {

class KvPoolController {
 public:
  explicit KvPoolController(KvPoolConfig config);
  ~KvPoolController();

  KvPoolController(const KvPoolController&) = delete;
  KvPoolController& operator=(const KvPoolController&) = delete;

  // --- session slots --------------------------------------------------------
  // Allocate a new session slot backed by a fresh pool sequence. Returns the
  // slot id, or -1 when the pool has no sequence capacity left (live sequences
  // are bounded by num_pages). The serve loop caps concurrency to max_slots().
  int new_slot();
  // Finish a session: free its private pages (cached pages are retained for
  // reuse) and reset the slot so the next request can reuse the same id.
  void release_slot(int slot);
  // Free everything and empty the cache (server shutdown).
  void shutdown();

  // --- per-request lifecycle -----------------------------------------------
  // Plan a prefill for a slot: adopt any cached page-aligned prefix, reserve
  // the private tail, and mark the full tail pages writable-for-cache.
  // `reused` reports the page-aligned token count the GPU may skip (its KV is
  // already in the shared arena rows the row table points at).
  struct Prefill {
    int seq = -1;
    std::size_t reused = 0;  // page-aligned tokens satisfied from the cache
    bool ok = false;
  };
  Prefill prefill(int slot, const PagedKV::Tokens& prompt);

  // Commit the session's full token list (prompt + generated so far) after a
  // generated token: extend capacity if a page boundary was crossed and mark
  // any newly-full private page writable-for-cache. Cheap no-op on a token
  // that did not cross a page boundary.
  void commit(int slot, const PagedKV::Tokens& all_tokens);

  // --- resident-session lifecycle (no prefix cache) ---------------------------
  // The serve loop keeps one resident conversation per slot and resumes it
  // from its own watermark, so it needs exclusive pages it can grow and give
  // back - not the content-addressed sharing prefill() sets up. These never
  // match or register cached pages, so a page is only ever written by the
  // slot that owns it.
  //
  // Make the slot live (idempotent) and return its id, or -1 when the pool
  // has no sequence capacity left.
  int open_slot();
  // Grow the slot to hold at least `n_tokens` tokens (page-granular, the new
  // pages are private and exclusive). Existing pages and their rows are never
  // moved, so a resident conversation's KV stays valid. false = the pool
  // cannot hold it; nothing changed.
  bool ensure(int slot, std::size_t n_tokens);
  // Hand back every page past the first `n_tokens` tokens (the slot keeps the
  // pages that cover them).
  void trim(int slot, std::size_t n_tokens);
  // Drop all of the slot's pages (its resident KV is no longer valid).
  void drop(int slot);
  // Pages needed beyond what the slot holds to reach `n_tokens`.
  std::size_t pages_missing(int slot, std::size_t n_tokens) const;
  std::size_t slot_pages(int slot) const;
  // --- device row table (the paged kernels' d_pt) ----------------------------
  // Fill `out` with the physical arena row for each logical position [0, n)
  // of the slot's sequence. n must be <= committed(slot). The mapping is a
  // valid permutation of arena rows (each page's rows stay contiguous, token
  // order within a page is preserved), and it is stable: extending the
  // sequence (commit) never reorders rows already committed. The paged
  // kernels (d_pt) consume exactly this table.
  bool row_table(int slot, std::size_t n, int* out) const;
  std::size_t committed(int slot) const;       // tokens the sequence holds
  std::size_t last_reused(int slot) const;     // page-aligned tokens reused at the last prefill

  // --- capacity / diagnostics ------------------------------------------------
  std::size_t max_slots() const { return pool_.pool_pages(); }
  std::size_t pool_pages() const { return pool_.pool_pages(); }
  std::size_t free_pages() const { return pool_.free_pages(); }
  std::size_t used_pages() const { return pool_.used_pages(); }
  std::size_t shared_pages() const { return pool_.shared_pages(); }
  std::size_t cached_pages() const { return pool_.cached_pages(); }
  std::size_t retained_pages() const { return pool_.retained_pages(); }
  std::size_t evicted_pages() const { return pool_.evicted_pages(); }
  std::size_t tokens_per_page() const { return pool_.page_rows(); }

 private:
  struct Slot {
    int seq = -1;
    std::size_t marked_pages = 0;  // pages already mark_written (incl. shared)
    std::size_t reused = 0;        // page-aligned tokens reused at last prefill
    bool in_use = false;
  };

  void mark_new_full(Slot& s, std::size_t n_tokens);

  PagedKV pool_;
  std::vector<Slot> slots_;
};

}  // namespace granite4
