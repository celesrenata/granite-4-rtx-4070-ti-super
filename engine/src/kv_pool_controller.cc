// Phase 4 (Integration): host controller implementation.
// See kv_pool_controller.h for the design and invariants.

#include "kv_pool_controller.h"

#include <utility>

namespace granite4 {

KvPoolController::KvPoolController(KvPoolConfig config)
    : pool_(std::move(config)) {}

KvPoolController::~KvPoolController() { shutdown(); }

int KvPoolController::new_slot() {
  // Recycle an idle slot first: its pool sequence stays alive (reset, not
  // released) so the slot id is stable across requests.
  for (Slot& s : slots_) {
    if (!s.in_use) {
      s.marked_pages = 0;
      s.reused = 0;
      return static_cast<int>(&s - slots_.data());
    }
  }
  const int seq = pool_.create_sequence();
  if (seq < 0) return -1;  // live sequences capped at num_pages
  slots_.push_back(Slot{seq, 0, 0, true});
  return static_cast<int>(slots_.size()) - 1;
}

void KvPoolController::release_slot(int slot) {
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return;
  Slot& s = slots_[slot];
  // Keeps the pool sequence alive so the next prefill on this slot re-adopts
  // any pages still cached; private pages are freed (cached ones retained).
  pool_.reset_sequence(s.seq);
  s.in_use = false;
  s.marked_pages = 0;
  s.reused = 0;
}

void KvPoolController::shutdown() {
  for (Slot& s : slots_)
    if (s.in_use) pool_.release(s.seq);
  pool_.clear();
  slots_.clear();
}

KvPoolController::Prefill
KvPoolController::prefill(int slot, const PagedKV::Tokens& prompt) {
  Prefill r;
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return r;
  Slot& s = slots_[slot];
  // Fresh start: any pages from a prior request on this slot are freed
  // (cached ones stay for reuse). Idempotent on a just-allocated slot.
  pool_.reset_sequence(s.seq);
  s.marked_pages = 0;
  s.reused = 0;

  s.reused = pool_.match_prefix(s.seq, prompt);
  if (!pool_.reserve(s.seq, prompt.size(), prompt)) {
    // Roll back any pages adopted by match_prefix so the slot (and its pages)
    // is left clean for the next request.
    pool_.reset_sequence(s.seq);
    s.in_use = false;
    s.reused = 0;
    return r;  // ok == false
  }
  s.in_use = true;
  mark_new_full(s, prompt.size());
  r.seq = s.seq;
  r.reused = s.reused;
  r.ok = true;
  return r;
}

void KvPoolController::commit(int slot, const PagedKV::Tokens& all_tokens) {
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return;
  Slot& s = slots_[slot];
  if (!s.in_use) return;
  const std::size_t have = pool_.sequence_tokens(s.seq);
  if (all_tokens.size() <= have) return;  // no-op: no new tokens
  if (!pool_.reserve(s.seq, all_tokens.size(), all_tokens)) return;
  mark_new_full(s, all_tokens.size());
}

bool KvPoolController::row_table(int slot, std::size_t n, int* out) const {
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return false;
  const Slot& s = slots_[slot];
  if (!s.in_use) return false;
  return pool_.row_table(s.seq, n, out);
}

std::size_t KvPoolController::committed(int slot) const {
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return 0;
  const Slot& s = slots_[slot];
  if (!s.in_use) return 0;
  return pool_.sequence_tokens(s.seq);
}

std::size_t KvPoolController::last_reused(int slot) const {
  if (slot < 0 || slot >= static_cast<int>(slots_.size())) return 0;
  return slots_[slot].reused;
}

void KvPoolController::mark_new_full(Slot& s, std::size_t n_tokens) {
  const std::size_t B = pool_.page_rows();
  const std::size_t full = n_tokens / B;
  for (std::size_t i = s.marked_pages; i < full; ++i) {
    pool_.mark_written(s.seq, i);
  }
  s.marked_pages = full;
}

}  // namespace granite4
