// Session selection for the persistent serve loop. See serve_resume.h.
#include "serve_resume.h"

#include <algorithm>
#include <cstddef>

namespace granite4 {

ServeDecision serve_decide(const std::vector<ServeSessionState>& sessions, int next,
                           const std::vector<std::int32_t>& prompt, int min_resume) {
  ServeDecision d;
  const int n = static_cast<int>(sessions.size());
  if (n == 0) return d;
  const int rrobin = ((next % n) + n) % n;
  // At least one prompt token must be forwarded (it produces the first
  // generated token), so no session can cover more than plen - 1 positions.
  const int cap = static_cast<int>(prompt.size()) - 1;
  const int floor = std::max(1, min_resume);
  int best = -1;
  int best_len = 0;
  for (int i = 0; i < n && cap > 0; ++i) {
    const auto& h = sessions[static_cast<std::size_t>(i)].history;
    const int hlen = static_cast<int>(h.size());
    if (hlen == 0) continue;
    if (sessions[static_cast<std::size_t>(i)].watermark != hlen) continue;  // stale
    const int lim = std::min(hlen, cap);
    int lcp = 0;
    while (lcp < lim && h[static_cast<std::size_t>(lcp)] == prompt[static_cast<std::size_t>(lcp)])
      ++lcp;
    if (lcp >= floor && lcp > best_len) {
      best = i;
      best_len = lcp;
    }
  }
  if (best >= 0) {
    d.picked = best;
    d.resume = best_len;
    d.next = (best + 1) % n;
    return d;
  }
  // Miss: prefer an empty session (starting at the pointer) over evicting a
  // live conversation; otherwise the pointer's session.
  int pick = rrobin;
  for (int k = 0; k < n; ++k) {
    const int i = (rrobin + k) % n;
    if (sessions[static_cast<std::size_t>(i)].history.empty()) {
      pick = i;
      break;
    }
  }
  d.picked = pick;
  d.resume = 0;
  d.next = (pick + 1) % n;
  return d;
}

}  // namespace granite4
