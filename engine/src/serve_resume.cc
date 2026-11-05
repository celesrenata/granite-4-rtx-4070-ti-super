// Session selection for the persistent serve loop. See serve_resume.h.

#include "serve_resume.h"

#include <cstddef>

namespace granite4 {

ServeDecision serve_decide(const std::vector<ServeSessionState>& sessions, int next,
                           const std::vector<std::int32_t>& prompt) {
  ServeDecision d;
  const int n = static_cast<int>(sessions.size());
  if (n == 0) return d;
  const int rrobin = next % n;
  int best = -1;
  int best_len = 0;
  for (int i = 0; i < n; ++i) {
    const int hlen = static_cast<int>(sessions[i].history.size());
    if (hlen == 0) continue;
    if (hlen > static_cast<int>(prompt.size())) continue;
    if (sessions[i].watermark != hlen) continue;  // stale: last turn failed
    bool match = true;
    for (int j = 0; j < hlen; ++j)
      if (sessions[i].history[static_cast<std::size_t>(j)] !=
          prompt[static_cast<std::size_t>(j)]) {
        match = false;
        break;
      }
    if (match && hlen > best_len) {
      best = i;
      best_len = hlen;
    }
  }
  d.resume = best_len;
  if (best >= 0) {
    d.picked = best;
    d.next = (best + 1) % n;
  } else {
    d.picked = rrobin;
    d.next = (rrobin + 1) % n;
  }
  return d;
}

}  // namespace granite4
