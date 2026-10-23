# Memory budget — 16 GB RTX 4070 Ti Super

**Planning figures only.** Nothing below is measured; P2 establishes the
measured baseline (bandwidth micro-benchmark, per-layer residency) and P3
verifies residency with `nvidia-smi`. The card is assumed to expose
~14.9 GiB of usable HBM (16 GB minus driver/CUDA overhead — to be measured).

## Model size (from the spec constants)

Per layer (GQA: 32 Q heads, 8 KV heads, head dim 128 ⇒ KV width 1024):

| Block | Params |
| --- | ---: |
| Q proj 4096×4096 | 16,777,216 |
| K proj 4096×1024 | 4,194,304 |
| V proj 4096×1024 | 4,194,304 |
| O proj 4096×4096 | 16,777,216 |
| gate / up / down (3 × 4096×12800) | 157,286,400 |
| **per layer** | **199,229,440** |
| × 40 layers | 7,969,177,600 |
| token embedding 100352×4096 (tied with LM head, assumed) | 411,041,792 |
| **total (tied)** | **8,380,219,392** |

Untied adds another 411,041,792. **P1 confirms untied:** the checkpoint has
363 tensors with both `model.embed_tokens.weight` and `lm_head.weight`
present, `tie_word_embeddings: false`. Layernorms add
`40 × (4096 + 4096) + 4096 = 331,776` elements the per-layer table above
omits. The measured checkpoint total is **17,583,185,920 bytes**
(= 8,791,592,960 elements × 2; verified byte-for-byte by
`granite4-checkpoint verify`).

## Weights by precision

| Precision | Raw weight bytes | Usable on 16 GB? |
| --- | ---: | --- |
| BF16 | 15.62 GiB | **No** — leaves < ~0.3 GiB for KV + activations |
| FP8 (e4m3) + scales | 7.81 GiB + ~1% scales ≈ **~8.0 GiB** | Yes |
| 4-bit + scales | 3.90 GiB + ~0.1–0.2 GiB | Yes, with headroom |

**Decision (working default): FP8 weights.** BF16 cannot coexist with any
KV cache on this card; 4-bit is the fallback if KV must be large, at the cost
of extra dequant work per token.

## KV cache

Per token, K+V: `2 × 40 layers × 8 KV heads × 128 head dim = 81,920` elements.

| KV precision | Per token | 16K ctx | 32K ctx | full 131072 ctx |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 160 KiB | 2.5 GiB | 5.0 GiB | 20.0 GiB |
| FP8 | 80 KiB | 1.25 GiB | 2.5 GiB | 10.0 GiB |

### Paged KV management (P3)

The runtime allocates KV as a **paged pool** (`engine/src/kv_pool.{h,cc}`,
host-only, no CUDA). A page holds `tokens_per_page` consecutive tokens of KV
for *all* layers and KV heads (FP8 e4m3, one byte per element); a page's bytes
are laid out token-major — `[token][layer][k|v][kv_head][head_dim]` — so a
token's KV is contiguous and a full page is one contiguous region. A
sequence's KV is an ordered block table of page ids; `KvPoolConfig::
from_budget` sizes the pool from the KV byte budget above
(`num_pages = floor(budget / page_bytes)`).

Two sharing rules:

- **Prefix cache** — only *complete* pages are shared. A page is
  content-addressable (a chained hash over its page-aligned token prefix,
  re-checked token-by-token on lookup, so a hash collision can never wrongly
  share pages). A new sequence whose prompt shares a cached page-aligned
  prefix reuses those physical pages (ref-counted) instead of allocating.
- **Eviction** — LRU over the *retained* pages: a page that no live sequence
  references but that is still cached (`ref_count == 0`) is evictable; when
  the free pool is empty the allocator evicts the least-recently-used one. A
  page a live sequence references is never evicted, so an allocation that
  cannot be satisfied fails rather than clobbering. This retains prompts
  across *sequential* requests (the common long-system-prompt case) as well
  as *concurrent* ones.

The GPU writes KV into a (page, token) slot and, at attention time, compacts a
sequence's pages into the contiguous `[Hk, kv_len, d]` layout the P2 attention
kernel consumes; the byte placement it compacts from is exactly the
`KvLayout::offset` layout above.

## Budget (FP8 weights + FP8 KV), ~14.9 GiB usable

| Item | Budget |
| --- | ---: |
| Weights (FP8 + scales) | ~8.0 GiB |
| Activations / workspace / CUDA runtime | ~1.0–1.5 GiB (measured in P3) |
| Remaining for KV | ~5.4–5.9 GiB |
| ⇒ KV capacity (FP8, 80 KiB/token) | **~68–74K tokens** single sequence |

Consequences:

- **Full 131072 context does not fit at FP8 KV** in Phase P3 with one
  sequence. Options, to be decided with measured numbers in P3/P4: FP8 KV
  with prefix-cache offload, 4-bit KV, or a context budget of ~64K tokens
  per sequence with the remainder via paged eviction.
- Multi-sequence: KV is sliced across sequences by the scheduler;
  ~16–32K tokens × 2–4 sequences fits comfortably.

## Roofline (planning, unverified)

Assumed GDDR6 bandwidth ~670 GB/s (product spec; **measure first in P2**).
Single-sequence decode streams the non-recomputable weights once per token:

- FP8: ~7.8–8.0 GiB/token ⇒ **≈ 80–85 tok/s theoretical ceiling**,
  realistically 40–60% of that with attention + KV reads and kernel
  efficiency. No number is claimed until P4 measures it.

## Verification obligations

| Phase | What must be measured |
| --- | --- |
| P1 | Actual loaded weight residency (`nvidia-smi`), scales overhead |
| P2 | Memory-bandwidth micro-benchmark; per-kernel peak memory |
| P3 | Steady-state residency vs this plan (≤5% drift); KV capacity check at max sequence count |
