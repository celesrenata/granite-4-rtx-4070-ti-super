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

## Measured KV ceiling (implemented GPU direct-KV, gremlin-1)

The above layout is what the live serve path uses (bf16 KV, one persistent
`[L][Hk][max_pos][d]` K and V buffer per session; weights shared across
sessions). Measured on the RTX 4070 Ti SUPER (16376 MiB) by launching
`serve --sessions N --max-context C`, waiting for the `ready` line (printed
after weights + all session KV + per-session warm-up), and sampling the
process's `nvidia-smi` `used_memory`:

| sessions | max_context | pool (positions) | used (MiB) |
| ---: | ---: | ---: | ---: |
| 1 | 8192 | 8192 | 10350 |
| 1 | 16384 | 16384 | 11698 |
| 1 | 24576 | 24576 | 13000 |
| 1 | 32768 | 32768 | 14332 |
| 2 | 16384 | 32768 | 14408 |
| 4 | 8192 | 32768 | 14432 |
| 1 | 40960 | 40960 | **OOM** (`gpu_ctx_create failed`) |
| 2 | 24576 | 49152 | **OOM** |
| 5 | 8192 | 40960 | **OOM** |

Findings:

- **Weights + CUDA runtime + fixed per-session scratch ≈ 8.8 GiB** (the
  10350 MiB baseline minus the 8192-position KV and compact buffers). This is
  the floor: even a zero-KV engine holds ~9 GiB. (The live production process
  adds the EAGLE r8 head, ~420 MiB — hence its ~10.8 GiB footprint.)
- **KV + compact costs ~162 MiB per 1000 positions per session** (theoretical
  KV is 160 KiB/position; the extra ~2 KiB is the per-layer compact buffer).
  Each 8192-position block costs ~1.30–1.35 GiB, so the pool grows linearly
  and predictably.
- **Practical ceiling ≈ 32768 pool positions** (~14.3–14.4 GiB, leaving ~2 GiB
  headroom). Anything at a 40960-position pool OOMs at `cudaMalloc`. The
  pool is `sessions × max_context`; the same 32768 positions can be shaped as
  1×32768, 2×16384, or 4×8192 — session count and context trade off one-for-
  one against each other.
- **Max concurrency at 8192 context = 4 sessions** (5 OOMs); at 16384 it is
  2 sessions; single long context tops out at 32768.
- Consequence for the serve loop: `--sessions` is bounded by VRAM, not just by
  the `kMaxSessions = 16` code limit. On this card, `--sessions N` is only
  safe for `N × max_context ≤ ~32768`. Note the live deployment's serve
  command never passes `--sessions` (API default 1), and EAGLE requires
  `--sessions 1` (the head's scratch/KV are per-session), so **context
  length is the only production axis**. Production (deployed 2026-11-05) now
  runs `--sessions 1` with `maxContext = 32768` at the measured ceiling:
  ~14.5–14.9 GiB with the EAGLE r8 head (the 32768 rows above are headless;
  the head adds ~420 MiB), leaving ~1.5–2 GiB for the co-resident venv
  python; the first prefill then lazily claims a bounded ~0.8 GiB of
  activation scratch (one-time, does not scale with prompt length — a
  ~29000-token prompt holds at the same ~15.2 GiB), so steady state is
  ~15.2 GiB for the engine (~0.4 GiB free on the card after the
  co-resident venv python). Verified live: a 14,838-token
  prompt (1.8× the old 8192 limit) and a ~29000-token prompt both served
  end-to-end through the EAGLE path, and the API's
  `context_length_exceeded` error now reports the 32768 bound.

## Reader-tier deployment target (2026-11-04)

Granite 4.2 8B is deployed as a dedicated reader on this card. Context is a
tier, not a capacity: readers are bounded investigations returning short
structured summaries, so slots are sized per job class.

| Reader tier | Context | Parallelism target |
| --- | ---: | ---: |
| `reader/fast` | 32K | 4 slots |
| `reader/standard` | 64K | 2 slots |
| `reader/long` | 128K | 1 slot (escalation only) |

These tiers share one pool: 4×32K, 2×64K, and 1×128K are all 131072 pool
positions (the same 32K-shape trade-off as before, re-expressed). Output
budget per slot: reserve 2–4K tokens for the response. Do not carry the
hybrid-attention Qwen readers' 262K/slot configuration over: Granite's full
attention makes that ~41 GiB at Q4_0 KV.

The working deployment (llama.cpp, outside this engine) baselines on:

- Weights **Q4_K_M** (~5 GiB) with Q5_K_M as the quality benchmark.
- KV **Q4_0** (llama.cpp block storage, 4.25 bits/element + 64-element
  block scales): 5.63 GiB at 128K pool positions, 1.41 GiB at 32K.
- Full-attention reference: 40 layers × 8 KV heads × 128 head dim, 131072
  configured context. llama.cpp table, per sequence: FP16 20 GiB @ 128K,
  Q8_0 10.63 GiB, Q4_0 5.63 GiB.

Reconciliation against *this* engine (measured 8.8 GiB floor, ~14.9 GiB
usable, pool = sessions × context):

| Engine KV format | Bytes/pos | Capacity at FP8 weights |
| --- | ---: | --- |
| BF16 (live path, measured) | 160 KiB | 32768 positions (the ceiling above) |
| int8 Q8_0-style (KV8; `engine/src/kernel_math.h`, device + host tests) | ~82.5 KiB | ~78K ⇒ **64K × 1 / 32K × 2** |
| Q4_0-style (follow-up) | ~41 KiB | ~156K ⇒ **64K × 2 / 128K × 1 / 32K × 4** |

So in-engine: int8 KV8 unlocks `reader/standard` (64K) today; Q4_0 KV is the
follow-up that unlocks the full 2-slot baseline. A Q4_K_M *weight* path is
not yet supported by this engine (FP8/BF16 only), so the llama.cpp server
remains the reference deployment until it lands. The FP8 paged pool
(`kv_pool.h`, `kernels_phase3.cu`) stays test-only until the live path
switches to it; the EAGLE draft head keeps its 1-layer BF16 KV (per-session
footprint is negligible; conversion is deferred with the quality evidence).

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
