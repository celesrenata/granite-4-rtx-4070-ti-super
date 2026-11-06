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
| int8 Q8_0-style (KV8; `engine/src/kernel_math.h`, device + host tests) | ~82.5 KiB (measured **78.2 KiB/pos** pool, "Measured KV8 A/B") | ~78K ⇒ **64K × 1 / 32K × 2** (measured: slightly better) |
| Q4_0-style (KV4; `G4_KV8=4`, `engine/src/kernel_math.h`, device + host tests) | 42.5 KiB (exact arithmetic: 40 KiB codes + 2.5 KiB scales) | ~156K ⇒ **64K × 2 / 128K × 1 / 32K × 4** |

So in-engine: int8 KV8 unlocks `reader/standard` (64K) today, and the
int4 KV4 path (`G4_KV8=4`, flake option `kv4 = true`) is the one that
unlocks the full 131072-position reader pool on a 16 GiB card — the
capacity arithmetic is exact ("Q4_0-KV (int4) footprint" below) and the
serve-level A/B is pending a free GPU. A Q4_K_M *weight* path is
not yet supported by this engine (FP8/BF16 only), so the llama.cpp server
remains the reference deployment until it lands. The FP8 paged pool
(`kv_pool.h`, `kernels_phase3.cu`) stays test-only until the live path
switches to it; the EAGLE draft head keeps its 1-layer BF16 KV (per-session
footprint is negligible; conversion is deferred with the quality evidence).

## Measured KV8 A/B (gremlin-1, 2026-11-05)

`scripts/kv8-ab-bench.sh` runs `kv-bench` per mode (G4_KV8=0 bf16 vs G4_KV8=1
int8 Q8_0; deterministic pseudo-tokens, identical workload, 3 repeats, best
kept; binary at worktree HEAD `bd14c2e`, checkpoint `/home/celes/kv8-bench`):

| max_pos | KV pool delta (int8 smaller) | prefill ms bf16→int8 | decode t/s bf16→int8 |
| ---: | ---: | --- | --- |
| 2305 (2048 ctx + 256 decode) | 184.0 MiB (192,937,984 B) | 523.2 → 542.7 (+3.7%) | 68.4 → 67.1 (−1.9%) |
| 4353 (4096 ctx + 256 decode) | 347.9 MiB (364,904,448 B) | 1064.2 → 1150.2 (+8.1%) | 65.6 → 64.1 (−2.3%) |
| 8321 (8192 ctx + 128 decode) | 664.2 MiB (696,254,464 B) | 2302.9 → 2622.9 (+13.9%) | 60.7 → 58.5 (−3.5%) |

Read:

- **Capacity: int8 pool is ~52% smaller** — the measured bf16−int8 delta is
  ~81.8 KiB per position, i.e. the int8 pool is **~78.2 KiB/pos** vs 160 KiB
  bf16. That is ~4.3 KiB/pos better than the 82.5 KiB planning estimate
  (the Q8_0 per-row scale packs tighter than the estimate assumed), so the
  capacity ceiling is slightly above the ~78K-position figure in the table
  above. This is what unlocks `reader/standard` (64K × 2 slots).
- **Latency: int8 is slightly slower, not faster** — prefill pays
  quantize-on-append (+3.7…13.9%), decode pays dequant-on-load (−1.9…3.5%
  tok/s). At these contexts the KV traffic is small next to the weights, so
  the halved KV bandwidth does not pay back the quantize/dequant cost. KV8 is
  a *capacity* optimization; do not expect it to speed up serving. The
  prefill gap grows with context (the int8 tile-fill path is the longer
  critical section at 8K ctx).
- The bf16 numbers above are a clean run on the same card and agree with the
  ~60–70 t/s decode band; int8 sits at −2…4% of that.

Quality (teacher-forced, same checkpoint, 24 reference positions): run the
gate prompt as `forward --tokens <prompt+ref[:-1]> --last 24` under
`G4_KV8=0` and `G4_KV8=1` and compare the two logit sets directly (the
confound-free measurement — a reference comparison would mix in the FP8
checkpoint's own drift, ~3 ULP at pos 3 on this export). Result: **argmax
identical at 23/24 positions; every position within 2 ULP** of the bf16 logit
(max 2 ULP, at pos 4). The one argmax flip (pos 13) is a 1-ULP near-tie
(11190 vs 3544), i.e. the int8 round-trip moves the top-1 only when the top
two logits are within rounding of each other. So on this prompt the int8 KV
storage cost is invisible at the token level; the kernel unit tests bound the
per-element round-trip error, and the gate keeps validating the bf16
arithmetic path (it pins G4_KV8=0 — see `docs/deployment.md`).

Caveat: this is one short prompt. The free-running 512-token A/B on the same
prompt does diverge (168/512 tokens from position 340), but that prompt
degenerates into a repetition loop in *both* modes, so it measures chaotic
loop sensitivity, not int8 error — the teacher-forced comparison above is the
honest quality number.
## Q4_0-KV (int4) footprint (exact arithmetic + unit-test bounds, 2026-11-04)

`G4_KV8=4` stores the persistent K/V cache as Q4_0-style int4: two 4-bit
codes per byte + one per-row float scale, dequantized on load (the int4
analogue of the KV8 int8 path; `engine/src/kernel_math.h`, host + device
tests; the int4 split-KV attention test in `engine/tests/test_kernels.cu`).

Per-position cost (GreedyAttention shape: 40 layers, Hk=8 KV heads, d=64):

    codes:  2 (K+V) * 8 heads * (64/2) code-bytes * 40 layers = 40,000 B
    scales: 2 (K+V) * 8 heads * 4 B            * 40 layers =   2,560 B
    total: 42,560 B = 42.5 KiB/pos

vs bf16 160 KiB/pos (10x smaller) and int8 KV8 82.5 KiB/pos (1.94x smaller).
A full 131072-position reader pool is 131072 * 42560 B = 5,576,763,392 B
= **5436 MiB** — exact, no measurement required. Against the card budget
(~14.3–14.9 GiB measured floor + weights/scratch, "Measured KV ceiling"):
5436 MiB KV fits the 16376 MiB card for the *greedy* path with roughly
~4–5 GiB left for weights + scratch + headroom. (The live serve also
carries the EAGLE r8 head + its bf16 draft KV, which scales with the pool
and is a separate budget — int4 unlocks 131K for the reader path only.)

Quality bounds (unit tests, gremlin-1 CUDA build, binary at worktree HEAD):

- int4 round-trip: worst device dequant error = **1.0 half-step** (bound
  1.0) on the random 4-bit codes; device dequant is bit-exact against the
  host twin on every code value 0..15.
- int4 vs int8 per-row error: int4 is ~16x the int8 quantization step
  (half-step = scale/8 vs scale/256); the per-row scale makes the absolute
  error proportional to the row magnitude, same contract as KV8.
- int4 split-KV attention: max |delta| vs the bf16 fused-GQA reference on
  the quantized cache = **0.02283** (test_kv4_attn_split_device, Hq=8
  Hk=2 d=128 rows=8). This is within the int4 dequantization noise floor;
  it is the attention-level analogue of the KV8 test_attn_split bound.

Pending: the serve-level A/B (prefill/decode latency, full-card margin with
the EAGLE head co-resident) needs a free GPU — gremlin-1's card was held by
the live serve during this work. `scripts/kv8-ab-bench.sh 4` runs it
(int4 vs bf16, 3 repeats, same harness as the int8 table above).


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
