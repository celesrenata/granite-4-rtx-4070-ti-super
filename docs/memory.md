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
