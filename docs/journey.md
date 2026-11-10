# Granite 4.2 8B on one RTX 4070 Ti SUPER: from bf16 to K5:V4 and tensor-core attention

A recap of the serving engine's path from the first scaffold (2026-10-22) to the current production build
(2026-11-10, NixOS generation 800 on gremlin-1). Every number here was measured on gremlin-1 (RTX 4070 Ti SUPER,
16 GB, sm_89, 632 GB/s measured read bandwidth) unless marked otherwise. Detailed tables and reproduction commands
live in `phases.md`, `memory.md`, `deployment.md` and `kv-k5v4.md`; Menagerie-side quality runs live in Menagerie
`docs/performance/`.

Note on "FP16": the starting point was a **bf16** KV cache (and a bf16 checkpoint packed to FP8 e4m3 weights at
load). The engine never used fp16 storage.

## The short version

| Milestone | Single-stream decode | Best aggregate | KV pool | Notes |
| --- | ---: | ---: | ---: | --- |
| First GPU forward (P4 start) | 55.7 ms/token (18 tok/s) | - | bf16, ~32K max | O(cols²) RMSNorm |
| Kernel fixes + CUDA graph | 13.95 ms/token (**72 tok/s**) | - | bf16 | ~95% of the weight-stream floor |
| EAGLE head r3 | 105.6 tok/s (k=3, 6 prompts) | - | bf16 | output == greedy |
| KV8 int8 KV | same decode | - | 64K | 82.5 KiB/pos |
| Concurrent sessions + skinny GEMM | 68 tok/s plain | 141 tok/s (2 clients) | KV8 64K | |
| Tree drafting + r10/r11 heads | 126.8 tok/s live API | 233.8 tok/s (2 clients) | KV8 | |
| KV4 3 x 64K oversubscribed | 126 tok/s | 311.6 tok/s (3 slots) | int4 120K | capacity, but lossy |
| **K5:V4** | ~120 tok/s live | 211 tok/s (3 clients, live) | 114,688 | int8-level accuracy |
| **Tensor-core attention** | unchanged at short ctx | - | 114,688 | 16K verify step 38.8 → 23.9 ms |
| **Code-tile prefill attention** | unchanged | - | 114,688 | 32K prefill 22.1 → 13.4 s |
| **VRAM diet** (embedding to host, shared scratch/RoPE, pieced catch-up) | unchanged | - | **147,456 at 98K ctx** | −1.7 GiB outside the pool |
| **r15long head** (long-context training data) | 32K decode 49.7 → 67.5 tok/s | serving 139 tok/s | 147,456 | long-record acceptance +27% offline |

Two rules held the whole way: speculative output must equal plain greedy token for token, and every speed claim
must come from a same-harness A/B.

## 1. Making a single stream fast (P0-P4, 2026-10-22 to 10-29)

The 8B model streams 8.38 GB of FP8 weights per token, so the floor is 632 GB/s / 8.38 GB ≈ 75 tok/s.

| Fix | Step time |
| --- | ---: |
| First working GPU forward | 55.7 ms |
| RMSNorm one block per row (was O(cols²), re-read the row per element) | 24.5 ms |
| Branchless e4m3 decode | 21.7 ms |
| 16-byte GEMV loads, 128-thread blocks, fused QKV, residual in epilogue, fused rope/append, fused GQA, direct-cache GQA, warp-tree RMSNorm, parallel softmax | 15.2 ms |
| Hardware FP8 convert, CUDA-graph decode with on-device argmax, split-KV attention, f32 RMSNorm | **13.95 ms (72 tok/s)** |

Prefill moved to tensor-core GEMMs with FP8 weight tiles decoded in shared memory: 2000 tokens went from 78 s to
0.5 s (~67 TFLOPS, ~76% of BF16 peak). Load time went from 131 s to 7.4 s by packing weights straight from the shards.

Learnings:
- After the fixes, decode GEMVs ran at 590-630 GB/s, so **single-stream speed is bandwidth-bound**. More speed means
  fewer bytes per token or more tokens per weight pass.
- **INT4 weights were rejected:** 83.6% top-1 agreement with FP8 (int8 control: 97.2%). Weight precision is a quality
  decision, not a free speed lever.
- Several "speedups" turned out to be measurement bugs: unsynchronized timers, a driver that ate the `ready` line,
  dirty L2 lines billed to GEMMs. Bench harnesses got syncs and L2 eviction because of this.

## 2. More tokens per weight pass: drafting (P4 rounds 3-6, 11-03 to 11-08)

- **Exact wide verify.** The verify GEMVs reproduce the decode arithmetic, so a verify row is bit-identical to a
  decode step. Rows 2-8 cost about 0.4 ms each instead of 3.5 ms.
- **Prompt-lookup drafting:** up to 1.65x on copy-heavy replies, but a small loss on novel text.
- **EAGLE head (r3):** 68.3 → 105.6 tok/s at k=3, with output identical to greedy.
- **Adaptive policy:** EAGLE accepted 75-81% of drafts on trained styles but ~8% on novel prose, so a per-request
  probe now picks draft or plain.
- **Skinny tensor-core GEMM for 1-16 rows:** a step costs 16.3 ms at 1 row and 16.6 ms at 16 rows, so verify width
  is nearly free.
- **Tree drafting:** 7 nodes when batched, 13 nodes solo. Each tree row is bit-identical to a decode of its path.
  Chat at 2 in flight went 205 → 226 tok/s.
- **Heads r8 → r12** were self-distilled on chat the model generated itself. Offline tokens/round went from about
  1.7 to 2.19 on chat. The live API went from 80.7 / 141.2 tok/s (1 / 2 clients) to 126.8 / 233.8.

Learnings:
- **Acceptance depends on the text**, and it falls with context length: 0.128 at 541 tokens, 0.079 at 16K and
  0.056 at 32K.
- The highest-yield lever was more self-distilled data. A multi-layer EAGLE-3-style head gave only +0.2% and was
  not adopted.

## 3. Fitting more context: KV formats (11-04 to 11-09)

| Format | KiB/position | Pool in ~5 GiB | Decode vs bf16 | Prefill vs bf16 | Quality |
| --- | ---: | ---: | ---: | ---: | --- |
| bf16 | 160 | ~32K | 1.00 | 1.00 | reference |
| KV8 (int8) | 82.5 | ~64K | 0.96-0.98 | 0.88-0.96 | KL 0.0019, 1% top-1 flips |
| KV4 (int4) | 42.5 | ~123K | ≈ int8 | 0.80-0.94 | KL 0.29, **14.6% flips** |
| **K5:V4** | **47.5** | **~110K (114,688 deployed)** | ≈ int4 | ≈ int4 | KL 0.084, 7.1% flips (sim); int8-level on tasks |

- **KV8 was nearly free.** It halved the KV for about 2-4% decode, and greedy output differs from bf16 only on
  near-ties.
- **KV4 was a pure capacity play.** It enabled 3 x 64K slots on a 120K pool, with two ~55K agents resident and
  311.6 tok/s aggregate on short chats.
- **But KV4 was the accuracy problem.** On the Menagerie golden set, int4 scored recall 80.3%, contradiction
  recall 63% and effective risk accuracy 91.7%. int8 scored 87.8%, 80% and 100%.
- **The simulator showed K carries the damage.** With K at int4 and V at int8 (4:8), the error is as bad as full
  int4. With K at int8 and V at int4 (8:4), 95% of the damage goes away. Spend bits on K first.
- **K5:V4 is the answer for this card.** K uses 5-bit codes, stored as int4 nibbles plus a high-bit plane; V stays
  int4. It still holds two ~55K agents.
  - Golden set: recall 86.9%, contradiction recall 100%, false positives 0%, risk accuracy 100%.
  - Needle-in-haystack at 7K/21K/55.7K: 9 of 9 correct.
  - Deployed as generation 793.
- **The live int4-vs-K5 gap was mostly the harness, not the format.**
  - Plain decode is equal (68.9 vs 68.8 tok/s at 512).
  - On held-out families, EAGLE is −1.6% / +0.7% / +3.4%.
  - Serving-style was −9%, but that set holds only 9 distinct prompts, so it is per-content variance.
- Measuring it surfaced a determinism finding: int4 rounding noise hurt answers, but forcing zero noise
  (re-prefill) also hurt, because retries and votes need some variation.

## 4. Heads on K5 features (11-09): nothing left to win

| Head | chat (K5 eval) | serving (K5 eval) |
| --- | ---: | ---: |
| r12 (int8 features, deployed) | **2.186** | **2.455** |
| r13k5 (3 of 5 sets, K5 features) | 2.154 | 2.389 |
| r14k5 (full mix, K5 features) | 2.182 | 2.431 |

- K5 features had to be dumped on the 4070 with the production binary. Dumps from the 5090 diverge (cosine 0.984).
- Training ran on the 5090 under its GPU lock, about 57 min per run. The Qwen3 runner was stopped for training and
  restarted automatically.
- r12 loses only ~1% when moved from int8 to K5 features, so a matched head has little room. Future head gains need
  more data, a bigger head, or different trees.

## 5. Verify attention on tensor cores (11-09, generation 795)

`granite4-attn-bench` showed the remaining engine lever. On the scalar kernels, a 14-row verify ran at 11% of the
K5:V4 bytes floor, and it took the same time in every format, so it was compute-bound. The weight GEMMs were already
at ~91% of roofline.

`attn_mma_kernel` is one `mma.m16n8k16` kernel for decode, chain verify, multi-session rows and tree rows. How it
works:
- Quantized tiles hold the raw integer codes. The K scale is applied to the logits and the V scale is folded into P.
- V fragments are loaded with `ldmatrix.trans`.
- The next tile's loads are pipelined behind the current one, and page-table rows are staged in shared memory.

| K5:V4, ms per 40 layers | 4K | 16K | 32K | 55K |
| --- | ---: | ---: | ---: | ---: |
| Tree verify, scalar → MMA | 5.56 → 2.36 | 22.96 → 8.54 | 46.12 → 17.63 | 77.67 → 30.15 |
| Decode, scalar → MMA | 0.87 → 0.80 | 3.35 → 3.21 | 7.33 → 7.11 | 11.96 → 11.50 |

| Live verify step (r12 tree, 14 rows) | 541 | 4K | 16K | 32K |
| --- | ---: | ---: | ---: | ---: |
| Attention, ms | 1.50 → 1.66 | 6.38 → 2.63 | 22.21 → 7.31 | – → 14.77 |
| Whole step, ms | 17.0 → 17.2 | 22.2 → 18.5 | 38.8 → 23.9 | – → 31.1 |

- The long-prompt family (10-13K-token prompts) went from 30.8 to 34.2 tok/s. Short chats are unchanged because
  they are weight-bound.
- Correctness checks:
  - The bit-identity device tests pass in all four formats.
  - Speculative output equals plain on every request.
  - The parity gate passes for K5 and bf16, with MMA on and off.
- Greedy text differs from the scalar build, because the arithmetic changed and K is no longer rounded to bf16
  after scaling. Error against the original values went down slightly.
- Things tried and kept or dropped:
  - More splits: no help; 256 stays.
  - Unrolled combine: no gain; reverted.
  - Masking instead of save/restore for tree rows: kept, since it removed register spills.

## 6. Prefill attention on code tiles (11-09)

The same trick applied to prefill. The quantized prefill kernel used to dequantize every K/V element with a scalar load, a multiply and a bf16 round, for every 32-query block of every 512-row chunk. It now stores raw codes and applies the scales to S and P, with vector loads, `ldmatrix.trans` for V and prefetch.

| K5:V4 prefill (`kv-bench`) | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| Before | 2.78 s | 7.39 s | 22.14 s |
| After | 2.26 s | 5.24 s | **13.39 s** |

A real 32K-token request (prefill plus 256 tokens) went from 27.1 to 18.1 s. The long held-out family went from 34.2 to 40.9 tok/s.

Also checked: draft-KV width (int4 / int8 / bf16) at long context changes acceptance by under 0.005. Long-context acceptance is a head and data problem.

## Where the time goes now (16K context, one verify step)

GEMMs 15.2 ms (the weight pass, ~590 GB/s), attention 7.3 ms, lm_head 0.7 ms, norms 0.5 ms, RoPE 0.2 ms.

Remaining levers, roughly in order:
1. **EAGLE acceptance at long context.** Done for now: r15long (trained on 3K–48K records) lifts long-record acceptance 27% offline and 32K decode 36% live. More long data, and longer records (64K+), are the next steps.
2. **Decode / verify attention at very long context.** It is ~25-30% of the bytes floor with MMA. Running more query rows per block,
   or reading K5 codes directly, could help.
3. **Heads.** Options are more self-distilled data (especially long-context and agent traffic), deeper heads, or
   retuned trees.
4. **Weights.** They are at ~95% of roofline. The only remaining lever is fewer bytes per token, and INT4 weights
   were rejected on quality.

## Production today (gremlin-1, generation 801)

- Engine 96d3459, flake input on `main`.
- K5:V4 KV (`kv5`), pool 147,456 positions, 3 sessions x 128K (the full checkpoint window; pool oversubscribed).
- EAGLE r15long (r12 + long-context records), k=4, tree drafting, draft KV int4.
- Tensor-core attention on (`G4_ATTN_MMA`), plus code-tile prefill attention (`G4_PREFILL_ATTN_CODES`). The parity gate runs at every start.
- Rollback: the previous generation, or set either env flag to `0`.
- Live, K5:V4 before the MMA change:

  | Measure | Result |
  | --- | ---: |
  | Prefill at 1K / 16K / 56K | 3,342 / 2,177 / 987 tok/s |
  | Decode at 545 / 7.9K ctx | 120.6 / 75.3 tok/s |
  | 3-client aggregate | 211 tok/s |
  | 56K prefix resume | 0.60 s, vs 58 s cold |

## Lessons that generalized

- **Measure the floor first.** The bandwidth roofline (GEMV) and the bytes floor (attention) told us which kernels
  were worth touching. Most wins came from finding a kernel far from its floor.
- **Make exactness structural.** One kernel per arithmetic path, plus per-row independence, means verify equals
  decode by construction. Tests then confirm it instead of chasing ULPs.
- **Quality proxies must reach the task.** int4 KV looked fine in unit tests (1 half-step error) and broke the golden
  set. The cheap simulator found K5:V4 before any kernel was written.
- **Small eval sets lie.** The 9-prompt serving set produced a −9% "regression". Check distinct-prompt counts before
  believing a delta.
- **Hardware topology matters for data.** Features dumped on a different GPU are different features.
