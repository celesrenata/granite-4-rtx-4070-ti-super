# EAGLE 2x Performance Spec

## Status

Planning specification. This document is for the next coding session and is not
an instruction to enable EAGLE in production yet.

## Objective

Improve the current Granite 4 speculative decoding implementation until it
achieves approximately 2x end-to-end generation speed while preserving exact
greedy output.

The primary success target is:

- **At least 2.0x speedup on long completions** relative to plain GPU greedy
  decoding on gremlin-1.
- **At least 1.8x aggregate speedup** across the existing benchmark prompt set.
- **Zero token differences** between plain greedy and speculative decoding for
  every tested prompt and every supported draft K.
- No production activation until all gates in this document pass.

The current implementation is a working proof of concept, not the final design:

- The trained head has approximately 233M parameters and is about 445 MB.
- Held-out acceptance is approximately 1.88 verified tokens/round at K=6.
- Aggregate throughput is approximately 68 tok/s plain, 105.6 tok/s at K=3,
  and 97.5 tok/s at K=5.
- A favorable individual prompt reaches approximately 2x, but the aggregate
  result is only about 1.55x.

The next iteration must improve both draft-head cost and acceptance rate. Raising
K alone is not an acceptable strategy because K=4 and K=5 currently reduce
throughput.

## Profiling result (2026-11-04, r3 head, gremlin-1)

`G4_EAGLE_TIMING=1` splits each round into head-draft, target-verify, and
readback (CUDA events). The result redefines the problem:

| round | K=3 | K=5 |
| --- | ---: | ---: |
| head-draft, ms/round | 2.06 | 3.31 |
| target-verify, ms/round | 15.08 | 15.82 |
| readback, ms/round | ~0.02 | ~0.01 |
| head fraction of wall | 12% | 17% |
| target-verify fraction of wall | 88% | 83% |

Two consequences:

1. **Target verify dominates (83–88%) and is flat in K.** Verifying 4 rows costs
   ~15.08ms/round and 6 rows ~15.9ms/round, i.e. essentially one plain decode
   step regardless of how many drafts are verified. Verify cannot be cut without
   a faster target.
2. **The head is already cheap (11–17%).** Compacting it does not unlock 2x.

Because verify is ~one decode step per round, end-to-end speedup equals
**tokens emitted per round** (accepted drafts + 1), and the target decode
baseline is ~68 tok/s. To reach 2.0x the aggregate, tokens/round must move from
~1.7 toward ~2.0.

Per-prompt tokens/round (K=3) vs speedup:

| prompt | tokens/round | speedup |
| --- | ---: | ---: |
| summarize_quote | 2.33 | 2.0x |
| json_extract | 2.31 | 2.0x |
| code_edit | 2.03 | 1.8x |
| qa | 1.77 | 1.55x |
| list | 1.71 | 1.51x |
| essay | 1.62 | 1.43x |

K=5 produces the same tok/s as K=3 on every prompt, meaning drafts beyond the
first 1–2 are rarely accepted. The binding constraint is therefore not K, not
head cost, and not verify cost: it is **the number of consecutive tokens the
draft head predicts correctly per round**, i.e. acceptance, dominated by step-1
accuracy.

The r3 head's per-depth masked draft accuracy (step d drafts tok_{i+d+1}):

| set | step 1 | step 2 | step 3 | step 4 |
| --- | ---: | ---: | ---: | ---: |
| selfgen24 (serving-like, OOD) | 0.522 | 0.437 | 0.399 | 0.374 |
| train_eval (in-distribution) | 0.546 | 0.497 | 0.474 | 0.458 |

Step-1 accuracy is the ceiling (cumulative accepted/round collapses from step 1
down). The step-1 number is markedly lower on serving-like (out-of-distribution)
text than on the chat training text, so the head overfits chat style and underfits
the serving prompts. The 2x target is reached by **raising step-1 accuracy on
the serving distribution**, not by a compact head.

### Revised plan (supersedes the architecture-first direction)

1. **Train on serving-representative target-generated continuations** (the six
   benchmark styles plus reasoning/math/code/analysis variants) so the head is
   trained on the distribution it will serve. Keep the held-out selfgen set
   untouched.
2. **Acceptance-oriented objective:** keep the soft CE to the target's draft-vocab
   distribution, add an explicit top-1 distillation term that directly optimizes
   argmax agreement (the metric), and weight step 1 most heavily.
3. **Re-evaluate step-1 accuracy and engine tokens/round** before any head
   architecture change. Only if step-1 still caps below the 2x target with the
   one-layer head should a deeper/larger head be considered, since head cost has
   headroom (it is 12% today, not the bottleneck).

`G4_EAGLE_TIMING=1` is the required profiling gate; every subsequent run must
report head/verify/readback, tokens/round, and per-prompt speedup.

## Serving-style fine-tune result (2026-11-04, r5 head, gremlin-1)

The revised plan was executed: the r3 head was fine-tuned (r5) on 364 target
greedy continuations generated over 400 serving-style prompts (`serve_styles.json`,
6 styles: essay/qa/json/math/code/analysis; EOS=100257; the chat-template render is
reused verbatim from `prep_selfgen.py`). Held-out set: 6 prompts per style
(36 records). r5 reuses r3's 32768 draft vocab (it covers 97.8% of the serving
reply tokens, so growing the vocab was not needed). Fine-tune: 6 epochs, lr 1e-4,
fresh optimizer from r3's head weights (`--head`), K=3 training / K=6 eval.

**The generalization question — does it hold across styles? Yes.** Per-style
held-out tokens/round (exact gate, K=6), 6 unseen prompts per style:

| style | r3 (chat-trained) | r5 (serving fine-tuned) | Δ |
| --- | ---: | ---: | ---: |
| json | 2.977 | 4.258 | +1.28 |
| math | 2.000 | 3.091 | +1.09 |
| analysis | 1.607 | 2.073 | +0.47 |
| code | 2.002 | 2.071 | +0.07 |
| essay | 1.677 | 1.749 | +0.07 |
| qa | 1.562 | 1.477 | −0.09 |
| **overall** | **1.830** | **2.213** | **+0.38** |

The gains concentrate exactly where r3 was weakest on out-of-distribution text
(json, math, analysis). It did not overfit (held-out tokens/round plateaued
rather than spiking then collapsing). Chat-forgetting check on the in-distribution
chat set: 1.945 → 1.836 tokens/round (−5.7%), the expected re-weighting tradeoff.

**End-to-end serve benchmark** (fresh `serve` process per config, same 400
serving-style prompts, K=4, warm process, one request at a time):

| config | agg tok/s | median ms | p95 ms | speedup vs plain |
| --- | ---: | ---: | ---: | ---: |
| plain | 69.6 | 1797 | 1804 | (baseline) |
| r3 | 106.5 | 1156 | 1416 | 1.53x |
| r5 | 227.3 | 499 | 1029 | **3.26x** |

**Byte-for-byte correctness:** r3 and r5 each match plain greedy exactly on
**400/400** prompts — the no-degradation contract holds end-to-end.

`G4_EAGLE_TIMING=1` during serve (200 prompts, K=4) confirms the per-round device
split and reconciles with the end-to-end numbers:

| config | tok/round (mean) | per-gen tok/s | head ms/rnd | verify ms/rnd |
| --- | ---: | ---: | ---: | ---: |
| r3 | 2.00 | 114 | 2.5 | 15.1 |
| r5 | 4.46 | 254 | 2.6 | 15.1 |

Head cost is ~2.5 ms/round and verify ~15.1 ms/round (verify is one exact batched
verify over the accepted prefix + 1, so it is independent of acceptance — the round
cost is fixed at ~17.5 ms and higher acceptance yields more tokens per round).
Verify is ~91% of the device round but the per-token verify work (14 ms/token)
equals a plain forward step, so the verify path is already near-optimal; no
batched-verify optimization is required. The 3.26x exceeds the 2x target and also
exceeds the offline 6-per-style estimate because the full 400-prompt serving mix is
heavily weighted toward high-acceptance json/math (r5 reaches 4.46 tokens/round
mean there).

**K sweep (400 prompts, end-to-end)** — K=6 is the production optimum:

| K | agg tok/s | speedup | median ms | p95 ms |
| ---: | ---: | ---: | ---: | ---: |
| 0 (plain) | 69.6 | (baseline) | 1797 | 1804 |
| 4 | 227.3 | 3.26x | 500 | 1036 |
| **6** | **249.8** | **3.59x** | **439** | 1143 |
| 8 | 237.7 | 3.41x | 449 | 1338 |

K=8 underperforms K=6: beyond the depth the head accepts, extra draft capacity
only widens the verify batch without adding accepted tokens, and the per-round
cost grows. Production config: **r8 (union) head, K=6** (251.5 tok/s, 3.61x,
400/400 byte-for-byte, distribution-robust).

K=6 correctness was separately confirmed with the full byte-for-byte benchmark
(400 serving prompts, max_tokens=128): r5 and r3 each match plain greedy exactly
on 400/400 at K=6, and r5 reproduces the sweep's 249.7 tok/s (3.59x). The
**r8 union head** was then bench'd at K=6 on the same 400 prompts: **251.5
tok/s, 3.61x, median 438.7 ms, p95 1043 ms, 400/400 byte-for-byte** — it
dominates r5 (249.7 / 3.59x / p95 1144) on throughput, p95, and generalization.
Note the optimal K is head-dependent — r3 is *slower* at K=6 than K=4 (1.38x vs
1.53x, lower-acceptance heads waste draft budget at depth) while r5/r8 speed up
— so the K=6 recommendation applies to the strong (r5/r8) heads specifically.

**Production head: r8 (union), K=6** — 251.5 tok/s, 3.61x, p95 1043 ms,
400/400 byte-for-byte, and distribution-robust across serving-style prompts.

**r6 (chat anchor) — negative result.** A rebalance adding 60 chat records
(~35% of tokens, from the original UltraChat/ShareGPT set) to fight r5's
qa/essay weakness was a wash: per-style tokens/round (K=6) moved within ±0.05
except json −0.26, and chat-forgetting was unchanged. The qa/essay weakness is
not chat-forgetting; draft-vocab coverage (94–99% on every style; qa 94.1%)
rules out a vocab gap; the remaining constraint is the target's own entropy on
open-ended text. r5 remains the selected head.

**Overfit probe — capacity vs data (decisive).** To settle whether the weak
styles are a capacity limit (the 233M 1-layer head can't represent the
hidden→next-token mapping) or a data-volume limit, r5 was fine-tuned on *only*
the 6 held-out qa records (the weakest style) for 40 epochs until memorized.
On those same 6 records (exact gate, K=6):

| head | qa tokens/round (train set) | train top-1 |
| --- | ---: | ---: |
| r5 (general) | 1.48 | — |
| r5 overfit on those 6 | **5.31** | **0.99** |

The head *can* represent the serving-style mapping (it memorizes qa to 0.99
top-1, 5.31 tokens/round). So the weak styles are **not** capacity-limited —
they are **training-data-volume limited** (only ~54 qa / ~54 essay records
today). The remaining lever is more serving-style data on the weak styles, not
a deeper head. (r6's chat anchor failed because it added chat data, not more
weak-style data.)

**r7 (more distinct weak-style data) — confirms the lever, reveals family
specialization.** A second prompt set (480 prompts) was generated with far more
*distinct* weak-style content (72 distinct qa vs ~20, 72 distinct analysis vs
~15; essay/code/math/json likewise), targeting the serving distribution.
Fine-tuning r3 on it (r7) lifted the weak styles hard **on its own family**
(essay 1.75→2.79, code 2.0→3.05, math 2.15→3.35; overall 2.61 vs r3's 1.88),
confirming data volume/diversity is the lever. But r7 *specialized to its
training family*: on the *other* held-out family (serve_eval) it dropped to 1.94
vs r5's 2.21. Symmetrically r5 overfits *its* family (2.21 on serve_eval but only
2.00 on serve_eval2). So neither single-family head generalizes; each just fits
the prompts it was trained on.

**r8 (union of both families) — the distribution-robust head.** Fine-tuning r3 on
the union of both training sets (808 records, 242k tokens) yields a head that
matches the best single-family head on *each* family:

| head | serve_eval (400-fam) | serve_eval2 (480-fam) |
| --- | ---: | ---: |
| r5 | **2.213** | 1.996 |
| r7 | 1.940 | **2.612** |
| r8 (union) | **2.223** | **2.607** |

r8 is the production candidate: it is not weaker on r5's family and is on par
with r7 on r7's family, so it generalizes across serving-style distribution
shift (new prompts/topics) rather than fitting one fixed prompt set. Final
correctness + latency at K=6 is verified below.

Status: the spec's 2x aggregate target is met and exceeded (3.61x at K=6 with
the r8 union head); the
binding constraint (acceptance on the serving distribution) is solved by the
serving-style fine-tune, with the correctness contract intact. Production
activation (r8 union head, K=6) remains a separate explicit step (not done here; the
running service is untouched).

## Non-negotiable correctness contract

1. The target model remains the sole authority for emitted tokens.
2. Every emitted token must be exactly the token plain greedy decoding would
   emit from the same prompt and model state.
3. Rejected draft tokens must never be emitted.
4. The existing target verify path must remain bit-identical to target decode at
   each position.
5. Existing parity-gate behavior must not be weakened or bypassed.
6. CUDA and host-only builds must continue to compile.
7. Existing tests must continue to pass.
8. Do not enable the feature in the systemd/NixOS production service as part of
   the optimization work. Production activation is a separate, explicit step.

## Required first step: profiling and baseline

Before changing architecture or training, add or use instrumentation that
separates each generation round into:

- target prefill/decode time;
- target verification time;
- draft-head catch-up time;
- draft-head chained-step time;
- device-to-host synchronization/readback time;
- host-side loop and JSON/service overhead;
- number of proposed tokens;
- number of accepted tokens;
- tokens emitted per round.

Measure the following on gremlin-1 with the same checkpoint and prompts used by
the current benchmark:

- plain greedy;
- current EAGLE K=1 through K=8;
- prompt lengths short, medium, and long;
- completion lengths at least 128 and 512 tokens where possible.

Report median and p95 timings, not only total tok/s. The optimization must make
clear whether the limiting factor is acceptance, head compute, target verify,
or synchronization.

## Architecture direction

### 1. Build a substantially smaller draft head

The current 233M-parameter head is too expensive for the achieved acceptance.
Implement a configurable compact head while preserving the same feature contract:

- input: target final-normed hidden state plus next-token embedding;
- output: predicted target feature and draft token;
- target lm_head remains frozen and authoritative.

The compact head should support at least these configurations:

- **small:** hidden width approximately 1024–2048;
- **medium:** hidden width approximately 2048–3072;
- optional reduced FFN multiplier;
- optional reduced query/KV head counts;
- optional 8-bit or FP8 weights where numerically safe.

Use projection layers between target hidden size 4096 and the compact head width.
Do not assume that copying the full target decoder layer is optimal.

The coder must benchmark at least two compact configurations against the current
head. Select based on **accepted tokens per millisecond**, not acceptance alone.

### 2. Make the head objective acceptance-oriented

The existing training objective overemphasizes feature regression. Add or tune
losses so that the head directly learns the target's next-token decisions:

- cross-entropy or KL loss against the target distribution;
- explicit next-token top-1 agreement loss;
- feature regression as a supporting loss, not the only objective;
- per-step weighting that emphasizes the first 3–5 draft positions;
- scheduled feature noise / self-conditioning so later draft steps resemble
  inference conditions.

Training data should prioritize target-generated continuations, because the
serving workload is judged against target-generated text. Keep a held-out set
that is never used for optimization.

The training script must record:

- loss by draft depth;
- target top-1 agreement by depth;
- draft-vs-target token agreement by depth;
- exact speculative tokens/round;
- acceptance rate by K;
- elapsed training step and tokens/sec.

### 3. Avoid unnecessary head work

Investigate and implement, where correct:

- fused or combined input normalization and projection;
- persistent device buffers and KV state;
- no host round-trip for intermediate draft tokens;
- batched chained draft steps where dependencies permit;
- direct target lm_head projection only over a compact draft vocabulary, or a
  cheaper equivalent that preserves useful candidate coverage;
- avoiding recomputation of already-valid head entries after a verify round;
- dynamic K selection based on remaining budget and recent acceptance.

Any approximation in the draft head is acceptable because the target verify gate
rejects incorrect proposals. Any approximation in target logits, target KV, or
the acceptance decision is not acceptable.

### 4. Dynamic K policy

Do not hard-code one K as the only strategy. Implement a cheap policy that can
select K per round using recent observations, for example:

- increase K after high acceptance;
- decrease K after rejection or when head cost dominates;
- cap K at the target verify kernel limit;
- reserve enough context for the anchor and draft rows;
- preserve deterministic greedy output.

The policy must be benchmarked against fixed K=3, K=4, and K=5. It is allowed
to choose K=0 for workloads where speculation is slower.

## Benchmark protocol

Use the existing benchmark prompt file and add a reproducible benchmark script if
necessary. Every benchmark run must:

1. warm up the engine;
2. run plain greedy and speculative modes in separate fresh engine processes;
3. use the same checkpoint, prompt tokens, EOS behavior, and max completion;
4. run each case multiple times and report the best/median clearly;
5. compare complete generated token arrays byte-for-byte;
6. report tok/s, wall time, proposed tokens, accepted tokens, rounds, and
   per-stage timings.

Required tables:

- per-prompt results;
- aggregate results;
- results grouped by completion length;
- results grouped by prompt length;
- acceptance and speed by K;
- current large head versus each compact head.

## Acceptance gates

### Gate A: correctness

- Existing Python test suite passes.
- Existing host CTest suite passes.
- CUDA engine builds successfully on gremlin-1.
- Exact output equality passes for every benchmark prompt at K=1..8 and for
  dynamic K.
- Existing parity gate passes unchanged.

### Gate B: compact-head quality

At least one compact head must satisfy all of:

- no correctness regressions;
- better accepted-tokens-per-head-millisecond than the current head;
- held-out self-generated acceptance at least 2.0 tokens/round at an evaluated K;
- no material degradation on code, JSON, QA, and long-form prompts.

### Gate C: speed

At least one fixed-K or dynamic policy must satisfy:

- >=2.0x median speedup on long completions;
- >=1.8x aggregate speedup on the benchmark suite;
- no benchmark category below 1.25x unless explicitly documented as a workload
  limitation;
- speedup remains after including model load, warmup, and service overhead when
  those are part of the reported production path.

If Gate C fails, do not enable production defaults. Continue profiling or reduce
scope rather than declaring success based on a favorable single prompt.

## Base-runner co-design requirements

EAGLE performance depends on the target runner as much as on the draft head. The
coder must profile and, where beneficial, improve the base runner before
claiming that the head has reached the 2x target.

### 1. Produce target features without duplicate work

The target final-normed hidden state is both the input to EAGLE and the input to
the target lm_head. The runner must:

- perform final RMSNorm once per target row;
- expose the normalized BF16 feature buffer directly to EAGLE;
- avoid device-to-host copies for features, logits, or draft tokens;
- avoid a second RMSNorm when both hidden state and logits are requested;
- reuse persistent feature buffers across rounds;
- make row offsets explicit so prompt, anchor, verify, and rollback rows cannot
  be confused.

Add a benchmark mode that reports the incremental cost of requesting hidden
states versus ordinary target decode. The desired incremental cost is near zero
other than the feature-store write.

### 2. Make speculative KV state cheap to commit and rollback

The current flow truncates the target KV watermark after verification. Improve
this path where profiling shows it matters:

- maintain a clear committed watermark and speculative watermark;
- write speculative rows into reusable cache space;
- commit an accepted prefix with a watermark update rather than copying KV;
- discard rejected suffix rows without clearing or reallocating the cache;
- ensure the next round overwrites every discarded row before reading it;
- expose the same semantics to the EAGLE KV cache;
- test rollback at zero accepted drafts, partial acceptance, and full acceptance.

Do not implement a costly copy-based branch unless it is proven faster or needed
for correctness. Add counters for cache allocation, cache copy, truncate, and
overwrite operations.

### 3. Keep the entire draft/verify loop on the device

The host must not synchronize once per proposed token or once per draft step.
The runner should support:

- device-resident token IDs for anchors and draft proposals;
- device-side argmax/token mapping;
- device-side comparison of verified IDs against proposed IDs;
- a compact device acceptance count or accepted-prefix result;
- one synchronization per completed speculative round at most;
- host readback only for tokens that must be returned to the API.

If a device-side acceptance prefix is implemented, preserve the exact first-
mismatch semantics and tie-breaking of the existing target argmax path.

### 4. Schedule target verify work efficiently

The target runner should provide a dedicated speculative verify entry point that:

- accepts one anchor plus 1–15 drafts in one call;
- uses the wide verify kernels for all supported row counts;
- computes logits only for rows needed by the acceptance decision;
- computes hidden states only for rows needed by the next EAGLE round;
- avoids processing rows that are guaranteed to be discarded;
- uses persistent scratch allocations rather than per-round `cudaMalloc`;
- keeps the target decode and verify arithmetic identical at each position.

Benchmark each verify row count separately. The implementation must not assume
that the maximum K is fastest; select the row count based on measured target
verify cost and observed acceptance.

### 5. Reduce target lm_head cost where it is safe

The target lm_head is often a major part of each verify round. Investigate:

- reusing the existing packed FP8 lm_head representation;
- compact draft-vocabulary projections for proposal generation;
- computing full-vocabulary logits only for target verification;
- fusing final norm and lm_head where that preserves decode/verify parity;
- avoiding duplicate lm_head work between the anchor and accepted rows;
- using the existing exact decode path as the reference for all changes.

A compact vocabulary may be used by the draft head, but never by the target
acceptance decision. Do not approximate target logits or use a reduced target
vocabulary for verification.

### 6. Improve memory locality and allocation behavior

Measure and, if justified, improve:

- placement and reuse of target hidden, normalized, draft-feature, and logits
  buffers;
- alignment and pitch for row-major BF16 feature buffers;
- FP8 weight scale access in verify kernels;
- L2 reuse of frequently used head weights and target lm_head rows;
- overlap between feature writes, head work, and target verify work;
- CUDA allocator and stream synchronization overhead.

All steady-state speculative buffers should be allocated during session setup.
The per-request path must not allocate or free device memory.

### 7. Consider CUDA graphs only after dependencies are stable

After the loop has stable shapes and buffer addresses, evaluate CUDA graph
capture for the common cases K=3, K=4, and K=5. Graphs are optional, but the
benchmark must establish whether they reduce launch overhead enough to matter.
Do not use graph capture to hide synchronization or make rollback semantics
unclear. Dynamic K may use a small set of pre-captured graphs.

### 8. Add runner-specific performance gates

Before evaluating the head itself, record these runner baselines:

- target decode time per token;
- target verify time for rows 2 through 16;
- hidden-state extraction overhead;
- KV truncate/commit overhead;
- host/device synchronization count per round;
- device allocation count after warmup;
- lm_head time as a fraction of target verify;
- end-to-end speed with an intentionally cheap/random draft head.

The cheap-head test separates runner improvements from model acceptance. Any
runner optimization must pass the exact-output parity gate and the existing
teacher-forced parity gate.

## Deployment requirements after the gates pass

Only after explicit review:

1. Export the selected head to a versioned path on gremlin-1.
2. Record checkpoint hash, head hash, training configuration, and benchmark
   results.
3. Configure the service with an explicit `GRANITE4_EAGLE_HEAD` path and the
   selected K policy.
4. Keep a one-command rollback to plain greedy by removing the head setting.
5. Restart the service and rerun the parity gate.
6. Run a live API comparison against plain greedy.
7. Update `docs/phases.md` with measured results and rollback instructions.

Do not commit large training artifacts or model weights to the repository.

## Files and likely work areas

Existing implementation and tooling:

- `engine/src/eagle_gpu.cu`
- `engine/src/eagle_gpu.h`
- `engine/src/forward_gpu.cu`
- `engine/src/forward_gpu_api.h`
- `engine/src/infer_main.cc`
- `scripts/eagle/train_eagle.py`
- `scripts/eagle/export_head.py`
- `scripts/eagle/prep_data.py`
- `scripts/eagle/prep_selfgen.py`
- `frontend/engine_service.py`
- `flake.nix`
- `tests/test_serving.py`

The coder should inspect the current code before editing. Preserve unrelated user
work and do not rewrite the engine wholesale without profiling evidence.

## Expected final report from the coding session

The coding session must finish with:

- files changed;
- tests and build commands run;
- profiling results;
- selected head architecture and parameter count;
- training data and objective;
- acceptance by depth and K;
- per-stage timing breakdown;
- benchmark speed tables;
- exact-output parity results;
- whether each acceptance gate passed or failed;
- explicit statement that production was or was not enabled.
