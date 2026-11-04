# Phases

Each phase has exit criteria; a phase is done only when its criteria are met
on a real sm_89 machine. Until P3 exits, the API keeps answering
`unavailable` / 501.

## P0 — Scaffold (this commit) ✅

- API surface with validation and an explicit unavailable boundary (400/501).
- Model constants in both languages, cross-checked by tests.
- C ABI metadata/status placeholder; host-only C++ test.
- CMake builds the host side without a CUDA toolkit; CUDA path targets sm_89.
- Docs: architecture, phases, memory budget (planning figures only).

Exit criteria: `pytest` and `ctest` (host-only) pass on a machine without CUDA.

## P1 — Weights (host side complete)

- Safetensors loader; parse checkpoint `config.json` and reject on any
  deviation from the spec (hard fail, listed field-by-field).
- Decide storage: FP8 (weights) is the working default — see
  [memory.md](memory.md); BF16 does not fit the card.
- Export a quantization pass for activations where kernels need it.

Exit criteria: full checkpoint loaded and checksummed; `nvidia-smi` memory
matches the plan within 5%; config mismatch on a perturbed checkpoint fails
fast with the exact field list.

**Done (host, no CUDA):** `granite4-checkpoint verify <dir>` streams the
whole checkpoint with bounded memory and reports per-shard and per-tensor
SHA-256, exact tensor count / shapes / dtypes against the derived expected
set, and a strict config.json check (field-by-field deviations, non-fatal —
a wrong config is reported, it does not abort the checksum pass). The real
8B checkpoint verifies clean: 4 shards, 363 tensors, 17,583,185,920 bytes,
config_ok, zero issues. `granite4-checkpoint export <dir> <out.safetensors>
[--per-channel]` writes a self-describing FP8 e4m3 checkpoint (weights +
per-tensor or per-channel F32 scales) that parses back through the
safetensors manifest. Both are covered by `ctest` (5 suites) and cross-checked
against `python hashlib`. `head_dim` is derived (`hidden_size / heads`) in
both the C++ parser and `scripts/inspect_checkpoint.py`, matching the real
config.json which omits it. The remaining exit criterion — `nvidia-smi`
residency within 5% of the plan — is deferred to the first sm_89 run.

## P2 — Kernels (sm_89) ✅

- GEMM (FP8 e4m3 weights; Ada has no Blackwell-style block-scaled MMA, so
  scales are per-tensor/channel — decided here), attention with GQA
  (32 Q / 8 KV, head dim 128), RMSNorm, rotary, activations, sampling.
- Each kernel unit-tested against a CPU reference on the same seed.

Exit criteria: per-kernel correctness on randomized inputs; micro-benchmark
script with the bandwidth/roofline baseline measured (not assumed).

**Done (verified on gremlin-1, RTX 4070 Ti Super / sm_89, CUDA 13.3):** the
portable reference math (`engine/src/kernel_math.h`) and the device kernels
(`engine/src/kernels_phase2.cu`) for the full P2 set — **RMSNorm, RoPE
(half-split), scaled softmax (attention-multiplier logits), FP8-e4m3 × BF16
GEMM with per-tensor dequant, GQA attention (Q·Kᵀ + value accumulation, gh =
Hq/Hk), fused SwiGLU gate/up, and sampling (deterministic greedy + top-p)** —
each checked against the reference with seeded randomized inputs (1–4 ULP
tolerance; sampling is an exact discrete match), plus the sm_89 arch probe.
Greedy is additionally verified at the full 100352-vocab size. Host reference:
`kernel_math` ctest (runs everywhere). Device: `kernels_device` ctest
(`GRANITE_ENABLE_CUDA=ON`).

**Micro-benchmark / roofline baseline:** `granite4-kernel-bench`
(`engine/src/kernels_bench.cu`) measures the card's *actual* streaming read
and copy bandwidth with CUDA events and prints the roofline note (decode is
read-bound; tokens/s = read_gbps / fp8 weight bytes per token) as JSON. The
baseline is measured on the card, not assumed. Per the cross-phase rule, the
measured numbers are kept out of the repo until P4; they are reproducible
from the binary (below).

**GPU forward pass (sm_89):** `engine/src/forward_gpu.cu` composes the
device-verified primitives into a full model forward with two speed-critical
structures — `gpu_weights_load` uploads the ~9.5 GB packed weights
(`GpuWeightsLayout`: FP8 e4m3 GEMM weights + BF16 embed/norms) once and
persists them on the device (dequant scales stay in pinned host memory;
kernels take them by value), and `gpu_forward_range` decodes **incrementally**:
only positions `[written, end)` run the stack, the KV history persisting in
the per-sequence ctx caches. `granite4-infer forward/generate --gpu 1` use it;
`--gpu 0` is the host oracle (the parity reference). Verified on gremlin-1:
5 positions x 100352-vocab forward matches the host oracle (max |dlogit|
1.18 BF16, top-1 equal at every position) and 32-token greedy generation is
token-for-token identical to the host path. Decode speed is ~255 ms/position
(~4 tps) with the one-thread-per-element GEMMs; the 50 tps target is a P4
item (tiled GEMM + fused attention), not a P2/P3 gate (per the roofline note
above).

**Folded into the GPU host's flake:** `packages.x86_64-linux.engine-cuda`
compiles the whole sm_89 build — kernels, the device-test binary, and the
micro-benchmark — and `doCheck` runs every ctest suite *except* the GPU-only
`kernels_device` run (that needs the host driver's `libcuda` plus a live
device, which a build sandbox cannot reach). The GPU host re-verifies the
device kernels with one command:
```
granite4-kernel-test     # device correctness (== the kernels_device ctest)
granite4-kernel-bench    # measured bandwidth / roofline baseline (JSON)
```

## P3 — Integration

- Scheduler (N concurrent sequences), prefill/decode loop, paged KV, prefix
  cache; memory manager wired to the plan in [memory.md](memory.md).
- Golden parity: fixed prompts, seeded sampling, compared token-for-token
  against the reference implementation (transformers/other engine) with
  quantization noise bounded per spec.

Exit criteria: parity gate passes; the API flips to serving; `/health`
reports `inference: available` only after the gate.

**In progress:** the paged KV memory manager + prefix cache
(`engine/src/kv_pool.{h,cc}`) and the GPU paged KV compaction kernel
(`engine/src/kernels_phase3.cu`) are done and verified:

- The host manager (7/7 ctest, host-only build) is a paged pool sized from
  the [memory.md](memory.md) budget, token-major byte layout, page-aligned
  prefix sharing (ref-counted, content-addressable with a token-verified
  collision guard), and LRU eviction of retained (unreferenced, still-cached)
  pages.
- The device kernel gathers one layer's K/V out of the paged FP8 pool into
  the contiguous `[Hk, kv_len, d]` BF16 the P2 attention kernels consume
  (FP8→BF16 dequant folded in). It is checked bit-exactly against the host
  reference `paged_kv_compact_bf16` - device-verified on gremlin-1
  (`granite4-kernel-test`), and the paged pool's byte layout is pinned by the
  host test.
- The deterministic **forward pass** `forward_logits_f32` (all 40 layers,
  float) is implemented and host-verified as the parity oracle. It composes
  only the device-verified primitives (RMSNorm, Llama-half-split RoPE,
  `gqa_attention_bf16`, SwiGLU), takes the dequantized FP8 weights, and runs
  causal attention over the whole history at each position. Three tests pin
  it: a zero-projection passthrough (output must equal
  `lm_head @ rmsnorm(embedding)` computed by hand, proving the full 40-layer
  loop + residual + final norm + head are wired correctly), a causality test
  (a later token must not change any earlier position's logits), and an
  independent float64 reference on a tiny model (catches weight-row mapping
  bugs that trivial 1-token cases hide).
- The **weight loader** (`engine/src/weights.{h,cc}`) reads the original BF16
  checkpoint and the P1 FP8 export (per-tensor and per-channel `__scale`
  dequant), and `granite4-infer forward|generate` exposes the host oracle.
  `granite4-weights-test` pins BF16 decode and both FP8 dequant paths against
  hand-computed values.

**Parity gate (token-for-token, P3 exit criterion):**
`scripts/parity_gate.py` compares `granite4-infer generate` (greedy) against
a float32 transformers reference on a fixed prompt, token-for-token. On the
real checkpoint the engine matches the float32 oracle
**24/24 tokens** for "The capital of France is" (greedy next-token
decisions identical at every step), so `--min-match 1.0` passes. The gate
writes `.gt/parity_gate.json` on PASS; the API loads that artifact and only
then reports `inference: available` (docs: `frontend/api.py`,
`frontend/engine_service.py`).

**Storage path, end to end:** the P1 FP8 export (`granite4-checkpoint
export`) produces the 8.2GB `granite4.fp8.v1` file, and `granite4-infer`
loads it (per-tensor `__scale` dequant) and runs the forward from it -
argmax on a real prompt matches the BF16-checkpoint run, so the FP8
storage format is a first-class serving input, not just a P1 artifact.
The tiny-model regression test (`test_forward_reference_tiny`) now also
pins the QKV row mapping against an independent float64 reference - it
caught a reference bug on first run (the reference indexed embeddings by
position instead of token id) and verifies the fix.

Not yet done (the GPU inference half of P3): loading the FP8 weights onto
the GPU, running the forward pass on the GPU with the paged KV pool, the
prefill/decode loop, and the concurrent-sequence scheduler. The host oracle
and the parity gate stand in for it; on a real sm_89 machine the gate must
pass there too before P3 exits (the API flips to serving off the gate
artifact either way). P4 (benchmarking) then tunes the GPU path.

## P4 — Performance

- Kernel and scheduling tuning at a fixed power limit.
- Benchmark harness: decode (code/JSON/prose), concurrent throughput, long
  context; identical conditions and scripts per comparison.

Exit criteria: benchmark table lands in `docs/` **with reproduction
commands**; only numbers measured in this phase may be stated anywhere.

### P4 result (measured on gremlin-1, RTX 4070 Ti SUPER, sm_89, CUDA 13.3,
16 GB, stock clocks/power; build `nix build .#engine-cuda`)

| Quantity | Value | Conditions |
|---|---|---|
| HBM read bandwidth | 631.9 GB/s | `granite4-kernel-bench`, 256 MB read, 100 iters |
| HBM copy bandwidth | 602.7 GB/s | `granite4-kernel-bench`, 256 MB copy, 100 iters |
| Per-token weight traffic | 8.38 GB | FP8 8B: 40 layers x 3.22 GB + lm_head 0.41 GB |
| Decode step (GPU, steady) | **15.2 ms/step ≈ 65 tps** | 5-token prompt, greedy, `--gpu 1`, steady-state steps 27–28 of a 24-token generate with `G4_TIMESTEP=1` (one event pair per call; the `G4_TIMESTEP=2` bucket breakdown adds ~440 event records and reads ~2.3 ms slower, so use it for attribution only); same-run A/B: 18.5 ms (P4 build) → 18.3 ms (16-byte GEMV build) → 18.1 ms (128-thread launch policy) → 17.8 ms (fused QKV) → 17.7 ms (residual adds fused into the O/down GEMV epilogues) → 17.5 ms (fused rope q/k and KV append+compact) → 16.5 ms (fused decode GQA) → 16.1 ms (direct-cache GQA) → 15.75 ms (warp-tree RMSNorm) → 15.2 ms (parallel softmax exp), token-identical |
| GEMV-only decode | 20.3 ms/token (bench) | `granite4-gemv-bench --iters 120`, L2-flushed, real per-layer shapes |
| Roofline ceiling | ≈75 tps | 631.9 GB/s ÷ 8.38 GB/token |

GEMV effective bandwidth by per-layer shape (`granite4-gemv-bench
--iters 120`, GB/s): q 381 / k 261 / v 258 / o 374 / gate 433 / up 433 /
down 417 / lm_head 565.

Per-bucket attribution, `G4_TIMESTEP=2` (ms/step, steady step 28, current
build; instrumented - the ~440 event records add ~2.3 ms to the 17.5 ms
total, and the bucket run itself drifts ~0.1 ms run-to-run):
`attn_g=2.47 attn_a=0.24 attn_o=1.89 ffn_g=7.40 ffn_o=3.85 lm=0.69` plus
per-GEMV: `gqkv=1.93 go=1.37 ggateup=6.85 gdown=3.64 glm=0.68`. The GEMV
spans sum to ~14.5 ms (83% of the instrumented total); the rest is
attention, RMSNorm/residual glue, and KV append.

The Q/K/V projections now share one launch over the concatenated q/k/v row
ranges, the O-projection and FFN-down GEMVs fold their residual adds into
the thread-0 epilogue (writing `d_hidden` in place), the attention block
fuses rope q/k into one launch and the KV append+compact (k and v) into
one, and the decode GQA (logits -> softmax -> value) runs as one
block-per-head kernel that keeps the bf16 logits and probabilities in shared
memory. The fused GQA needs `d + 2*kv_len` bf16 of shared memory (48 KiB
default limit, ~24 K of context); beyond that a `gqa_fusable` flag falls
back to the classic three-kernel scratch path. The `attn_o` bucket fell
3.31 → 2.44 ms.

The three decisive P4 fixes (all re-verified by `granite4-kernel-test`; the
last two are additionally token-identical end-to-end):

1. **rmsnorm block-reduction** — the one-thread-per-element form re-read the
   whole row per element (O(cols²), ~33 MB redundant HBM traffic per call at
   cols=4096). One block per row with a shared-memory tree reads the row
   once; step went 55.7 ms → 24.5 ms (bucket-instrumented, as are the P4
   A/B numbers below; the uninstrumented step reads ~2.4 ms lower).
2. **Branchless e4m3 decode** — the branchy `ldexp`-based `fp8_to_f32` was
   the per-element ALU cost of every dequant GEMM. The bit-mapped form
   (`((120+e)<<23)|(m<<20)`, predicated selects for subnormals/top/sNaN) is
   bit-identical on all 256 codepoints (`test_fp8_decode_all`) and lifts the
   mid shapes from ~330 to ~400 GB/s effective; in the bucket-instrumented
   same-run A/B the step went 24.5 ms → 21.7 ms (−2.75 ms,
   token-identical output).
3. **16-byte (uint4) GEMV loads + 128-thread blocks** — each lane loads 16
   fp8 bytes per iteration instead of one uint32 of 4, so the kernel issues
   4x fewer global loads and each lane has 4x the bytes in flight; effective
   read bandwidth rose from 381 to 413 GB/s per token (q 330→381, k
   234→261, gate 400→433, lm_head 517→565 GB/s) and the GEMV-only bench
   from 22.1 to 20.3 ms/token. With 16-byte loads a lane covers one uint4
   per iteration at K=4096, so 128 threads (not 256) leave no idle lanes;
   the small-grid Q/K/V shapes gain most. The bf16 outputs are bit-identical
   to the uint32 form on every decode shape (exhaustive per-shape
   comparison), and a same-run A/B (uninstrumented step, `G4_TIMESTEP=1`)
   went 18.5 ms → 18.3 ms (16-byte loads) → 18.1 ms (128 threads), all
   token-identical 24-token generates.
4. **Fused Q/K/V GEMV launch** — Q, K, and V now share one grid over their
   concatenated row ranges while retaining separate weights, scales, and
   output buffers. This removes two launch ramps per layer; the in-context
   qkv span went 2.21 → 1.92 ms and the full uninstrumented step went 18.1
   → 17.8 ms, token-identical.
5. **Residual add fused into GEMV epilogues** — the O-projection and
   FFN-down GEMVs each wrote a full bf16 vector only for `add_resid_kernel`
   to read it back and overwrite `d_hidden`; both now compute
   `bf16(hidden + resid_scale * y)` in the same thread-0 epilogue (the
   `gemv_row_dequant<true>` template path), removing the extra launch and
   the intermediate buffer per layer. Same-run A/B: 17.78-17.83 →
   17.61-17.72 ms/step, token-identical (the epilogue keeps the intermediate
   `bf16(scale * sum)` rounding, so outputs stay bit-identical).
6. **Fused rope q/k and KV append+compact** — the attention block launched
   rope for q and k separately and then four KV micro-kernels (append k,
   append v, compact k, compact v); at Hk=8 those copies move only ~1 KB
   each and were pure launch latency. `rope_at_pos_qk_kernel` handles both
   projections in one grid, and `kv_append_compact_kernel` appends the
   current slot and gathers the causal prefix for k and v in one grid (each
   thread owns one (hkv, i) column, so the appended value is the one it just
   loaded). The `attn_a` span fell 0.98 → 0.80 ms and the full step
   17.7 → 17.5 ms, token-identical.
7. **Fused decode GQA** — `gqa_fused_kernel` (kernels_phase2.cu) computes
   logits -> softmax -> value in one block per Q head, keeping the bf16
   logits and probabilities in shared memory instead of two global scratch
   round-trips per layer. The per-stage arithmetic (bf16 rounding, the
   double-precision softmax denominator, the row-sequential sums) is
   reproduced verbatim, and a device test asserts the output is bit-identical
   to the three-kernel pipeline at the real head shape. Same-run A/B:
   17.44-17.48 → 16.50-16.63 ms/step; `attn_o` 3.31 → 2.44 ms. Beyond
   `d + 2*kv_len` bf16 in 48 KiB of shared memory (~24 K context) the
   `gqa_fusable` flag falls back to the classic scratch path.
8. **SwiGLU gate/up at 128 threads** — the fused gate/up kernel reads two
   fp8 rows per output row, so its 12,800 blocks have substantially more
   work than the Q/K/V rows. Nevertheless, reducing the launch width from
   256 to 128 lowers the FFN-gate bucket 6.93 → 6.86 ms and the steady
   decode 16.50-16.63 → 16.52-16.58 ms (within run noise, token-identical).
   Keep the smaller launch: it is not a regression and saves about 70 µs in
   the instrumented bucket.
9. **Direct-cache GQA** — the append step now writes only the current K/V
   slot to the persistent `[Hk][max_pos][d]` cache, and fused GQA indexes
   that cache directly with `max_pos` as its row stride. This removes the
   per-layer causal-prefix gather and its full-prefix BF16 read/write traffic.
   Same-run A/B at the 5-token prompt and 24-token decode: `attn_a`
   0.80 → 0.24 ms and full step 16.51-16.56 → 16.06-16.09 ms, about
   0.47 ms/step faster. Output is token-identical. The direct path is the
   default; `G4_DIRECT_KV=0` restores the compact-prefix rollback path.
   A padded-stride device test checks direct-cache GQA bit identity.
10. **Warp-tree RMSNorm reduction** — the decode RMSNorm ended with a
   serial thread-0 loop over 256 double partials; it now does a deterministic
   warp-shuffle tree in double and thread 0 combines only the eight warp
   totals. Same double-precision accumulation, only the tree order changes
   (inside bf16 rounding; output token-identical). Same-run A/B:
   `attn_g` 2.68 → 2.47 ms, `ffn_g` 7.61 → 7.39 ms, full step
   16.06-16.09 → 15.74-15.77 ms (~0.32 ms/step).
11. **Parallel softmax exponentials in fused GQA** — Phase B's serial
   thread-0 loop ran ~58 double-precision `exp()` calls per layer, which are
   slow on consumer Ada. The exp terms are now computed across the block and
   accumulated in ascending kv order by one thread, so the double denominator
   and the bf16 probabilities are the same sequence of additions as the
   reference (bit-identical, token-identical). Parallel form covers
   `kv_len <= 512`; longer contexts keep the exact serial loop. Same-run
   A/B: `attn_o` 2.50 → 1.89 ms, full step 15.74-15.77 → 15.16-15.21 ms
   (~0.55 ms/step).
12. **Fused-k speculative verify GEMV primitive** — the standalone
   `granite4-verify-gemv-bench` streams each FP8 weight row once while
   accumulating up to four BF16 activation rows. The bench flushes the L2
   cache before every timed launch, so each call reads its weights from HBM
   like decode. On `gremlin-1`, with a 4096×4096 FP8 projection and 20
   iterations: k=2 is 1.91× faster than two single-token launches, k=3 is
   2.47×, k=4 is 3.05× (44.0 → 14.4 µs/token). `--model` runs the real
   Granite 4.2 per-layer shapes (q/k/v/o/gate/up/down/lm_head, L=40,
   V=100352): one k=4 verify round costs 27.22 ms of GEMV work against
   80.43 ms for four single-token rounds, a 2.95× reduction; fused k=4
   effective bandwidth is 307.8 GB/s (the k=4 path is activation/latency
   bound, not at the 631.9 GB/s roofline). The device correctness test
   passes. The fused-k GEMV is now switched into the production path:
   `forward_range_batched_impl` forwards a 2–4-row suffix in one
   token-major batch (batched rmsnorm / QKV dequant + position-aware RoPE /
   persistent KV append / fused causal GQA / O-down residual GEMV / fused-K
   lm_head GEMV), so the verify no longer replays the prefix position by
   position. Numerically the batched path is equivalent to the per-position
   loop (the batched GQA stage is bit-identical; the batched QKV-dequant,
   residual projection, and fused-K GEMV are each within 2 ULP of their
   single-token twins, and greedy argmax is stable — see
   `test_batched_forward_range_parity_device`). Measured end-to-end on
   `gremlin-1` (K=4, 24-token budget, repeated-list chat prompt,
   `G4_TIMESTEP` device time): each width-4 verify round drops from 60.5 ms
   (per-position) to 36.1 ms (batched), **1.67×** per round; over a whole
   24-token generation the batched draft path is 365 ms of device time
   versus 409 ms for the per-position draft path (**~1.12×**, diluted
   because only 2 of 21 rounds are width-4). The whole-round speedup is
   below the standalone 2.95× GEMV number on purpose: a verify round is
   attention + FFN + RoPE + GEMV, not GEMV alone. Token output is identical
   across batched / per-position / no-draft (24/24), so the draft-verify
   gate preserves exact greedy output through the truncate+recompute
   rejection path.
13. **Prompt-lookup drafter with measured acceptance** — `granite4-infer
   generate --draft-lookup K` (greedy only) proposes up to K tokens by
   copying the continuation of the most recent non-overlapping occurrence of
   the longest repeated suffix (≥2 tokens), then verifies the proposals
   against the target's greedy argmax rows in one batched suffix forward.
   It carries no weights and no extra VRAM, and emits `G4_DRAFT steps=...
   proposed=... accepted=... rate=...` on stderr, and mirrors the same
   counters in the stdout
   JSON (`{"generated":[...],"draft":{"steps":S,"proposed":P,"accepted":A,
   "accept_rate":R}}`) so a benchmark harness parses one document; that shape
   is produced by the unit-tested `generate_document` in `speculative_verify`.
   Measured on `gremlin-1` with a 24-token budget
   and K=4: repeated text/list 7/8 accepted (0.875) in 16 steps, repeated
   Python code 13/17 (0.765) in 9 steps, repeated JSON 11/12 (0.917) in 12
   steps; ordinary chat produced no proposals (0/0) because the suffix never
   repeats. Baseline and draft outputs are token-identical on all three
   structured prompts. A rejected proposal now emits the target's corrective
   token at the first mismatch (previously it was dropped and recomputed,
   which cost extra steps); consensus voting across occurrences was tried
   and measured identical acceptance, so the simpler most-recent copy is
   kept. Verification is now batched: the 2–4-row suffix forward
   (`forward_range_batched_impl`) is numerically equivalent to the
   per-position loop (greedy argmax stable; see item 12 and
   `test_batched_forward_range_parity_device`). Measured end-to-end on
   `gremlin-1` (K=4, 24-token budget, repeated-list chat prompt): a
   width-4 verify round is 60.5 ms per-position vs 36.1 ms batched
   (**1.67×** per round); over a whole generation the batched draft path is
   365 ms device time vs 409 ms for the per-position draft path
   (**~1.12×** — only 2 of 21 rounds are width-4, the rest are single-token
   decode that the batching does not touch).

   Granite 4.2 chat-template prompts (rendered from the checkpoint's
   `<|im_start|>…<|im_end|>` template with `enable_thinking`, tokenized by
   the checkpoint byte-level BPE), same conditions, token-identical to
   greedy: reasoning + repeated list 8/11 accepted (0.727) in 15 steps;
   tool-call JSON 7/12 (0.583) in 14 steps; repeated Python code 8/16
   (0.500) in 13 steps; ordinary chat 4/8 (0.500) in 18 steps. Ordinary
   chat still yields proposals because the chat template itself repeats
   structural markers (`<|im_start|>`, newlines); identity holds there too.
   Chat framing lowers code acceptance versus the raw prompt (0.765 →
   0.500) because the assistant preamble and turn markers dilute the
   copied run. Tests: `tests/test_granite42_drafter_cases.py` (always-on
   token/contract corpus) and `tests/test_granite42_gpu_drafter.py`
   (opt-in, `GRANITE4_GPU_TESTS=1` plus `GRANITE4_INFER`/`GRANITE4_CKPT`).

Parity: the GPU forward matches the host oracle within bf16 rounding
(max |Δlogit| ≈ 1.1 bf16, same as P3); greedy `generate --gpu 1` (24
tokens) is token-identical across the P4 kernel builds (uint32 4-element
GEMV, branchless decode, 16-byte GEMV). The
token-6 tie vs the BF16 checkpoint reference (7754 vs 1217, both rounding
to 17.625) is bounded bf16 quantization noise within the spec's allowed
band, not an arithmetic defect.

On longer / more repetitive prompts the bf16-vs-f32 gap can flip a greedy
argmax: on the repeated-list chat prompt above (24 tokens) the GPU
(bf16) and host (f32) oracle diverge at the first generated token
after the shared prefix (578 vs 34863) even with **no draft at all** —
the divergence is in the base decode loop, not the draft-verify gate.
Each side is self-consistent under `--draft-lookup 4`
(GPU draft == GPU no-draft; host draft == host no-draft, 24/24),
so the draft loop faithfully reproduces its own oracle on both
paths; what it cannot do is make bf16 and f32 agree where the logits
are within ~1 bf16 of each other. The parity claim that matters for
the draft work is therefore GPU-side self-consistency, which holds.

Reproduction (GPU host, from the flake checkout):
```
nix build .#engine-cuda          # engine binaries (incl. gemv-bench)
./result/bin/granite4-kernel-bench        # bandwidth / roofline (JSON)
./result/bin/granite4-gemv-bench --iters 120   # per-shape GEMV bandwidth (JSON)
G4_TIMESTEP=1 ./result/bin/granite4-infer generate \
  --checkpoint <ckpt> --tokens 791,6864,315,9822,374 \
  --max-tokens 24 --gpu 1        # per-step GPU ms (one event pair; headline)
G4_TIMESTEP=2 ./result/bin/granite4-infer generate \
  --checkpoint <ckpt> --tokens 791,6864,315,9822,374 \
  --max-tokens 24 --gpu 1        # per-bucket ms/step on stderr (instrumented)
```

Remaining headroom (optional): the isolated GEMV bench reports ~413 GB/s
aggregate (65% of the 631.9 GB/s roofline; the L2-flush writeback inflates
its per-call times relative to the in-context run). The in-context step is
15.2 ms (≈65 tps) against the ≈75 tps measured ceiling; the GEMV spans are
now at 78–97% of the roofline per shape, so remaining gains are small and
need split-K reductions (the latter changes summation order and needs the
bit-exact token gate). The non-GEMV glue is ~0.9 ms of per-launch overhead
across ~160 small kernels (80 RMSNorm, 40 rope, 40 KV append), which CUDA
graphs would address; it was not pursued because uninstrumented wall time
already tracks GPU event time (the CPU is not the limiter).

P4 hardening validation (2026-10-29): the launcher-bound checks and shared
`kMaxHeadDim` definition were rebuilt through `nix build --impure --no-link
--builders '' .#engine-cuda` on gremlin-1. The resulting
`granite4-kernel-test` reported `all device kernel tests passed`;
`granite4-verify-gemv-test` reported `PASS (4 tokens x 9 rows x 256 K)`. A
real-checkpoint GPU smoke test with `--gpu 1 --max-tokens 2` completed
successfully; measured steps were 19.86 ms for the initial position and 15.02
ms for the incremental position. A fresh rebuild and smoke run on 2026-10-29
reported 77.80 ms for the five-token prefill, then 15.11 ms and 15.09 ms for
incremental steps, with generated IDs `[12366, 1210, 198]`. The host suite
remained 141 passed, 6 skipped, with host-only CTest 8/8; the gremlin CUDA
derivation separately ran the device kernel and verify-GEMV tests successfully. `nvidia-smi` itself
segfaulted on the host during this check, so GPU identity is established by the
CUDA test binaries rather than that utility. After centralizing the GQA
softmax limits, the latest gremlin-1 rebuild again passed both device tests;
the standalone 4096x4096 verify-GEMV benchmark measured k=4 at 3.02x speedup
and 14.67 us/token (k=2 1.92x, k=3 2.54x). On 2026-10-30 a further gremlin-1
rebuild (after the API doc and cleanup commits) again passed both device
tests; the same 791,6864,315,9822,374 smoke prompt generated the same IDs
[12366, 1210, 198] with 76.39 ms prefill and 15.09 ms / 15.09 ms incremental
steps, matching the 2026-10-29 figures within measurement noise.

Tried and rejected: 64-thread blocks for the small-grid QKV and O
projections (they are latency-limited at 128 threads). Same-run A/B read
15.13–15.17 ms/step vs 15.16–15.21 committed, and the bucket spans were
unchanged (`gqkv` 1.93→1.91, `go` 1.36→1.37 - both noise), so 128 threads
remain.

Tried and rejected: 128-thread RMSNorm (halving the serial thread-0
reduction but doubling the per-lane strided pass) read 16.65–16.71 ms/step
vs 16.06–16.09 committed. The warp-tree reduction later fixed the same
overhead without lengthening the strided pass.

Tried and rejected (same-run A/B, `G4_TIMESTEP=1`, token-identical): a
16-byte-load form of the fused SwiGLU gate/up kernel (`kernels_phase2.cu`)
read 18.64–18.78 ms/step vs 18.29–18.33 ms for the 256-thread baseline it
was compared against. That block already streams two fp8 rows (8 bytes/lane)
with a shared activation decode, so widening it doubles register pressure
for no extra bytes in flight.

Tried and rejected: pairing the two half-rows of the GEMV vector path
(each lane loads chunk g and g + K/2 in one iteration with separate
accumulators - deeper memory-level parallelism) read 16.81–16.85 ms/step
vs 16.50–16.63 ms committed. Weight and activation bytes are still each
read exactly once per block, so the cost is register pressure: the lane
holds two in-flight uint4 weight vectors and two decode pipelines at once,
which drops blocks-per-SM and hurts latency hiding more than the extra
MLP helps.

### P4 round 3 — EAGLE draft head (2026-11-04, gremlin-1)

The engine now supports an optional trained EAGLE-style draft head. The head
reads the target model's final-normed hidden states, proposes up to 15 tokens,
and the target verifies the anchor plus proposals in one exact batched forward.
The verify output is bit-identical to ordinary greedy decoding; rejected drafts
are never emitted. Training data and tooling live in `scripts/eagle/`, while
`dump-hidden` writes the target features used by training.

Training used 16.0M chat tokens (UltraChat SFT plus ShareGPT), a 232.8M
parameter head initialized from target layer 39, and two epochs on gremlin-1.
The held-out self-generated set reached 1.493/1.711/1.799/1.843/1.866/1.880
verified tokens per round for K=1..6 at the final checkpoint.

The exported head was tested live with the CUDA engine on the six existing
benchmark prompts. All K settings produced exactly the same token sequence as
plain greedy decoding. Aggregate throughput was 68.3 tok/s plain, 105.6 tok/s
at K=3, 102.6 at K=4, and 97.5 at K=5. A live serving check with K=5 measured
138.0 vs 67.9 tok/s on the summarize prompt, with 84 accepted proposals out
of 211. K=3 is the default because it provided the best aggregate throughput;
the service can select another value with `GRANITE4_EAGLE_K`.

The head is opt-in through `GRANITE4_EAGLE_HEAD`; when configured, the NixOS
module passes it to the resident CUDA engine. The frontend sends the per-request
K only when a head is loaded. `GRANITE4_EAGLE_K=0` or an unset head disables
speculation.

### P4 round 2 — end-to-end speed (2026-11-03, gremlin-1)

Same card and checkpoint as above (RTX 4070 Ti SUPER, CUDA 12.9 toolchain
from the flake, BF16 checkpoint packed to FP8 at load). Device times are
`G4_TIMESTEP=1` event times; wall times are `/usr/bin/time` of the whole
CLI run. Baseline = the parent commit, built the same way.

| Measurement | Baseline | Now |
| --- | ---: | ---: |
| Decode, ms/token, 5-token prompt (positions 5–68) | 15.11–15.39 | 13.95 |
| Decode, ms/token, 2000-token prompt | 65.6 | 14.54 |
| Prefill, 5 tokens | 76.3 ms | 24.4 ms |
| Prefill, 500 tokens | 8017 ms | 122 ms |
| Prefill, 2000 tokens | 77985 ms | 503 ms |
| `generate --gpu 1`, 5-token prompt, 64 tokens, wall | 131 s | 7.4 s |
| Repeated-text prompt, 96 tokens, `--draft-lookup 4`, device ms/token | — | 9.24 (vs 14.35 greedy) |

What changed, by size of effect:

1. **Load path** (131 s → 7.4 s wall). The GPU paths no longer build the
   host oracle's ~34 GB float copy and re-quantize it on one thread
   (`load_weights` + `pack_gpu_model` was 2:10 of the 2:11). `gpu_pack.cc`
   streams each shard tensor with parallel `pread`s straight into the packed
   blob; BF16 and per-tensor FP8 sources have at most 65536 / 256 distinct
   values, so the max-abs scan and `fp8_e4m3_encode` run once per distinct
   value through a lookup table. Byte-identical to the old packer (blob and
   scales; `test_weights` checks both packers on BF16, FP8 per-tensor and
   per-channel checkpoints). `G4_LOAD_TIMING=1` prints the phases (pack
   4.8 s, upload 1.0 s).
2. **Tensor-core prefill** (`prefill_gemm.cu`). Prompts and any suffix of
   5+ rows run as chunks of up to 512 rows: every projection is one
   `mma.sync.m16n8k16` BF16 GEMM with the FP8 weight tile decoded in shared
   memory (q|k|v and gate|up are contiguous in the packed layout and run as
   one GEMM each, with per-segment scales); a 4-stage `cp.async` pipeline and
   a 16-row tile for short chunks keep short prompts near the weight-stream
   floor, split-K fills the card when a chunk has too few output tiles.
   Causal attention is a flash-style tensor-core kernel over the chunk; the
   lm_head runs only for rows whose logits are requested. 500 and 2000-token
   prompts sustain ~67 TFLOPS (~76% of the card's BF16/FP32-accumulate
   peak).
3. **Split-KV decode attention** (`attention_split.cu`). The fused GQA kernel
   ran 32 blocks with a serial kv_len-long value loop: 190 µs/layer at 2 K
   context. Split-KV (64-position tiles, K/V staged once per KV head for its
   4 query heads, online softmax, ordered combine) takes ~4 µs/layer at short
   context and keeps 2 K-context decode at 14.5 ms/token (the old build: 65.6
   ms, where past 512 positions the fused kernel also ran its exp() terms
   serially; with the parallel-exp limit raised to 3584 it was 21.5 ms).
   Decode and verify rows use the same split layout (it depends only on the
   cache stride), so a verify row is bit-identical to the decode step at that
   position.
4. **Decode step as a CUDA graph, greedy on the device**. Every
   single-position kernel reads the position from device memory, so the step
   is captured once and replayed; `argmax_rows_kernel` writes the next token
   into the token buffer and advances the position, so `gpu_decode_greedy`
   enqueues all steps without a host round trip (no per-token 200 KB logits
   copy, host argmax, or token re-upload). With `--eos` the ids come back in
   8-token chunks, one chunk ahead.
5. **Hardware FP8 decode** in every GEMV (`cvt.rn.f16x2.e4m3x2`, exact for
   all finite codes): replaces ~10 integer ops per weight; the decode GEMVs
   were close to integer-issue bound. Bit-identical outputs.
6. **RMSNorm**: 1024 threads with the row in registers, and the sum of
   squares as a compensated float pair instead of FP64 (consumer Ada runs
   FP64 at 1/64 rate): 7.8 → 2.9 µs per call, 81 calls per token.
7. **Fixes found on the way**: `verify_gemv.cu` rounded with `0x7ff` instead
   of `0x7fff` (near-truncation instead of RNE — why the fused-K lm_head was
   only "within 2 ULP"; it is now bit-identical to the decode GEMV, and the
   test asserts that). The batched fused-GQA launcher under-sized shared
   memory once a batch straddled the parallel-softmax limit (rows below the
   limit used the exp buffer the allocation did not include). The serving
   layer compared generated ids against the tokenizer's eos *string*, so it
   never stopped at EOS; it now reads `eos_token_id` from `config.json` and
   passes `--eos` so the engine stops computing there.

Persistent engine: `granite4-infer serve --gpu 1` loads once and answers
JSON-line requests; the API uses one such process for a GPU gate
(`parity_gate.py --gpu` records `"gpu": true`), so a chat request costs its
prefill + decode instead of a 7 s load.

Parity: decode-path changes 1, 4, 5 are bit-identical. The tensor-core
prefill, split-KV attention and FP32 RMSNorm change summation order (and
the split attention keeps f32 probabilities), so logits move within bf16
rounding; device tests bound each kernel against its reference
(`test_prefill_gemm_device`, `test_prefill_attention_device`,
`test_attn_split_device`). Greedy output is self-consistent across paths:
draft vs no-draft is token-identical (5/5 `test_granite42_gpu_drafter.py`,
plus 48-token checks on three prompts), and the graph / no-graph /
batched-verify / compact-KV switches give identical ids. Against the old
build, the 64-token generate for `791,6864,315,9822,374` can flip at
generated token 12, a near tie (bf16 logits 18.0 vs 17.75–17.875); the fp32
transformers reference on the same prefix ranks the *other* candidate first
(18.42 vs 17.78), so neither build is "the" reference there. With
`G4_PREFILL_GEMM=0 G4_SPLIT_ATTN=0` the 64 ids match the old build exactly.

A/B switches (all default on): `G4_PREFILL_GEMM`, `G4_SPLIT_ATTN`,
`G4_GRAPH`, `G4_BATCH_VERIFY`, `G4_DIRECT_KV`.

Reproduction (gremlin-1, from a checkout):
```
nix build .#engine-cuda
B=./result/bin/granite4-infer C=/opt/granite4-rtx-4070-ti-super/ckpt
./result/bin/granite4-kernel-test                 # all device kernel tests passed
G4_TIMESTEP=1 $B generate --checkpoint $C --tokens 791,6864,315,9822,374 \
  --max-tokens 64 --gpu 1                         # prefill + per-step (graph) ms
G4_LOAD_TIMING=1 /usr/bin/time $B generate --checkpoint $C \
  --tokens 791,6864,315,9822,374 --max-tokens 64 --gpu 1   # load phases + wall
```

What is left: decode GEMVs run at 590–630 GB/s per shape against the
632 GB/s measured roofline, so the 13.95 ms step is ~95% of the
weight-stream floor (13.3 ms); more single-stream speed needs fewer bytes
per token (lower-precision weights, a quality decision) or more accepted
tokens per weight pass (drafting). Long prefill is MMA bound at ~76% of the
BF16 peak; FP8 activations would double the MMA rate at a numerics cost.

### P4 round 3 — exact wide verify (2026-11-03, gremlin-1)

The speculative verify forward (`forward_rows_batched`) now takes 2–16 rows
(`kMaxVerifyRows`, was 4) through four multi-token GEMVs in `verify_gemv.cu`
(Q|K|V, O/down with the residual add, SwiGLU gate/up, lm_head). Each streams
a weight row once for all rows and reproduces its decode twin's per-token
arithmetic (lane chunking at 128 threads, warp tree, warp totals in order),
so every verify row is bit-identical to the decode step at that position:
`test_verify_kernels_bit_identical_device` checks all four at widths 1–16
against the decode kernels, and `test_batched_forward_range_parity_device`
now requires the batched forward (widths 2, 3, 4, 16, plus truncate and
recompute) to match the per-position forward bit for bit (it allowed 32 ULP
before). The old batched Q/K/V and O/down kernels launched one block per
(token, row) and re-read every weight row per token; that, not attention,
was the ~3.5 ms per extra row. Suffixes of 17+ rows still run the
tensor-core prefill.

Device time of one verify forward (`G4_TIMESTEP=1`, `generate
--draft-lookup K` on the summarize_quote prompt, medians):

| rows | 1 (decode) | 2 | 3 | 4 | 6 | 7 | 8 | 12 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| before | 14.1 | 17.4 | 21.0 | 24.5 | — | 24.5 (prefill) | — | — | — |
| now (ms) | 14.1 | 14.5 | 15.0 | 15.0 | 15.7 | — | 17.2 | 23.7 | 29.6 |

Up to 8 rows an extra row costs ~0.4 ms. Past that the per-lane activation
loads (two bytes of activation per weight byte per row, from L1/L2) start to
dominate, so 9–16 rows cost ~1.5 ms each.

Prompt-lookup drafting through the persistent engine (`serve`, chat-format
prompts, up to 512 tokens, best of 2; outputs are token-identical to K=0 in
every cell, which they were not at K ≥ 6 before):

| prompt | K=0 | K=2 | K=4 | K=6 | K=8 | K=12 | K=15 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| essay | 71.2 | 67.8 | 67.7 | 67.5 | 67.0 | 65.4 | 63.9 |
| qa | 69.8 | 68.1 | 67.9 | 67.6 | 67.0 | 64.4 | 62.3 |
| code_edit | 54.7 | 66.8 | 68.7 | 70.6 | 72.0 | 67.0 | 63.0 |
| json_extract | 58.6 | 63.9 | 76.3 | 75.3 | 73.6 | 67.3 | 62.5 |
| summarize_quote | 67.3 | 81.4 | 97.9 | 107.1 | 111.3 | 106.6 | 100.8 |
| list | 68.4 | 62.7 | 62.3 | 62.0 | 61.8 | 61.3 | 61.0 |

(tokens/s). Copy-heavy replies gain up to 1.65×; novel text still loses a
few percent, because lookup proposes rarely there and each miss costs a
verify round. A model drafter (EAGLE-style head) is the next step; it can
now verify chains of up to 8 tokens at ~0.4 ms per extra row.

## Rules across phases

- No kernel or code is copied from `reference/qwen3-27b-5090/`; it is kept
  for layout and conventions only.
- No benchmark number is written into this repo before P4, and none is
  claimed without its conditions + commands.
- Status stays honest: any endpoint that cannot do a thing says so with a
  machine-readable code, never a placeholder value.
