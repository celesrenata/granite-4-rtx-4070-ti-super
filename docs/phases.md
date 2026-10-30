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
   passes. This is not yet an end-to-end speculative path: k-token RMSNorm,
   RoPE, causal GQA/KV append, candidate generation, and the exact
   accept/reject gate remain to be implemented; the primitive is
   deliberately not linked into production decode.
13. **Prompt-lookup drafter with measured acceptance** — `granite4-infer
   generate --draft-lookup K` (greedy only) proposes up to K tokens by
   copying the continuation of the most recent non-overlapping occurrence of
   the longest repeated suffix (≥2 tokens), then verifies each proposal
   against the target's greedy argmax one position at a time. It carries no
   weights and no extra VRAM, and emits `G4_DRAFT steps=... proposed=...
   accepted=... rate=...`. Measured on `gremlin-1` with a 24-token budget
   and K=4: repeated text/list 7/8 accepted (0.875) in 16 steps, repeated
   Python code 13/17 (0.765) in 9 steps, repeated JSON 11/12 (0.917) in 12
   steps; ordinary chat produced no proposals (0/0) because the suffix never
   repeats. Baseline and draft outputs are token-identical on all three
   structured prompts. A rejected proposal now emits the target's corrective
   token at the first mismatch (previously it was dropped and recomputed,
   which cost extra steps); consensus voting across occurrences was tried
   and measured identical acceptance, so the simpler most-recent copy is
   kept. This is an acceptance measurement, not a throughput win: the
   verifications still run one position at a time, so the fused-k GEMV
   (2.29× at k=4) and batched causal attention/KV append are the next
   integration step.

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
and 14.67 us/token (k=2 1.92x, k=3 2.54x).

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

## Rules across phases

- No kernel or code is copied from `reference/qwen3-27b-5090/`; it is kept
  for layout and conventions only.
- No benchmark number is written into this repo before P4, and none is
  claimed without its conditions + commands.
- Status stays honest: any endpoint that cannot do a thing says so with a
  machine-readable code, never a placeholder value.
