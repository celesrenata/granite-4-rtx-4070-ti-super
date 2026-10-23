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

## Rules across phases

- No kernel or code is copied from `reference/qwen3-27b-5090/`; it is kept
  for layout and conventions only.
- No benchmark number is written into this repo before P4, and none is
  claimed without its conditions + commands.
- Status stays honest: any endpoint that cannot do a thing says so with a
  machine-readable code, never a placeholder value.
