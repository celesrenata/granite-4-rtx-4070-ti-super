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

## P1 — Weights

- Safetensors loader; parse checkpoint `config.json` and reject on any
  deviation from the spec (hard fail, listed field-by-field).
- Decide storage: FP8 (weights) is the working default — see
  [memory.md](memory.md); BF16 does not fit the card.
- Export a quantization pass for activations where kernels need it.

Exit criteria: full checkpoint loaded and checksummed; `nvidia-smi` memory
matches the plan within 5%; config mismatch on a perturbed checkpoint fails
fast with the exact field list.

## P2 — Kernels (sm_89)

- GEMM (FP8 e4m3 weights; Ada has no Blackwell-style block-scaled MMA, so
  scales are per-tensor/channel — decided here), attention with GQA
  (32 Q / 8 KV, head dim 128), RMSNorm, rotary, activations, sampling.
- Each kernel unit-tested against a CPU reference on the same seed.

Exit criteria: per-kernel correctness on randomized inputs; micro-benchmark
script with the bandwidth/roofline baseline measured (not assumed).

## P3 — Integration

- Scheduler (N concurrent sequences), prefill/decode loop, paged KV, prefix
  cache; memory manager wired to the plan in [memory.md](memory.md).
- Golden parity: fixed prompts, seeded sampling, compared token-for-token
  against the reference implementation (transformers/other engine) with
  quantization noise bounded per spec.

Exit criteria: parity gate passes; the API flips to serving; `/health`
reports `inference: available` only after the gate.

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
