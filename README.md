# Granite 4.2 8B on one RTX 4070 Ti Super

A single-purpose inference engine for Granite 4.2 8B on one NVIDIA RTX 4070 Ti
Super (16 GB, sm_89), written in C++20 / CUDA with a dependency-light Python
stdlib API in front. It is not a general engine: the constants, memory plan and
kernels are written for this one model on this one GPU.

**Status: P4 engine, EAGLE speculative decoding and quantized KV wired in.** The CUDA engine loads
FP8 weights, runs the full Granite 4.2 8B forward pass, and serves deterministic
GPU generation through the native inference binary. A prompt is one
tensor-core prefill call, greedy decode is a CUDA graph chained on the device
(no host round trip per token), and `generate --draft-lookup K` verifies
prompt-lookup drafts in one batched forward. On an RTX 4070 Ti Super, steady
batch-1 decode is about **13.95 ms/token (~72 tps)** at short context and
14.5 ms/token at 2 K context (the weight-streaming roofline is 13.3 ms); a
500-token prompt prefills in ~122 ms. See [`docs/phases.md`](docs/phases.md)
for commands and conditions. The stdlib Python HTTP layer serves
OpenAI-style chat completions through a resident engine process, gated by a
GPU parity check against the fp32 transformers reference; gremlin-1 runs it
as a NixOS service (~0.23 s for a short answer).

Optional features, all off unless configured (details in the linked docs):

- **EAGLE draft head** (`GRANITE4_EAGLE_HEAD`, `GRANITE4_EAGLE_K`): a trained
  ~233M-parameter head drafts tokens and the target verifies them in one
  exact batched forward, so output stays identical to plain greedy. Measured
  ~1.55x aggregate (105.6 vs 68.3 tok/s at K=3). See
  [docs/eagle-2x-performance-spec.md](docs/eagle-2x-performance-spec.md).
- **Quantized KV cache** (`G4_KV8`): `1` = int8 (Q8_0, 82.5 KiB/position),
  `4` = int4 (Q4_0, 42.5 KiB/position) versus 164 KiB/position for bf16.
  int4 is what lets a 131072-position pool fit 16 GiB. See
  [docs/memory.md](docs/memory.md).
- **Quantized draft KV** (`G4_DRAFT_KV`): `16` (bf16, default), `8` or `4`
  for the EAGLE head's own KV, needed for the head to fit beside a 131K
  reader pool. int8 is lossless on accept rate; int4 costs ~1%. See
  [docs/deployment.md](docs/deployment.md).

`reference/qwen3-27b-5090/` is a different model/GPU kept for layout
conventions only. No kernels, code or numbers were copied from it.

## Model spec (planning constants)

| Constant | Value |
| --- | ---: |
| Layers | 40 |
| Hidden size | 4096 |
| Intermediate (MLP) size | 12800 |
| Attention heads / KV heads | 32 / 8 (GQA, ratio 4) |
| Head dim | 128 |
| Vocab size | 100352 |
| Max context | 131072 |

Source of truth: [`frontend/config.py`](frontend/config.py) and
[`engine/src/model_config.h`](engine/src/model_config.h), cross-checked by
tests. The weight-loading phase (P1) re-validates these against the
checkpoint's `config.json` and rejects it on any mismatch.

## API

| Endpoint | Response |
| --- | --- |
| `GET /health` | 200 — `{"status":"ok","inference":"available"\|"unavailable",...}` |
| `GET /metrics` | 200 — model constants, GPU target, engine status |
| `GET /v1/models` | 200 — model card; `"status"` follows the parity gate |
| `POST /v1/chat/completions` | 200 completion when a passing parity-gate artifact is loaded; 400 for invalid requests; **501** `inference_unavailable` without a gate |

Serving is gate-driven: `scripts/parity_gate.py` writes an artifact on PASS
and `granite4-api --gate-artifact <path>` serves only with it. A GPU gate
(`--gpu`) makes the API keep one `granite4-infer serve --gpu 1` process
resident. The NixOS module wires all of this up (see
[docs/deployment.md](docs/deployment.md)).

```sh
python3 -m frontend.api --port 8080 --gate-artifact .gt/parity_gate.json
curl -s -X POST localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"granite-4.2-8b","messages":[{"role":"user","content":"hi"}]}'
```

## Build

```sh
# Host-only (works without the CUDA toolkit; used by CI and the Python tests):
cmake -S . -B build -G Ninja -DGRANITE_ENABLE_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure

# Full build (requires nvcc; CUDA objects target sm_89 explicitly):
cmake -S . -B build -G Ninja
cmake --build build
```

Nix users can build the pieces with `nix build .#engine-cuda` (CUDA engine,
`granite4-infer`) and `.#api` (`granite4-api`, `granite4-parity-gate`);
`nix develop` gives a shell with cmake, ninja, uv and `GRANITE4_LIB` preset.

CUDA objects target sm_89 only. The device tests (`granite4-kernel-test`,
`granite4-verify-gemv-test`) need a live GPU, so the Nix build skips them and
they are re-run on the GPU host.

## Environment variables

| Variable | Read by | Meaning |
| --- | --- | --- |
| `GRANITE4_GATE` | API | Parity-gate artifact path (default `.gt/parity_gate.json`) |
| `GRANITE4_MAX_CONTEXT` | API | Positions per request for the resident engine (default 8192) |
| `GRANITE4_EAGLE_HEAD` | API | EAGLE head safetensors; unset disables speculation |
| `GRANITE4_EAGLE_K` | API | Tokens drafted per round, 0-15 (frontend default 3; the NixOS `eagle.k` option defaults to 5) |
| `GRANITE4_LIB` | API | Path to the `granite4` shared library for the metadata probe |
| `G4_KV8` | engine | KV format: unset/`0` bf16, `1` int8, `4` int4 (split-KV attention shape only) |
| `G4_DRAFT_KV` | engine | EAGLE head KV width: `16` (default), `8`, `4` |
| `G4_SPLIT_ATTN`, `G4_DIRECT_KV`, `G4_GRAPH`, `G4_BATCH_VERIFY`, `G4_PREFILL_GEMM` | engine | A/B switches, default on; `0` disables |
| `G4_TIMESTEP` | engine | `1` per-call GPU time, `2` per-bucket attribution (stderr) |
| `G4_EAGLE_TIMING`, `G4_LOAD_TIMING`, `G4_SERVE_DEBUG`, `G4_FORCE_REPREFILL` | engine | Diagnostics and A/B for the serve loop |

## Engine CLI

`granite4-infer` subcommands: `forward`, `generate`, `serve`, `dump-hidden`
(writes target features for EAGLE training) and `kv-bench` (KV format A/B).
Common flags: `--checkpoint`, `--gpu 1`, `--tokens`, `--max-tokens`,
`--draft-lookup K`, `--eagle HEAD --eagle-k K`, `--sessions N`,
`--max-context N`, `--bench-ctx`, `--bench-decode`. EAGLE requires
`--sessions 1`. Training and export tooling for the head is in
`scripts/eagle/`.

## Tests

```sh
uv run --group dev pytest        # or: python3 -m pytest if pytest is installed
```

All tests run without a GPU or CUDA toolkit:

- [`tests/test_api.py`](tests/test_api.py) — live HTTP checks of the four endpoints,
  including the full 400-validation matrix and the 501 boundary.
- [`tests/test_config.py`](tests/test_config.py) — Python model constants, config
  validation, and a cross-check that the C++ header holds the same values and
  that CMake targets arch 89.
- [`engine/tests/test_config_validation.cc`](engine/tests/test_config_validation.cc) —
  host-only C++ checks of the constants, validation logic and the C ABI
  (via `ctest`).

## Docs

- [docs/architecture.md](docs/architecture.md) — components, request flow, status model, ABI
- [docs/phases.md](docs/phases.md) — P0→P4 plan with exit criteria and measurements
- [docs/memory.md](docs/memory.md) — weight/KV budget math and measured KV8/KV4 footprints
- [docs/deployment.md](docs/deployment.md) — NixOS module, parity gate, KV and draft-KV options
- [docs/eagle-2x-performance-spec.md](docs/eagle-2x-performance-spec.md) — EAGLE head results and 2x plan
- [docs/chat-context.md](docs/chat-context.md), [docs/golden.md](docs/golden.md), [docs/remote-checkpoint.md](docs/remote-checkpoint.md) — chat context bound, golden reference, remote checkpoint inspection

## Limitations

- Batch 1: the engine serves one request at a time (the API serializes them).
- Greedy only on the GPU serving path; seeded top-p exists in the CLI.
- `maxContext` (default 8192; production runs 32768) bounds prompt + completion per request.
- EAGLE speculation needs `--sessions 1`.
- Weights are FP8 e4m3 with per-tensor scales; outputs agree with the fp32
  reference within bf16 rounding, not bit for bit (see docs/phases.md).
