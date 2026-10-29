# Granite 4.2 8B on one RTX 4070 Ti Super

A single-purpose inference engine for Granite 4.2 8B on one NVIDIA RTX 4070 Ti
Super (16 GB, sm_89), written in C++20 / CUDA with a dependency-light Python
stdlib API in front. It is not a general engine: the constants, memory plan and
kernels are written for this one model on this one GPU.

**Status: P4 engine / speculative-decoding preparation.** The CUDA engine loads
FP8 weights, runs the full Granite 4.2 8B forward pass, and serves deterministic
GPU generation through the native inference binary. Phase 4 launcher hardening
has been rebuilt and device-tested on gremlin-1; the fused speculative GEMV
primitive and attention/QKV staging APIs are verified, but full batched
speculative generation is not yet wired into the production decode loop. On an RTX 4070 Ti Super,
steady batch-1 decode is currently about **15.2 ms/token (~65 tps)** for the
5-token prompt benchmark; see [`docs/phases.md`](docs/phases.md) for commands
and conditions. The Python HTTP layer remains a thin API scaffold while the
native engine and speculative-decoding kernels evolve.

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

| Endpoint | Response (Phase 0) |
| --- | --- |
| `GET /health` | 200 — `{"status":"ok","inference":"unavailable",...}` |
| `GET /metrics` | 200 — model constants, GPU target, engine status (probes the optional built `.so`) |
| `GET /v1/models` | 200 — model card with `"status":"unavailable"` |
| `POST /v1/chat/completions` | 400 + error body when the request is invalid; **501** `inference_unavailable` when valid |

Run it (no dependencies beyond Python ≥ 3.10):

```sh
python3 -m frontend.api --port 8080
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -s -X POST localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"granite-4.2-8b","messages":[{"role":"user","content":"hi"}]}'
# -> 501 {"error":{"message":"...inference is not implemented...", "code":"inference_unavailable"}}
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

The CUDA path is compile-only in Phase 0: [`engine/src/kernels.cu`](engine/src/kernels.cu)
contains a single arch-probe kernel so the toolchain and arch selection are
exercised before any real kernel lands.

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
- [docs/phases.md](docs/phases.md) — P0→P4 plan with exit criteria
- [docs/memory.md](docs/memory.md) — weight/KV budget math for the 16 GB card (planning figures)

## Limitations (by design, at this phase)

- No inference, no weight loading, no real kernels — endpoints say so explicitly.
- No performance claims anywhere in this tree.
- The CUDA path is unverified until it is compiled and run on an sm_89 machine.
