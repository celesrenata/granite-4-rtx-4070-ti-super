# Granite 4.2 8B on one RTX 4070 Ti Super

A single-purpose inference engine for Granite 4.2 8B on one NVIDIA RTX 4070 Ti
Super (16 GB, sm_89), written in C++20 / CUDA with a dependency-light Python
stdlib API in front. It is not a general engine: the constants, memory plan and
kernels are written for this one model on this one GPU.

**Status: P4 engine / speculative-decoding wired in.** The CUDA engine loads
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

CUDA objects target sm_89 only. The device tests (`granite4-kernel-test`,
`granite4-verify-gemv-test`) need a live GPU, so the Nix build skips them and
they are re-run on the GPU host.

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

## Limitations

- Concurrency equals `sessions`: up to N requests run at once and decode
  batched (plain and EAGLE), further requests queue. Conversations stay
  resident (`serve --sessions N`, NixOS `sessions`), each with its own EAGLE
  state, on one shared paged KV pool plus
  one draft-KV pool (`--kv-pool-tokens`, default N x `maxContext`). A request
  resumes the session sharing the longest prefix with its prompt, so
  interleaved chats skip re-prefilling their history. gremlin-1 runs 2 x 32768
  with KV8 (5.2 GiB pool).
- Greedy only on the GPU serving path; seeded top-p exists in the CLI.
- `maxContext` (default 8192) bounds prompt + completion per request.
- Weights are FP8 e4m3 with per-tensor scales; outputs agree with the fp32
  reference within bf16 rounding, not bit for bit (see docs/phases.md).
