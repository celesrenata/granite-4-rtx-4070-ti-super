# Architecture

A single-purpose engine (this page started as the Phase 0 scaffold description and has been updated through P4): **Granite 4.2 8B → one RTX 4070 Ti Super
(sm_89)**. One model, one GPU, one shared library, one small Python server.
There is deliberately no general-purpose layer.

## Components

```
                +--------------------------------------------+
                | frontend/  (Python 3 stdlib only)         |
                |  api.py          HTTP server + validation|
                |  config.py       model constants (P-side)|
                |  engine_bridge   optional ctypes probe   |
                +-------------------+------------------------+
                                    | C ABI (capi.h)
                +-------------------v------------------------+
                | engine/  (C++20, shared lib `granite4`)    |
                |  model_config.h      constants (C++ side) |
                |  config_validation   spec + structure     |
                |  capi.cc             C ABI: metadata +   |
                |                      status placeholder  |
                |  kernels.cu          sm_89 objects       |
                |                      (arch probe only)   |
                +--------------------------------------------+
```

- **frontend/** — `http.server.ThreadingHTTPServer`, `json`, `time`,
  `ctypes`. No third-party packages. The request lifecycle is
  validate → reject (4xx) → infer. `infer` is the 501 placeholder unless a
  passing parity-gate artifact is loaded; with a GPU gate,
  `engine_service.py` keeps one `granite4-infer serve --gpu 1` process
  resident (serialized requests), `tokenizer.py` / `chat_template.py`
  handle Granite tokenization and chat formatting, and
  `GRANITE4_EAGLE_HEAD` / `GRANITE4_EAGLE_K` enable speculative decoding.
- **engine/** — one shared library with a flat C ABI
  ([`engine/src/capi.h`](../engine/src/capi.h)). The host side (constants,
  validation, ABI) compiles with any C++20 compiler; CUDA objects are
  added only when `GRANITE_ENABLE_CUDA=ON`, always for arch 89
  (`CMAKE_CUDA_ARCHITECTURES 89`, no `native`, no auto-detect).

## Request flow (POST /v1/chat/completions)

1. Parse body as JSON; must be an object.
2. `model` must equal `granite-4.2-8b`; anything else → 400 `unknown_model`.
3. `messages` must be a non-empty array of `{role, content}` with
   `role ∈ {system, user, assistant}`; unknown roles and type errors → 400
   with the offending index.
4. Optional fields (`max_tokens`, `temperature`) type-checked;
   `temperature ∈ [0, 2]`.
5. Valid request → completion from the resident GPU engine when a passing
   parity gate is loaded (prompt + completion must fit `maxContext`, else
   400 `context_length_exceeded`); otherwise **501** `inference_unavailable`
   with an OpenAI-shaped error body.

All other routes that do not exist → 404 with the same error shape.

## Status model

One vocabulary, used by the API and the C ABI:

| Value | Meaning |
| --- | --- |
| `ok` | server reachable, API responsive |
| `unimplemented` | library loaded, C ABI inference placeholder (inference runs through `granite4-infer`, not the ABI) |
| `no-device` | library loaded, no usable CUDA device / wrong arch |
| `unavailable` | library missing or ABI drift (for `/metrics.engine`; the API itself stays up) |

Honesty rule: no endpoint ever reports inference-ready or returns sampled
tokens until a real model is loaded and the golden parity test (P3) passes.

## C ABI (Phase 0 surface)

- `granite4_get_model_info(out)` — copies the target model constants and the
  `sm_89` target. Spec values, not loaded state.
- `granite4_status(out)` — status code, device-present flag, text. Always
  `unimplemented` in Phase 0. Device probing is compiled in only for CUDA
  builds; host-only builds report `cuda_device_present = 0`.

No allocation, no locking, NULL-tolerant. Everything after the weight loader
(P1) grows this surface additively.

## Threading and lifetime

- Frontend: `ThreadingHTTPServer`; each request is independent, no shared
  mutable state (the engine probe is re-resolved per `/metrics` call).
- Engine: Phase 0 has no GPU state; P3 will own the device, KV pool and
  scheduler behind one lock-protected runtime object — the ABI will stay
  flat (create/destroy/step) rather than a class hierarchy.

## GPU engine (beyond the C ABI)

Inference is not exposed through `capi.h`. The `granite4-infer` binary
(`engine/src/infer_main.cc`) loads FP8 weights (`weights.cc`, `gpu_pack.cc`)
and runs the forward pass in `forward_gpu.cu` with split-KV attention
(`attention_split.cu`), tensor-core prefill (`prefill_gemm.cu`), a CUDA-graph
greedy decode loop, batched speculative verify, prompt-lookup drafting
(`draft_lookup.cc`) and the optional EAGLE head (`eagle_gpu.cu`). The persistent
KV cache is bf16, int8 (`G4_KV8=1`) or int4 (`G4_KV8=4`); the EAGLE head's
KV is bf16 or quantized via `G4_DRAFT_KV`. Subcommands and env vars are listed
in the README. `serve_resume.cc` reuses a session's cached prefix between
requests.

## File map

| Path | Role |
| --- | --- |
| `CMakeLists.txt` | build root; CUDA on/off switch |
| `engine/CMakeLists.txt` | `granite4` library + host-only test |
| `engine/src/model_config.h` | constants, C++ single source of truth |
| `engine/src/config_validation.{h,cc}` | spec match + structural checks |
| `engine/src/capi.{h,cc}` | C ABI metadata/status |
| `engine/src/kernels*.cu`, `attention_split.cu`, `prefill_gemm.cu` | sm_89 kernels |
| `engine/src/forward_gpu.cu`, `eagle_gpu.cu` | GPU forward pass, KV formats, EAGLE head |
| `engine/src/infer_main.cc` | `granite4-infer` CLI and serve loop |
| `engine/tests/test_config_validation.cc` | host-only ABI/validation test |
| `frontend/api.py` | HTTP API + validation |
| `frontend/config.py` | constants, Python single source of truth |
| `frontend/engine_bridge.py` | optional ctypes probe |
| `frontend/engine_service.py` | resident `granite4-infer serve` process manager |
| `frontend/tokenizer.py`, `chat_template.py` | tokenizer and Granite chat template |
| `scripts/parity_gate.py` | GPU parity gate |
| `scripts/eagle/` | EAGLE head training, export, serving benchmarks |
| `tests/` | pytest suite (no CUDA needed) |
