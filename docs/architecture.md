# Architecture

Phase 0 of a single-purpose engine: **Granite 4.2 8B → one RTX 4070 Ti Super
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
  validate → reject (4xx) → infer; in Phase 0 the "infer" step is the
  501 placeholder.
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
5. Valid request → **501** `inference_unavailable` with an OpenAI-shaped
   error body, so clients surface the reason cleanly instead of a raw error.

All other routes that do not exist → 404 with the same error shape.

## Status model

One vocabulary, used by the API and the C ABI:

| Value | Meaning |
| --- | --- |
| `ok` | server reachable, API responsive |
| `unimplemented` | library loaded, inference not yet implemented (Phase 0 ⇒ always) |
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

## File map

| Path | Role |
| --- | --- |
| `CMakeLists.txt` | build root; CUDA on/off switch |
| `engine/CMakeLists.txt` | `granite4` library + host-only test |
| `engine/src/model_config.h` | constants, C++ single source of truth |
| `engine/src/config_validation.{h,cc}` | spec match + structural checks |
| `engine/src/capi.{h,cc}` | C ABI metadata/status |
| `engine/src/kernels.cu` | sm_89 object stub (arch probe) |
| `engine/tests/test_config_validation.cc` | host-only ABI/validation test |
| `frontend/api.py` | HTTP API + validation |
| `frontend/config.py` | constants, Python single source of truth |
| `frontend/engine_bridge.py` | optional ctypes probe |
| `tests/` | pytest suite (no CUDA needed) |
