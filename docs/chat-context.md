# Chat context — speculative decoding (draft-lookup) integration

Handoff document so a new session can continue the work without re-deriving state.
Snapshot taken: 2026-11-03, branch `main` (batched k-token verify forward
wired in; see "Just completed" below).

## What this project is

`granite-4-rtx-4070-ti-super`: a minimal C++/CUDA inference engine for
IBM Granite-4 (text) models on an RTX 4070 Ti Super. Core deliverables:
FP8 (E4M3) weights + BF16 compute, a fused single-kernel decode GEMV
(`forward_gpu.cu`), a host reference/oracle (`forward_logits_f32`), and a
CLI (`infer_main.cc`) with subcommands `inspect`, `logits`, and `generate`.

The host path doubles as a correctness oracle: it is the ground truth that
the CUDA path is compared against.

## Current work: greedy speculative decoding via draft-lookup

The active effort is speculative decoding where the "draft model" is a cheap
n-gram / repeated-span lookup over the generated sequence (`draft_lookup.{h,cc}`),
proposing up to K tokens that the target then verifies in one forward pass.

### What is already implemented and working

- `draft_lookup.h` / `draft_lookup.cc` — proposes up to K repeated spans from the
  running `seq`. Pure host, no weights. Unit-tested (`tests/test_draft_lookup.cc`).
- `speculative_verify.{h,cc}` — the pure-host gate + helpers, no CUDA:
  - `verify_greedy(target_argmax, draft, K) -> VerifyResult{accepted, correction}`.
    Semantics: walk j in [0,K); first mismatch accepts j tokens and sets
    `correction = target[j]`; all-match accepts K with `correction = -1`.
  - `commit_speculative(v, draft, K, bonus, budget) -> CommitPlan{commit, tokens}` —
    which tokens the step commits, clamped to the output budget (the accepted
    prefix, then the gate's tail token: the correction on rejection, the bonus
    when the draft was fully accepted). Extracted from the loop for unit testing.
  - `DraftReport{steps, proposed, accepted}` + `draft_accept_rate(report)` — the
    cumulative counters and their ratio.
  - `generate_document(generated, report)` — the single producer of the
    `generate` stdout JSON. With `report` non-null it emits
    `{"generated":[...],"draft":{"steps":S,"proposed":P,"accepted":A,
    "accept_rate":R}}`; with `report == nullptr` it emits `{"generated":[...]}`
    only. All four `generate` emit sites (CUDA draft, CUDA plain, host draft,
    host plain) call this, so the shape is pinned by `test_speculative_verify.cc`
    and can't drift between the CUDA and host paths.
- `infer_main.cc` `generate`:
  - **CUDA path** (`--gpu 1`): draft-verify-commit loop is integrated. Per step it
    proposes K, runs one target forward over `accepted prefix + draft`, verifies each
    proposal row, commits the accepted prefix + correction, then drops KV rows
    written for rejected drafts (`kv_watermark_restore`).
  - **Host oracle path** (`--gpu 0`, the default): a host mirror of the same
    loop, using `forward_logits_f32` and argmax rows. Because there is no KV
    watermark on the host path, each step replays the whole accepted prefix from
    position 0 (correct, just slow). Used to *measure acceptance rate* before the
    fused k-token verify CUDA path exists.
  - CLI flag `--draft-lookup K` (0 = off). Draft path requires greedy
    (`--seed` must be omitted; enforced with a clear error).
  - Reports cumulative counters as `G4_DRAFT steps=... proposed=... accepted=...`
    on stderr **and** mirrors them in the stdout JSON `draft` object above
    (the structured metric a benchmark harness parses; see `DraftReport`).
  - `max_pos` is sized to `tokens + max_tokens + draft_k` to fit the draft window.

### Design notes / invariants to respect

- Verification is **exact only under greedy**. That is why the draft path rejects
  `--seed != 0` (top-p sampling can't be re-derived from argmax rows).
- The CUDA verify loop and the host mirror must stay in lockstep. If you change
  the accept/reject logic, change both (and `speculative_verify.{h,cc}` + the
  `test_speculative_verify.cc` cases that pin the document shape).
- The K-fused target forward produces target-argmax for each draft position in
  one pass; the host path approximates this by reading per-position rows from a
  single `forward_logits_f32` call.

### Just completed: k-token batched verify forward wired into the draft loop

- `forward_range_batched_impl` in `forward_gpu.cu` now forwards a 2–4-row
  suffix in one token-major batch: batched rmsnorm → QKV dequant +
  position-aware RoPE (cos/sin pre-offset by `start*half`, no double-rope) →
  persistent KV append at `start+r` → fused causal GQA (direct cache) →
  O/down residual GEMV → fused-K `verify_gemv` lm_head. Dispatch is in
  `forward_range_impl` when `2 ≤ rows ≤ 4 && direct_kv && gqa_fusable`;
  `G4_BATCH_VERIFY=0` forces the per-position loop; on any failure the caller
  falls through to the loop (KV append is idempotent).
- Parity: the batched path is **numerically equivalent**, not bit-identical.
  The batched GQA stage is bit-identical to its single-token twin; the batched
  QKV-dequant, residual projection, SiLU, and fused-K GEMV are each within
  2 ULP of their single-token twins (per the primitive-level tests), and that
  tolerance compounds across layers. `test_batched_forward_range_parity_device`
  asserts logits within 32 ULP **and greedy-argmax stability** (the invariant
  the draft-verify loop actually relies on), plus truncate+recompute for the
  draft-loop rejection path. Verified on gremlin-1: full device suite passes;
  `G4_TIMESTEP` shows the batched path taken at widths 4/3/3/2 plus the
  truncate redos, including `start=0` prefill.
- Candidate next actions (ordered by effort):
  1. **End-to-end speedup of the batched verify**: `generate --draft-lookup K`
     on a repeated prompt, `G4_BATCH_VERIFY=0` (per-position) vs default
     (batched), wall time + `G4_DRAFT` counters on gremlin-1; confirm the
     generated tokens are identical between the two. The per-step verify is now
     O(suffix) instead of O(suffix positions over the full prefix), so the
     win should show up as step wall time at K=4 / high accept rate.
  2. **CUDA-vs-host token comparison under `--draft-lookup`**: run `generate
     --draft-lookup K` on both `--gpu 0` and `--gpu 1` over a fixed repeated
     prompt and diff the `generated` token lists, to confirm the GPU draft path
     reproduces the oracle. Needs gremlin-1; the stdout `draft` metric now makes
     acceptance-rate comparison a one-line JSON parse.
  3. Optionally extend the `draft` JSON object with per-step timing (wall clock
     per verify round) once the end-to-end path is measured.

## Key files (engine/src/)

| File | Role |
|---|---|
| `infer_main.cc` | CLI + CUDA/host generate + draft-verify-commit loop |
| `forward_gpu.cu` | fused decode GEMV, `gpu_ctx_create`, `gpu_weights_load` |
| `kernels_phase2.cu`, `kernels_phase3.cu`, `kernels.cu` | kernel variants/phases |
| `draft_lookup.{h,cc}` | draft proposal (repeated spans) |
| `speculative_verify.{h,cc}` | greedy accept/reject gate + `commit_speculative` + `DraftReport`/`generate_document` (pure host) |
| `verify_gemv.cu` | fused-k verify GEMV primitive, used by the batched verify forward |
| `kv_pool.cc` | KV pool + watermark restore (drop rejected-draft rows) |
| `fp8_convert.cc`, `gemv_bench.cu`, `kernels_bench.cu` | quant + micro-benchmarks |
| `capi.{h,cc}` | C API surface |
| `safetensors.h`, `checkpoint_sha256.cc`, `checkpoint_cli.cc` | weight loading / integrity |
| `model_config.h`, `config_validation.{h,cc}`, `kernel_math.h` | config + math helpers |

Supporting docs: `docs/architecture.md`, `docs/phases.md`, `docs/golden.md`,
`docs/memory.md`, `docs/remote-checkpoint.md`, `docs/deployment.md`.

## Reproduce

- Build (host-only, no CUDA toolkit): `cmake -S . -B build -DGRANITE_ENABLE_CUDA=OFF
  -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build`.
  The CUDA path needs nvcc + sm_89 (gremlin-1); the host oracle + all pure-host
  tests build and pass without a GPU.
- Host oracle (no GPU): `granite-4-infer logits --checkpoint <ckpt> ...`
- GPU generate: `granite-4-infer generate ... --gpu 1 --max-tokens N`
- Speculative: add `--draft-lookup K` (omit `--seed`). The acceptance counters
  land in the stdout JSON `draft` object and on stderr as `G4_DRAFT ...`.
- Golden/correctness: see `docs/golden.md` for the host-vs-CUDA comparison target.
