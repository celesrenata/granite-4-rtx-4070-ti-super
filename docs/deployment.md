# Deployment runbook

> **Status (2026-11-03): deployed on gremlin-1.** `services.granite4-rtx`
> serves chat completions from the GPU engine behind a parity gate (Step 5).
> The staging/copy steps below remain the way to try an unpushed tree.

## What is true right now

- gremlin-1 runs the flake's NixOS module with a checkpoint configured. Each
  service start runs the parity gate on the GPU; on PASS the API reports
  `"inference": "available"` and answers `POST /v1/chat/completions` through
  one resident `granite4-infer serve --gpu 1` process. On FAIL the API stays
  up in its Phase-0 state (501 `inference_unavailable`) - a gate failure is
  never "fixed" into serving.
- Without `services.granite4-rtx.checkpoint` the module is still the Phase-0
  scaffold (validates requests, 501 for completions, `PrivateDevices=true`).
- No secrets are stored anywhere in this repository; do not add any as part
  of deployment.

## Target

| Item | Value |
| --- | --- |
| Remote host | `root@10.1.1.12` |
| Staging directory | `/opt/granite4-rtx-4070-ti-super/staging-<timestamp>` — clearly named, created fresh per copy |
| NixOS flake | `/etc/nixos/flake.nix` — **never a write target of the copy step** |
| Service / API port | placeholders (`granite4-rtx`, `8080`) until the flake module lands |

**Hard rule:** the copy step writes *only* into the staging directory. The
NixOS flake is touched in a separate, later step (Step 5) and only after
explicit user approval.

## Step 1 — Local checks (all must pass before anything else)

```sh
uv run --group dev pytest            # or: python3 -m pytest
cmake -S . -B build -G Ninja -DGRANITE_ENABLE_CUDA=OFF
cmake --build build
ctest --test-dir build --output-on-failure
```

These are host-only: no GPU, no CUDA toolkit, no network access.

## Step 2 — Review the working tree

```sh
git status
git diff
git diff --staged
```

Confirm only the intended files changed.

## Step 3 — Commit / push (placeholders)

```sh
git add docs/deployment.md scripts/deploy-dry-run.sh
git commit -m "phase0: deployment runbook and dry-run script"
# PLACEHOLDER — run only after review and explicit user approval:
# git push origin <branch>
```

## Step 4 — Copy the pushed commit to the remote staging directory

Fresh, clearly named directory per copy. Ship **only the git-tracked tree
of the pushed commit**, via `git archive` (≈ 1 MB). This is deterministic
and structurally excludes everything else in the local checkout — the local
`.gt/` checkpoint (~34 GB, "Never stage this", `.gitignore:12`), `build/`,
and any future junk directory:

```sh
STAMP=$(date +%Y%m%d-%H%M%S)
REMOTE_STAGING=/opt/granite4-rtx-4070-ti-super/staging-$STAMP

ssh root@10.1.1.12 "mkdir -p '$REMOTE_STAGING'"
git archive --format=tar HEAD \
  | ssh root@10.1.1.12 "tar -x -C '$REMOTE_STAGING'"
```

- **Do not** copy into `/etc/nixos/` and **do not overwrite
  `/etc/nixos/flake.nix`**. The only write target in this step is
  `$REMOTE_STAGING`.
- The `reference/qwen3-27b-5090` submodule is a different model/GPU kept
  for layout conventions only; it is not expanded by `git archive` and is
  not needed on the RTX host. The small tracked `reference/granite/`
  Python files do ship.
- **Do not** use a raw `rsync ./` here. This checkout contains `.gt/`
  (34 GB) and the old `--exclude '.git' --exclude 'build/' --exclude
  'reference/'` list did not cover it — a plain copy hangs (observed:
  10-minute timeout killed mid-transfer, 32 GB already sent).
- Verify the copy (read-only): sentinels plus a byte-exact sha256
  manifest, computed identically on both ends:

```sh
ssh root@10.1.1.12 "test -f '$REMOTE_STAGING/CMakeLists.txt' && test -f '$REMOTE_STAGING/frontend/api.py' && echo copy-ok"

TMP=$(mktemp -d); git archive --format=tar HEAD | tar -x -C "$TMP"
( cd "$TMP" && find . -type f | sort | xargs -n50 shasum -a 256 | sort ) > /tmp/m_local.txt
ssh root@10.1.1.12 "( cd '$REMOTE_STAGING' && find . -type f | sort | xargs -n50 sha256sum | sort )" > /tmp/m_remote.txt
diff /tmp/m_local.txt /tmp/m_remote.txt && echo manifest-ok
rm -rf "$TMP"
```

Expected: 90 files (91 tracked entries = 90 files + 1 gitlink for the
`reference/qwen3-27b-5090` submodule, which `git archive` does not expand).
`diff` must be empty.

## Step 5 — Flake integration

The host flake (`/etc/nixos/flake.nix` on gremlin-1) takes this repo as the
`granite4` input and enables the module:

```nix
granite4.nixosModules.granite4Rtx
({ ... }: {
  services.granite4-rtx = {
    enable = true;
    package = granite4.packages.${system}.api;
    enginePackage = granite4.packages.${system}.engine-cuda;
    checkpoint = "/opt/granite4-rtx-4070-ti-super/ckpt";  # BF16 safetensors + tokenizer
    maxContext = 8192;            # positions per request (160 KiB VRAM each, bf16 KV)
    kv8 = true;                   # int8 KV cache (opt-in; ~52% less KV VRAM, gate stays bf16)
    # 64K context x 2 sessions (131072-position pool) needs int4 KV and a
    # quantized EAGLE draft KV instead (see the table below):
    #   kv4 = true; maxContext = 65536; sessions = 2; eagle.draftKv = 8;
    listenAddress = "127.0.0.1";
    listenPort = 2701;
  };
})
```

What the module does with a checkpoint:

| Unit step | What runs |
| --- | --- |
| `ExecStartPre` | remove the previous gate artifact, then `granite4-parity-gate --gpu --mode teacher-forced --reference <api>/lib/reference/granite/parity_ref.json` (one GPU forward, ~10 s); writes `/var/lib/granite4-rtx/parity_gate.json` on PASS and `parity_report.json` always. A failing gate does not fail the unit (`-` prefix). |
| `ExecStart` | `granite4-api` with `GRANITE4_GATE` / `GRANITE4_MAX_CONTEXT`; on a passing GPU gate it starts the resident engine in the background (load + warm-up ~8 s) |
| sandbox | `ProtectSystem=strict`, `ProtectHome`, `NoNewPrivileges`, `DevicePolicy=closed` with only the NVIDIA device nodes allowed, `StateDirectory=granite4-rtx` |

The gate (`scripts/parity_gate.py`) compares the engine against a shipped
fp32 transformers greedy reference for "The capital of France is" (24
tokens), teacher-forced: each position passes when the reference token is the
engine's top-1 or within `gate.ulpTolerance` (default 2) bf16 ULPs of it;
`gate.minMatch` (default 1.0) of positions must pass. On gremlin-1: 24/24
within 2 ULP, 21/24 exact top-1 (the three misses are two exact bf16 ties and
one 2-ULP gap). Free-running token matching is still available
(`--mode free`, the original P3 gate) for the f32 host oracle; for the bf16
GPU path one near-tie would flip the rest of the sequence.

KV8 (int8 KV storage) is opt-in per deploy with `kv8 = true` (the NixOS
module then sets `G4_KV8=1` in the unit environment). The unit environment is
shared by `ExecStartPre` (the gate) and `ExecStart` (the API + its serve
engine), so the gate pins `G4_KV8=0` in its own engine subprocesses
(`scripts/parity_gate.py`, `_gate_env`): the gate always validates the bf16
*arithmetic* path, and a `kv8` deploy only switches the serve engine's KV
*storage*.

What KV8 buys (measured on gremlin-1, `docs/memory.md` "Measured KV8 A/B"):
the KV pool is 82.5 KiB/pos (80 KiB int8 codes + 2.5 KiB per-row scales) vs
164 KiB/pos bf16 (160 KiB codes + the 4 KiB/pos split-attention compact
prefix) — ~48% smaller, so a larger context pool fits — at the cost of
prefill +3.7…13.9% and decode −1.9…3.5% tok/s: a *capacity* win, not a
speed win. Quality: the teacher-forced bf16-vs-int8 comparison flips the
argmax only on 1-ULP near-ties (23/24 identical, every position within 2 ULP
on the gate prompt). The int8 round-trip error is also bounded by the kernel
unit tests.

`G4_KV8=4` / `kvFormat = "int4"` (per-row-scale int4, 2 codes/byte) goes
further: 42.5 KiB/pos (40 KiB packed codes + 2.5 KiB scales, ~74% smaller
than bf16). It is the only KV format that lets the full 131072-position
reader pool fit a 16 GiB card alongside the FP8 weights for the *greedy*
path: 131072 positions × 42.5 KiB = **5436 MiB** of KV, which sits under
the ~16376 MiB card alongside the ~9.2–9.8 GiB weights/scratch floor
(see `docs/memory.md`, "Q4_0-KV (int4) footprint"). Serve-level A/B
(measured on a freed gremlin-1, `scripts/kv8-ab-bench.sh 4`): int4 holds
~122 KiB/pos less than bf16 (~1.5× int8's saving, ~2× the positions for
the same budget) but is **slower, not faster** — prefill +6.7…24.8%
(worse than int8's +3.7…13.9%, the penalty grows with context) and decode
−1.8…3.1% t/s (tracking int8). Pick int8 when latency matters and the pool
fits int8 (~64K positions: 64K × 1 or 32K × 2); pick int4 only when the context pool must exceed what int8 can hold.
The speculative EAGLE drafter keeps its own persistent KV, which scales
with the pool and is a separate budget — at 131072 positions the draft
head's bf16 KV alone is ~537 MiB, which OOMs the 131K int4-reader config
(the head dies on its last allocation, "eagle head: rope tables").
`G4_DRAFT_KV` quantizes that draft KV (16 = bf16, the default; 8 = Q8_0;
4 = Q4_0), reusing the reader's int8/int4 append + split-KV attention
kernels; both quantized widths require the split-attention shape
(d==128, Hq==4*Hk) and fall back to bf16 otherwise. Measured fit on
gremlin-1 (int4 reader pool @ 131072, r8 head, 1 session):

| draft KV | mem used | free |
| --- | ---: | ---: |
| 16 (bf16) | **OOM** | — |
| 8 (int8) | 15795 MiB | 233 MiB |
| 4 (int4) | 15667 MiB | 361 MiB |

with the default path unchanged (bf16 draft @ 32768: 15359 MiB, the live
footprint). Drafts are still target-verified in full precision, so a
quantized draft KV can only lower the acceptance rate, never change the
output. Measured quality (int4 reader @ 32768, r8 head, eagle_k=4, 3
prompts × 120 tokens):

| draft KV | accept rate | speedup | output |
| --- | ---: | ---: | --- |
| 16 (bf16) | 0.161 | 1.594× | ref |
| 8 (int8) | 0.161 | 1.592× | byte-identical |
| 4 (int4) | 0.159 | 1.581× | byte-identical |

So int8 draft KV is lossless (same accept rate as bf16) and int4 costs
~1% acceptance — negligible. The per-row int4 error is ~16× the int8
one, yet the draft KV cache only ever holds the current request's
*generated* tokens (a 1-layer head, not the long reader context), so
the quantization error stays small and barely moves the accept rate.
`G4_DRAFT_KV=8` is the lossless 131K default; `4` is the headroom
option if you need the last ~128 MiB.

Deploy a pushed commit:

```sh
ssh root@10.1.1.12 'cd /etc/nixos && nix flake update granite4 && \
  nixos-rebuild build --flake .#gremlin-1 && \
  nix store diff-closures /run/current-system ./result && \
  nixos-rebuild switch --flake .#gremlin-1'
```

Try an unpushed tree (Step 4 copy at `$D`, committed in a throwaway git repo):
`nixos-rebuild build --flake .#gremlin-1 --override-input granite4 git+file://$D`
then `./result/bin/switch-to-configuration test` (not added to the boot menu).

## Step 6 — Verification

```sh
ssh root@10.1.1.12 'journalctl -u granite4-rtx -b -o cat | grep -E "PARITY|engine ready"'
# -> PARITY PASS (min-match 100.0%) ... granite4: GPU engine ready
ssh root@10.1.1.12 'curl -s 127.0.0.1:2701/health'      # "inference":"available"
ssh root@10.1.1.12 "curl -s -X POST 127.0.0.1:2701/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{\"model\":\"granite-4.2-8b\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one sentence.\"}]}'"
# -> "The capital of France is Paris.", finish_reason "stop", ~0.23 s
ssh root@10.1.1.12 'cat /var/lib/granite4-rtx/parity_report.json'   # per-position detail
```

Full live check (25 assertions: endpoints, answers, determinism, limits,
validation, 8 concurrent requests; stdlib only):

```sh
scp scripts/service_smoke.py root@10.1.1.12:/tmp/
ssh root@10.1.1.12 'nix shell nixpkgs#python3 -c python3 /tmp/service_smoke.py'
# -> 25/25 passed
```

Request limits on the GPU path: `max_tokens` defaults to 1024 and is capped
at `maxContext - prompt_tokens`; a prompt longer than the context is a 400
`context_length_exceeded`. Generation stops at the checkpoint's
`eos_token_id` (`<|im_end|>`), which is counted in `completion_tokens` but
not included in the text.

## Step 7 — Rollback

In reverse order, least destructive first:

```sh
# 1. Roll back to the previous NixOS generation:
ssh root@10.1.1.12 "sudo nixos-rebuild --rollback"
# 2. Stop the service if it should not keep running:
ssh root@10.1.1.12 "sudo systemctl stop <service>"
# 3. Remove the staging copy only after the rollback is confirmed
#    and the user approves:
ssh root@10.1.1.12 "rm -rf '$REMOTE_STAGING'"
# 4. If the local commit itself was the problem, revert it locally:
git revert <commit>
```

Rollback relies on NixOS generations; it must never require hand-editing
`/etc/nixos/flake.nix` back.

## Fixed box issue: `nvidia-smi` segfaulted (patchelf, 2026-11-03)

`nvidia-smi`, `nvidia-powerd` and `nvidia-cuda-mps-control` from the 615.78.08
driver package crashed in `ld-linux-x86-64.so.2` before `main` ("error 7" =
write fault at 0x401568). Cause: nixpkgs' driver builder rewrites the
binaries' interpreter and RUNPATH with patchelf 0.18/0.19, which moved
`.dynamic` into a LOAD segment that is mapped read-only; ld.so writes
`DT_DEBUG` there at startup. Making that segment writable, or re-patching the
untouched `origBin/` copies with patchelf 0.15, fixes it (the earlier
PKU/kernel theory was wrong). gremlin-1's `modules/graphics.nix` now
overrides the driver package's `postFixup` to re-patch those binaries with
`nixpkgs-stable`'s patchelf (0.15.2); the kernel module derivation is
unchanged. Drop the override once nixpkgs ships a patchelf that keeps
`.dynamic` writable (check: `nvidia-smi -L` on a build without it).

## P2 — GPU re-verification (gremlin-1)

The flake's `packages.x86_64-linux.engine-cuda` compiles the full sm_89
engine (kernels, `granite4-kernel-test`, `granite4-kernel-bench`) and runs
every ctest suite *except* `kernels_device` — the device *run* needs the host
driver's `libcuda` and a live GPU, which a build sandbox cannot reach. On the
GPU box it is one command each (binaries ship in the package's `bin/`):

```sh
ssh root@10.1.1.12 "granite4-kernel-test"     # "all device kernel tests passed"
ssh root@10.1.1.12 "granite4-kernel-bench"    # JSON: measured read/copy GB/s + roofline note
```

`granite4-kernel-bench` prints the measured bandwidth baseline (the P2 exit
criterion: measured, not assumed). Per the cross-phase rule, the numbers are
not committed to this repo before P4; they are reproducible from the binary
with the commands above.
