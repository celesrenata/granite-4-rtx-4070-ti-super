# Deployment runbook — Phase 0 scaffold

> **Status: planning only.** Every command below is written for a later,
> separate deployment session. Nothing in this document has been executed as
> part of the Phase 0 scaffold work. The companion script
> [`scripts/deploy-dry-run.sh`](../scripts/deploy-dry-run.sh) prints these
> commands without executing any of them.

## What is true right now (Phase 0)

- The API is **intentionally unavailable**: `POST /v1/chat/completions`
  returns **501 `inference_unavailable`** for valid requests (400 for
  invalid ones), and `GET /health` reports `"inference": "unavailable"`.
  This is the documented Phase 0 boundary ([`frontend/api.py`](../frontend/api.py),
  [phases.md](phases.md)) — it is not a deployment bug and must not be
  "fixed" into serving tokens.
- **CUDA and remote checks are unverified.** The CUDA path
  ([`engine/src/kernels.cu`](../engine/src/kernels.cu)) has not been
  compiled or run on an sm_89 machine, and nothing in this tree has been
  tested on the remote host. No claim about GPU behavior or remote state may
  be made until a real build runs there in a later phase.
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

## Step 4 — Copy to the remote staging directory

Fresh, clearly named directory per copy:

```sh
STAMP=$(date +%Y%m%d-%H%M%S)
REMOTE_STAGING=/opt/granite4-rtx-4070-ti-super/staging-$STAMP

ssh root@10.1.1.12 "mkdir -p '$REMOTE_STAGING'"
rsync -avz --delete --exclude '.git' --exclude 'build/' --exclude 'reference/' \
  ./ root@10.1.1.12:"$REMOTE_STAGING/"
# (if rsync is unavailable on the remote: scp -r ./ root@10.1.1.12:"$REMOTE_STAGING/"
#  then remove .git/, build/, reference/ inside the staging dir)
```

- **Do not** copy into `/etc/nixos/` and **do not overwrite
  `/etc/nixos/flake.nix`**. The only write target in this step is
  `$REMOTE_STAGING`.
- `--exclude reference/` — that subtree is a different model/GPU kept for
  layout conventions only and is not part of the deployable artifact.
- Verify the copy (read-only, on the remote):

```sh
ssh root@10.1.1.12 "test -f '$REMOTE_STAGING/CMakeLists.txt' && test -f '$REMOTE_STAGING/frontend/api.py' && echo copy-ok"
```

## Step 5 — Flake integration (separate step, only after explicit user approval)

1. Review the current remote flake before touching it:

   ```sh
   ssh root@10.1.1.12 "cat /etc/nixos/flake.nix"
   ```

2. Review the proposed flake module against it and present the exact diff
   to the user. **Integration happens only after the user approves the
   diff.**
3. Apply the module on the remote, then rebuild and restart:

   ```sh
   ssh root@10.1.1.12 "sudo nixos-rebuild build --use-rollback-switch"
   ssh root@10.1.1.12 "sudo nixos-rebuild switch"
   ssh root@10.1.1.12 "sudo systemctl restart <service>"
   ```

## Step 6 — Verification (expected Phase 0 behavior)

```sh
curl -s http://10.1.1.12:8080/health     # -> 200, "inference":"unavailable"
curl -s http://10.1.1.12:8080/metrics    # -> 200, engine status as built
curl -s http://10.1.1.12:8080/v1/models  # -> 200, status "unavailable"
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  http://10.1.1.12:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"granite-4.2-8b","messages":[{"role":"user","content":"hi"}]}'
# -> 501 (intentional in Phase 0; see tests/test_api.py)
```

A healthy Phase 0 deployment is exactly this: the API process is up,
endpoints respond, and inference is **explicitly** unavailable. Do not
treat the 501 as a failure, and do not declare CUDA or remote success from
any of this — GPU behavior remains unverified until Phase 1+ work lands and
is exercised on the sm_89 machine.

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

## Known box issue: `nvidia-smi` segfaults (not an NVIDIA problem)

`nvidia-smi` (from `nix/store/…-nvidia-x11-615.78.08-bin`) crashes with
SIGSEGV *inside `ld-linux-x86-64.so.2` during loader init* (fault at the
binary's ELF program-header region, before any library is loaded). Diagnosis
on gremlin-1 (2026-10-23): the CPU has Intel **PKU** (`pku`/`ospke`), the
box runs a bleeding-edge **7.2.11** kernel, and the loader is the store's
**glibc 2.44-57** dynamic linker — a loader/kernel interaction, *not* the
GPU. The store's glibc 2.42 loader runs fine, the CUDA runtime talks to the
615 driver cleanly (`cudaGetDeviceCount()` → RTX 4070 Ti SUPER, sm 8.9,
driver 13.4), and the device tests + benchmark pass. Until the box's kernel
or the nixpkgs glibc changes (re-evaluate after either), use these instead:

```sh
granite4-kernel-test    # exercises the driver + GPU
granite4-kernel-bench   # bandwidth + memory baseline (JSON below)
cat /proc/driver/nvidia/version   # kernel module version (615.78.08)
```
The P1 residency criterion (measured memory) is read from
`granite4-kernel-bench`'s `mem_total_mib` / `mem_free_mib` fields — the
CUDA runtime's `cudaMemGetInfo` view, which works where `nvidia-smi` does
not. Run it after the model is loaded (P3) and the difference from the
pre-load free memory is the residency figure.

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
