# Remote checkpoint inspection — Phase 0 planning

> **Status: helper prepared, nothing executed.**
> [`scripts/remote_checkpoint_inspect.sh`](../scripts/remote_checkpoint_inspect.sh)
> defaults to **dry-run**: it prints the exact `ssh` command it would run and
> opens **no** network connection. No ssh session, no copy, no git operation,
> no deployment, and no remote command of any kind has been executed as part
> of preparing this helper. This document describes what *must* be true
> before an explicit `--execute` run can succeed.

## What the helper does

One script, one ssh session, metadata only:

| Step | Remote action | What it touches |
| --- | --- | --- |
| 1 | Read `config.json` and `tokenizer_config.json` | presence, JSON validity, **top-level key names only** — values are never printed |
| 2 | List the checkpoint directory; hash `*.json`, `*.txt`, `*.model` files | name, size, `sha256` (streamed in 1 MiB chunks, stdlib `hashlib` only) |
| 3 | Run `scripts/inspect_checkpoint.py <checkpoint>` from the staged repo copy | the inspector's own stdlib-only report + exit code |

Hard guarantees (by construction of the script):

- **Dry-run is the default.** `--execute` is required to connect; every
  run without it is pure local printing.
- **Model weights are never opened.** Files whose extension is not
  `.json` / `.txt` / `.model` (e.g. `*.safetensors`, `*.bin`, `*.pt`) are
  listed by name and skipped.
- **Nothing is downloaded** to the local machine. The only thing that comes
  back over ssh is the small text report on stdout.
- **No remote file is created, modified, or deleted.** The payload is
  read-only Python (stdlib only: `json`, `os`, `sys`, `hashlib`,
  `subprocess`) plus the inspector itself, which is also read-only.
- **`ssh -n`** pins stdin to `/dev/null` so no local file can leak into the
  session; the target and payload are shell-quoted (`%q`) by the script.
- The local checkpoint argument must be an absolute path free of shell
  metacharacters, and the script exits `2` (before any output mode) if the
  argument, `REMOTE_PROJECT_PATH`, or `REMOTE_PYTHON` fails validation.

## Prerequisites (all must hold before `--execute` can succeed)

1. **SSH access to `root@10.1.1.12`** (the default `REMOTE`; override via the
   `REMOTE` environment variable). Key-based, non-interactive ssh is
   assumed — the helper never reads a password and runs `ssh -n`.
2. **`python3` on the remote** (default interpreter; override with
   `REMOTE_PYTHON`). The payload and the inspector are **stdlib only** — no
   pip packages, no network, no torch/transformers.
3. **A checkpoint directory on the remote** with at least
   `config.json` and ideally `tokenizer_config.json` at its top level.
   Metadata files are capped at 5 MiB before parsing (a normal
   `config.json` is a few KB).
4. **The repository staged on the remote** — see below; this is the one
   prerequisite the helper *cannot* do for you.

## Exact staging requirement

Step 3 of the payload runs the inspector **from the staged copy of this
repository**, not from the local machine:

```
REMOTE_PROJECT_PATH/scripts/inspect_checkpoint.py   # must exist on the remote
```

- Default staging path: **`/opt/granite4-rtx-4070-ti-super`** (variable
  `REMOTE_PROJECT_PATH` in
  [`scripts/remote_checkpoint_inspect.sh`](../scripts/remote_checkpoint_inspect.sh)).
  It must contain a byte-for-byte current `scripts/inspect_checkpoint.py`,
  i.e. the copy must be made **after** any local change to that file.
- The staging copy is created by a *separate, explicitly approved* copy
  step (see [`docs/deployment.md`](deployment.md)). The inspection helper
  performs **no copy** — it only checks for the inspector's presence and
  fails with `STAGING REQUIRED` (remote exit 3) when the copy is missing.
- The staging directory is a copy destination, never a write target of the
  inspection run itself; the flake at `/etc/nixos/flake.nix` must never be
  part of it (same hard rule as the deployment runbook).
- If the staged copy is stale, the inspector's verdict reflects the stale
  spec — refresh the copy before trusting step 3.

## Command examples

Dry run (default — prints the exact command, connects to nothing):

```sh
scripts/remote_checkpoint_inspect.sh /data/checkpoints/granite-4.2-8b
```

Dry run with overrides (different host / staging path / interpreter):

```sh
REMOTE=root@10.1.1.13 \
REMOTE_PROJECT_PATH=/srv/granite4 \
REMOTE_PYTHON=/usr/bin/python3.11 \
  scripts/remote_checkpoint_inspect.sh /data/checkpoints/granite-4.2-8b
```

Execute the one read-only inspection (explicit opt-in):

```sh
scripts/remote_checkpoint_inspect.sh /data/checkpoints/granite-4.2-8b --execute
```

Reading the result: the report ends with the inspector's verdict. A
planning-field **mismatch** (remote exit 1) is a *finding*, not a failure of
the helper — it means the checkpoint disagrees with the spec baked into
`REQUIRED_FIELDS` in
[`scripts/inspect_checkpoint.py`](../scripts/inspect_checkpoint.py).

## Expected behavior when remote access is unavailable

Until ssh access to the remote exists, an `--execute` run is **expected to
fail** and that failure is the correct, honest outcome:

- ssh cannot connect → ssh exits `255` → the helper prints
  `remote access UNAVAILABLE` and exits **4**. No payload was reached;
  nothing about the checkpoint has been learned. Do **not** record any
  planning value from such a run.
- Dry-run mode never touches the network, so it always succeeds locally —
  it proves the command to run, not the remote state.

If the connection drops mid-run, the same exit 4 applies; re-run after
restoring access. A partial report is never to be treated as verified.

## Verification loop — bring values back *before* touching planning constants

The only purpose of this helper is to **verify** remote reality and carry
verified facts into this repository. The rule:

1. Run the inspection (dry-run first, then `--execute` once access exists).
2. Record the report verbatim (top-level key names, metadata file names +
   `sha256`, inspector verdict) in this repository — e.g. as a dated
   artifact next to [`docs/golden.md`](golden.md) — so it is auditable.
3. **Only then** may planning constants be modified:
   - `REQUIRED_FIELDS` in [`scripts/inspect_checkpoint.py`](../scripts/inspect_checkpoint.py),
   - and the matching C++ side in [`engine/src/model_config.h`](../engine/src/model_config.h)
     / [`engine/src/config_validation.cc`](../engine/src/config_validation.cc).
4. Never edit a planning constant "from memory of what the remote showed".
   The constant change must cite the recorded report; the report must cite
   the `sha256` values and the inspector exit code that justified it.

Until this loop has produced a recorded report, **no** claim in the
repository may assert that the checkpoint at `root@10.1.1.12` matches the
Granite 4.2 8B planning spec — the spec match is *unverified*, exactly as
stated in [`docs/deployment.md`](deployment.md).

## Safety & auditability checklist

- [x] Dry-run default; `--execute` is the only path to the network
- [x] Single ssh session, `ssh -n`, shell-quoted target + payload
- [x] Stdlib-only remote payload; no pip, no torch, no network on the remote
- [x] Metadata JSONs: key names only, values never printed
- [x] Weights listed by name, never opened; no downloads either direction
- [x] No remote writes of any kind
- [x] Local argument validation with fail-fast exit 2 before any output
- [x] Exit codes documented: `0` ok · `2` bad local args · `3` remote
      failure (mismatch / invalid input / not staged) · `4` no remote access
- [x] Known limitation: this helper verifies *metadata and the spec check
      only*; it says nothing about weight correctness, GPU behavior, or
      serving performance — those remain out of scope by design
