#!/usr/bin/env bash
# remote_checkpoint_inspect.sh — safe remote checkpoint metadata verifier.
#
# DRY RUN BY DEFAULT: without `--execute` this script only PRINTS the exact
# ssh command that would be run against the remote host. It never opens a
# network connection, and no remote command is executed.
#
# With `--execute` it runs exactly ONE read-only ssh session. The remote
# payload is stdlib/Python metadata inspection only:
#   * config.json and tokenizer_config.json: presence, JSON validity,
#     top-level key names (key NAMES only — values are never printed);
#   * metadata files (*.json, *.txt, *.model): top-level name + size +
#     sha256 (streamed in 1 MiB chunks; model weight files are never opened);
#   * this repository's scripts/inspect_checkpoint.py (stdlib only), run
#     from the staged remote copy of the repository.
# The payload performs no downloads, no writes, and no other remote work.
# Remote files are never altered.
#
# Usage:
#   scripts/remote_checkpoint_inspect.sh CHECKPOINT_DIR [--execute]
#   scripts/remote_checkpoint_inspect.sh --execute CHECKPOINT_DIR
#
# Configurable via environment (defaults shown):
#   REMOTE                root@10.1.1.12
#   REMOTE_PROJECT_PATH   /opt/granite4-rtx-4070-ti-super  (staged repo copy)
#   REMOTE_PYTHON         python3
#
# Exit codes:
#   0  dry-run printed (default) / remote inspection passed (remote exit 0)
#   2  bad local arguments (missing/invalid checkpoint or path)
#   3  remote inspection failed (planning-field mismatch, invalid input,
#      or repository not staged on the remote)
#   4  remote access unavailable (ssh could not connect or returned an
#      unexpected status; the payload was not successfully reached)

set -eu
LC_ALL=C

usage() {
  cat <<'EOF'
usage: remote_checkpoint_inspect.sh CHECKPOINT_DIR [--execute]

CHECKPOINT_DIR  absolute remote checkpoint directory, e.g.
                /data/checkpoints/granite-4.2-8b
--execute       actually run the one read-only ssh inspection
                (default is dry-run: print the exact command, connect to nothing)
-h, --help      show this help

Environment overrides:
  REMOTE               (default: root@10.1.1.12)
  REMOTE_PROJECT_PATH  (default: /opt/granite4-rtx-4070-ti-super)
  REMOTE_PYTHON        (default: python3)

Safety contract:
  * Default mode is dry-run: the exact ssh command is printed and nothing is
    executed. No network connection is opened.
  * With --execute, exactly one read-only ssh session is opened.
  * The remote payload uses only the Python stdlib: metadata JSON structure
    (top-level key names only, no values), metadata file names + sizes +
    sha256, and scripts/inspect_checkpoint.py from the staged repo copy.
  * Model weight files are never opened or read. Nothing is downloaded.
    No remote file is created, modified, or deleted.
EOF
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
CHECKPOINT_ARG=""
EXECUTE=0
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --execute)
      EXECUTE=1
      ;;
    --)
      ;;
    -*)
      printf 'ERROR: unknown option: %s\n\n' "$arg" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [ -n "$CHECKPOINT_ARG" ]; then
        printf 'ERROR: unexpected extra argument: %s\n' "$arg" >&2
        exit 2
      fi
      CHECKPOINT_ARG="$arg"
      ;;
  esac
done

if [ -z "$CHECKPOINT_ARG" ]; then
  printf 'ERROR: missing CHECKPOINT_DIR argument.\n\n' >&2
  usage >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# configuration (override via environment when the layout changes)
# ---------------------------------------------------------------------------
REMOTE="${REMOTE:-root@10.1.1.12}"
REMOTE_PROJECT_PATH="${REMOTE_PROJECT_PATH:-/opt/granite4-rtx-4070-ti-super}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# local guards — fail fast, before anything is printed or executed
# ---------------------------------------------------------------------------

# REMOTE must be a non-empty target and must not look like an ssh option
case "$REMOTE" in
  ""|-*)
    printf 'ERROR: REMOTE must be a non-empty ssh target like user@host (got: "%s")\n' "$REMOTE" >&2
    exit 2
    ;;
esac

# checkpoint must be an absolute path of conservative characters. It is
# embedded in the remote payload inside double quotes, so anything that
# carries shell/ssh metacharacters is rejected outright.
case "$CHECKPOINT_ARG" in
  /*) ;;
  *)
    printf 'ERROR: CHECKPOINT_DIR must be an absolute remote path (got: %s)\n' "$CHECKPOINT_ARG" >&2
    exit 2
    ;;
esac
case "$CHECKPOINT_ARG" in
  *[!A-Za-z0-9._/-]*|*#*|*\\*)
    printf 'ERROR: CHECKPOINT_DIR contains characters forbidden by this script (spaces, quotes, $, `; and other metacharacters):\n  %s\n' "$CHECKPOINT_ARG" >&2
    exit 2
    ;;
esac
for seg in ${CHECKPOINT_ARG//\// }; do
  if [ "$seg" = "." ] || [ "$seg" = ".." ]; then
    printf 'ERROR: CHECKPOINT_DIR must not contain "." or ".." segments (got: %s)\n' "$CHECKPOINT_ARG" >&2
    exit 2
  fi
done

# the staged repository copy must be an absolute path of conservative
# characters; "%" is rejected too because it participates in printf formats
case "$REMOTE_PROJECT_PATH" in
  /*) ;;
  *)
    printf 'ERROR: REMOTE_PROJECT_PATH must be an absolute path (got: %s)\n' "$REMOTE_PROJECT_PATH" >&2
    exit 2
    ;;
esac
case "$REMOTE_PROJECT_PATH" in
  *[!A-Za-z0-9._/-]*|*#*|*\\*|*%*)
    printf 'ERROR: REMOTE_PROJECT_PATH contains characters forbidden by this script:\n  %s\n' "$REMOTE_PROJECT_PATH" >&2
    exit 2
  ;;
esac

# REMOTE_PYTHON is interpolated unquoted into the remote payload, so it must
# be a bare executable name like "python3"
case "$REMOTE_PYTHON" in
  ""|*[!A-Za-z0-9._/-]*|*#*|*\\*|*%*)
    printf 'ERROR: REMOTE_PYTHON must be a bare executable name like "python3" (got: %s)\n' "$REMOTE_PYTHON" >&2
    exit 2
  ;;
esac

# the local repository must contain the inspector; the staged remote copy is
# expected to be a copy of this repository
if [ ! -f "${ROOT}/scripts/inspect_checkpoint.py" ]; then
  printf 'ERROR: scripts/inspect_checkpoint.py not found under %s\n' "$ROOT" >&2
  printf 'ERROR: the staged remote copy of the repository would lack the inspector.\n' >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# remote payload — built ONCE, identical for the dry-run print and for
# execution. The Python source itself contains no shell interpolation: the
# checkpoint and repo paths are passed via environment variables exported on
# the ssh command line (each quoted individually). Only $REMOTE_PYTHON is
# expanded into the payload, and it is validated above.
# ---------------------------------------------------------------------------
REMOTE_PY_PAYLOAD="$(
  cat <<'PY'
import hashlib, json, os, sys

CK = os.environ["G4CK"]
REPO = os.environ["G4REPO"]
META_NAMES = ("config.json", "tokenizer_config.json")
META_SIZE_LIMIT = 5 * 1048576
METADATA_EXTS = (".json", ".txt", ".model")
HASH_CHUNK = 1048576

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def top_keys(p):
    with open(p, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict):
        return None
    return [str(k) for k in doc.keys()]

print("=== remote-checkpoint-inspect (read-only; model weights never opened) ===")
print("checkpoint     : " + CK)
print("repository copy: " + REPO)
if not os.path.isdir(CK):
    print("ERROR: checkpoint directory not found: " + CK)
    sys.exit(2)

# 1) metadata JSON files: structure only — top-level key names, never values
for name in META_NAMES:
    p = os.path.join(CK, name)
    if not os.path.isfile(p):
        print(name + ": MISSING")
        continue
    if os.path.getsize(p) > META_SIZE_LIMIT:
        print(name + ": PRESENT but larger than 5 MiB - not parsed (not a normal metadata file)")
        continue
    try:
        keys = top_keys(p)
    except Exception:
        print(name + ": PRESENT but UNPARSEABLE (not valid JSON)")
        continue
    if keys is None:
        print(name + ": PRESENT, valid JSON, but top level is not an object")
    else:
        print(name + ": PRESENT, valid JSON object, %d top-level keys" % len(keys))
        print("  top-level keys: " + ", ".join(keys))

# 2) top-level names of every entry + sha256 for metadata files only
print("top-level entries: " + str(len(os.listdir(CK))))
for name in sorted(os.listdir(CK)):
    p = os.path.join(CK, name)
    if not os.path.isfile(p):
        print("  [dir] " + name)
        continue
    ext = os.path.splitext(name)[1].lower()
    if ext not in METADATA_EXTS:
        print("  [skip weights/large data] " + name)
        continue
    try:
        digest = sha256_file(p)
    except Exception:
        print("  [read error] " + name)
        continue
    print("  " + name + "  " + str(os.path.getsize(p)) + " bytes  sha256=" + digest)

# 3) this repository's inspector, run from the staged copy (stdlib only)
if not os.path.isfile(os.path.join(REPO, "scripts", "inspect_checkpoint.py")):
    print("ERROR: repository not staged at " + REPO + " (scripts/inspect_checkpoint.py missing)")
    print("STAGING REQUIRED: copy this repository (without .git) to " + REPO + " first;")
    print("this script performs no copy, download, or any other remote work.")
    sys.exit(3)
print("--- running inspector: scripts/inspect_checkpoint.py on " + CK + " ---")
import subprocess
try:
    rc = subprocess.call([sys.executable,
                          os.path.join(REPO, "scripts", "inspect_checkpoint.py"),
                          CK])
    if rc is None:
        rc = 3
except Exception:
    print("ERROR: inspector could not be run")
    rc = 3
if rc == 0:
    print("inspector verdict: all planning fields match the spec")
elif rc == 1:
    print("inspector verdict: PLANNING-FIELD MISMATCH - bring the reported")
    print("values back to this repository before modifying planning constants")
else:
    print("inspector verdict: INVALID INPUT (remote exit " + str(rc) + ")")
print("--- remote-checkpoint-inspect complete ---")
sys.exit(rc if rc in (0, 1, 2, 3) else 3)
PY
)"

# Single read-only remote command:
#   -n            stdin is /dev/null — nothing local can leak into the session
#   <<'PY'        quoted heredoc: the Python source reaches python3 literally
# %q              shell-safe quoting for the target, env values, and payload
REMOTE_CMD="$(
  printf "%s - <<'PY'\n%s\nPY" "$REMOTE_PYTHON" "$REMOTE_PY_PAYLOAD"
)"
SSH_QUOTED="$(
  printf 'ssh -n %q %q' "$REMOTE" \
    "$(printf 'G4CK=%q G4REPO=%q sh -c %q' \
        "$CHECKPOINT_ARG" "$REMOTE_PROJECT_PATH" "$REMOTE_CMD")"
)"

# ---------------------------------------------------------------------------
# mode: dry-run (default) prints only; --execute runs the one ssh command
# ---------------------------------------------------------------------------
if [ "$EXECUTE" -ne 1 ]; then
  printf '[dry-run] mode              : DRY RUN — nothing below is executed; no network connection is opened\n'
  printf '[dry-run] local repository  : %s\n' "$ROOT"
  printf '[dry-run] remote target     : %s\n' "$REMOTE"
  printf '[dry-run] remote checkpoint : %s\n' "$CHECKPOINT_ARG"
  printf '[dry-run] staged repo path  : %s\n' "$REMOTE_PROJECT_PATH"
  printf '[dry-run] remote python     : %s\n' "$REMOTE_PYTHON"
  printf '[dry-run]\n'
  printf '[dry-run] would execute exactly this ONE command:\n'
  printf '[dry-run] %s\n' "$SSH_QUOTED"
  printf '[dry-run]\n'
  printf '[dry-run] remote payload scope (whitelist — stdlib only):\n'
  printf '[dry-run]   1. config.json / tokenizer_config.json: presence, JSON validity, top-level key names (no values)\n'
  printf '[dry-run]   2. metadata files (*.json, *.txt, *.model): name, size, sha256 (streamed)\n'
  printf '[dry-run]   3. %s/scripts/inspect_checkpoint.py on the checkpoint\n' "$REMOTE_PROJECT_PATH"
  printf '[dry-run]   nothing else: no downloads, no writes, no weight reads, no remote mutation\n'
  printf '[dry-run]\n'
  printf '[dry-run] to actually run it: re-invoke with --execute.\n'
  printf '[dry-run] expect an ssh connection failure until access to %s exists (see docs/remote-checkpoint.md).\n' "$REMOTE"
  exit 0
fi

printf '[execute] opening ONE read-only ssh session to %s\n' "$REMOTE"
printf '[execute] remote checkpoint: %s\n' "$CHECKPOINT_ARG"
printf '[execute] staged repo path : %s\n' "$REMOTE_PROJECT_PATH"
set +e
eval "$SSH_QUOTED"
status=$?
set -e

case "$status" in
  0)
    printf '[execute] done: remote inspection passed (remote exit 0 — planning fields match spec).\n'
    ;;
  1)
    printf '[execute] done with FAILURES: planning-field mismatch (remote exit 1).\n' >&2
    printf '[execute] bring the reported values back to this repository BEFORE modifying planning constants.\n' >&2
    exit 3
    ;;
  2)
    printf '[execute] done with FAILURES: checkpoint or metadata invalid (remote exit 2).\n' >&2
    exit 3
    ;;
  3)
    printf '[execute] done with FAILURES: repository not staged at %s (remote exit 3).\n' "$REMOTE_PROJECT_PATH" >&2
    printf '[execute] stage the repository copy there first; this script performs no copy.\n' >&2
    exit 3
    ;;
  255)
    printf '[execute] remote access UNAVAILABLE: ssh could not connect to %s (status 255).\n' "$REMOTE" >&2
    printf '[execute] this is the expected result until remote access is established (see docs/remote-checkpoint.md).\n' >&2
    exit 4
    ;;
  *)
    printf '[execute] unexpected ssh status %s — the remote payload did not complete successfully.\n' "$status" >&2
    exit 4
    ;;
esac
exit 0
