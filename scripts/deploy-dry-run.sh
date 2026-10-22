#!/usr/bin/env bash
# deploy-dry-run.sh — DRY RUN ONLY for the Phase 0 scaffold deployment.
#
# This script PRINTS the later deployment workflow. It never executes
# ssh, scp, rsync, git push, nixos-rebuild, or anything else remote. The
# only commands it actually runs are local, read-only file-existence
# checks.
#
# - set -eu: any failed local check aborts with a clear message.
# - Remote host, user, and paths are variables (overridable via env).
# - The copy target is a clearly-named staging directory; the script
#   refuses to point it anywhere under /etc/nixos. The NixOS flake is a
#   separate, later, user-approved step.
# - No secrets are stored, read, or transmitted.

set -eu

# --- variables (override via environment when the layout changes) ----------
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-10.1.1.12}"
REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
STAMP="$(date +%Y%m%d-%H%M%S)"
REMOTE_STAGING_DIR="${REMOTE_STAGING_DIR:-/opt/granite4-rtx-4070-ti-super/staging-${STAMP}}"
NIXOS_FLAKE_PATH="/etc/nixos/flake.nix"        # never a write target of the copy step
SERVICE_NAME="${SERVICE_NAME:-granite4-rtx}"   # placeholder, set by the flake module
API_PORT="${API_PORT:-8080}"                   # placeholder, set by the flake module

say() { printf '[dry-run] %s\n' "$*"; }
# Print a command the later session will execute; this script does not run it.
cmd() { printf '[dry-run] would execute: %s\n' "$*"; }

# --- guard: the staging dir must not touch /etc/nixos ----------------------
case "$REMOTE_STAGING_DIR" in
  /etc/nixos*|*/etc/nixos*)
    printf 'ERROR: REMOTE_STAGING_DIR=%s is inside /etc/nixos.\n' "$REMOTE_STAGING_DIR" >&2
    printf 'The copy step must target a staging directory; /etc/nixos/flake.nix is only touched in the separate, user-approved flake step.\n' >&2
    exit 1
    ;;
esac

# --- local pre-flight: required files must exist ---------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
required_files=(
  "CMakeLists.txt"
  "engine/CMakeLists.txt"
  "engine/src/model_config.h"
  "engine/src/config_validation.cc"
  "engine/src/config_validation.h"
  "engine/src/capi.h"
  "engine/src/capi.cc"
  "engine/src/kernels.cu"
  "engine/tests/test_config_validation.cc"
  "frontend/__init__.py"
  "frontend/api.py"
  "frontend/config.py"
  "frontend/engine_bridge.py"
  "conftest.py"
  "pyproject.toml"
  "README.md"
  "docs/architecture.md"
  "docs/phases.md"
  "docs/deployment.md"
  "tests/test_api.py"
  "tests/test_config.py"
)
missing=()
for f in "${required_files[@]}"; do
  [ -e "${ROOT}/${f}" ] || missing+=("$f")
done
if [ "${#missing[@]}" -gt 0 ]; then
  printf 'ERROR: required local file(s) missing under %s:\n' "$ROOT" >&2
  for f in "${missing[@]}"; do
    printf '  - %s\n' "$f" >&2
  done
  exit 1
fi
say "local pre-flight OK: all ${#required_files[@]} required files present under ${ROOT}"

say "remote target : ${REMOTE_TARGET}"
say "staging dir   : ${REMOTE_STAGING_DIR} (fresh per copy; NOT ${NIXOS_FLAKE_PATH})"
say "service/port  : ${SERVICE_NAME} : ${API_PORT} (placeholders until the flake module lands)"
say
say "=== later workflow (nothing below is executed by this script) ==="

say
say "1) Local checks — must all pass first (host-only: no GPU, no CUDA toolkit)"
cmd "cd ${ROOT} && uv run --group dev pytest"
cmd "cmake -S ${ROOT} -B ${ROOT}/build -G Ninja -DGRANITE_ENABLE_CUDA=OFF"
cmd "cmake --build ${ROOT}/build"
cmd "ctest --test-dir ${ROOT}/build --output-on-failure"

say
say "2) Review the working tree"
cmd "git -C ${ROOT} status"
cmd "git -C ${ROOT} diff"
cmd "git -C ${ROOT} diff --staged"

say
say "3) Commit / push — PLACEHOLDERS; push only after review + user approval"
cmd "git -C ${ROOT} add docs/deployment.md scripts/deploy-dry-run.sh"
cmd "git -C ${ROOT} commit -m 'phase0: deployment runbook and dry-run script'"
cmd "git -C ${ROOT} push origin <branch>   # PLACEHOLDER — do not run before approval"

say
say "4) Copy to the remote staging directory (only write target; never ${NIXOS_FLAKE_PATH})"
cmd "ssh ${REMOTE_TARGET} \"mkdir -p '${REMOTE_STAGING_DIR}'\""
cmd "rsync -avz --delete --exclude '.git' --exclude 'build/' --exclude 'reference/' ${ROOT}/ ${REMOTE_TARGET}:${REMOTE_STAGING_DIR}/"
say "     fallback if rsync is unavailable on the remote: scp -r ${ROOT}/ ${REMOTE_TARGET}:${REMOTE_STAGING_DIR}/ then prune .git/ build/ reference/ inside staging"
cmd "ssh ${REMOTE_TARGET} \"test -f '${REMOTE_STAGING_DIR}/CMakeLists.txt' && test -f '${REMOTE_STAGING_DIR}/frontend/api.py' && echo copy-ok\""

say
say "5) Flake integration — SEPARATE STEP; only after explicit user approval"
cmd "ssh ${REMOTE_TARGET} \"cat ${NIXOS_FLAKE_PATH}\"   # review the current flake first"
say "     present the proposed flake-module diff to the user; integrate only after approval"
cmd "ssh ${REMOTE_TARGET} \"sudo nixos-rebuild build --use-rollback-switch\""
cmd "ssh ${REMOTE_TARGET} \"sudo nixos-rebuild switch\""
cmd "ssh ${REMOTE_TARGET} \"sudo systemctl restart ${SERVICE_NAME}\""

say
say "6) Verification (expected Phase 0: server up, inference explicitly unavailable)"
cmd "curl -s http://${REMOTE_HOST}:${API_PORT}/health"
cmd "curl -s http://${REMOTE_HOST}:${API_PORT}/metrics"
cmd "curl -s http://${REMOTE_HOST}:${API_PORT}/v1/models"
cmd "curl -s -o /dev/null -w '%{http_code}\\n' -X POST http://${REMOTE_HOST}:${API_PORT}/v1/chat/completions -H 'content-type: application/json' -d '{\"model\":\"granite-4.2-8b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'   # expect 501 in Phase 0"
say "     NOTE: 501 is the correct Phase 0 result; CUDA/remote behavior is unverified."

say
say "7) Rollback (reverse order, least destructive first)"
cmd "ssh ${REMOTE_TARGET} \"sudo nixos-rebuild --rollback\""
cmd "ssh ${REMOTE_TARGET} \"sudo systemctl stop ${SERVICE_NAME}\""
cmd "ssh ${REMOTE_TARGET} \"rm -rf '${REMOTE_STAGING_DIR}'\"   # only after rollback confirmed + user approval"
cmd "git -C ${ROOT} revert <commit>   # local revert if the commit itself was the problem"

say
say "dry run complete: no ssh/scp/rsync/git push/nixos-rebuild command was executed."
exit 0
