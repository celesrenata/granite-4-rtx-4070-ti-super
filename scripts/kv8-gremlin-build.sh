#!/usr/bin/env bash
# kv8-gremlin-build.sh — build the sm_89 CUDA engine on gremlin-1 (RTX 4070 Ti
# SUPER) from a plain source tree, using the Nix store's toolchain (no
# `nix develop` needed: the store already has cmake / ninja / clang /
# cuda-merged).
#
#   bash scripts/kv8-gremlin-build.sh             # sync + build + GPU test
#   GREMLIN_HOST=gremlin-1 GREMLIN_DIR=~/src/kv8 bash scripts/kv8-gremlin-build.sh
#
# GREMLIN_DIR is expanded on the remote host (via ~).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${GREMLIN_HOST:-gremlin-1}"
REMOTE_DIR="${GREMLIN_DIR:-/home/celes/src/kv8}"

say() { printf '[kv8-gremlin] %s\n' "$*"; }

say "syncing -> $HOST:$REMOTE_DIR"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude '.git' --exclude 'build' --exclude 'build-host' \
  --exclude '__pycache__' --exclude '.gt' --exclude '*.pyc' \
  "$REPO/" "$HOST:$REMOTE_DIR/"

say "uploading remote build driver"
scp -q "$(dirname "$0")/kv8-gremlin-remote.sh" "$HOST:$REMOTE_DIR/kv8-gremlin-remote.sh"

say "build + GPU test on $HOST"
ssh "$HOST" "bash '$REMOTE_DIR/kv8-gremlin-remote.sh' '$REMOTE_DIR'"
