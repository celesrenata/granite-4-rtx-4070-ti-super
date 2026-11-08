#!/usr/bin/env bash
# Local wrapper: sync tree to gremlin-1, push the one-shot build script, run it.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${GREMLIN_HOST:-gremlin-1}"
REMOTE_DIR="${GREMLIN_DIR:-/home/celes/src/kv8}"
printf '[kv8-p4] syncing -> %s:%s\n' "$HOST" "$REMOTE_DIR"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude '.git' --exclude 'build' --exclude 'build-host' \
  --exclude '__pycache__' --exclude '*.gt' --exclude '*.pyc' \
  "$REPO/" "$HOST:$REMOTE_DIR/"
scp -q "$REPO/scripts/kv8-p4-remote-build.sh" "$HOST:$REMOTE_DIR/kv8-p4-remote-build.sh"
printf '[kv8-p4] building + testing on %s\n' "$HOST"
ssh "$HOST" "bash '$REMOTE_DIR/kv8-p4-remote-build.sh'"
