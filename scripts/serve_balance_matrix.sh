#!/usr/bin/env bash
# Runs serve_balance_bench.py over candidate serve configs on gremlin-1 (GPU
# free: stop granite4-rtx first). Results: $OUT/<label>.json (+ .stderr).
#   serve_balance_matrix.sh ENGINE [OUT] [LABEL...]   (no labels: all)
set -uo pipefail
ENGINE="$1"
OUT="${2:-/tmp/balance}"
shift $(( $# >= 2 ? 2 : $# ))
ONLY=" $* "
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python3}"
CKPT=/opt/granite4-rtx-4070-ti-super/ckpt
HEAD=/opt/granite4-rtx-4070-ti-super/eagle/heads/r12.safetensors
mkdir -p "$OUT"

run() {  # label kv sessions max_context pool draft_kv eagle_k
  [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $1 "* ]] || return 0
  echo "=== $1"
  "$PY" "$SRC/scripts/serve_balance_bench.py" --engine "$ENGINE" --ckpt "$CKPT" --eagle "$HEAD" \
    --label "$1" --kv "$2" --sessions "$3" --max-context "$4" --pool "$5" --draft-kv "$6" \
    --eagle-k "$7" --out "$OUT/$1.json" || echo "=== $1 FAILED (see $OUT/$1.json.stderr)"
}

# label                kv sess max_ctx  pool  draft k
run kv4-2x64k          4  2   65536  131072  4   4   # every slot can fill at once
run kv4-3x64k-96k      4  3   65536   98304  4   4   # 3 slots on the margin
run kv4-3x64k-120k     4  3   65536  122880  4   4   # 3 slots, two long agents resident
run kv4-3x64k          4  3   65536  131072  4   4   # (does not fit: EAGLE OOM)
run kv8-2x64k-64k      1  2   65536   65536  4   4   # KV8 speed, 2 slots on the margin
run kv8-2x64k-oversub  1  2   65536   73728  4   4   # (does not fit: OOM)
run kv8-2x32k          1  2   32768   65536 16   4   # the old deploy
echo "=== matrix done"
