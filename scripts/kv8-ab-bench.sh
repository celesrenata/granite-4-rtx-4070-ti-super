#!/usr/bin/env bash
# A/B benchmark for the quantized KV cache (G4_KV8=1 int8 / G4_KV8=4 int4)
# vs the bf16 KV cache.
#
# Runs granite4-infer kv-bench twice per (ctx, decode) pair:
#   G4_KV8=0  -> bf16 persistent KV cache
#   G4_KV8=$MODE -> int8 (Q8_0) or int4 (Q4_0) persistent KV cache
# Each mode is run REPEATS times; the run with the lowest decode_ms is kept.
# The memory delta is read from the "free_after_load" field (free device
# memory after weights + KV pool + scratch are resident; the bf16-vs-quantized
# delta at the same max_pos is exactly the persistent KV pool size delta).
#
# Usage: scripts/kv8-ab-bench.sh [MODE] [CHECKPOINT_DIR]
#   MODE = 1 (default, int8) or 4 (int4); the bf16 run is always the control.
set -euo pipefail

MODE="${1:-1}"
case "$MODE" in
  1) MODE_LABEL="int8" ;;
  4) MODE_LABEL="int4" ;;
  *) echo "MODE must be 1 (int8) or 4 (int4), got '$MODE'" >&2; exit 2 ;;
esac
[ "$#" -gt 0 ] && shift  # drop MODE; the remaining $1 is the checkpoint dir
BIN="${KV8_BIN:-/home/celes/src/kv8/build/engine/granite4-infer}"
CKPT="${1:-/home/celes/kv8-bench}"
REPEATS="${KV8_REPEATS:-3}"
# gremlin-1 has no python3 on the default PATH; use the Nix env.
PY="${KV8_PY:-/nix/store/0kb3r23cflhjb0fknss1x7mxx8aqgsnj-python3-3.14.8-env/bin/python3}"

run() {  # run <kvmode:0|1|4> <ctx> <decode> -> one JSON line (best decode_ms)
  local kvmode="$1" ctx="$2" dec="$3"
  local best="" best_ms=1e9 i
  for ((i = 0; i < REPEATS; ++i)); do
    line=$(G4_KV8="$kvmode" "$BIN" kv-bench --checkpoint "$CKPT" --gpu 1 \
      --bench-ctx "$ctx" --bench-decode "$dec" --seed 1234 2>/dev/null | tail -1)
    ms=$(sed -n 's/.*"decode_ms":\([0-9.]*\).*/\1/p' <<<"$line")
    if awk "BEGIN{exit !($ms < $best_ms)}"; then
      best_ms=$ms
      best=$line
    fi
  done
  printf '%s\n' "$best"
}

for pair in "2048 256" "4096 256" "8192 128"; do
  set -- $pair
  ctx=$1 dec=$2
  echo "# ctx=$ctx decode=$dec repeats=$REPEATS mode=G4_KV8=$MODE ($MODE_LABEL)"
  bf16=$(run 0 "$ctx" "$dec")
  quant=$(run "$MODE" "$ctx" "$dec")
  printf 'bf16   %s\n' "$bf16"
  printf '%-6s %s\n' "$MODE_LABEL" "$quant"
  "$PY" - "$bf16" "$quant" "$MODE_LABEL" <<'PYSUM' || true
import json, sys
b, k, mode = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
pool_delta = k["free_after_load"] - b["free_after_load"]
print("KV pool:      %s pool is %d B smaller than bf16 at max_pos=%d"
      % (mode, pool_delta, b["max_pos"]))
print("prefill ms:   %12.3f vs %.3f  (%.3fx)"
      % (b["prefill_ms"], k["prefill_ms"], b["prefill_ms"] / k["prefill_ms"]))
print("decode tps:   %12.3f vs %.3f  (%.3fx)"
      % (b["decode_tps"], k["decode_tps"], k["decode_tps"] / b["decode_tps"]))
PYSUM
  echo
done
