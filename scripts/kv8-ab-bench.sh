#!/usr/bin/env bash
# A/B benchmark for the KV8 int8 KV cache vs the bf16 KV cache.
#
# Runs granite4-infer kv-bench twice per (ctx, decode) pair:
#   G4_KV8=0  -> bf16 persistent KV cache
#   G4_KV8=1  -> int8 (Q8_0) persistent KV cache
# Each mode is run REPEATS times; the run with the lowest decode_ms is kept.
# The memory delta is read from the "kv_delta_bytes" field (free memory before
# load -> after decode; deterministic pool allocation makes it exact).
#
# Usage: scripts/kv8-ab-bench.sh [CHECKPOINT_DIR]
set -euo pipefail

BIN="${KV8_BIN:-/home/celes/src/kv8/build/engine/granite4-infer}"
CKPT="${1:-/home/celes/kv8-bench}"
REPEATS="${KV8_REPEATS:-3}"
# gremlin-1 has no python3 on the default PATH; use the Nix env.
PY="${KV8_PY:-/nix/store/0kb3r23cflhjb0fknss1x7mxx8aqgsnj-python3-3.14.8-env/bin/python3}"

run() {  # run <kv8:0|1> <ctx> <decode> <reps> -> one JSON line (best decode_ms)
  local kv8="$1" ctx="$2" dec="$3"
  local best="" best_ms=1e9 i
  for ((i = 0; i < REPEATS; ++i)); do
    line=$(G4_KV8="$kv8" "$BIN" kv-bench --checkpoint "$CKPT" --gpu 1 \
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
  echo "# ctx=$ctx decode=$dec repeats=$REPEATS"
  bf16=$(run 0 "$ctx" "$dec")
  kv8=$(run 1 "$ctx" "$dec")
  printf 'bf16 %s\n' "$bf16"
  printf 'int8 %s\n' "$kv8"
  "$PY" - "$bf16" "$kv8" <<'PYSUM' || true
import json, sys
b, k = json.loads(sys.argv[1]), json.loads(sys.argv[2])
# free_after_load is identical in both runs except for the persistent KV
# pool (weights + fixed scratch are the same), so its delta IS the pool-size
# delta at this max_pos.
pool_delta = k["free_after_load"] - b["free_after_load"]
print("KV pool:      int8 pool is %d B smaller than bf16 at max_pos=%d"
      % (pool_delta, b["max_pos"]))
print("prefill ms:   %12.3f vs %.3f  (%.3fx)"
      % (b["prefill_ms"], k["prefill_ms"], b["prefill_ms"] / k["prefill_ms"]))
print("decode tps:   %12.3f vs %.3f  (%.3fx)"
      % (b["decode_tps"], k["decode_tps"], k["decode_tps"] / b["decode_tps"]))
PYSUM
  echo
done
