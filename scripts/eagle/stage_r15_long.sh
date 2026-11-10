#!/usr/bin/env bash
# r15long, gremlin side: long-context training data for the EAGLE head.
#   W wait until gen_chat.py finishes ds/long_prompts.json (gen_long_prompts.py: multi-file agent
#     reads of 3K-48K tokens + a task; replies are the live target's own greedy output)
#   P prep records (prep_chat.py, --max-len 65536; prompts >= eval_from are held out), shard the
#     train split by tokens (one dump window per shard)
#   D per shard: wait for the service to be idle (no non-local client for 30 s; local generators
#     are paused), stop granite4-rtx, dump K5:V4 features with the PRODUCTION engine at
#     --max-context 65536, restart granite4-rtx (also on ANY exit). Each shard gets a DONE marker;
#     the 5090 pulls them (train_remote_r15long.sh).
# Downtime: ~10-15 min per shard, never while an outside client is connected.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
S=/opt/granite4-rtx-4070-ti-super/staging-k5v4-20261109-144713
ENG=/run/current-system/sw/bin/granite4-infer
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
LOG=$E/stage_r15_long.log
SHARDS=${SHARDS:-6}
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
. /root/idle_gate.sh
log "== W waiting for gen_chat (ds/long_r15/gen.jsonl)"
while pgrep -f "[g]en_chat.py --prompts ds/long_prompts.json" > /dev/null; do sleep 120; done
log "gen: $(wc -l < ds/long_r15/gen.jsonl) replies; $(tail -1 gen_long_r15.log)"
log "== P prep + shard"
EVAL_FROM=$($PY -c "import json;print(sum(r['split']=='train' for r in json.load(open('ds/long_prompts.json'))))")
rm -rf ds/long_r15_train ds/long_r15_eval ds/k5/long_r15_*
PYTHONPATH=$S $PY prep_chat.py --data ds/long_r15/gen.jsonl --tokenizer $CK/tokenizer.json \
  --train-out ds/long_r15_train --eval-out ds/long_r15_eval --eval-from "$EVAL_FROM" --max-len 65536 \
  >> $LOG 2>&1 || { log "prep failed"; exit 1; }
$PY - "$SHARDS" >> $LOG 2>&1 <<'EOF' || { log "shard failed"; exit 1; }
import json, struct, sys
from pathlib import Path
import numpy as np
shards = int(sys.argv[1])
def records(d):
    raw = np.fromfile(Path(d) / "tokens.bin", dtype=np.int32)
    mask = np.fromfile(Path(d) / "mask.bin", dtype=np.uint8)
    out, i, o = [], 0, 0
    while i < len(raw):
        n = int(raw[i]); out.append((raw[i + 1:i + 1 + n], mask[o:o + n])); i += 1 + n; o += n
    return out
def write(d, recs):
    d = Path(d); d.mkdir(parents=True, exist_ok=True)
    with open(d / "tokens.bin", "wb") as ft, open(d / "mask.bin", "wb") as fm:
        for t, m in recs:
            ft.write(struct.pack("<i", len(t))); ft.write(t.astype("<i4").tobytes()); fm.write(m.tobytes())
    meta = {"records": len(recs), "tokens": int(sum(len(t) for t, _ in recs)),
            "loss_tokens": int(sum(int(m.sum()) for _, m in recs))}
    (d / "meta.json").write_text(json.dumps(meta)); print(d, meta)
    # styles: one per record (per_style_eval.py), by length bucket
    (d / "styles.json").write_text(json.dumps([f"len{min(len(t) // 8192, 6) * 8}k" for t, _ in recs]))
tr = records("ds/long_r15_train")
total = sum(len(t) for t, _ in tr)
cur, acc, k = [], 0, 0
for r in tr:                      # contiguous shards of ~equal tokens
    cur.append(r); acc += len(r[0])
    if acc >= total * (k + 1) / shards and k < shards - 1:
        write(f"ds/k5/long_r15_train_s{k}", cur); cur = []; k += 1
write(f"ds/k5/long_r15_train_s{k}", cur)
write("ds/k5/long_r15_eval", records("ds/long_r15_eval"))
EOF
restart() { pkill -f "[d]ump-hidden" && sleep 3; systemctl start granite4-rtx; gen_resume; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
trap restart EXIT
for s in ds/k5/long_r15_eval ds/k5/long_r15_train_s*; do
  log "== D $s ($(cat $s/meta.json))"
  idle_wait 7200 >> $LOG 2>&1
  systemctl stop granite4-rtx
  sleep 3
  t0=$(date +%s)
  G4_KV8=5 $ENG dump-hidden --checkpoint $CK --gpu 1 --in $s/tokens.bin --out $s/hidden.bin \
    --max-context 65536 >> $LOG 2>&1 || { log "dump $s failed"; exit 1; }
  want=$(( $(grep -o '"tokens": *[0-9]*' $s/meta.json | head -1 | grep -o '[0-9]*$') * 4096 * 2 ))
  [ "$(stat -c %s $s/hidden.bin)" = "$want" ] || { log "size mismatch $s (want $want)"; exit 1; }
  touch $s/DONE
  log "dumped $s in $(( $(date +%s) - t0 ))s"
  systemctl start granite4-rtx; gen_resume
  sleep 120   # let queued traffic through between windows
done
touch ds/k5/R15_DONE
log "R15 DUMP DONE"
