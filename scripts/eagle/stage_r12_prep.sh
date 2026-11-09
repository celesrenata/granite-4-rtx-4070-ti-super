#!/usr/bin/env bash
# r12, gremlin side (data): wait for gen_chat (ds/chat_prompts3.json via the
# live API) to finish, clean + dedupe against every earlier split, prep, then
# stop granite4-rtx only for the hidden-state dump (the served engine's KV8
# features) and write ds/chat_r12/DUMP_DONE. Training runs on the RTX 5090
# host (train_remote_r12.sh), which pulls the records and pushes the head
# back. granite4-rtx is restarted on ANY exit.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=/run/current-system/sw/bin/granite4-infer
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
LOG=$E/stage_r12.log
GEN=ds/chat_r12/gen.jsonl
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
restart() { pkill -f "[d]ump-hidden" && sleep 3; systemctl start granite4-rtx; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
rm -f ds/chat_r12/DUMP_DONE
log "== W waiting for gen_chat to exit"
while pgrep -f "[g]en_chat.py" > /dev/null; do sleep 120; done
tail -1 gen_chat_r12.log | grep -q "^DONE" || log "gen_chat exited without DONE (using what was written)"
log "gen: $(tail -1 gen_chat_r12.log); $(wc -l < $GEN) replies"
log "== C clean + dedupe"
$PY - >> $LOG 2>&1 <<'EOF' || { log "clean failed"; exit 1; }
import json
recs = []
for line in open("ds/chat_r12/gen.jsonl"):
    try:
        recs.append(json.loads(line))
    except ValueError:
        pass  # a reply cut by a stop
first = lambda r: r["messages"][0]["content"]
eval_from = int(max(r["conv"] for r in recs) * 0.9)
p1 = [c["turns"][0] for c in json.load(open("ds/chat_prompts.json"))]
p2 = [c["turns"][0] for c in json.load(open("ds/chat_prompts2.json"))]
trained = set(p1[:2159]) | set(p2[:5423])           # r10/r11 training prompts
held = set(p1[2159:]) | set(p2[5423:6027])          # chat_r9_eval / chat_r11_eval
r12_train = {first(r) for r in recs if r["conv"] < eval_from}
r12_eval = {first(r) for r in recs if r["conv"] >= eval_from}
keep = []
for r in recs:
    f = first(r)
    if r["conv"] >= eval_from:
        if f in trained or f in r12_train:
            continue                                 # not actually held out
    elif f in held or f in r12_eval:
        continue                                     # keep every eval set unseen
    keep.append(r)
with open("ds/chat_r12/gen_clean.jsonl", "w") as o:
    for r in keep:
        o.write(json.dumps(r) + "\n")
open("ds/chat_r12/eval_from", "w").write(str(eval_from))
print("clean:", len(recs), "replies ->", len(keep), "kept; eval from conv", eval_from)
EOF
EVAL_FROM=$(cat ds/chat_r12/eval_from)
log "== P prep records (eval: conv >= $EVAL_FROM)"
$PY prep_chat.py --data ds/chat_r12/gen_clean.jsonl --tokenizer $CK/tokenizer.json \
  --train-out ds/chat_r12_train --eval-out ds/chat_r12_eval --eval-from $EVAL_FROM --max-len 4096 >> $LOG 2>&1 || { log "prep failed"; exit 1; }
N=$($PY -c "import json;print(json.load(open('ds/chat_r12_eval/meta.json'))['records'])")
$PY -c "import json;json.dump(['chat']*$N,open('ds/chat_r12_eval/styles.json','w'))"
log "== stopping granite4-rtx (dump only)"
trap restart EXIT
systemctl stop granite4-rtx
sleep 3
log "== D dump hidden (KV8)"
for s in chat_r12_eval chat_r12_train; do
  G4_KV8=1 $ENG dump-hidden --checkpoint $CK --gpu 1 --in ds/$s/tokens.bin --out ds/$s/hidden.bin \
    --max-context 4096 >> $LOG 2>&1 || { log "dump $s failed"; exit 1; }
done
touch ds/chat_r12/DUMP_DONE
log "DUMP DONE (training continues on the 5090 host)"
