#!/usr/bin/env bash
# r12 generation split (gremlin side): the live API generates conversations
# [0, SPLIT) of ds/chat_prompts3.json, the RTX 5090 host [SPLIT, TAIL) with
# gen_chat_hf.py (pushed here as ds/chat_r12/gen_hf.jsonl + HF_DONE), and the
# API then generates the held-out tail [TAIL, end) so every eval conversation
# is engine-written. The three parts are merged into ds/chat_r12/gen.jsonl
# (conversation ids of the original file) and stage_r12_prep.sh runs.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
SPLIT=${SPLIT:-3000}
TAIL=${TAIL:-5374}
LOG=$E/stage_r12.log
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
max_conv() { $PY -c "
import json
m = -1
for l in open('$1'):
    try: m = max(m, json.loads(l)['conv'])
    except ValueError: pass
print(m)" 2>/dev/null || echo -1; }
pkill -f "[s]tage_r12_prep.sh" && log "replaced the waiting stage_r12_prep (generation is split)"
log "== G1 API generates [0, $SPLIT)"
while [ "$(max_conv ds/chat_r12/gen.jsonl)" -lt "$SPLIT" ] && pgrep -f "[g]en_chat.py" > /dev/null; do sleep 60; done
pkill -f "[g]en_chat.py"; sleep 15
log "API part: $(wc -l < ds/chat_r12/gen.jsonl) replies (max conv $(max_conv ds/chat_r12/gen.jsonl))"
log "== G2 API generates the held-out tail [$TAIL, end)"
$PY -c "import json;p=json.load(open('ds/chat_prompts3.json'));json.dump(p[$TAIL:],open('ds/chat_prompts3_tail.json','w'));print(len(p[$TAIL:]))" >> $LOG
$PY gen_chat.py --prompts ds/chat_prompts3_tail.json --out ds/chat_r12/gen_tail.jsonl --limit 100000 \
  --max-tokens 448 --workers 2 > gen_chat_r12_tail.log 2>&1
log "tail: $(tail -1 gen_chat_r12_tail.log)"
log "== waiting for the 5090 part [$SPLIT, $TAIL)"
until [ -f ds/chat_r12/HF_DONE ]; do sleep 60; done
log "== M merge"
cp ds/chat_r12/gen.jsonl ds/chat_r12/gen_api_raw.jsonl
$PY - >> $LOG 2>&1 <<EOF || { log "merge failed"; exit 1; }
import json
def recs(p):
    for l in open(p):
        try: yield json.loads(l)
        except ValueError: pass
out = [r for r in recs("ds/chat_r12/gen_api_raw.jsonl") if r["conv"] < $SPLIT]
hf = [r for r in recs("ds/chat_r12/gen_hf.jsonl") if $SPLIT <= r["conv"] < $TAIL]
tail = []
for r in recs("ds/chat_r12/gen_tail.jsonl"):
    r["conv"] += $TAIL
    tail.append(r)
with open("ds/chat_r12/gen.jsonl", "w") as o:
    for r in out + hf + tail:
        o.write(json.dumps(r) + "\n")
print("merged: api", len(out), "hf", len(hf), "tail", len(tail))
EOF
log "== stage_r12_prep"
exec bash stage_r12_prep.sh
