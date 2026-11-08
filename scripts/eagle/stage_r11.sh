#!/usr/bin/env bash
# r11 EAGLE head: r10 fine-tuned on ~2.7x more self-distilled chat
# (ds/chat_prompts2.json, generated through the live API by gen_chat.py).
#   W wait until gen_chat reaches STOP_AT conversations (then stop it) or exits
#   C clean: drop a truncated last line; dedupe first turns across the
#     train/eval splits of both prompt sets
#   P prep (last 10% of the new conversations held out) -> stop granite4-rtx
#   D dump hidden (KV8) -> M merge -> T fine-tune r11 from r10
#   E offline eval r10 vs r11 (chat_r9_eval, chat_r11_eval, serve_eval)
#   X export -> A live A/B with the deployed engine (tree drafting)
# granite4-rtx is restarted on ANY exit (trap). It keeps serving r10;
# deploying r11 is a separate, reviewed step.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=/run/current-system/sw/bin/granite4-infer
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
LOG=$E/stage_r11.log
STOP_AT=${STOP_AT:-6000}
GEN=ds/chat_r11/gen.jsonl
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
restart() { pkill -f "[d]ump-hidden" && sleep 3; systemctl start granite4-rtx; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
podrun() {
  podman run --rm --network host --gpus all --shm-size=8g \
    -v $E:/w/eagle -v $CK:/w/ckpt --entrypoint "" localhost/granite4-eagle:latest \
    /opt/conda/bin/python "$@"
}
last_conv() { tail -1 $GEN | $PY -c "import json,sys;print(json.loads(sys.stdin.read())['conv'])" 2>/dev/null || echo 0; }
log "== W waiting for gen_chat (stop at conversation $STOP_AT)"
while pgrep -f "[g]en_chat.py" > /dev/null; do
  if [ "$(last_conv)" -ge "$STOP_AT" ]; then
    pkill -f "[g]en_chat.py"; sleep 15; log "stopped gen_chat at conversation $(last_conv)"; break
  fi
  sleep 120
done
log "gen: $(tail -1 gen_chat_r11.log); $(wc -l < $GEN) replies"
log "== C clean + dedupe"
$PY - >> $LOG 2>&1 <<'EOF' || { log "clean failed"; exit 1; }
import json
recs = []
for line in open("ds/chat_r11/gen.jsonl"):
    try:
        recs.append(json.loads(line))
    except ValueError:
        pass  # a reply cut by the stop
first = lambda r: r["messages"][0]["content"]
eval_from = int(max(r["conv"] for r in recs) * 0.9)
p1 = [c["turns"][0] for c in json.load(open("ds/chat_prompts.json"))]
r9_eval = set(p1[2159:])           # chat_r9_eval prompts (r10's held-out split)
r9_train = set(p1[:2159])          # trained by r10 (and again by r11)
r11_eval = {first(r) for r in recs if r["conv"] >= eval_from}
keep = []
for r in recs:
    f = first(r)
    if r["conv"] >= eval_from:
        if f in r9_train:          # seen in training already: not held out
            continue
    elif f in r9_eval or f in r11_eval:
        continue                   # keep both eval sets unseen
    keep.append(r)
with open("ds/chat_r11/gen_clean.jsonl", "w") as o:
    for r in keep:
        o.write(json.dumps(r) + "\n")
open("ds/chat_r11/eval_from", "w").write(str(eval_from))
print("clean:", len(recs), "replies ->", len(keep), "kept; eval from conv", eval_from)
EOF
EVAL_FROM=$(cat ds/chat_r11/eval_from)
log "== P prep records (eval: conv >= $EVAL_FROM)"
$PY prep_chat.py --data ds/chat_r11/gen_clean.jsonl --tokenizer $CK/tokenizer.json \
  --train-out ds/chat_r11_train --eval-out ds/chat_r11_eval --eval-from $EVAL_FROM --max-len 4096 >> $LOG 2>&1 || { log "prep failed"; exit 1; }
log "== stopping granite4-rtx (GPU stages)"
trap restart EXIT
systemctl stop granite4-rtx
sleep 3
log "== D dump hidden (KV8)"
for s in chat_r11_eval chat_r11_train; do
  G4_KV8=1 $ENG dump-hidden --checkpoint $CK --gpu 1 --in ds/$s/tokens.bin --out ds/$s/hidden.bin \
    --max-context 4096 >> $LOG 2>&1 || { log "dump $s failed"; exit 1; }
done
log "== M merge (chat_r9_train + chat_r11_train + serve_train3 + train:2000)"
$PY merge_records.py --base ds --spec chat_r9_train chat_r11_train serve_train3 train:2000 --out ds/r11_train >> $LOG 2>&1 || { log "merge failed"; exit 1; }
cat ds/r11_train/meta.json >> $LOG 2>/dev/null
log "== T fine-tune r11 from r10"
podrun /w/eagle/train_eagle.py --ckpt /w/ckpt --train /w/eagle/ds/r11_train \
  --eval-self /w/eagle/ds/chat_r11_eval --out /w/eagle/runs/r11 \
  --depth 3 --eval-depth 6 --head /w/eagle/runs/r10/ckpt.pt \
  --draft-vocab /w/eagle/runs/r3/draft_vocab.npy \
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 400 > ft_r11.log 2>&1 || { log "training failed (ft_r11.log)"; exit 1; }
log "train: $(grep eval_tokens_per_round ft_r11.log | head -1)"
log "train: $(grep eval_tokens_per_round ft_r11.log | tail -1)"
log "== E offline eval (depth 6): r10 vs r11"
for d in chat_r9_eval chat_r11_eval; do
  N=$($PY -c "import json;print(json.load(open('ds/$d/meta.json'))['records'])")
  $PY -c "import json;json.dump(['chat']*$N,open('ds/$d/styles.json','w'))"
done
for r in r10 r11; do
  for d in chat_r9_eval chat_r11_eval; do
    log "-- $r $d"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/$d \
      --styles /w/eagle/ds/$d/styles.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -1 >> $LOG
  done
  log "-- $r serve_eval"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/serve_eval \
    --styles /w/eagle/ds/serve_eval/styles_all.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -1 >> $LOG
done
log "== X export r11"
podrun /w/eagle/export_head.py --ckpt /w/eagle/runs/r11/ckpt.pt --vocab /w/eagle/runs/r3/draft_vocab.npy \
  --out /w/eagle/heads/r11.safetensors >> $LOG 2>&1 || { log "export failed"; exit 1; }
chmod 644 $E/heads/r11.safetensors  # the service user must be able to read it
log "== A live A/B (deployed engine: tree drafting; held-out prompts)"
rm -f /tmp/ab_plain.json
for h in plain r10 r11; do
  H=$h; [ $h != plain ] && H=$E/heads/$h.safetensors
  $PY $E/ab_heads.py $ENG $h $H >> $LOG 2>&1
done
log "ALL DONE"
