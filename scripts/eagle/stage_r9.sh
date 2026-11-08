#!/usr/bin/env bash
# r9 EAGLE head: self-distilled general chat. Unattended GPU stages, run after
# gen_chat.py (which uses the live API) finishes:
#   P prep records -> stop granite4-rtx -> D dump hidden (KV8, the served
#   features) -> M merge with serving + chat replay data -> T fine-tune r8 ->
#   E offline eval (r8 vs r9) -> X export -> A live A/B -> restart granite4-rtx.
# The service is restarted on ANY exit (trap). It keeps serving r8; deploying
# r9 is a separate, reviewed step.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=/run/current-system/sw/bin/granite4-infer
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
LOG=$E/stage_r9.log
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
restart() { systemctl start granite4-rtx; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
podrun() {
  podman run --rm --network host --gpus all --shm-size=8g \
    -v $E:/w/eagle -v $CK:/w/ckpt --entrypoint "" localhost/granite4-eagle:latest \
    /opt/conda/bin/python "$@"
}
log "waiting for gen_chat to finish"
until grep -q "^DONE" gen_chat_r9.log; do sleep 60; done
log "gen: $(tail -1 gen_chat_r9.log); $(wc -l < ds/chat_r9/gen.jsonl) replies"
log "== P prep records"
$PY prep_chat.py --data ds/chat_r9/gen.jsonl --tokenizer $CK/tokenizer.json \
  --train-out ds/chat_r9_train --eval-out ds/chat_r9_eval --eval-from 1100 --max-len 4096 >> $LOG 2>&1 || { log "prep failed"; exit 1; }
log "== stopping granite4-rtx (GPU stages)"
trap restart EXIT
systemctl stop granite4-rtx
sleep 3
log "== D dump hidden (KV8)"
for s in chat_r9_eval chat_r9_train; do
  G4_KV8=1 $ENG dump-hidden --checkpoint $CK --gpu 1 --in ds/$s/tokens.bin --out ds/$s/hidden.bin \
    --max-context 4096 >> $LOG 2>&1 || { log "dump $s failed"; exit 1; }
done
log "== M merge (chat_r9_train + serve_train3 + train:2000)"
$PY merge_records.py --base ds --spec chat_r9_train serve_train3 train:2000 --out ds/r9_train >> $LOG 2>&1 || { log "merge failed"; exit 1; }
cat ds/r9_train/meta.json >> $LOG 2>/dev/null
log "== T fine-tune r9 from r8"
podrun /w/eagle/train_eagle.py --ckpt /w/ckpt --train /w/eagle/ds/r9_train \
  --eval-self /w/eagle/ds/chat_r9_eval --out /w/eagle/runs/r9 \
  --depth 3 --eval-depth 6 --head /w/eagle/runs/r8/ckpt.pt \
  --draft-vocab /w/eagle/runs/r3/draft_vocab.npy \
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 100 > ft_r9.log 2>&1 || { log "training failed (ft_r9.log)"; exit 1; }
log "train: $(grep eval_tokens_per_round ft_r9.log | head -1)"
log "train: $(grep eval_tokens_per_round ft_r9.log | tail -1)"
log "== E offline eval (depth 6): r8 vs r9"
N=$($PY -c "import json;print(json.load(open('ds/chat_r9_eval/meta.json'))['records'])")
$PY -c "import json;json.dump(['chat']*$N,open('ds/chat_r9_eval/styles.json','w'));json.dump(['serving']*36,open('ds/serve_eval/styles_all.json','w'))"
for r in r8 r9; do
  log "-- $r chat eval"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/chat_r9_eval \
    --styles /w/eagle/ds/chat_r9_eval/styles.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -4 >> $LOG
  log "-- $r serving eval"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/serve_eval \
    --styles /w/eagle/ds/serve_eval/styles_all.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -4 >> $LOG
done
log "== X export r9"
podrun /w/eagle/export_head.py --ckpt /w/eagle/runs/r9/ckpt.pt --vocab /w/eagle/runs/r3/draft_vocab.npy \
  --out /w/eagle/heads/r9.safetensors >> $LOG 2>&1 || { log "export failed"; exit 1; }
log "== A live A/B (warm serve, held-out prompts)"
rm -f /tmp/ab_plain.json
for h in plain r8 r9; do
  H=$h; [ $h != plain ] && H=$E/heads/$h.safetensors
  $PY $E/ab_heads.py $ENG $h $H >> $LOG 2>&1
done
$PY $E/ab_heads.py $ENG r8_forced $E/heads/r8.safetensors G4_EAGLE_MIN_ACCEPT=0 >> $LOG 2>&1
$PY $E/ab_heads.py $ENG r9_forced $E/heads/r9.safetensors G4_EAGLE_MIN_ACCEPT=0 >> $LOG 2>&1
log "ALL DONE"
