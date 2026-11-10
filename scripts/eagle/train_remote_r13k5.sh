#!/usr/bin/env bash
# r13k5, RTX 5090 side: pull gremlin's K5:V4 feature dumps (stage_r13k5_dump.sh) as they finish,
# then (holding the shared 5090 GPU lock, qwen27b stopped) offline-eval r12 on the K5 evals,
# fine-tune r13k5 from r12 on K5 features only, offline-eval r13k5, export and push
# heads/r13k5.safetensors to gremlin. qwen27b is restarted on ANY exit after it was stopped.
#
# Why K5 features from gremlin and not the 5090: on identical input, the same engine binary gives
# K5:V4 features that differ much more between the RTX 4070 and the RTX 5090 (cosine 0.984) than
# int8 ones do (0.9994), so only the production card's dump represents the deployed target.
set -uo pipefail
W=/root/g4eagle
PYT=/nix/store/yl54awzzpv6mf497bm6mhfh5v58ncysm-python3-3.12.15-env/bin/python3
G=root@10.1.1.12
GE=/opt/granite4-rtx-4070-ti-super/eagle
LOG=$W/train_r13k5.log
cd $W
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
SETS="serve_eval serve_train3 chat_r12_eval train2000 chat_r12_train"
mkdir -p ds/k5
log "== P pulling K5 dumps as they finish"
for s in $SETS; do
  until ssh $G "test -f $GE/ds/k5/$s/DONE"; do
    if ssh $G "grep -q failed $GE/stage_r13k5.log"; then log "gremlin dump failed"; exit 1; fi
    sleep 30
  done
  rsync -a "$G:$GE/ds/k5/$s" ds/k5/ || { log "pull $s failed"; exit 1; }
  log "pulled $s ($(du -sh ds/k5/$s | cut -f1))"
done
log "== G taking the 5090 (stop qwen27b, hold GPU lock)"
LOCKPID=""
restore() {
  [ -n "$LOCKPID" ] && kill "$LOCKPID" 2>/dev/null
  systemctl start qwen27b
  log "qwen27b restarted ($(systemctl is-active qwen27b))"
}
trap restore EXIT
systemctl stop qwen27b
sleep 5
flock /run/arcane-gpu/5090.lock -c "exec sleep infinity" &
LOCKPID=$!
sleep 2
eval_head() {  # eval_head <label> <ckpt>
  log "-- $1 chat_r12_eval(K5): $($PYT per_style_eval.py --head "$2" --records ds/k5/chat_r12_eval --styles ds/k5/chat_r12_eval/styles.json --vocab runs/r3/draft_vocab.npy --ckpt ckpt --depth 6 2>&1 | tail -1)"
  log "-- $1 serve_eval(K5):    $($PYT per_style_eval.py --head "$2" --records ds/k5/serve_eval --styles ds/k5/serve_eval/styles_all.json --vocab runs/r3/draft_vocab.npy --ckpt ckpt --depth 6 2>&1 | tail -1)"
}
log "== E0 offline eval r12 on K5 features (baseline)"
eval_head r12 runs/r12/ckpt.pt
log "== T fine-tune r13k5 from r12 on K5 features"
TRAIN=ds/k5/chat_r12_train,ds/k5/serve_train3,ds/k5/train2000
$PYT train_eagle.py --ckpt ckpt --train "$TRAIN" --eval-self ds/k5/chat_r12_eval --out runs/r13k5 \
  --depth 3 --eval-depth 6 --head runs/r12/ckpt.pt --draft-vocab runs/r3/draft_vocab.npy \
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 500 > ft_r13k5.log 2>&1 || { log "training failed (ft_r13k5.log)"; exit 1; }
log "train first: $(grep eval_tokens_per_round ft_r13k5.log | head -1)"
log "train last:  $(grep eval_tokens_per_round ft_r13k5.log | tail -1)"
log "== E1 offline eval r13k5 on K5 features"
eval_head r13k5 runs/r13k5/ckpt.pt
log "== X export + push"
$PYT export_head.py --ckpt runs/r13k5/ckpt.pt --vocab runs/r3/draft_vocab.npy \
  --out runs/r13k5/r13k5.safetensors >> "$LOG" 2>&1 || { log "export failed"; exit 1; }
scp -q runs/r13k5/r13k5.safetensors "$G:$GE/heads/r13k5.safetensors" \
  && ssh $G "chmod 644 $GE/heads/r13k5.safetensors" || { log "push failed"; exit 1; }
log "ALL DONE (head pushed to gremlin: heads/r13k5.safetensors)"
