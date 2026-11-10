#!/usr/bin/env bash
# r14k5, RTX 5090 side: r12's full training mix, every set re-dumped with the production K5:V4 engine.
# r13k5 (3 of r12's 5 sets) lost to r12 offline (2.154 vs 2.186 chat, 2.389 vs 2.455 serve), most likely
# because it forgot the r9/r11 chat data. This run pulls gremlin's K5 dumps of chat_r9_train and
# chat_r11_train (stage_r13k5_dump.sh with K5_SETS=..., marker ds/k5/R14_DONE), then (qwen27b stopped,
# shared 5090 GPU lock held) fine-tunes r14k5 from r12 with r12's exact recipe, evaluates r12 and r14k5
# on the K5 evals, exports and pushes heads/r14k5.safetensors. qwen27b is restarted on ANY exit after it
# was stopped.
set -uo pipefail
W=/root/g4eagle
PYT=/nix/store/yl54awzzpv6mf497bm6mhfh5v58ncysm-python3-3.12.15-env/bin/python3
G=root@10.1.1.12
GE=/opt/granite4-rtx-4070-ti-super/eagle
LOG=$W/train_r14k5.log
cd $W
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
for s in chat_r12_train serve_train3 train2000 chat_r12_eval serve_eval; do
  [ -f ds/k5/$s/hidden.bin ] || { log "missing local K5 set $s"; exit 1; }
done
log "== P pulling K5 dumps of chat_r9_train, chat_r11_train"
for s in chat_r9_train chat_r11_train; do
  until ssh $G "test -f $GE/ds/k5/$s/DONE"; do
    if ssh $G "grep -q failed $GE/stage_r14k5.log"; then log "gremlin dump failed"; exit 1; fi
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
log "== T fine-tune r14k5 from r12 (r12's full mix, K5 features)"
TRAIN=ds/k5/chat_r9_train,ds/k5/chat_r11_train,ds/k5/chat_r12_train,ds/k5/serve_train3,ds/k5/train2000
$PYT train_eagle.py --ckpt ckpt --train "$TRAIN" --eval-self ds/k5/chat_r12_eval --out runs/r14k5 \
  --depth 3 --eval-depth 6 --head runs/r12/ckpt.pt --draft-vocab runs/r3/draft_vocab.npy \
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 500 > ft_r14k5.log 2>&1 || { log "training failed (ft_r14k5.log)"; exit 1; }
log "train first: $(grep eval_tokens_per_round ft_r14k5.log | head -1)"
log "train last:  $(grep eval_tokens_per_round ft_r14k5.log | tail -1)"
log "== E offline eval on K5 features: r12 (baseline), r13k5, r14k5"
eval_head r12 runs/r12/ckpt.pt
eval_head r13k5 runs/r13k5/ckpt.pt
eval_head r14k5 runs/r14k5/ckpt.pt
log "== X export + push"
$PYT export_head.py --ckpt runs/r14k5/ckpt.pt --vocab runs/r3/draft_vocab.npy \
  --out runs/r14k5/r14k5.safetensors >> "$LOG" 2>&1 || { log "export failed"; exit 1; }
scp -q runs/r14k5/r14k5.safetensors "$G:$GE/heads/r14k5.safetensors" \
  && ssh $G "chmod 644 $GE/heads/r14k5.safetensors" || { log "push failed"; exit 1; }
log "ALL DONE (head pushed to gremlin: heads/r14k5.safetensors)"
