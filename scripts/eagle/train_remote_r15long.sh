#!/usr/bin/env bash
# r15long, RTX 5090 side: EAGLE head trained on long contexts.
# Every deployed head so far was trained on records of <= 4096 tokens (and RoPE tables of 8192
# positions), while the service drafts at up to 98K positions; acceptance there falls from ~0.13
# to ~0.05-0.08. This run:
#   P pulls gremlin's K5 dumps of the long-context set (stage_r15_long.sh: multi-file agent reads
#     of 3K-48K tokens + a task, replies by the live target)
#   G takes the 5090 (qwen27b stopped, shared GPU lock held; restored on ANY exit)
#   A diagnostic: r12 on the long eval with the head's attention windowed to the last N entries
#   T fine-tunes r15long from r12: r12's full K5 mix + the long shards (x4), 1 epoch, r12's lr,
#     --rope-len 65536 (long records train from the reply region, train_eagle.py LONG_RECORD)
#   E offline eval r12 vs r15long: chat_r12_eval, serve_eval, long_r15_eval
#   X export + push heads/r15long.safetensors (deploying is a separate, reviewed step)
set -uo pipefail
W=/root/g4eagle
PYT=/nix/store/yl54awzzpv6mf497bm6mhfh5v58ncysm-python3-3.12.15-env/bin/python3
G=root@10.1.1.12
GE=/opt/granite4-rtx-4070-ti-super/eagle
LOG=$W/train_r15long.log
cd $W
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
log "== P pulling the long-context K5 dumps (each set as soon as gremlin marks it DONE)"
until ssh $G "test -d $GE/ds/k5/long_r15_eval"; do sleep 60; done
SETS=$(ssh $G "cd $GE/ds/k5 && ls -d long_r15_*")
for s in $SETS; do
  until ssh $G "test -f $GE/ds/k5/$s/DONE"; do
    if ssh $G "grep -q -E 'dump .* failed|size mismatch' $GE/stage_r15_long.log 2>/dev/null"; then log "gremlin dump failed"; exit 1; fi
    sleep 60
  done
  [ -f ds/k5/$s/DONE ] || rsync -a "$G:$GE/ds/k5/$s" ds/k5/ || { log "pull $s failed"; exit 1; }
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
ev() {  # ev <label> <ckpt> <records> <styles> [extra args]
  local label=$1 ck=$2 recs=$3 st=$4; shift 4
  log "-- $label $(basename $recs): $($PYT per_style_eval.py --head "$ck" --records "$recs" --styles "$st" \
      --vocab runs/r3/draft_vocab.npy --ckpt ckpt --depth 6 --rope-len 65536 "$@" 2>&1 | tail -1)"
}
log "== A r12 on the long eval, attention windowed (diagnostic)"
for w in 0 1024 4096; do
  ev "r12 window=$w" runs/r12/ckpt.pt ds/k5/long_r15_eval ds/k5/long_r15_eval/styles.json --window $w
done
log "== T fine-tune r15long from r12"
LONG=$(ls -d ds/k5/long_r15_train_s* | tr '\n' ',')
TRAIN=ds/k5/chat_r9_train,ds/k5/chat_r11_train,ds/k5/chat_r12_train,ds/k5/serve_train3,ds/k5/train2000,$LONG$LONG$LONG$LONG
$PYT train_eagle.py --ckpt ckpt --train "${TRAIN%,}" --eval-self ds/k5/long_r15_eval --out runs/r15long \
  --depth 3 --eval-depth 6 --head runs/r12/ckpt.pt --draft-vocab runs/r3/draft_vocab.npy \
  --epochs 1 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 500 --rope-len 65536 > ft_r15long.log 2>&1 || { log "training failed (ft_r15long.log)"; exit 1; }
log "train first: $(grep eval_tokens_per_round ft_r15long.log | head -1)"
log "train last:  $(grep eval_tokens_per_round ft_r15long.log | tail -1)"
log "== E offline eval: r12 vs r15long"
for h in r12 r15long; do
  ev "$h" runs/$h/ckpt.pt ds/k5/chat_r12_eval ds/k5/chat_r12_eval/styles.json
  ev "$h" runs/$h/ckpt.pt ds/k5/serve_eval ds/k5/serve_eval/styles_all.json
  ev "$h" runs/$h/ckpt.pt ds/k5/long_r15_eval ds/k5/long_r15_eval/styles.json
done
log "== X export + push"
$PYT export_head.py --ckpt runs/r15long/ckpt.pt --vocab runs/r3/draft_vocab.npy \
  --out runs/r15long/r15long.safetensors >> "$LOG" 2>&1 || { log "export failed"; exit 1; }
scp -q runs/r15long/r15long.safetensors "$G:$GE/heads/r15long.safetensors" \
  && ssh $G "chmod 644 $GE/heads/r15long.safetensors" || { log "push failed"; exit 1; }
log "ALL DONE (head pushed to gremlin: heads/r15long.safetensors)"
