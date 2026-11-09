#!/usr/bin/env bash
# r12, RTX 5090 side (training): wait for the static records (pull_static.sh)
# and gremlin's DUMP_DONE, pull chat_r12_{train,eval}, fine-tune r12 from r11
# on every chat set + serving + base (several Records dirs, no merged copy),
# offline-evaluate r11 vs r12, export, push heads/r12.safetensors to gremlin
# and run the live A/B there (granite4-rtx is stopped for it and restarted).
# Deploying r12 stays a separate step.
set -uo pipefail
W=/root/g4eagle
PYT=${PYT:-/nix/store/yl54awzzpv6mf497bm6mhfh5v58ncysm-python3-3.12.15-env/bin/python3}
G=root@10.1.1.12
GE=/opt/granite4-rtx-4070-ti-super/eagle
LOG=$W/train_r12.log
cd $W
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
log "== W waiting for static records and gremlin's dump"
until grep -q STATIC_DONE pull_static.log 2>/dev/null; do sleep 60; done
until ssh $G "test -f $GE/ds/chat_r12/DUMP_DONE"; do sleep 120; done
log "== pull chat_r12_eval / chat_r12_train"
for d in chat_r12_eval chat_r12_train; do
  rsync -a $G:$GE/ds/$d ds/ || { log "pull $d failed"; exit 1; }
done
log "== T fine-tune r12 from r11 (RTX 5090)"
TRAIN=ds/chat_r9_train,ds/chat_r11_train,ds/chat_r12_train,ds/serve_train3,ds/train2000
$PYT train_eagle.py --ckpt ckpt --train $TRAIN --eval-self ds/chat_r12_eval --out runs/r12 \
  --depth 3 --eval-depth 6 --head runs/r11/ckpt.pt --draft-vocab runs/r3/draft_vocab.npy \
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 \
  --eval-every 500 > ft_r12.log 2>&1 || { log "training failed (ft_r12.log)"; exit 1; }
log "train: $(grep eval_tokens_per_round ft_r12.log | head -1)"
log "train: $(grep eval_tokens_per_round ft_r12.log | tail -1)"
log "== E offline eval (depth 6): r11 vs r12"
for r in r11 r12; do
  for d in chat_r9_eval chat_r11_eval chat_r12_eval; do
    log "-- $r $d"; $PYT per_style_eval.py --head runs/$r/ckpt.pt --records ds/$d \
      --styles ds/$d/styles.json --vocab runs/r3/draft_vocab.npy --ckpt ckpt --depth 6 2>&1 | tail -1 >> $LOG
  done
  log "-- $r serve_eval"; $PYT per_style_eval.py --head runs/$r/ckpt.pt --records ds/serve_eval \
    --styles ds/serve_eval/styles_all.json --vocab runs/r3/draft_vocab.npy --ckpt ckpt --depth 6 2>&1 | tail -1 >> $LOG
done
log "== X export r12 -> gremlin"
$PYT export_head.py --ckpt runs/r12/ckpt.pt --vocab runs/r3/draft_vocab.npy \
  --out runs/r12/r12.safetensors >> $LOG 2>&1 || { log "export failed"; exit 1; }
scp -q runs/r12/r12.safetensors $G:$GE/heads/r12.safetensors && ssh $G "chmod 644 $GE/heads/r12.safetensors" \
  || { log "push failed"; exit 1; }
log "== A live A/B on gremlin (deployed engine, trees; held-out prompts)"
ssh $G "cat > /tmp/ab_r12.sh" <<'EOF'
E=/opt/granite4-rtx-4070-ti-super/eagle
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
ENG=/run/current-system/sw/bin/granite4-infer
P=$(pgrep -f "[g]en_chat.py"); [ -n "$P" ] && kill -STOP $P
trap "systemctl start granite4-rtx; [ -n \"$P\" ] && kill -CONT $P" EXIT
systemctl stop granite4-rtx; sleep 3
cd $E; rm -f /tmp/ab_plain.json
for h in plain r11 r12; do
  H=$h; [ $h != plain ] && H=$E/heads/$h.safetensors
  $PY $E/ab_heads.py $ENG $h $H
done
echo ABDONE
EOF
ssh $G "bash /tmp/ab_r12.sh > /tmp/ab_r12.out 2>&1"
ssh $G "cat /tmp/ab_r12.out" >> $LOG
log "ALL DONE"
