#!/usr/bin/env bash
# Multi-layer (EAGLE-3 style) head experiment, run after stage_r10.sh:
#   D dump layer features (new engine build) + bit-check its final hidden
#   T train ml1 (r10 + fuse over layers $FEAT_LAYERS) and a control r10c
#     (same recipe, final features only), both from r10
#   E offline eval r10 / r10c / ml1 (+ tree estimate)
# The service is stopped for the GPU stages and restarted on ANY exit.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=${ENG:-/home/celes/src/paged-shared/build/engine/granite4-infer}
PY=/nix/store/a1ix6fp6hinpbnra6vwn43zvkvhsyam3-python3-3.13.15-env/bin/python3
FEAT_LAYERS=${FEAT_LAYERS:-2,20}
LOG=$E/stage_ml.log
cd $E
export LD_LIBRARY_PATH="$(ls -d /nix/store/*cuda-merged-12.9 | head -1)/lib64:/run/opengl-driver/lib"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
restart() { pkill -f "[d]ump-hidden" && sleep 3; systemctl start granite4-rtx; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
podrun() {
  podman run --rm --network host --gpus all --shm-size=8g \
    -v $E:/w/eagle -v $CK:/w/ckpt --entrypoint "" localhost/granite4-eagle:latest \
    /opt/conda/bin/python "$@"
}
log "waiting for stage_r10 to exit"
while pgrep -f "[s]tage_r10.sh" > /dev/null; do sleep 60; done
grep -q "ALL DONE" stage_r10.log || { log "stage_r10 did not finish"; exit 1; }
[ -f runs/r10/ckpt.pt ] || { log "no runs/r10/ckpt.pt"; exit 1; }
cp train_eagle_ml.py train_eagle.py
log "== stopping granite4-rtx (GPU stages)"
trap restart EXIT
systemctl stop granite4-rtx
sleep 3
log "== D dump features (layers $FEAT_LAYERS)"
for s in chat_r9_eval serve_eval r10_train; do
  G4_KV8=1 $ENG dump-hidden --checkpoint $CK --gpu 1 --in ds/$s/tokens.bin --out ds/$s/hidden_chk.bin \
    --feat-layers $FEAT_LAYERS --feat-out ds/$s/feat.bin --max-context 4096 >> $LOG 2>&1 \
    || { log "dump $s failed"; exit 1; }
  if cmp -s ds/$s/hidden.bin ds/$s/hidden_chk.bin; then log "$s: final hidden bit-identical"; \
  else log "$s: final hidden DIFFERS from the stored dump"; fi
  rm -f ds/$s/hidden_chk.bin
done
COMMON="--ckpt /w/ckpt --train /w/eagle/ds/r10_train --eval-self /w/eagle/ds/chat_r9_eval
  --depth 3 --eval-depth 6 --head /w/eagle/runs/r10/ckpt.pt --draft-vocab /w/eagle/runs/r3/draft_vocab.npy
  --epochs 2 --lr 1e-4 --warmup 50 --accum 8 --distill 1.0 --step-decay 0.8 --eval-every 200"
log "== T ml1 (multi-layer, from r10)"
podrun /w/eagle/train_eagle.py $COMMON --extra-layers 2 --out /w/eagle/runs/ml1 > ft_ml1.log 2>&1 \
  || { log "ml1 training failed"; exit 1; }
log "ml1: $(grep eval_tokens_per_round ft_ml1.log | tail -1)"
log "== T r10c (control, final features only)"
podrun /w/eagle/train_eagle.py $COMMON --out /w/eagle/runs/r10c > ft_r10c.log 2>&1 \
  || { log "r10c training failed"; exit 1; }
log "r10c: $(grep eval_tokens_per_round ft_r10c.log | tail -1)"
log "== E offline eval (depth 6)"
for r in r10 r10c ml1; do
  log "-- $r chat"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/chat_r9_eval \
    --styles /w/eagle/ds/chat_r9_eval/styles.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -1 >> $LOG
  log "-- $r serving"; podrun /w/eagle/per_style_eval.py --head /w/eagle/runs/$r/ckpt.pt --records /w/eagle/ds/serve_eval \
    --styles /w/eagle/ds/serve_eval/styles_all.json --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt --depth 6 2>&1 | tail -1 >> $LOG
done
log "-- ml1 tree estimate (chat)"
podrun /w/eagle/tree_estimate.py --head /w/eagle/runs/ml1/ckpt.pt --records /w/eagle/ds/chat_r9_eval \
  --vocab /w/eagle/runs/r3/draft_vocab.npy --ckpt /w/ckpt 2>&1 | grep -E "chain|budget" >> $LOG
log "ALL DONE"
