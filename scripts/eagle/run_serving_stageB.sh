#!/usr/bin/env bash
# Stage B: fine-tune the r3 head on the serving-style target-generated records
# (serve_train), evaluating on the held-out serve_eval set. Runs in the
# granite4-eagle podman container (torch+CUDA, --gpus all). Requires stage A to
# have produced ds/serve_train + ds/serve_eval (with hidden.bin). GPU must be
# free (serve stopped).
set -euo pipefail
cd /opt/granite4-rtx-4070-ti-super/eagle

EPOCHS=${EPOCHS:-6}
LR=${LR:-1e-4}
WARMUP=${WARMUP:-30}
ACCUM=${ACCUM:-8}
EVAL_EVERY=${EVAL_EVERY:-25}
RUN=${RUN:-r5}
TRAIN=${TRAIN:-serve_train}

echo "== [B] fine-tune $RUN from runs/r3/ckpt.pt on $TRAIN =="
podman run --rm --network host --gpus all --shm-size=8g \
  -v /opt/granite4-rtx-4070-ti-super/eagle:/w/eagle \
  -v /opt/granite4-rtx-4070-ti-super/ckpt:/w/ckpt \
  --entrypoint "" localhost/granite4-eagle:latest \
  /opt/conda/bin/python /w/eagle/train_eagle.py \
    --ckpt /w/ckpt \
    --train /w/eagle/ds/$TRAIN \
    --eval-self /w/eagle/ds/serve_eval \
    --out /w/eagle/runs/$RUN \
    --depth 3 --eval-depth 6 \
    --head /w/eagle/runs/r3/ckpt.pt \
    --draft-vocab /w/eagle/runs/r3/draft_vocab.npy \
    --epochs "$EPOCHS" --lr "$LR" --warmup "$WARMUP" --accum "$ACCUM" \
    --distill 1.0 --step-decay 0.8 \
    --eval-every "$EVAL_EVERY"
echo "STAGE B done -> runs/$RUN"
