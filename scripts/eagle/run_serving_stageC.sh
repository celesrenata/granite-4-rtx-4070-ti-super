#!/usr/bin/env bash
# Stage C: per-style exact-gate acceptance for a head, on the held-out serving
# eval set (6 unseen prompts per style) plus a chat-forgetting check on the
# in-distribution chat eval set. Runs in the granite4-eagle container.
# serve_eval record order = style-sorted (split_serving.py), 6 per style.
set -euo pipefail
cd /opt/granite4-rtx-4070-ti-super/eagle

RUN=${RUN:-r6}
CHAT_LIMIT=${CHAT_LIMIT:-100}

echo "== [C] per-style eval for $RUN (serve_eval, 6 prompts/style) =="
podman run --rm --network host --gpus all --shm-size=8g \
  -v /opt/granite4-rtx-4070-ti-super/eagle:/w/eagle \
  -v /opt/granite4-rtx-4070-ti-super/ckpt:/w/ckpt \
  --entrypoint "" localhost/granite4-eagle:latest \
  /opt/conda/bin/python /w/eagle/per_style_eval.py \
    --head /w/eagle/runs/$RUN/ckpt.pt \
    --records /w/eagle/ds/serve_eval \
    --styles /w/eagle/ds/serving/eval_styles.json \
    --vocab /w/eagle/runs/r3/draft_vocab.npy \
    --ckpt /w/ckpt --depth 6

echo "== [C] chat-forgetting check (first $CHAT_LIMIT chat records) =="
HOSTPY=${HOSTPY:-/nix/store/0kb3r23cflhjb0fknss1x7mxx8aqgsnj-python3-3.14.8-env/bin/python3}
$HOSTPY -c "import json; json.dump(['chat']*$CHAT_LIMIT, open('ds/serve_eval/chat_styles.json','w'))"
podman run --rm --network host --gpus all --shm-size=8g \
  -v /opt/granite4-rtx-4070-ti-super/eagle:/w/eagle \
  -v /opt/granite4-rtx-4070-ti-super/ckpt:/w/ckpt \
  --entrypoint "" localhost/granite4-eagle:latest \
  /opt/conda/bin/python /w/eagle/per_style_eval.py \
    --head /w/eagle/runs/$RUN/ckpt.pt \
    --records /w/eagle/ds/train_eval \
    --styles /w/eagle/ds/serve_eval/chat_styles.json \
    --vocab /w/eagle/runs/r3/draft_vocab.npy \
    --ckpt /w/ckpt --depth 6 --limit $CHAT_LIMIT
echo "STAGE C done -> $RUN"
