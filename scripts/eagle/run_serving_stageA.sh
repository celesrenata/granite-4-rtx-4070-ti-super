#!/usr/bin/env bash
# Stage A (host, GPU free): turn serving_data.json into training/eval records and
# dump hidden states, then report draft-vocab coverage. Run AFTER gen_serving
# finishes and the serve process is stopped.
set -euo pipefail
cd /opt/granite4-rtx-4070-ti-super/eagle
CKPT=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=/opt/granite4-rtx-4070-ti-super/dev/build/engine/granite4-infer
GPU=1
# Non-interactive ssh shell has no generic python3 on PATH; use the Nix env
# that ran gen_serving.py (has tokenizers + numpy).
PY=/nix/store/0kb3r23cflhjb0fknss1x7mxx8aqgsnj-python3-3.14.8-env/bin/python3

echo "== [A1] verify serving_data.json =="
$PY -c "import json;d=json.load(open('serving_data.json'));print('records',len(d));assert all(d[i].get('reply') for i in range(len(d))),'missing reply';assert d[0]['finish'] in ('stop','length');print('finish sample',d[0]['finish']);import collections;print('finishes',collections.Counter(x['finish'] for x in d))"

echo "== [A2] split into eval (6/style) + train =="
$PY split_serving.py --data serving_data.json --per-style 6 --out-dir ds/serving

echo "== [A3] prep records (tokenize) =="
$PY prep_selfgen.py --tokenizer $CKPT/tokenizer.json --data ds/serving/eval.json  --out ds/serve_eval
$PY prep_selfgen.py --tokenizer $CKPT/tokenizer.json --data ds/serving/train.json --out ds/serve_train

echo "== [A4] dump hidden (host, GPU) =="
$ENG dump-hidden --checkpoint $CKPT --gpu $GPU --in ds/serve_eval/tokens.bin  --out ds/serve_eval/hidden.bin  --max-context 4096
$ENG dump-hidden --checkpoint $CKPT --gpu $GPU --in ds/serve_train/tokens.bin --out ds/serve_train/hidden.bin --max-context 4096

echo "== [A5] draft-vocab coverage (r3 vocab on serving reply tokens) =="
$PY vocab_coverage.py --tokens ds/serve_train/tokens.bin --mask ds/serve_train/mask.bin --styles ds/serving/train_styles.json --vocab runs/r3/draft_vocab.npy
echo "STAGE A done"
