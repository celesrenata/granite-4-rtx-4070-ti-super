#!/usr/bin/env bash
# r13k5, gremlin side: re-dump the training/eval features with the PRODUCTION K5:V4 engine
# (G4_KV8=5), so the next EAGLE head learns the deployed target. granite4-rtx is stopped for
# the dump and restarted on ANY exit. Each finished dataset gets a DONE marker; the 5090 pulls them.
set -uo pipefail
E=/opt/granite4-rtx-4070-ti-super/eagle
CK=/opt/granite4-rtx-4070-ti-super/ckpt
ENG=/run/current-system/sw/bin/granite4-infer
LOG=$E/stage_r13k5.log
cd $E
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
restart() { pkill -f "[d]ump-hidden" && sleep 3; systemctl start granite4-rtx; log "granite4-rtx restarted ($(systemctl is-active granite4-rtx))"; }
mkdir -p ds/k5
rm -f ds/k5/ALL_DONE
log "== engine $(readlink -f $ENG)"
log "== stopping granite4-rtx (K5 dump only)"
trap restart EXIT
systemctl stop granite4-rtx
sleep 3
for s in serve_eval serve_train3 chat_r12_eval train2000 chat_r12_train; do
  mkdir -p ds/k5/$s
  rm -f ds/k5/$s/DONE ds/k5/$s/hidden.bin
  for f in tokens.bin mask.bin meta.json styles.json styles_all.json; do [ -f ds/$s/$f ] && cp ds/$s/$f ds/k5/$s/; done
  t0=$(date +%s)
  G4_KV8=5 $ENG dump-hidden --checkpoint $CK --gpu 1 --in ds/$s/tokens.bin --out ds/k5/$s/hidden.bin \
    --max-context 4096 >> $LOG 2>&1 || { log "dump $s failed"; exit 1; }
  [ "$(stat -c %s ds/k5/$s/hidden.bin)" = "$(stat -c %s ds/$s/hidden.bin)" ] || { log "size mismatch $s"; exit 1; }
  touch ds/k5/$s/DONE
  log "dumped $s in $(( $(date +%s) - t0 ))s"
done
touch ds/k5/ALL_DONE
log "K5 DUMP DONE"
