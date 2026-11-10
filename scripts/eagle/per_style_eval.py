#!/usr/bin/env python3
"""Per-style exact-gate acceptance for a head checkpoint.

Loads the frozen target + a head checkpoint and runs the exact chain-acceptance
loop (identical to train_eagle.eval_acceptance) per record, grouped by style.
styles.json is a list aligned to record order. Reports tokens/round at each
draft depth K; the serve config uses K=3.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eagle as te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="ckpt.pt (head state)")
    ap.add_argument("--records", required=True, help="Records dir (tokens/mask/hidden)")
    ap.add_argument("--styles", required=True, help="json list of per-record style")
    ap.add_argument("--ckpt", default="/w/ckpt")
    ap.add_argument("--vocab", required=True, help="draft_vocab.npy")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N records (0=all)")
    ap.add_argument("--rope-len", type=int, default=8192, help="at least the longest record")
    ap.add_argument("--window", type=int, default=0,
                    help="diagnostic: the head attends only to its last N step-1 entries (0 = all)")
    a = ap.parse_args()

    dev = "cuda"
    fr = te.load_frozen(a.ckpt, dev)
    st = torch.load(a.head, map_location="cpu")
    n_extra = sum(1 for k in st["head"] if k.startswith("ln_x."))
    head = te.EagleHead(n_extra=n_extra)
    head.load_state_dict(st["head"], strict=False)
    head = head.to(dev).to(torch.bfloat16)
    vocab = np.load(a.vocab)
    vt = torch.as_tensor(vocab, device=dev)
    lm_sub = fr["lm_head.weight"][vt]
    emb = fr["model.embed_tokens.weight"]
    cos, sin = te.rope_tables(a.rope_len, dev)

    rec = te.Records(Path(a.records))
    styles = json.load(open(a.styles))
    assert len(styles) == len(rec), f"styles {len(styles)} != records {len(rec)}"
    styles_sorted = sorted(set(styles))
    # per-style: K -> [produced, rounds]
    acc = {s: {k: [0, 0] for k in range(1, a.depth + 1)} for s in styles_sorted}
    for r in range(min(len(rec), a.limit) if a.limit else len(rec)):
        toks, mask, hid = rec.get(r)
        toks, mask, hid = toks.to(dev), mask.to(dev), hid.to(dev)
        n = toks.shape[0]
        s = styles[r]
        with torch.no_grad():
            ex = rec.get_extra(r) if n_extra else None
            start = int(mask.nonzero()[0]) - 1 if (mask is not None and mask.any()) else 0
            # Long records: chains from the reply start only (te.unroll q0; exact for this loop).
            q0 = max(start, 0) if n > te.LONG_RECORD else 0
            te.ATTN_WINDOW = a.window
            preds = te.unroll(head, emb, toks, hid, a.depth, cos, sin,
                              extra=ex.to(dev) if ex is not None else None, q0=q0)
            draft = [vt[(p.to(torch.bfloat16) @ lm_sub.t()).argmax(-1)] for p in preds]
            hit = []
            for d, dr in enumerate(draft, start=1):
                m = n - 1 - d
                if m - q0 <= 0:
                    hit.append(np.zeros(max(m, 0), bool))
                    continue
                h = (dr[: m - q0] == toks[q0 + d + 1: d + 1 + m]).cpu().numpy()
                hit.append(np.concatenate([np.zeros(q0, bool), h]))
            for K in range(1, a.depth + 1):
                p = max(start, 0)
                while p < n - 2:
                    a2 = 0
                    while a2 < K and a2 < len(hit) and p < len(hit[a2]) and hit[a2][p]:
                        a2 += 1
                    acc[s][K][0] += a2 + 1
                    acc[s][K][1] += 1
                    p += a2 + 1

    print("per-style tokens/round (exact gate):")
    for K in range(1, a.depth + 1):
        row = f"  K={K}: " + "  ".join(
            f"{s[:9]:9s}={acc[s][K][0]/acc[s][K][1]:5.3f}" if acc[s][K][1]
            else f"{s[:9]:9s}=   n/a" for s in styles_sorted)
        print(row)
    # overall per-depth table
    agg = {k: [sum(acc[s][k][0] for s in styles_sorted),
               sum(acc[s][k][1] for s in styles_sorted)] for k in range(1, a.depth + 1)}
    print("overall:", " ".join(f"d{k}={agg[k][0]/agg[k][1]:.3f}" if agg[k][1]
                               else f"d{k}=n/a" for k in range(1, a.depth + 1)))


if __name__ == "__main__":
    main()
