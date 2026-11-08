#!/usr/bin/env python3
"""Offline estimate of static draft trees vs a chain for a head checkpoint.

For each start position, unroll the head along the TRUE continuation (the
path every tree must contain to accept anything) and record the rank of the
true token among the head's draft logits at each depth. A tree node is a
tuple of ranks (r1..rd); it is accepted iff the true ranks start with that
tuple. Expected tokens/round of a tree = 1 + sum of its node probabilities.
Trees are grown greedily from the root by highest node probability (child
probability <= parent, so this is the best prefix-closed tree per budget).
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eagle as te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--ckpt", default="/w/ckpt")
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--max-rank", type=int, default=4)
    ap.add_argument("--budgets", default="3,5,6,7,10,14,15")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda"
    fr = te.load_frozen(a.ckpt, dev)
    st = torch.load(a.head, map_location="cpu")
    head = te.EagleHead()
    head.load_state_dict(st["head"], strict=False)
    head = head.to(dev).to(torch.bfloat16)
    vt = torch.as_tensor(np.load(a.vocab), device=dev)
    lm_sub = fr["lm_head.weight"][vt]
    emb = fr["model.embed_tokens.weight"]
    cos, sin = te.rope_tables(8192, dev)
    # full-vocab id -> draft-vocab index (-1 if absent: never draftable)
    inv = torch.full((int(fr["lm_head.weight"].shape[0]),), -1, dtype=torch.long, device=dev)
    inv[vt.long()] = torch.arange(vt.numel(), device=dev)

    rec = te.Records(Path(a.records))
    big = a.max_rank  # rank value meaning ">= max_rank" (stops every tree path)
    paths = Counter()
    total = 0
    nrec = min(len(rec), a.limit) if a.limit else len(rec)
    for r in range(nrec):
        toks, mask, hid = rec.get(r)
        toks, mask, hid = toks.to(dev), mask.to(dev), hid.to(dev)
        n = toks.shape[0]
        start = int(mask.nonzero()[0]) - 1 if (mask is not None and mask.any()) else 0
        start = max(start, 0)
        with torch.no_grad():
            preds = te.unroll(head, emb, toks, hid, a.depth, cos, sin)
            ranks = []
            for d, p in enumerate(preds, start=1):
                m = n - 1 - d
                if m <= 0:
                    ranks.append(None)
                    continue
                logits = (p[:m].to(torch.bfloat16) @ lm_sub.t()).float()
                tgt = inv[toks[d + 1: d + 1 + m].long()]
                tl = logits.gather(1, tgt.clamp(min=0)[:, None])
                rk = (logits > tl).sum(1)
                rk = torch.where(tgt < 0, torch.full_like(rk, big), rk.clamp(max=big))
                ranks.append(rk.cpu().numpy())
        for p in range(start, n - 2):
            tup = []
            for d in range(a.depth):
                rd = ranks[d]
                if rd is None or p >= len(rd) or rd[p] >= big:
                    break
                tup.append(int(rd[p]))
            # count every prefix: node (r1..rk) accepted iff truth starts with it
            for k in range(1, len(tup) + 1):
                paths[tuple(tup[:k])] += 1
            total += 1

    prob = {k: v / total for k, v in paths.items()}
    chain = [1 + sum(prob.get((0,) * k, 0.0) for k in range(1, K + 1)) for K in range(1, a.depth + 1)]
    print(f"positions={total}")
    print("chain tokens/round:", " ".join(f"K{K}={c:.3f}" for K, c in enumerate(chain, 1)))
    print("depth-1 top-b:", " ".join(
        f"top{b}={sum(prob.get((i,), 0.0) for i in range(b)):.3f}" for b in range(1, a.max_rank + 1)))
    res = {"positions": total, "chain": chain, "trees": {}}
    for B in [int(x) for x in a.budgets.split(",")]:
        tree, frontier = [], [(i,) for i in range(a.max_rank)]
        while len(tree) < B and frontier:
            best = max(frontier, key=lambda t: prob.get(t, 0.0))
            frontier.remove(best)
            tree.append(best)
            if len(best) < a.depth:
                frontier += [best + (i,) for i in range(a.max_rank)]
        exp = 1 + sum(prob.get(t, 0.0) for t in tree)
        depth = max(len(t) for t in tree)
        ck = min(B, a.depth)
        print(f"budget {B:2d}: tree {exp:.3f} tok/round (depth {depth}) vs chain K{ck} {chain[ck - 1]:.3f}")
        res["trees"][B] = {"tokens_per_round": exp, "nodes": [list(t) for t in tree]}
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
