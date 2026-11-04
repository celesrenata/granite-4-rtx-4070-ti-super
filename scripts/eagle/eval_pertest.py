#!/usr/bin/env python3
"""Per-depth draft accuracy: localizes where the draft chain loses tokens.

For a head checkpoint, over a held-out record set, computes for each draft
depth d = 1..6 the fraction of *masked* (assistant) target positions where the
step-d chain draft equals the target's own next token (toks[i + d + 1]).
Also reports step-1 draft-vs-target-argmax agreement (decoupling head error
from target error) and the cumulative accepted-prefix length the exact gate
would give.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eagle as te


def main():
    ck, sets = sys.argv[1], sys.argv[2:]
    dev = "cuda"
    fr = te.load_frozen("/w/ckpt", dev)
    st = torch.load(ck, map_location="cpu")
    head = te.EagleHead()
    head.load_state_dict(st["head"], strict=False)
    head = head.to(dev).to(torch.bfloat16)
    vocab = np.load(Path(ck).with_name("draft_vocab.npy"))
    vt = torch.as_tensor(vocab, device=dev)
    lm = fr["lm_head.weight"]
    emb = fr["model.embed_tokens.weight"]
    cos, sin = te.rope_tables(8192, dev)
    for s in sets:
        rec = te.Records(Path(s))
        n = len(rec)
        step_match = {d: [0, 0] for d in range(1, 7)}   # d: [hit, tot] on masked
        d1_targ = [0, 0]                                 # step-1 draft == target argmax
        d1_seq = [0, 0]                                  # target argmax == next token
        for r in range(n):
            toks, mask, hid = rec.get(r)
            toks, mask, hid = toks.to(dev), mask.to(dev), hid.to(dev)
            mlen = toks.shape[0]
            with torch.no_grad():
                preds = te.unroll(head, emb, toks, hid, 6, cos, sin)
            tgt = (hid @ lm.t())  # [n, V] target full logits
            for d, p in enumerate(preds, start=1):
                m = mlen - 1 - d  # entries i < m have a token at i+d+1
                if m <= 0:
                    continue
                draft = vt[(p[:m].to(torch.bfloat16) @ lm[vt].t()).argmax(-1)]
                nxt = toks[d + 1: d + 1 + m]
                w = mask[d + 1: d + 1 + m]
                step_match[d][0] += int(((draft == nxt) & w).sum())
                step_match[d][1] += int(w.sum())
            # step-1 diagnostics on masked
            m = mlen - 2
            p1 = preds[0][:m]
            draft1 = vt[(p1.to(torch.bfloat16) @ lm[vt].t()).argmax(-1)]
            targ1 = tgt[1:1 + m].argmax(-1)
            w1 = mask[2:2 + m]
            d1_targ[0] += int(((draft1 == targ1) & w1).sum())
            d1_targ[1] += int(w1.sum())
            d1_seq[0] += int(((targ1 == toks[2:2 + m]) & w1).sum())
            d1_seq[1] += int(w1.sum())
        print(f"== {s} ({n} records) ==")
        print("  per-depth draft==next-token (masked):",
              {d: round(step_match[d][0] / max(step_match[d][1], 1), 4) for d in step_match})
        print(f"  step1 draft==target_argmax {d1_targ[0]/max(d1_targ[1],1):.4f}  "
              f"target_argmax==next {d1_seq[0]/max(d1_seq[1],1):.4f}")
        cum = 1.0
        print("  cumulative accepted/round:", end=" ")
        for d in range(1, 7):
            cum *= step_match[d][0] / max(step_match[d][1], 1)
            print(round(cum + 1, 3), end=" ")
        print()


if __name__ == "__main__":
    main()
