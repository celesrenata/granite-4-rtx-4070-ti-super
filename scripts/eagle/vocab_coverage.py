#!/usr/bin/env python3
"""Per-style draft-vocab coverage over the held-out serving eval replies.

For each style, count the reply tokens (mask==1) and what fraction are inside
the draft vocab (the set of ids the head can emit). If a weak style's replies
are mostly out-of-vocab, acceptance is capped by vocab coverage, not head
quality — and the fix is to grow the draft vocab for those tokens, not to
retrain. Pure host-side numpy; no GPU/model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def parse_blocks(raw: np.ndarray):
    lengths, offsets = [], []
    i = 0
    while i < len(raw):
        lengths.append(int(raw[i]))
        offsets.append(i)
        i += 1 + int(raw[i])
    return lengths, offsets


def main():
    root = Path("/opt/granite4-rtx-4070-ti-super/eagle/ds/serve_eval")
    raw = np.fromfile(root / "tokens.bin", dtype=np.int32)
    mask = np.fromfile(root / "mask.bin", dtype=np.uint8)
    vocab = set(np.load("/opt/granite4-rtx-4070-ti-super/eagle/runs/r3/draft_vocab.npy").tolist())
    styles = json.load(open("/opt/granite4-rtx-4070-ti-super/eagle/ds/serving/eval_styles.json"))
    lengths, offsets = parse_blocks(raw)
    by_style = {}
    for r, (L, off) in enumerate(zip(lengths, offsets)):
        toks = raw[off + 1: off + 1 + L]
        m = mask[tok_sum_start(r, lengths): tok_sum_start(r, lengths) + L]
        st = styles[r]
        a = by_style.setdefault(st, [0, 0])
        for t, mk in zip(toks, m):
            if mk == 1:
                a[0] += 1
                if int(t) in vocab:
                    a[1] += 1
    print(f"{'style':9s} reply_tok  in_vocab  coverage")
    tot_in = tot_all = 0
    for st in sorted(by_style):
        all_n, in_n = by_style[st]
        tot_all += all_n; tot_in += in_n
        print(f"{st:9s} {all_n:6d}   {in_n:7d}  {in_n/all_n*100:5.1f}%")
    print(f"{'overall':9s} {tot_all:6d}   {tot_in:7d}  {tot_in/tot_all*100:5.1f}%")


def tok_sum_start(r, lengths):
    return sum(lengths[:r])


if __name__ == "__main__":
    main()
