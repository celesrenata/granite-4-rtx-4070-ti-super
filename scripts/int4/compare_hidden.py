#!/usr/bin/env python3
"""Teacher-forced quality of a weight format vs a baseline, from dump-hidden.

Both runs dump the final-normed hidden states of the same records
(tokens.bin / mask.bin); logits = hidden @ lm_head^T (BF16 lm_head from the
checkpoint). Reports, over every position and over the reply (mask) positions:
top-1 agreement with the baseline, mean NLL of the true next token for both,
and the mean KL(baseline || candidate).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

H = 4096


def records(root: Path):
    raw = np.fromfile(root / "tokens.bin", dtype=np.int32)
    i, out = 0, []
    while i < len(raw):
        n = int(raw[i])
        out.append(raw[i + 1:i + 1 + n])
        i += 1 + n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--base", required=True, help="baseline hidden.bin")
    ap.add_argument("--cand", required=True, nargs="+", help="candidate hidden.bin(s)")
    ap.add_argument("--ckpt", required=True)
    a = ap.parse_args()
    root = Path(a.records)
    recs = records(root)
    mask = np.fromfile(root / "mask.bin", dtype=np.uint8)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    idx = json.load(open(Path(a.ckpt) / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(Path(a.ckpt) / idx["lm_head.weight"]), "pt") as f:
        lm = f.get_tensor("lm_head.weight").to(dev)
    base = np.memmap(a.base, dtype=np.int16, mode="r").reshape(-1, H)
    for cpath in a.cand:
        cand = np.memmap(cpath, dtype=np.int16, mode="r").reshape(-1, H)
        st = {k: [0.0, 0] for k in ("agree", "nll_b", "nll_c", "kl")}
        stm = {k: [0.0, 0] for k in st}
        off = 0
        for toks in recs:
            n = len(toks)
            hb = torch.from_numpy(np.ascontiguousarray(base[off:off + n - 1])).view(torch.bfloat16).to(dev)
            hc = torch.from_numpy(np.ascontiguousarray(cand[off:off + n - 1])).view(torch.bfloat16).to(dev)
            tgt = torch.from_numpy(toks[1:].astype(np.int64)).to(dev)
            m = torch.from_numpy(mask[off + 1:off + n].astype(np.bool_)).to(dev)
            lb = torch.log_softmax((hb @ lm.T).float(), -1)
            lc = torch.log_softmax((hc @ lm.T).float(), -1)
            agree = (lb.argmax(-1) == lc.argmax(-1)).float()
            nb = -lb.gather(1, tgt[:, None])[:, 0]
            nc = -lc.gather(1, tgt[:, None])[:, 0]
            kl = (lb.exp() * (lb - lc)).sum(-1)
            for k, v in (("agree", agree), ("nll_b", nb), ("nll_c", nc), ("kl", kl)):
                st[k][0] += v.sum().item()
                st[k][1] += v.numel()
                stm[k][0] += v[m].sum().item()
                stm[k][1] += int(m.sum().item())
            off += n
        f = lambda d, k: d[k][0] / max(d[k][1], 1)
        print(json.dumps({
            "candidate": cpath, "positions": st["agree"][1], "reply_positions": stm["agree"][1],
            "top1_agree_all": round(f(st, "agree"), 4), "top1_agree_reply": round(f(stm, "agree"), 4),
            "nll_base_reply": round(f(stm, "nll_b"), 4), "nll_cand_reply": round(f(stm, "nll_c"), 4),
            "ppl_ratio_reply": round(float(np.exp(f(stm, "nll_c") - f(stm, "nll_b"))), 4),
            "kl_reply": round(f(stm, "kl"), 5)}), flush=True)


if __name__ == "__main__":
    main()
