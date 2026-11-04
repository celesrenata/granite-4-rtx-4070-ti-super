#!/usr/bin/env python3
"""Export a train_eagle.py checkpoint to the engine's head format.

safetensors with bf16 fc.{weight,bias}, ln1/ln2/lnf.weight,
q/k/v/o/gate/up/down.weight and an int32 draft_vocab (eagle_gpu.h).
`--random` writes an untrained head instead (pipeline tests: the target's
exact verify keeps the output identical to plain greedy whatever it drafts).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eagle as te  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="head_final.pt / ckpt.pt")
    ap.add_argument("--vocab", help="draft_vocab.npy (default: next to --ckpt)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.random:
        torch.manual_seed(0)
        sd = te.EagleHead().state_dict()
        vocab = np.arange(32768, dtype=np.int64)
    else:
        st = torch.load(args.ckpt, map_location="cpu")
        sd = dict(st["head"])
        # Runs before the attn_scale buffer used the target's 1/128.
        sd.setdefault("attn_scale", torch.tensor([te.ATTN_SCALE]))
        vocab = np.load(args.vocab or Path(args.ckpt).with_name("draft_vocab.npy"))
    out = {k: v.to(torch.float32 if k == "attn_scale" else torch.bfloat16).contiguous()
           for k, v in sd.items()}
    out["draft_vocab"] = torch.from_numpy(vocab.astype(np.int32))
    save_file(out, args.out)
    print(f"wrote {args.out}: {len(out)} tensors, draft vocab {len(vocab)}")


if __name__ == "__main__":
    main()
