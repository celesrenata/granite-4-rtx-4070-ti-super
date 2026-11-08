#!/usr/bin/env python3
"""Fake-quantize a Granite checkpoint's layer matrices to int4 (weight-only).

Writes a BF16 checkpoint whose q/k/v/o/gate/up/down weights are
dequant(quant_int4(W)): symmetric int4 codes in [-8, 7], one scale per group
of --group input columns. The engine runs it unchanged (it packs BF16 to its
own format), so its outputs measure the int4 rounding error before any int4
kernel exists. Embeddings, norms and lm_head are copied as-is.

  --method rtn : scale = amax / 7
  --method mse : per group, the clip ratio in [0.70, 1.00] that minimises the
                 squared reconstruction error (cheap, usually much better)
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def quant(w: torch.Tensor, group: int, method: str) -> torch.Tensor:
    if method == "none":  # pipeline control: a re-save must change nothing
        return w.clone()
    n, k = w.shape
    if method == "int8":  # control: symmetric int8 per group (near lossless)
        g = w.float().reshape(n, k // group, group)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 127.0
        return ((g / s).round().clamp(-127, 127) * s).reshape(n, k).to(torch.bfloat16)
    g = w.float().reshape(n, k // group, group)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    if method == "rtn":
        best = amax / 7.0
    else:
        best, best_err = None, None
        for c in torch.linspace(0.70, 1.00, 16):
            s = amax * c / 7.0
            dq = (g / s).round().clamp(-8, 7) * s
            err = (dq - g).pow(2).sum(-1, keepdim=True)
            if best is None:
                best, best_err = s, err
            else:
                m = err < best_err
                best = torch.where(m, s, best)
                best_err = torch.where(m, err, best_err)
    dq = (g / best).round().clamp(-8, 7) * best
    return dq.reshape(n, k).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--method", choices=("rtn", "mse", "int8", "none"), default="rtn")
    ap.add_argument("--only", choices=("all", "mlp"), default="all",
                    help="mlp: quantize gate/up/down only (attention stays as-is)")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors" and f.is_file():
            shutil.copy2(f, dst / f.name)
    err_num = err_den = 0.0
    for shard in sorted(src.glob("*.safetensors")):
        t = load_file(str(shard))
        for name in list(t):
            sel = PROJ if a.only == "all" else ("gate_proj", "up_proj", "down_proj")
            if any(f".{p}." in name for p in sel):
                w = t[name]
                q = quant(w, a.group, a.method)
                err_num += (q.float() - w.float()).pow(2).sum().item()
                err_den += w.float().pow(2).sum().item()
                t[name] = q
        save_file(t, str(dst / shard.name), metadata={"format": "pt"})
        print(f"{shard.name}: done", flush=True)
    print(json.dumps({"method": a.method, "group": a.group, "only": a.only,
                      "relative_weight_error": (err_num / err_den) ** 0.5}))


if __name__ == "__main__":
    main()
