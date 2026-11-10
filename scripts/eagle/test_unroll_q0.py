#!/usr/bin/env python3
"""unroll(q0 > 0) == unroll(q0 = 0)[entries >= q0], on a random head and record (f32)."""
import sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
import train_eagle as te

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
head = te.EagleHead().to(dev).float()
n, depth, q0 = 96, 4, 37
V = 1000
emb = torch.randn(V, te.H, device=dev) * 0.02
toks = torch.randint(0, V, (n,), device=dev)
feats = torch.randn(n, te.H, device=dev)
cos, sin = te.rope_tables(256, dev)
with torch.no_grad():
    full = te.unroll(head, emb, toks, feats, depth, cos, sin)
    part = te.unroll(head, emb, toks, feats, depth, cos, sin, q0=q0)
worst = 0.0
for d, (a, b) in enumerate(zip(full, part), start=1):
    m = n - d
    diff = (a[q0:m] - b[: m - q0]).abs().max().item()
    worst = max(worst, diff)
    print(f"depth {d}: {b.shape[0]} chains, max |diff| = {diff:.3g}")
assert len(full) == len(part)
assert worst < 1e-3, worst
print("unroll q0 OK")
