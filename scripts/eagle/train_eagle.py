#!/usr/bin/env python3
"""EAGLE-style draft head for Granite 4.2 8B: model, training, evaluation.

Head (one Granite decoder layer on top of the target's features):

    x_j  = fc([ln_e embed(tok_{j+1}) ; ln_h feat_j])   (2H -> H, bias)
    h_j  = x_j + o(attn(ln1(x_j)))                      causal over head entries
    h_j  = h_j + down(silu(gate(ln2 h_j)) * up(ln2 h_j))
    f^_j = lnf(h_j)                                     predicted feature j+1
    draft token = argmax(lm_head_sub @ f^_j)            (frozen target lm_head,
                                                         rows of the draft vocab)

feat_j is the target's final-normed hidden state at position j (what the
engine's `dump-hidden` writes; lm_head(feat_j) predicts tok_{j+1}) for
accepted positions, and the head's own f^ for drafted ones. Attention scale,
RoPE (theta 1e7, half split) and head layout follow the target, so the
engine runs the head with its existing kernels.

Training-time test: the head is unrolled `depth` steps. Step 1 sees true
features everywhere; step d of the chain started at entry i takes
(tok_{i+d}, f^_{d-1,i}) and attends to the true entries 0..i plus the chain's
own earlier steps - exactly what it sees when drafting d tokens past an
accepted position. Loss per step: soft cross-entropy against the target's
distribution over the draft vocab + 0.1 * smooth-L1 to the true feature,
weighted 0.8^(d-1), on assistant tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

H, HQ, HK, D, INTER = 4096, 32, 8, 128, 12800
ATTN_SCALE = 0.0078125  # Granite attention_multiplier (the target's layers)
ROPE_THETA = 1e7
EPS = 1e-5


# ---------------------------------------------------------------- data ----
class Records:
    """tokens.bin / mask.bin / hidden.bin written by prep_data.py + dump-hidden."""

    def __init__(self, root: Path, need_hidden: bool = True):
        raw = np.fromfile(root / "tokens.bin", dtype=np.int32)
        self.offsets, self.lengths, self.tok_pos = [], [], []
        i = tok = 0
        while i < len(raw):
            n = int(raw[i])
            self.tok_pos.append(i + 1)
            self.lengths.append(n)
            self.offsets.append(tok)
            i += 1 + n
            tok += n
        self.raw = raw
        self.total = tok
        self.mask = np.fromfile(root / "mask.bin", dtype=np.uint8) if (root / "mask.bin").exists() else None
        self.hidden = (np.memmap(root / "hidden.bin", dtype=np.int16, mode="r").reshape(-1, H)
                       if need_hidden else None)
        if self.hidden is not None and self.hidden.shape[0] != tok:
            raise SystemExit(f"{root}: hidden rows {self.hidden.shape[0]} != tokens {tok}")

    def __len__(self):
        return len(self.lengths)

    def get(self, r):
        n, p, o = self.lengths[r], self.tok_pos[r], self.offsets[r]
        toks = torch.from_numpy(self.raw[p:p + n].astype(np.int64))
        mask = torch.from_numpy(self.mask[o:o + n].astype(np.bool_)) if self.mask is not None else None
        hid = torch.from_numpy(np.ascontiguousarray(self.hidden[o:o + n])).view(torch.bfloat16)
        return toks, mask, hid


class RecordDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.rec = None
        self.n = len(Records(self.root, need_hidden=False))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.rec is None:  # open per worker
            self.rec = Records(self.root)
        return self.rec.get(i)


# --------------------------------------------------------------- model ----
class RMSNorm(nn.Module):
    def __init__(self, n, init=None):
        super().__init__()
        self.weight = nn.Parameter(init.clone().float() if init is not None else torch.ones(n))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * self.weight).to(x.dtype)


def rope_tables(max_pos, device):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, D, 2, device=device, dtype=torch.float64) / D))
    ang = torch.arange(max_pos, device=device, dtype=torch.float64)[:, None] * inv[None]
    return ang.cos().float(), ang.sin().float()


def apply_rope(x, cos, sin):  # x [heads, n, D]; cos/sin [n, D/2]
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).to(x.dtype)


class EagleHead(nn.Module):
    def __init__(self, final_norm_init=None, attn_scale=D ** -0.5):
        super().__init__()
        # Exported with the weights (engine eagle_gpu.cu reads it).
        self.register_buffer("attn_scale", torch.tensor([float(attn_scale)]))
        self.fc = nn.Linear(2 * H, H, bias=True)
        self.ln1, self.ln2 = RMSNorm(H), RMSNorm(H)
        # Input norms: the target's features are ~1000x the embeddings' scale.
        self.ln_e, self.ln_h = RMSNorm(H), RMSNorm(H)
        self.lnf = RMSNorm(H, final_norm_init)
        self.q = nn.Linear(H, HQ * D, bias=False)
        self.k = nn.Linear(H, HK * D, bias=False)
        self.v = nn.Linear(H, HK * D, bias=False)
        self.o = nn.Linear(HQ * D, H, bias=False)
        self.gate = nn.Linear(H, INTER, bias=False)
        self.up = nn.Linear(H, INTER, bias=False)
        self.down = nn.Linear(INTER, H, bias=False)
        for m in (self.fc, self.q, self.k, self.v, self.o, self.gate, self.up, self.down):
            nn.init.normal_(m.weight, std=0.02)
        for m in (self.o, self.down):
            m.weight.data.mul_(0.5)
        nn.init.zeros_(self.fc.bias)

    def qkv(self, x, cos, sin):
        xn = self.ln1(x)
        n = x.shape[0]
        q = self.q(xn).view(n, HQ, D).transpose(0, 1)
        k = self.k(xn).view(n, HK, D).transpose(0, 1)
        v = self.v(xn).view(n, HK, D).transpose(0, 1)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def tail(self, x, attn):  # attn [HQ, n, D]
        n = x.shape[0]
        h = x + self.o(attn.transpose(0, 1).reshape(n, HQ * D))
        hn = self.ln2(h)
        h = h + self.down(F.silu(self.gate(hn)) * self.up(hn))
        return self.lnf(h)


def _attn_chunk(q, k, v, mask, scale):
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    s = s.masked_fill(~mask[None], float("-inf"))
    return (torch.softmax(s, -1).to(v.dtype) @ v)


def masked_attention(q, k, v, mask, scale, chunk=256):
    """Attention with an arbitrary [m, K] boolean mask, query-chunked and
    recomputed in backward (the fused SDPA kernels need a causal mask)."""
    outs = [ckpt.checkpoint(_attn_chunk, q[:, s:s + chunk], k, v, mask[s:s + chunk], scale,
                            use_reentrant=False) for s in range(0, q.shape[1], chunk)]
    return torch.cat(outs, 1)


def unroll(head, emb_w, toks, feats, depth, cos, sin, noise=0.0, grad_ckpt=False):
    """Predicted features per step: preds[d-1][i] ~ feature i + d, for the
    chain started at entry i (i < n - d). toks [n], feats [n, H] (bf16)."""
    n = toks.shape[0]
    keys, vals, preds = [], [], []
    feat_in = feats[: n - 1]
    if noise > 0:
        feat_in = feat_in + (torch.rand_like(feat_in, dtype=torch.float32) * 2 - 1).to(feat_in.dtype) * noise
    m1 = n - 1
    for d in range(1, depth + 1):
        m = n - d
        if m <= 1:
            break
        tok_in = toks[d: d + m]
        fin = feat_in[:m] if d == 1 else preds[-1][:m]
        pos = torch.arange(d - 1, d - 1 + m, device=toks.device)

        def block(fin, tok_in, pos=pos, m=m, d=d):
            x = head.fc(torch.cat([head.ln_e(F.embedding(tok_in, emb_w)), head.ln_h(fin)], -1))
            q, k, v = head.qkv(x, cos[pos], sin[pos])
            return x, q, k, v

        x, q, k, v = block(fin, tok_in)
        keys.append(k)
        vals.append(v)
        kk = torch.cat(keys, 1).repeat_interleave(HQ // HK, 0)
        vv = torch.cat(vals, 1).repeat_interleave(HQ // HK, 0)
        # Query i (this step) sees step-1 entries 0..i and entry i of steps 2..d.
        i = torch.arange(m, device=toks.device)
        cols = [torch.arange(m1, device=toks.device)[None, :] <= i[:, None]]
        for e in range(2, d + 1):
            me = n - e
            cols.append(torch.arange(me, device=toks.device)[None, :] == i[:, None])
        mask = torch.cat(cols, 1)
        scale = float(head.attn_scale)
        if d == 1:
            att = F.scaled_dot_product_attention(q[None], kk[None], vv[None], is_causal=True,
                                                 scale=scale)[0]
        else:
            att = masked_attention(q, kk, vv, mask, scale)
        pred = ckpt.checkpoint(head.tail, x, att, use_reentrant=False) if grad_ckpt else head.tail(x, att)
        preds.append(pred)
    return preds


def soft_ce(pred, target_logits, lm_sub, w, distill=0.0):
    """sum_i w_i * (CE(softmax(target_i), lm_sub pred_i) + distill * D) where
    D = -log p_draft(y*) with y* = target argmax over the draft vocab (a
    differentiable top-1 distillation term that optimizes the acceptance
    metric directly). Also returns weighted top-1 matches."""
    lp = F.log_softmax((pred @ lm_sub.t()).float(), -1)
    p = F.softmax(target_logits, -1)
    loss = -(p * lp).sum(-1)
    y = target_logits.argmax(-1)
    if distill:
        loss = loss - distill * lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    correct = (lp.argmax(-1) == y).float()
    return (loss * w).sum(), (correct * w).sum().detach()


def chunked_ce(pred, target_logits, lm_sub, w, distill=0.0, chunk=512):
    tot = pred.new_zeros((), dtype=torch.float32)
    cor = torch.zeros((), device=pred.device)
    for s in range(0, pred.shape[0], chunk):
        l, c = ckpt.checkpoint(soft_ce, pred[s:s + chunk], target_logits[s:s + chunk], lm_sub,
                               w[s:s + chunk], distill, use_reentrant=False)
        tot = tot + l
        cor = cor + c
    return tot, cor


# ------------------------------------------------------------- weights ----
def init_from_layer(head, ckpt_dir, layer):
    """Start the head's layer from target layer `layer` (its attention scale
    is then the target's) and fc as identity on the feature half."""
    from safetensors import safe_open

    idx = json.load(open(Path(ckpt_dir) / "model.safetensors.index.json"))["weight_map"]
    pre = f"model.layers.{layer}."
    names = {"input_layernorm.weight": head.ln1.weight, "post_attention_layernorm.weight": head.ln2.weight,
             "self_attn.q_proj.weight": head.q.weight, "self_attn.k_proj.weight": head.k.weight,
             "self_attn.v_proj.weight": head.v.weight, "self_attn.o_proj.weight": head.o.weight,
             "mlp.gate_proj.weight": head.gate.weight, "mlp.up_proj.weight": head.up.weight,
             "mlp.down_proj.weight": head.down.weight}
    with torch.no_grad():
        for n, p in names.items():
            with safe_open(str(Path(ckpt_dir) / idx[pre + n]), framework="pt") as f:
                p.copy_(f.get_tensor(pre + n).float())
        head.fc.weight[:, H:].copy_(torch.eye(H))
        head.attn_scale.fill_(ATTN_SCALE)


def load_frozen(ckpt_dir, device):
    from safetensors import safe_open

    idx = json.load(open(Path(ckpt_dir) / "model.safetensors.index.json"))["weight_map"]
    out = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        with safe_open(str(Path(ckpt_dir) / idx[name]), framework="pt") as f:
            out[name] = f.get_tensor(name).to(device=device, dtype=torch.bfloat16)
    return out


def draft_vocab(train_root: Path, size: int, always=(100257,)):
    rec = Records(train_root, need_hidden=False)
    counts = np.bincount(rec.raw[np.concatenate([np.arange(p, p + n) for p, n in zip(rec.tok_pos, rec.lengths)])],
                         minlength=100352)
    order = np.argsort(-counts, kind="stable")
    keep = list(dict.fromkeys(list(always) + order.tolist()))[:size]
    keep = np.array(sorted(keep), dtype=np.int64)
    cover = counts[keep].sum() / counts.sum()
    return keep, float(cover)


# ---------------------------------------------------------------- eval ----
@torch.no_grad()
def eval_acceptance(head, frozen, vocab, rec: Records, depth, cos, sin, device, max_records=None,
                    reply_only=True):
    """Exact chain acceptance on target-generated sequences: from each accepted
    position p, drafts d = 1..depth are kept while they equal the sequence's
    next tokens; a round commits accepted + 1 tokens and moves on."""
    emb_w, lm_sub = frozen["model.embed_tokens.weight"], frozen["lm_head.weight"][vocab]
    vocab_t = torch.as_tensor(vocab, device=device)
    head.eval()
    produced = {k: 0 for k in range(1, depth + 1)}
    rounds = {k: 0 for k in range(1, depth + 1)}
    for r in range(len(rec) if max_records is None else min(max_records, len(rec))):
        toks, mask, hid = rec.get(r)
        toks, hid = toks.to(device), hid.to(device)
        n = toks.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = unroll(head, emb_w, toks, hid, depth, cos, sin)
        draft = [vocab_t[(p.to(torch.bfloat16) @ lm_sub.t()).argmax(-1)] for p in preds]
        # hit[d-1][i]: step d of the chain at entry i predicts toks[i + d + 1]
        hit = []
        for d, dr in enumerate(draft, start=1):
            m = n - 1 - d
            hit.append((dr[:m] == toks[d + 1: d + 1 + m]).cpu().numpy() if m > 0 else np.zeros(0, bool))
        start = int(mask.nonzero()[0]) - 1 if (reply_only and mask is not None and mask.any()) else 0
        for K in range(1, depth + 1):
            p = max(start, 0)  # entry p: tok_{p+1} known, drafts tok_{p+2}..
            while p < n - 2:
                a = 0
                while a < K and a < len(hit) and p < len(hit[a]) and hit[a][p]:
                    a += 1
                produced[K] += a + 1
                rounds[K] += 1
                p += a + 1
    head.train()
    return {K: produced[K] / max(rounds[K], 1) for K in produced}


# --------------------------------------------------------------- train ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval-self", help="target-generated sequences (prompt + 8B reply) with hidden")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--eval-depth", type=int, default=6)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--reg-weight", type=float, default=0.1)
    ap.add_argument("--distill", type=float, default=1.0,
                    help="top-1 distillation weight: -log p_draft(target argmax) per masked row, added to the CE")
    ap.add_argument("--step-decay", type=float, default=0.8,
                    help="per-draft-step loss weight decay (step d is weight^(d-1))")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--head",
                    help="fine-tune: load head weights only (no optimizer state), restart step/scheduler at 0")
    ap.add_argument("--draft-vocab",
                    help="reuse an existing draft_vocab.npy instead of recomputing from --train")
    ap.add_argument("--init-layer", type=int, default=-1,
                    help="initialize the head layer from this target layer (-1: random)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen(args.ckpt, device)
    if args.draft_vocab:
        vocab = np.load(args.draft_vocab)
        print(f"draft vocab {len(vocab)}: reusing {args.draft_vocab}", flush=True)
    else:
        vocab, cover = draft_vocab(Path(args.train), args.vocab)
        np.save(out / "draft_vocab.npy", vocab)
        print(f"draft vocab {len(vocab)}: covers {cover:.4%} of training tokens", flush=True)
    vocab_t = torch.as_tensor(vocab, device=device)
    lm_sub = frozen["lm_head.weight"][vocab_t]
    emb_w = frozen["model.embed_tokens.weight"]
    cos, sin = rope_tables(8192, device)

    head = EagleHead(frozen["model.norm.weight"])
    if args.init_layer >= 0:
        init_from_layer(head, args.ckpt, args.init_layer)
    head = head.to(device)
    print(f"head params: {sum(p.numel() for p in head.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    step0 = 0
    if args.resume:
        st = torch.load(args.resume, map_location=device)
        head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"])
        step0 = st["step"]
    elif args.head:
        st = torch.load(args.head, map_location=device)
        head.load_state_dict(st["head"])
        print(f"fine-tune from {args.head} (step 0, fresh optimizer)", flush=True)

    ds = RecordDataset(args.train)
    total_steps = int(math.ceil(len(ds) * args.epochs / args.accum))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    # Linear warmup (restarted on --resume) into a cosine decay to 10% at
    # total_steps.
    sched = lambda s: (min(1.0, (s - step0 + 1) / args.warmup) *
                       (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0)))))
    eval_rec = Records(Path(args.eval_self)) if args.eval_self else None
    log = open(out / "train_log.jsonl", "a")
    noise = args.noise * float(frozen["model.norm.weight"].float().abs().mean())

    step, micro, t0, seen = step0, 0, time.time(), 0
    acc_stats = np.zeros((args.depth, 2))
    loss_acc = 0.0
    epoch = 0
    while step < total_steps:
        g = torch.Generator().manual_seed(1000 + epoch)
        loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=True, generator=g,
                                             num_workers=4, pin_memory=True, prefetch_factor=4)
        for toks, mask, hid in loader:
            toks, mask, hid = toks.to(device, non_blocking=True), mask.to(device), hid.to(device, non_blocking=True)
            n = toks.shape[0]
            if n < 8:
                continue
            with torch.no_grad():
                tgt_logits = (hid @ lm_sub.t()).float()  # [n, V'] target dist. over the draft vocab
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = unroll(head, emb_w, toks, hid, args.depth, cos, sin, noise=noise, grad_ckpt=True)
                loss = 0.0
                for d, pred in enumerate(preds, start=1):
                    m = n - 1 - d  # entries with a next token to predict
                    w = mask[d + 1: d + 1 + m].float()
                    if w.sum() == 0:
                        continue
                    ce, cor = chunked_ce(pred[:m], tgt_logits[d: d + m], lm_sub, w,
                                         distill=args.distill)
                    reg = (F.smooth_l1_loss(pred[:m].float(), hid[d: d + m].float(), reduction="none").mean(-1) * w).sum()
                    loss = loss + (args.step_decay ** (d - 1)) * (ce + args.reg_weight * reg) / w.sum()
                    acc_stats[d - 1] += (cor.item(), w.sum().item())
            (loss / args.accum).backward()
            loss_acc += float(loss)
            seen += n
            micro += 1
            if micro % args.accum:
                continue
            for gp in opt.param_groups:
                gp["lr"] = args.lr * sched(step)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                dt = time.time() - t0
                rec = {"step": step, "of": total_steps, "loss": loss_acc / (25 * args.accum),
                       "lr": args.lr * sched(step), "tok_s": seen / dt,
                       "top1": [round(c / max(t, 1), 4) for c, t in acc_stats]}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                acc_stats[:] = 0
                loss_acc = 0.0
                seen, t0 = 0, time.time()
            if eval_rec is not None and (step % args.eval_every == 0 or step == total_steps):
                acc = eval_acceptance(head, frozen, vocab, eval_rec, args.eval_depth, cos, sin, device)
                rec = {"step": step, "eval_tokens_per_round": {k: round(v, 3) for k, v in acc.items()}}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                torch.save({"head": head.state_dict(), "opt": opt.state_dict(), "step": step},
                           out / "ckpt.pt")
            if step >= total_steps:
                break
        epoch += 1
    torch.save({"head": head.state_dict(), "step": step, "vocab": vocab}, out / "head_final.pt")


if __name__ == "__main__":
    main()
