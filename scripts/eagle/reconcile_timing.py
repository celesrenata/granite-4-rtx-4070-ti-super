#!/usr/bin/env python3
"""Reconcile the serve benchmark with G4_EAGLE_TIMING: launch serve --eagle with
G4_EAGLE_TIMING=1, send the serving prompts, and collect the per-generation
device-time lines (head/verify split + tok/rnd = tokens per round). Reports the
mean tok/rnd and tok/s so the end-to-end speedup is tied to a real acceptance
rate. Correctness is already established (bench_serving: 400/400 identical).

A background thread drains stderr so the request loop never blocks on a stderr
readline (the serve keeps stderr open for its whole lifetime).
"""
from __future__ import annotations
import argparse, json, os, queue, re, statistics, subprocess, threading
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
import bench_serving as bs  # render_template/load_template + EOS

def probe(engine, ckpt, gpu, max_ctx, head, toks, max_tokens, k):
    env = dict(os.environ, G4_EAGLE_TIMING="1")
    cmd = [engine, "serve", "--checkpoint", ckpt, "--gpu", str(gpu),
           "--max-context", str(max_ctx)]
    if head:
        cmd += ["--eagle", head]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
    err_q = queue.Queue()
    def drain():
        for line in p.stderr:
            err_q.put(line.strip())
        err_q.put(None)
    threading.Thread(target=drain, daemon=True).start()
    meta = json.loads(p.stdout.readline())
    assert meta.get("ready"), meta
    lines = []
    for i, t in enumerate(toks):
        req = json.dumps({"tokens": t, "max_tokens": max_tokens, "eos": bs.EOS, "eagle": k})
        p.stdin.write(req + "\n"); p.stdin.flush()
        json.loads(p.stdout.readline())
        # non-blocking harvest of G4_EAGLE lines emitted so far
        while True:
            try:
                l = err_q.get(timeout=0.05)
            except queue.Empty:
                break
            if l is None:
                break
            if l.startswith("G4_EAGLE"):
                lines.append(l)
    p.stdin.close()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    trnd, tps, head_rnd, verify_rnd = [], [], [], []
    for l in lines:
        m = re.search(r"tok/rnd=([\d.]+)", l)
        s = re.search(r"tok/s=([\d.]+)", l)
        h = re.search(r"head_ms=[\d.]+ \(([\d.]+)/rnd\)", l)
        v = re.search(r"verify_ms=[\d.]+ \(([\d.]+)/rnd\)", l)
        if m: trnd.append(float(m.group(1)))
        if s: tps.append(float(s.group(1)))
        if h: head_rnd.append(float(h.group(1)))
        if v: verify_rnd.append(float(v.group(1)))
    return trnd, tps, head_rnd, verify_rnd, len(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/granite4-rtx-4070-ti-super/dev/build/engine/granite4-infer")
    ap.add_argument("--checkpoint", default="/opt/granite4-rtx-4070-ti-super/ckpt")
    ap.add_argument("--prompts", default=str(HERE / "serve_styles.json"))
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--heads", default="r3,r5")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.checkpoint + "/tokenizer.json")
    tmpl = bs.load_template(HERE / "prep_selfgen.py")
    data = json.load(open(a.prompts))
    if a.limit:
        data = data[:a.limit]
    toks = [tok.encode(bs.render_prompt(tmpl, d["prompt"]), add_special_tokens=False).ids for d in data]

    heads = {}
    for h in a.heads.split(","):
        heads[h] = f"/opt/granite4-rtx-4070-ti-super/eagle/heads/{h}.safetensors"

    print(f"prompts={len(toks)} K={a.k}\n")
    for name, head in heads.items():
        trnd, tps, hr, vr, n = probe(a.engine, a.checkpoint, a.gpu, a.max_context,
                                     head, toks, a.max_tokens, a.k)
        if not trnd:
            print(f"{name}: no G4_EAGLE lines captured"); continue
        print(f"{name:5} gens={n}  tok/round mean={statistics.mean(trnd):.3f} "
              f"median={statistics.median(trnd):.3f} p95={sorted(trnd)[int(0.95*len(trnd))-1]:.3f}")
        print(f"        per-gen tok/s mean={statistics.mean(tps):.1f}  "
              f"head {statistics.mean(hr):.2f}ms/rnd  verify {statistics.mean(vr):.2f}ms/rnd")

if __name__ == "__main__":
    main()
