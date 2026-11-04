#!/usr/bin/env python3
"""K sweep for the serving benchmark: end-to-end tok/s, median/p95 per K.

Measures a plain baseline (K=0) and an EAGLE head at several eagle_k values on
the same 400 serving-style prompts in one persistent serve per (head, K). Picks
the production K that maximizes aggregate tok/s (verify is ~constant per round,
so higher acceptance at larger K should raise tok/s until drafts stop being
accepted). Correctness is already established per head (bench_serving).
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
import bench_serving as bs

def run_config(engine, ckpt, gpu, max_ctx, head, toks, max_tokens, k):
    p = bs.launch(engine, ckpt, gpu, max_ctx, head)
    per_prompt = []
    for i, t in enumerate(toks):
        r = bs.gen_one(p, t, max_tokens, k)
        if r is None:
            continue
        generated, wall = r
        if i < 3:  # warmup
            continue
        per_prompt.append((len(generated), wall))
        if i and i % 100 == 0:
            print(f"  {k} {i}/{len(toks)}", flush=True)
    p.stdin.close()
    try: p.wait(timeout=10)
    except Exception: p.kill()
    return bs.stats(per_prompt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/granite4-rtx-4070-ti-super/dev/build/engine/granite4-infer")
    ap.add_argument("--checkpoint", default="/opt/granite4-rtx-4070-ti-super/ckpt")
    ap.add_argument("--prompts", default=str(HERE / "serve_styles.json"))
    ap.add_argument("--head", default="/opt/granite4-rtx-4070-ti-super/eagle/heads/r5.safetensors")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--ks", default="4,6,8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "k_sweep_result.json"))
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.checkpoint + "/tokenizer.json")
    tmpl = bs.load_template(HERE / "prep_selfgen.py")
    data = json.load(open(a.prompts))
    if a.limit:
        data = data[:a.limit]
    toks = [tok.encode(bs.render_prompt(tmpl, d["prompt"]), add_special_tokens=False).ids for d in data]

    ks = [int(x) for x in a.ks.split(",")]
    results = {}
    for k in [0] + ks:
        head = None if k == 0 else a.head
        print(f"== K={k} ==" , flush=True)
        s = run_config(a.engine, a.checkpoint, a.gpu, a.max_context, head, toks, a.max_tokens, k)
        results[f"K={k}"] = s
        print(f"  K={k}: {s}", flush=True)

    base = results["K=0"]["agg_tps"]
    print("\n==== K SWEEP (agg tok/s, speedup vs plain) ====")
    for k in [0] + ks:
        r = results[f"K={k}"]
        sp = f"{r['agg_tps']/base:.3f}x" if k else "(baseline)"
        print(f"  K={k}: tok/s={r['agg_tps']:6.1f}  median={r['median_ms']:7.1f}ms  p95={r['p95_ms']:7.1f}ms  {sp}")
    json.dump(results, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
