#!/usr/bin/env python3
"""End-to-end serve benchmark: plain vs EAGLE (r3 / r5) over the same serving-style
prompts, in one persistent `serve` process per config.

Produces the spec's required numbers: aggregate speedup, per-prompt latency
median + p95, and byte-for-byte output equality (EAGLE must exactly match plain
greedy — the no-degradation contract). The prompt set is serve_styles.json
(400 serving-style single-turn prompts); the chat-template render is reused
verbatim from prep_selfgen.py (same as gen_serving.py).

Runs on the GPU host, GPU free. Sequential configs (one serve at a time).
"""
from __future__ import annotations
import argparse, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

EOS = 100257

# ---------------- same prompt render as gen_serving.py ----------------
def load_template(path: Path) -> str:
    src = path.read_text()
    m = re.search(r'prompt = f"(.*)"\n', src, re.S)
    if m is None:
        raise SystemExit(f"could not find template in {path}")
    return m.group(1)

def render_prompt(tmpl_src: str, prompt_text: str) -> str:
    return eval('f"' + tmpl_src + '"', {"__builtins__": {}},
                {"d": {"prompt": prompt_text}})

# ---------------- serve driver ----------------
def launch(engine: str, ckpt: str, gpu: int, max_ctx: int, head: str | None) -> subprocess.Popen:
    cmd = [engine, "serve", "--checkpoint", ckpt, "--gpu", str(gpu), "--max-context", str(max_ctx)]
    if head:
        cmd += ["--eagle", head]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, bufsize=1)
    ready = p.stdout.readline()
    try:
        meta = json.loads(ready)
        if not meta.get("ready"):
            raise SystemExit(f"serve not ready: {ready.strip()}")
    except Exception as e:
        p.kill()
        raise SystemExit(f"serve failed to start: {e}\n{ready}")
    return p

def gen_one(p: subprocess.Popen, tokens, max_tokens, eagle_k):
    req = {"tokens": tokens, "max_tokens": max_tokens, "eos": EOS, "eagle": eagle_k}
    t0 = time.perf_counter()
    p.stdin.write(json.dumps(req) + "\n"); p.stdin.flush()
    line = p.stdout.readline()
    wall = time.perf_counter() - t0
    if not line:
        return None
    doc = json.loads(line)
    if "error" in doc:
        return None
    return doc["generated"], wall

def bench_config(p, toks_list, max_tokens, eagle_k, warmup=3):
    per_prompt = []
    total_tok = 0.0
    total_wall = 0.0
    for i, tok in enumerate(toks_list):
        r = gen_one(p, tok, max_tokens, eagle_k)
        if r is None:
            print(f"  config eagle={eagle_k} prompt {i}: ERROR", flush=True)
            continue
        generated, wall = r
        if i < warmup:
            continue
        per_prompt.append((len(generated), wall))
        total_tok += len(generated)
        total_wall += wall
    return per_prompt, total_tok, total_wall

def stats(per_prompt):
    if not per_prompt:
        return {}
    lat = [w for _, w in per_prompt]
    agg_tps = sum(t for t, _ in per_prompt) / sum(lat)
    return {
        "n": len(lat),
        "median_ms": statistics.median(lat) * 1000,
        "p95_ms": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)] * 1000,
        "mean_ms": statistics.mean(lat) * 1000,
        "agg_tps": agg_tps,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/granite4-rtx-4070-ti-super/dev/build/engine/granite4-infer")
    ap.add_argument("--checkpoint", default="/opt/granite4-rtx-4070-ti-super/ckpt")
    ap.add_argument("--prompts", default=str(HERE / "serve_styles.json"))
    ap.add_argument("--r3-head", default="/opt/granite4-rtx-4070-ti-super/eagle/heads/r3.safetensors")
    ap.add_argument("--r5-head", default="/opt/granite4-rtx-4070-ti-super/eagle/heads/r5.safetensors")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--k", type=int, default=4, help="eagle_k for the EAGLE configs")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (0=all)")
    ap.add_argument("--out", default=str(HERE / "serve_bench_result.json"))
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.checkpoint + "/tokenizer.json")
    tmpl = load_template(HERE / "prep_selfgen.py")
    data = json.load(open(a.prompts))
    if a.limit:
        data = data[:a.limit]
    print(f"prompts: {len(data)}  max_tokens={a.max_tokens}  K={a.k}", flush=True)
    toks = [tok.encode(render_prompt(tmpl, d["prompt"]), add_special_tokens=False).ids for d in data]

    # config name -> (head file or None, eagle_k). One run each: capture timing
    # AND outputs in the same pass (saves a redundant full serve run).
    configs = [("plain", None, 0), ("r3", a.r3_head, a.k), ("r5", a.r5_head, a.k)]
    results, outputs = {}, {}
    for name, head, k in configs:
        p = launch(a.engine, a.checkpoint, a.gpu, a.max_context, head)
        per_prompt, gen = [], []
        warmup = 3
        for i, t in enumerate(toks):
            r = gen_one(p, t, a.max_tokens, k)
            if r is None:
                print(f"  {name} prompt {i}: ERROR", flush=True)
                gen.append(None); continue
            generated, wall = r
            gen.append(generated)
            if i < warmup:
                continue
            per_prompt.append((len(generated), wall))
            if i and i % 50 == 0:
                print(f"  {name} {i}/{len(toks)}", flush=True)
        p.stdin.close(); p.wait(timeout=10)
        results[name] = stats(per_prompt)
        outputs[name] = gen
        print(f"{name}: {results[name]}", flush=True)

    # correctness: every EAGLE output must equal plain greedy (no-degradation).
    print("== byte-for-byte correctness (EAGLE vs plain) ==")
    base = outputs["plain"]
    for name in ("r3", "r5"):
        g = outputs[name]
        match = sum(1 for x, y in zip(g, base) if x == y)
        print(f"  {name} vs plain: {match}/{len(base)} identical token lists", flush=True)

    plain_tps = results["plain"]["agg_tps"]
    summary = {
        "plain": results["plain"],
        "r3": {**results["r3"], "speedup_vs_plain": round(results["r3"]["agg_tps"] / plain_tps, 3)},
        "r5": {**results["r5"], "speedup_vs_plain": round(results["r5"]["agg_tps"] / plain_tps, 3)},
        "K": a.k, "max_tokens": a.max_tokens, "n_prompts": len(data),
    }
    json.dump(summary, open(a.out, "w"), indent=1)
    print("\n==== SUMMARY (aggregate tok/s + speedup vs plain) ====")
    for n in ("plain", "r3", "r5"):
        r = results[n]
        sp = f"  speedup={r['agg_tps']/plain_tps:.3f}x" if n != "plain" else "  (baseline)"
        print(f"  {n:6} agg_tps={r['agg_tps']:6.1f}  median={r['median_ms']:7.1f}ms  p95={r['p95_ms']:7.1f}ms{sp}")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
