#!/usr/bin/env python3
"""Repeat one prompt N times through `granite4-infer serve` (fresh prefill
each time) and check every output against a plain-greedy run: catches
nondeterministic paths that a single benchmark pass can miss.

  repeat_check.py --engine BIN --ckpt DIR [--eagle HEAD] [--doc 0] [-n 12]
                  [--env K=V ...]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from frontend import chat_template, tokenizer  # noqa: E402


def outputs(engine, ckpt, ids, n, eagle, env_extra, max_tokens):
    env = dict(os.environ, G4_KV8="1", G4_FORCE_REPREFILL="1")
    env.update(env_extra)
    cmd = [engine, "serve", "--checkpoint", ckpt, "--gpu", "1", "--max-context", "32768",
           "--sessions", "2"]
    if eagle:
        cmd += ["--eagle", eagle, "--eagle-k", "4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
    p.stdout.readline()
    outs = []
    for i in range(n):
        req = {"id": i + 1, "tokens": ids, "max_tokens": max_tokens, "eos": 100257}
        if eagle:
            req["eagle"] = 4
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        d = json.loads(p.stdout.readline())
        if "error" in d:
            raise SystemExit(f"engine error: {d}")
        outs.append(d["generated"])
    p.stdin.close()
    p.wait()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eagle", default="")
    ap.add_argument("--doc", type=int, default=0, help="engine source file index (conc_bench doc family)")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--env", nargs="*", default=[])
    a = ap.parse_args()
    tok = tokenizer.ByteBPE(Path(a.ckpt) / "tokenizer.json")
    from conc_bench import doc_sources
    src = doc_sources()[a.doc]
    q = (f"Here is a source file:\n\n{src.read_text()[:10000]}\n\nExplain what this code does, "
         "section by section.")
    ids = tok.encode(chat_template.render([{"role": "user", "content": q}]))
    env = dict(kv.split("=", 1) for kv in a.env)
    ref = outputs(a.engine, a.ckpt, ids, 1, "", {}, a.max_tokens)[0]
    got = outputs(a.engine, a.ckpt, ids, a.n, a.eagle, env, a.max_tokens)
    bad = [(i, next((k for k in range(min(len(g), len(ref))) if g[k] != ref[k]), None))
           for i, g in enumerate(got) if g != ref]
    print(f"prompt {len(ids)} tokens, {a.n} runs {a.env}: "
          + ("all equal to plain" if not bad else f"DIFFER (run, first token): {bad}"), flush=True)


if __name__ == "__main__":
    main()
