#!/usr/bin/env python3
"""Probe: prefill a long prompt, then re-send it for N decode tokens, and show
the engine's session/resume decision (G4_SERVE_DEBUG) and the wall times.

    serve_resume_probe.py --engine BIN --ckpt CKPT [--eagle HEAD] [--tokens 30000]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.argv, _argv = sys.argv[:1], sys.argv
import serve_balance_bench as sbb  # noqa: E402  (corpus + template helpers)
sys.argv = _argv
from frontend import chat_template  # noqa: E402
from frontend.tokenizer import ByteBPE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eagle", default="")
    ap.add_argument("--kv", default="4")
    ap.add_argument("--tokens", type=int, default=30000)
    ap.add_argument("--decode", type=int, default=256)
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--pool", type=int, default=0)
    a = ap.parse_args()
    tok = ByteBPE(Path(a.ckpt) / "tokenizer.json")
    corpus = "\n\n".join((REPO / p).read_text(encoding="utf-8", errors="replace") for p in sbb.LONG_SOURCES)
    ids = tok.encode(corpus, allowed_special=False)[:a.tokens - 200]
    prompt = tok.encode(chat_template.render([{"role": "user", "content": tok.decode(ids) +
                        "\n\nWrite a detailed summary of the material above, section by section."}]))
    cmd = [a.engine, "serve", "--checkpoint", a.ckpt, "--gpu", "1", "--max-context", "65536",
           "--sessions", "2"]
    if a.pool:
        cmd += ["--kv-pool-tokens", str(a.pool)]
    if a.eagle:
        cmd += ["--eagle", a.eagle, "--eagle-k", "4"]
    env = dict(os.environ, G4_KV8=a.kv, G4_DRAFT_KV="4", G4_SERVE_DEBUG="1")
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1, env=env)
    p.stdout.readline()
    for n, mt in enumerate((1, a.decode, a.decode)):
        r = {"id": n, "tokens": prompt, "max_tokens": mt}
        if a.eos:
            r["eos"] = sbb.EOS
        if a.eagle:
            r["eagle"] = 4
        t = time.time()
        p.stdin.write(json.dumps(r) + "\n")
        p.stdin.flush()
        d = json.loads(p.stdout.readline())
        wall = time.time() - t
        gen = d.get("generated", [])
        print(f"req {n}: max_tokens={mt} generated={len(gen)} wall={wall:.2f}s "
              f"draft={json.dumps(d.get('draft'))[:160] if d.get('draft') else None} err={d.get('error')}",
              flush=True)
        if n == 1:
            print("   text:", repr(tok.decode(gen)[:300]), flush=True)
    p.stdin.close()
    err = p.stderr.read()
    p.wait()
    print("\n".join(l for l in err.splitlines() if l.startswith("G4_SERVE")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
