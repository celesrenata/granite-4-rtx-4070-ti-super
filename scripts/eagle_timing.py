#!/usr/bin/env python3
"""Solo EAGLE round timing: runs `granite4-infer generate` with
G4_EAGLE_TIMING=1 on a few held-out chat prompts per k and prints the
engine's per-round head / verify split.

  eagle_timing.py --engine BIN --ckpt DIR --eagle HEAD [--ks 3,6]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from frontend import chat_template, tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eagle", required=True)
    ap.add_argument("--ks", default="3,6")
    ap.add_argument("--prompts", default="/opt/granite4-rtx-4070-ti-super/eagle/ds/chat_prompts.json")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    tok = tokenizer.ByteBPE(Path(a.ckpt) / "tokenizer.json")
    qs = [c["turns"][0] for c in json.load(open(a.prompts))[-a.n:]]
    env = dict(os.environ, G4_EAGLE_TIMING="1", G4_EAGLE_MIN_ACCEPT="0", G4_KV8="1")
    for k in [int(x) for x in a.ks.split(",")]:
        for q in qs:
            ids = tok.encode(chat_template.render([{"role": "user", "content": q}]))
            cmd = [a.engine, "generate", "--checkpoint", a.ckpt, "--gpu", "1",
                   "--tokens", ",".join(map(str, ids)), "--max-tokens", str(a.max_tokens),
                   "--eos", "100257", "--eagle", a.eagle, "--eagle-k", str(k)]
            p = subprocess.run(cmd, env=env, capture_output=True, text=True)
            line = [l for l in p.stderr.splitlines() if l.startswith("G4_EAGLE head_ms")]
            print(f"k={k} {line[-1] if line else p.stderr[-300:]}", flush=True)


if __name__ == "__main__":
    main()
