#!/usr/bin/env python3
"""Offline drafter acceptance study for exact greedy speculative decoding.

Runs on the GPU host (gremlin-1) with CPU torch; the GPU stays with the
service. Steps:

1. `collect`: send chat prompts to the live API and store the 8B's greedy
   replies (text).
2. `score`: for the target and each drafter, one teacher-forced CPU forward
   over prompt + reply, keeping the per-position argmax. With exact greedy
   verification the drafter's proposal at a position depends only on the
   accepted prefix, which is the target's own sequence, so the teacher-forced
   argmax is exactly what it would draft. Acceptance per round is the run of
   consecutive positions where drafter argmax == target argmax.

Drafters with another chat format (Granite 4.0: <|start_of_role|>) get the
prompt in their own template; reply tokens are shared (same BPE vocab and
merges, and <|im_end|> / <|end_of_text|> are both id 100257).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Write a detailed essay about the history of computing.",
    "Explain how vaccines train the immune system, in a few paragraphs.",
    "What are the pros and cons of remote work? Use bullet points.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the difference between TCP and UDP to a junior developer.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "How do I set up a Python virtual environment and install packages from requirements.txt?",
    "Summarize the causes of World War I.",
    "Write a SQL query that returns the top 5 customers by total order value, and explain it.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Give me a 7-day beginner running plan.",
    "Explain what a hash map is and how collisions are handled.",
    "Translate into French: 'The meeting has been moved to Thursday afternoon because the manager is travelling.'",
    "Write a bash script that finds the 10 largest files under a directory.",
    "Compare Rust and Go for writing network services.",
    "What is photosynthesis? Explain for a 10 year old.",
    "Draft a polite email asking my landlord to fix a leaking tap.",
    "Explain gradient descent with a simple numerical example.",
    "List 10 healthy breakfast ideas with one sentence each.",
    "Write a JavaScript function to debounce another function, and explain how it works.",
    "What are the main differences between a Roth IRA and a traditional IRA?",
    "Explain how a CPU cache works and why it matters for performance.",
    "Write a haiku sequence (three haikus) about autumn.",
    "Refactor this into idiomatic Python and explain the changes:\n\nresult = []\nfor i in range(len(xs)):\n    if xs[i] % 2 == 0:\n        result.append(xs[i] * xs[i])\nprint(result)",
]


def collect(args) -> None:
    out = []
    for i, p in enumerate(PROMPTS):
        body = json.dumps({"model": "granite-4.2-8b", "messages": [{"role": "user", "content": p}],
                           "max_tokens": args.max_tokens}).encode()
        req = urllib.request.Request(args.api + "/v1/chat/completions", data=body,
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read())
        out.append({"prompt": p, "reply": d["choices"][0]["message"]["content"],
                    "finish": d["choices"][0]["finish_reason"]})
        print(f"[{i + 1}/{len(PROMPTS)}] {d['usage']['completion_tokens']} tokens", flush=True)
    Path(args.data).write_text(json.dumps(out, indent=1))


def render_8b(prompt: str) -> str:
    # Must match frontend/chat_template.py (thinking off).
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>"


def score(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    data = json.loads(Path(args.data).read_text())
    tgt_tok = AutoTokenizer.from_pretrained(args.target)
    replies = [tgt_tok.encode(d["reply"], add_special_tokens=False) + [100257]
               if d["finish"] == "stop" else tgt_tok.encode(d["reply"], add_special_tokens=False)
               for d in data]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    def argmaxes(path: str, own_template: bool) -> list[list[int]]:
        key = cache / (Path(path).name + ".json")
        if key.is_file():
            return json.loads(key.read_text())
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
        model.eval()
        res = []
        t0 = time.time()
        for d, rep in zip(data, replies):
            if own_template:
                text = tok.apply_chat_template([{"role": "user", "content": d["prompt"]}],
                                               add_generation_prompt=True, tokenize=False)
                pids = tok.encode(text, add_special_tokens=False)
            else:
                pids = tgt_tok.encode(render_8b(d["prompt"]), add_special_tokens=False)
            ids = torch.tensor([list(pids) + rep[:-1]])
            with torch.no_grad():
                logits = model(ids).logits[0, len(pids) - 1:]
            res.append(logits.argmax(-1).tolist())
        print(f"{Path(path).name}: {time.time() - t0:.0f}s", flush=True)
        key.write_text(json.dumps(res))
        del model
        return res

    tgt = argmaxes(args.target, own_template=False)
    consistent = sum(sum(a == b for a, b in zip(t, r)) for t, r in zip(tgt, replies))
    total = sum(len(r) for r in replies)
    print(f"target argmax == API reply token: {consistent}/{total} "
          f"({consistent / total:.1%}; <100% = CPU bf16 vs engine near-ties / re-tokenization)")

    report = {}
    for spec in args.drafters:
        path, _, mode = spec.partition(":")
        drf = argmaxes(path, own_template=(mode == "own"))
        match = sum(sum(a == b for a, b in zip(x, t)) for x, t in zip(drf, tgt))
        row = {"top1_agreement": match / total}
        for k in args.ks:
            produced = rounds = 0
            for x, t in zip(drf, tgt):
                p = 0
                while p < len(t):
                    a = 0
                    while a < k and p + a < len(t) and x[p + a] == t[p + a]:
                        a += 1
                    # a accepted drafts plus the target's correction/bonus token.
                    produced += min(a + 1, len(t) - p)
                    p += a + 1
                    rounds += 1
            row[f"K{k}_tokens_per_round"] = produced / rounds
        report[Path(path).name] = row
        print(Path(path).name, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--api", default="http://127.0.0.1:2701")
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("--data", default="acceptance_data.json")
    s = sub.add_parser("score")
    s.add_argument("--data", default="acceptance_data.json")
    s.add_argument("--target", required=True)
    s.add_argument("--drafters", nargs="+", required=True,
                   help="PATH[:own]  (':own' = render the prompt with the drafter's chat template)")
    s.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--cache", default="acceptance_cache")
    s.add_argument("--out", default="acceptance_report.json")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
