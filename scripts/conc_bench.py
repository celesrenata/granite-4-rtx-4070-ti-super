#!/usr/bin/env python3
"""Concurrency benchmark for `granite4-infer serve` (2 sessions, shared pool).

Runs one prompt set against one engine process twice: with one request in
flight (sequential) and with `--inflight N` requests in flight (concurrent:
the engine batches the plain decodes of all running sessions). Every request
is fresh (G4_FORCE_REPREFILL=1), greedy, eos stop. Reports aggregate tok/s
(generated tokens / wall, prefill included) and checks the concurrent outputs
equal the sequential ones token for token.

  conc_bench.py --engine BIN --ckpt CKPT [--eagle HEAD] [--inflight 2]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from frontend import chat_template, tokenizer  # noqa: E402

NOVEL = ["Write a short story about a lighthouse keeper who discovers a message in a bottle.",
         "Explain how a refrigerator works to a ten year old.",
         "What are the trade-offs between microservices and a monolith? Be specific.",
         "Draft a polite email declining a meeting invitation and proposing two alternative times.",
         "Describe the water cycle and its effect on weather in about 200 words.",
         "List ten practical tips for writing maintainable Python code, with one sentence each.",
         "Summarize the causes of the French Revolution in three paragraphs.",
         "Explain the difference between TCP and UDP with examples of when to use each."]


def run(proc, reqs, inflight):
    """Send reqs keeping `inflight` in flight; returns ({id: generated}, wall_s)."""
    t0 = time.time()
    out, sent, i = {}, 0, 0
    acc = [0, 0]
    while len(out) < len(reqs):
        while sent - len(out) < inflight and i < len(reqs):
            proc.stdin.write(json.dumps(reqs[i]) + "\n")
            proc.stdin.flush()
            i += 1
            sent += 1
        d = json.loads(proc.stdout.readline())
        if "error" in d:
            raise SystemExit(f"engine error: {d}")
        out[d["id"]] = d["generated"]
        if d.get("draft"):
            acc[0] += d["draft"]["accepted"]
            acc[1] += d["draft"]["proposed"]
    run.accept = acc[0] / acc[1] if acc[1] else None
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eagle", default="")
    ap.add_argument("--eagle-k", type=int, default=6)
    ap.add_argument("--inflight", type=int, default=2)
    ap.add_argument("--prompts", default="/opt/granite4-rtx-4070-ti-super/eagle/ds/chat_prompts.json")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--kv", default="1")
    ap.add_argument("--env", nargs="*", default=[], help="extra engine env KEY=VAL")
    ap.add_argument("--families", default="chat,novel,serving")
    a = ap.parse_args()
    tok = tokenizer.ByteBPE(Path(a.ckpt) / "tokenizer.json")
    chat = [c["turns"][0] for c in json.load(open(a.prompts))[1700:1712]]
    serving = [d["prompt"] for d in json.load(open(
        "/opt/granite4-rtx-4070-ti-super/eagle/ds/serving/eval.json"))][:12]
    allf = {"chat": chat, "novel": NOVEL, "serving": serving}
    fams = {f: allf[f] for f in a.families.split(",")}
    env = dict(os.environ, G4_KV8=a.kv, G4_FORCE_REPREFILL="1")
    for kv in a.env:
        k, _, v = kv.partition("=")
        env[k] = v
    cmd = [a.engine, "serve", "--checkpoint", a.ckpt, "--gpu", "1", "--max-context", "32768",
           "--sessions", "2"]
    if a.eagle:
        cmd += ["--eagle", a.eagle, "--eagle-k", str(a.eagle_k)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
    proc.stdout.readline()
    rid = 0
    for fam, ps in fams.items():
        reqs = []
        for q in ps:
            rid += 1
            r = {"id": rid, "tokens": tok.encode(chat_template.render([{"role": "user", "content": q}])),
                 "max_tokens": a.max_tokens, "eos": 100257}
            if a.eagle:
                r["eagle"] = a.eagle_k
            reqs.append(r)
        seq, ws = run(proc, reqs, 1)
        acc_seq = run.accept
        for r in reqs:
            rid += 1
            r["id"] += 10000  # fresh ids for the second pass
        con, wc = run(proc, reqs, a.inflight)
        acc_con = run.accept
        n = sum(len(v) for v in seq.values())
        same = all(seq[r["id"] - 10000] == con[r["id"]] for r in reqs)
        print(f"{fam:6s} {len(reqs)} prompts {n} tok: sequential {n / ws:6.1f} tok/s, "
              f"{a.inflight} in flight {n / wc:6.1f} tok/s ({ws / wc:.2f}x), "
              f"outputs identical: {same}"
              + (f", accept seq {acc_seq:.3f} / conc {acc_con:.3f}" if acc_seq is not None and acc_con is not None else ""),
              flush=True)
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    main()
