#!/usr/bin/env python3
"""Oversubscribed KV pool check for `granite4-infer serve` (--kv-pool-tokens <
sessions x max-context).

Two long requests whose KV does not fit the pool together run first one at a
time (the reference: each gets the whole pool), then both sent at once. With
both in flight the second must wait for the first (the engine logs
"G4_SERVE defer") instead of taking a running session's pages, and every
output must equal its sequential reference token for token. Before the fix
the second request evicted the running one, so its KV was overwritten
mid-generation and the outputs diverged.

    serve_oversub_check.py --engine BIN --ckpt CKPT [--kv 4] [--eagle HEAD]

Runs on the GPU host with the GPU free. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from frontend.tokenizer import ByteBPE  # noqa: E402


def corpus(tok: ByteBPE, paths: list[str], want: int) -> list[int]:
    out: list[int] = []
    for p in paths:
        out += tok.encode((REPO / p).read_text(encoding="utf-8", errors="replace"),
                          allowed_special=False)
        if len(out) >= want:
            return out[:want]
    raise SystemExit(f"corpus too short: {len(out)} < {want} tokens")


def start(args: argparse.Namespace) -> subprocess.Popen:
    cmd = [args.engine, "serve", "--checkpoint", args.ckpt, "--gpu", "1",
           "--max-context", str(args.max_context), "--sessions", "2",
           "--kv-pool-tokens", str(args.pool)]
    if args.eagle:
        cmd += ["--eagle", args.eagle, "--eagle-k", str(args.eagle_k)]
    env = dict(os.environ, G4_SERVE_DEBUG="1", G4_KV8=args.kv)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=open(args.stderr, "w"), text=True, bufsize=1, env=env)
    print("ready:", proc.stdout.readline().strip(), flush=True)
    return proc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kv", default="4", help="G4_KV8 value (0 bf16, 1 int8, 4 int4)")
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--pool", type=int, default=16384, help="KV pool positions (< 2 x max-context)")
    ap.add_argument("--prompt", type=int, default=10000, help="tokens per request")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--eagle", default="")
    ap.add_argument("--eagle-k", type=int, default=4)
    ap.add_argument("--stderr", default="oversub.stderr")
    args = ap.parse_args()
    if 2 * (args.prompt + args.max_tokens) <= args.pool:
        raise SystemExit("both requests fit the pool together: nothing is oversubscribed")

    tok = ByteBPE(Path(args.ckpt) / "tokenizer.json")
    prompts = [corpus(tok, ["engine/src/forward_gpu.cu", "engine/src/kernels_phase2.cu"], args.prompt),
               corpus(tok, ["engine/src/infer_main.cc", "engine/src/prefill_gemm.cu"], args.prompt)]
    reqs = []
    for i, p in enumerate(prompts):
        r = {"id": i, "tokens": p, "max_tokens": args.max_tokens}
        if args.eagle:
            r["eagle"] = args.eagle_k
        reqs.append(r)

    proc = start(args)
    send = lambda r: (proc.stdin.write(json.dumps(r) + "\n"), proc.stdin.flush())  # noqa: E731

    def read() -> dict:
        line = proc.stdout.readline()
        if not line:
            raise SystemExit("engine died (see " + args.stderr + ")")
        return json.loads(line)

    ref = {}
    for r in reqs:  # sequential: each request alone in the pool
        send(r)
        d = read()
        if "error" in d:
            raise SystemExit(f"sequential request {r['id']}: {d['error']}")
        ref[d["id"]] = d["generated"]
    print("sequential:", {k: len(v) for k, v in ref.items()}, flush=True)

    # Concurrent: the same two prompts at once (ids offset to tell the replies
    # apart). Each still has to evict the other's idle cache or wait for it.
    for r in reqs:
        r["id"] += 10
    for r in reqs:
        send(r)
    got = {}
    for _ in reqs:
        d = read()
        if "error" in d:
            print(f"FAIL: concurrent request {d.get('id')}: {d['error']}")
            return 1
        got[d["id"] - 10] = d["generated"]
    proc.stdin.close()
    proc.wait(timeout=120)

    log = Path(args.stderr).read_text()
    deferred = log.count("G4_SERVE defer")
    bad = [k for k in ref if ref[k] != got.get(k)]
    print(f"deferred admissions: {deferred}")
    for k in ref:
        print(f"request {k}: {'IDENTICAL' if k not in bad else 'DIFFERENT'} ({len(got.get(k, []))} tokens)")
    if bad or deferred == 0:
        print("RESULT: FAIL" + ("" if deferred else " (the second request was never deferred)"))
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
