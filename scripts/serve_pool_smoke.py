#!/usr/bin/env python3
"""Multi-conversation smoke test for `granite4-infer serve` (shared KV pool).

Plays a fixed script of interleaved long conversations against one engine
process and records every generated token id, per-turn wall time, the
session/resume decision the engine made, and the engine's VRAM. Run it once
against a reference engine (e.g. the production binary: plain greedy, one
session) and once against the engine under test (N sessions + EAGLE + shared
pool), then diff the two transcripts:

    serve_pool_smoke.py run --engine REF --ckpt CKPT --out ref.json
    serve_pool_smoke.py run --engine NEW --ckpt CKPT --sessions 2 \
        --eagle HEAD --eagle-k 6 --out new.json
    serve_pool_smoke.py compare ref.json new.json

EAGLE's greedy gate is exact, so with the same KV format the generated ids
must match the plain-greedy reference token for token, whatever the session
routing, resume, or pool eviction did. Runs on the GPU host with the GPU free.
Stdlib only (the repo's own byte-BPE tokenizer renders the prompts).
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
from frontend.tokenizer import ByteBPE  # noqa: E402

# (conversation, new user tokens appended this turn, max new tokens). The first
# turn of a conversation carries its whole opening prompt. Interleaved so that
# every turn after the first resumes a cached session; the final big
# conversation (C) does not fit beside A and B in a pool sized for two full
# contexts' worth of *typical* use, so it forces an LRU eviction, and the
# closing A/B turns then re-prefill a dropped session.
SCRIPT = [
    ("A", 12000, 96), ("B", 12000, 96),
    ("A", 300, 96), ("B", 300, 96),
    ("A", 300, 96), ("B", 300, 96),
    ("C", 30000, 96),
    ("A", 300, 96), ("B", 300, 96),
]


def corpus(tok: ByteBPE) -> dict[str, list[int]]:
    """Three disjoint token streams of real text (source + docs)."""
    def toks(paths: list[str], want: int) -> list[int]:
        out: list[int] = []
        for p in paths:
            f = REPO / p
            if f.is_file():
                out += tok.encode(f.read_text(encoding="utf-8", errors="replace"),
                                  allowed_special=False)
            if len(out) >= want:
                break
        return out[:want]
    # Large enough for the whole script; text is plain source / prose.
    return {
        "A": toks(["engine/src/forward_gpu.cu", "engine/src/kernels_phase2.cu"], 16000),
        "B": toks(["engine/src/infer_main.cc", "engine/src/prefill_gemm.cu"], 16000),
        "C": toks(["docs/memory.md", "docs/phases.md", "docs/architecture.md",
                   "docs/eagle-2x-performance-spec.md", "README.md",
                   "engine/src/eagle_gpu.cu", "engine/src/kv_pool.cc"], 40000),
    }


def vram_mib(pid: int) -> int:
    """Device memory of `pid` per nvidia-smi (0 if unknown)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return 0
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return int(parts[1])
    return 0


def run(args: argparse.Namespace) -> int:
    tok = ByteBPE(Path(args.ckpt) / "tokenizer.json")
    text = corpus(tok)
    cmd = [args.engine, "serve", "--checkpoint", args.ckpt, "--gpu", "1",
           "--max-context", str(args.max_context), "--sessions", str(args.sessions)]
    if args.kv_pool_tokens:
        cmd += ["--kv-pool-tokens", str(args.kv_pool_tokens)]
    if args.eagle:
        cmd += ["--eagle", args.eagle, "--eagle-k", str(args.eagle_k)]
    env = dict(os.environ, G4_SERVE_DEBUG="1")
    if args.kv:
        env["G4_KV8"] = args.kv
    err_path = Path(args.out).with_suffix(".stderr")
    err_f = err_path.open("w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=err_f, text=True, bufsize=1, env=env)
    assert proc.stdin is not None and proc.stdout is not None
    ready = json.loads(proc.stdout.readline())
    load_s = time.time() - t0
    print(f"engine ready in {load_s:.0f}s: {ready}", flush=True)
    peak = vram_mib(proc.pid)
    print(f"VRAM after load: {peak} MiB", flush=True)

    history: dict[str, list[int]] = {}
    cursor = {k: 0 for k in text}
    turns = []
    for n, (conv, new_user, max_new) in enumerate(SCRIPT):
        src = text[conv]
        chunk = src[cursor[conv]:cursor[conv] + new_user]
        cursor[conv] += new_user
        if len(chunk) < new_user:
            raise SystemExit(f"corpus for {conv} too short for the script")
        prompt = history.get(conv, []) + chunk
        req = {"tokens": prompt, "max_tokens": max_new}
        if args.eagle:
            req["eagle"] = args.eagle_k
        t = time.time()
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        wall = time.time() - t
        if not line:
            raise SystemExit(f"engine died on turn {n} ({conv})")
        doc = json.loads(line)
        if "error" in doc:
            raise SystemExit(f"turn {n} ({conv}): engine error: {doc['error']}")
        gen = doc["generated"]
        history[conv] = prompt + gen
        peak = max(peak, vram_mib(proc.pid))
        rec = {"turn": n, "conv": conv, "prompt_tokens": len(prompt), "generated": gen,
               "wall_s": round(wall, 3), "draft": doc.get("draft")}
        turns.append(rec)
        acc = ""
        if rec["draft"]:
            acc = f" accept={rec['draft'].get('accept_rate', 0):.2f}"
        print(f"turn {n:2d} {conv} prompt={len(prompt):6d} new={len(gen):3d} "
              f"wall={wall:6.2f}s{acc}", flush=True)
    proc.stdin.close()
    proc.wait(timeout=60)
    err_f.close()
    # Session/resume decisions the engine logged, in request order.
    decisions = []
    for line in err_path.read_text().splitlines():
        if line.startswith("G4_SERVE "):
            decisions.append(line[len("G4_SERVE "):])
    for rec, dec in zip(turns, decisions):
        rec["decision"] = dec
    doc = {"cmd": cmd, "kv": args.kv, "ready": ready, "load_s": round(load_s, 1),
           "peak_vram_mib": peak, "turns": turns}
    Path(args.out).write_text(json.dumps(doc))
    print(f"peak VRAM {peak} MiB; transcript -> {args.out}")
    for rec in turns:
        print(f"  turn {rec['turn']:2d} {rec['conv']}: {rec.get('decision', '?')}")
    return 0


def compare(args: argparse.Namespace) -> int:
    a = json.loads(Path(args.ref).read_text())
    b = json.loads(Path(args.new).read_text())
    bad = 0
    for ra, rb in zip(a["turns"], b["turns"]):
        same = ra["generated"] == rb["generated"] and ra["prompt_tokens"] == rb["prompt_tokens"]
        sa, sb = ra["wall_s"], rb["wall_s"]
        print(f"turn {ra['turn']:2d} {ra['conv']} prompt={ra['prompt_tokens']:6d} "
              f"{'IDENTICAL' if same else 'DIFFERENT'}  ref={sa:6.2f}s new={sb:6.2f}s")
        if not same:
            bad += 1
            ga, gb = ra["generated"], rb["generated"]
            first = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
            print(f"    first divergence at generated[{first}] ref={ga[first:first+4]} new={gb[first:first+4]}")
    print(f"peak VRAM: ref={a['peak_vram_mib']} MiB new={b['peak_vram_mib']} MiB")
    print("RESULT:", "ALL IDENTICAL" if bad == 0 else f"{bad} turn(s) differ")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-context", type=int, default=32768)
    r.add_argument("--sessions", type=int, default=1)
    r.add_argument("--kv-pool-tokens", type=int, default=0)
    r.add_argument("--eagle", default="")
    r.add_argument("--eagle-k", type=int, default=6)
    r.add_argument("--kv", default="1", help="G4_KV8 value (0 bf16, 1 int8, 4 int4)")
    r.set_defaults(fn=run)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    c.set_defaults(fn=compare)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
