#!/usr/bin/env python3
"""Speed / concurrency / KV pool benchmark for one `granite4-infer serve` config.

One engine process per run (the GPU must be free), driven over its stdin
protocol with chat-template prompts, greedy + EAGLE like the live API:

  * VRAM after load and peak (the EAGLE feature buffer is allocated lazily);
  * short chats (12 prompts, max 256, eos stop): 1 in flight and `sessions`
    in flight, aggregate tok/s including prefill, EAGLE accept rate;
  * long context at 30K and 60K tokens (when they fit max-context): fresh
    prefill time, then 256 decode tokens resumed on the cached prefix;
  * two 30K conversations at once (two agents): wall and aggregate tok/s.

    serve_balance_bench.py --engine BIN --ckpt CKPT --kv 4 --sessions 2 \
        --max-context 65536 --pool 131072 --eagle HEAD --eagle-k 4 --draft-kv 4 \
        --label kv4-2x64k --out kv4-2x64k.json
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
from frontend import chat_template  # noqa: E402
from frontend.tokenizer import ByteBPE  # noqa: E402

EOS = 100257
CHAT_PROMPTS = "/opt/granite4-rtx-4070-ti-super/eagle/ds/chat_prompts.json"
LONG_SOURCES = ["docs/phases.md", "docs/memory.md", "docs/architecture.md", "docs/deployment.md",
                "docs/eagle-2x-performance-spec.md", "README.md", "engine/src/infer_main.cc",
                "engine/src/forward_gpu.cu", "engine/src/eagle_gpu.cu", "engine/src/prefill_gemm.cu",
                "engine/src/attention_split.cu", "engine/src/kernels_phase2.cu", "frontend/api.py",
                "frontend/engine_service.py", "engine/src/kv_pool.cc", "flake.nix"]


def vram_mib(pid: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0] == str(pid):
            return int(p[1])
    return 0


class Engine:
    def __init__(self, a: argparse.Namespace):
        cmd = [a.engine, "serve", "--checkpoint", a.ckpt, "--gpu", "1",
               "--max-context", str(a.max_context), "--sessions", str(a.sessions)]
        if a.pool:
            cmd += ["--kv-pool-tokens", str(a.pool)]
        if a.eagle:
            cmd += ["--eagle", a.eagle, "--eagle-k", str(a.eagle_k)]
        env = dict(os.environ, G4_KV8=a.kv, G4_DRAFT_KV=str(a.draft_kv))
        self.eagle_k = a.eagle_k if a.eagle else 0
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(a.out + ".stderr", "w"), text=True, bufsize=1, env=env)
        self.ready = self.p.stdout.readline().strip()
        if not self.ready:
            raise SystemExit("engine failed to start (see " + a.out + ".stderr)")
        self.rid = 0
        self.peak = vram_mib(self.p.pid)

    def batch(self, prompts: list[list[int]], max_tokens: int, inflight: int, eos: bool = True):
        """Runs prompts keeping `inflight` in flight: (generated counts, wall s, accepted, proposed)."""
        t0, sent, done = time.time(), 0, 0
        n, acc, prop = [], 0, 0
        while done < len(prompts):
            while sent - done < inflight and sent < len(prompts):
                self.rid += 1
                r = {"id": self.rid, "tokens": prompts[sent], "max_tokens": max_tokens}
                if eos:
                    r["eos"] = EOS
                if self.eagle_k:
                    r["eagle"] = self.eagle_k
                self.p.stdin.write(json.dumps(r) + "\n")
                self.p.stdin.flush()
                sent += 1
            line = self.p.stdout.readline()
            if not line:
                raise SystemExit("engine died")
            d = json.loads(line)
            if "error" in d:
                raise SystemExit(f"engine error: {d['error']}")
            done += 1
            n.append(len(d["generated"]))
            if d.get("draft"):
                acc += d["draft"].get("accepted", 0)
                prop += d["draft"].get("proposed", 0)
            self.peak = max(self.peak, vram_mib(self.p.pid))
        return n, time.time() - t0, acc, prop

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=120)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kv", default="4")
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--pool", type=int, default=0)
    ap.add_argument("--eagle", default="")
    ap.add_argument("--eagle-k", type=int, default=4)
    ap.add_argument("--draft-kv", type=int, default=4)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = ByteBPE(Path(a.ckpt) / "tokenizer.json")
    render = lambda text: tok.encode(chat_template.render([{"role": "user", "content": text}]))  # noqa: E731
    chats = [render(c["turns"][0]) for c in json.load(open(CHAT_PROMPTS))[-12:]]
    corpus = "\n\n".join((REPO / p).read_text(encoding="utf-8", errors="replace") for p in LONG_SOURCES)
    corpus_ids = tok.encode(corpus, allowed_special=False)

    def long_prompt(n_tokens: int, tag: str) -> list[int]:
        head = tok.encode(f"[{a.label} {tag}]\n", allowed_special=False)
        body = (corpus_ids * (n_tokens // len(corpus_ids) + 1))[:n_tokens - 200]
        text = tok.decode(head + body)
        return render(text + "\n\nWrite a detailed summary of the material above, section by section.")

    res: dict = {"label": a.label, "kv": a.kv, "sessions": a.sessions, "max_context": a.max_context,
                 "pool": a.pool or a.sessions * a.max_context, "eagle_k": a.eagle_k if a.eagle else 0,
                 "draft_kv": a.draft_kv}
    e = Engine(a)
    res["ready"], res["vram_load_mib"] = e.ready, e.peak
    print(f"[{a.label}] ready: {e.ready}  vram={e.peak} MiB", flush=True)

    e.batch(chats[:1], 32, 1)  # warm (allocates the EAGLE feature buffer)
    for inflight in sorted({1, a.sessions}):
        n, wall, acc, prop = e.batch(chats, 256, inflight)
        res[f"short_x{inflight}_tok_s"] = round(sum(n) / wall, 1)
        res[f"short_x{inflight}_accept"] = round(acc / prop, 3) if prop else None
        print(f"[{a.label}] short x{inflight}: {sum(n)} tok in {wall:.1f}s -> {sum(n) / wall:.1f} tok/s"
              f" accept={res[f'short_x{inflight}_accept']}", flush=True)

    for L in (30000, 60000):
        if L + 600 > a.max_context:
            continue
        p = long_prompt(L, f"long{L}")
        _, t_pre, _, _ = e.batch([p], 1, 1)
        # A 1-token request ends before the EAGLE head catches up on the
        # prompt, so the next EAGLE request re-syncs it (about one prefill);
        # time the second decode, the steady state of a conversation.
        e.batch([p], 256, 1)
        n, t_dec, acc, prop = e.batch([p], 256, 1)
        res[f"prefill_{L // 1000}k_s"] = round(t_pre, 2)
        res[f"decode_{L // 1000}k_tok_s"] = round(sum(n) / t_dec, 1)
        res[f"decode_{L // 1000}k_accept"] = round(acc / prop, 3) if prop else None
        print(f"[{a.label}] {len(p)} tok: prefill {t_pre:.1f}s, decode {sum(n)} tok -> {sum(n) / t_dec:.1f} tok/s"
              f" accept={res[f'decode_{L // 1000}k_accept']}", flush=True)

    if a.sessions >= 2 and 30600 <= a.max_context:
        ps = [long_prompt(30000, "agentA"), long_prompt(30000, "agentB")]
        n, wall, _, _ = e.batch(ps, 256, 2)
        res["two_30k_wall_s"] = round(wall, 1)
        print(f"[{a.label}] two 30K at once: {wall:.1f}s total", flush=True)

    if a.sessions >= 3:
        # Worst case the slots allow: two long conversations near the pool's
        # share plus a short chat, all in flight at once.
        big = min(int((a.pool or a.sessions * a.max_context) * 0.45), a.max_context - 2000)
        ps = [long_prompt(big, "stressA"), long_prompt(big, "stressB"), chats[0]]
        n, wall, _, _ = e.batch(ps, 256, 3)
        res["stress_two_long_tokens"] = big
        res["stress_wall_s"] = round(wall, 1)
        print(f"[{a.label}] stress 2x{big} + short at once: {wall:.1f}s, {n} tokens", flush=True)

    res["vram_peak_mib"] = e.peak
    e.close()
    print(f"[{a.label}] peak vram {e.peak} MiB", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
