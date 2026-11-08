#!/usr/bin/env python3
"""Live A/B of EAGLE heads in a warm `serve` (2 sessions, KV8, k=6).

Held-out prompt families only (none of these prompts were trained on):
  chat     - first turns of gen_chat conversations >= --eval-from
  novel    - five hand-written prompts
  serving  - ds/serving/eval.json (6 held-out prompts per serving style)
  long     - two 8-13K-token document questions
Every request is fresh (G4_FORCE_REPREFILL=1), greedy with eos, max 256.
Prints tok/s per family (incl. prefill) and checks that every head produces
the same tokens as plain greedy (the exact-verify contract).
Usage: ab_heads.py ENGINE LABEL HEAD|plain [ENV=VAL ...]
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/celes/src/paged-shared")
from frontend import chat_template, tokenizer  # noqa: E402

CK = "/opt/granite4-rtx-4070-ti-super/ckpt"
E = "/opt/granite4-rtx-4070-ti-super/eagle"
SRC = "/home/celes/src/paged-shared"
eng, label, head = sys.argv[1], sys.argv[2], sys.argv[3]
extra = dict(kv.split("=", 1) for kv in sys.argv[4:])
tok = tokenizer.ByteBPE(CK + "/tokenizer.json")

convs = json.load(open(E + "/ds/chat_prompts.json"))
# The last 30 conversations: held out by every head (r10 trains on < 2159).
chat = [c["turns"][0] for c in convs[-30:]]
serving = [d["prompt"] for d in json.load(open(E + "/ds/serving/eval.json"))][:24]
novel = ["Write a short story about a lighthouse keeper who discovers a message in a bottle.",
         "Explain how a refrigerator works to a ten year old.",
         "What are the trade-offs between microservices and a monolith? Be specific.",
         "Draft a polite email declining a meeting invitation and proposing two alternative times.",
         "Describe the water cycle and its effect on weather in about 200 words."]
long = ["Here is a CUDA source file:\n\n" + open(SRC + "/engine/src/forward_gpu.cu").read()[:40000] +
        "\n\nExplain in detail how this file implements decode, prefill and the KV cache.",
        "Here is an engineering document:\n\n" + open(SRC + "/docs/memory.md").read()[:30000] +
        "\n\nSummarize the memory plan and its measured numbers."]
fams = {"chat": chat, "novel": novel, "serving": serving, "long": long}

env = dict(os.environ, G4_KV8="1", G4_FORCE_REPREFILL="1", **extra)
cmd = [eng, "serve", "--checkpoint", CK, "--gpu", "1", "--max-context", "32768", "--sessions", "2"]
k = 0 if head == "plain" else 6
if k:
    cmd += ["--eagle", head, "--eagle-k", str(k)]
p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=open(f"/tmp/ab_{label}.err", "w"), text=True, bufsize=1, env=env)
p.stdout.readline()
res, outs = [], {}
for fam, ps in fams.items():
    n, w, acc, prop, o = 0, 0.0, 0, 0, []
    for q in ps:
        ids = tok.encode(chat_template.render([{"role": "user", "content": q}]))
        req = {"tokens": ids, "max_tokens": 256, "eos": 100257}
        if k:
            req["eagle"] = k
        t = time.time()
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        d = json.loads(p.stdout.readline())
        w += time.time() - t
        n += len(d["generated"])
        o.append(d["generated"])
        if d.get("draft"):
            acc += d["draft"]["accepted"]
            prop += d["draft"]["proposed"]
    outs[fam] = o
    res.append(f"{fam}: {n / w:6.1f} tok/s" + (f" (accept {acc / prop:.2f})" if prop else ""))
p.stdin.close()
p.wait()
json.dump(outs, open(f"/tmp/ab_{label}.json", "w"))
same = ""
if os.path.exists("/tmp/ab_plain.json") and label != "plain":
    ref = json.load(open("/tmp/ab_plain.json"))
    same = f"  outputs==plain: {all(ref[f] == outs[f] for f in ref)}"
print(f"{label:14s} " + "  ".join(res) + same, flush=True)
