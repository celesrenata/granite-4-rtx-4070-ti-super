#!/usr/bin/env python3
"""gen_chat.py's records from a local transformers copy of the target (bf16,
greedy, batched) instead of the live API: for a box with a spare GPU.

Prompts are rendered with the engine's own chat template (frontend/
chat_template.render, thinking off) and the replies decoded greedily, so the
text matches what the served engine would write up to FP8-vs-bf16 drift;
training targets still come from the engine's dump-hidden of these tokens.
Writes {"conv", "src", "messages", "reply", "finish"} lines for conversations
[--start, --end) of --prompts and resumes from --out; DONE marks completion.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# nixpkgs' triton returns libcuda_dirs() as a str (each char becomes a -L).
import triton.backends.nvidia.driver as _drv  # noqa: E402
_drv.libcuda_dirs = lambda: ["/run/opengl-driver/lib"]
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from frontend import chat_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, dtype=torch.bfloat16).to("cuda").eval()
    convs = json.load(open(a.prompts))
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["conv"], len(r["messages"])))
            except ValueError:
                pass
    f = out.open("a")
    hist = {i: [] for i in range(a.start, min(a.end, len(convs)))}
    t0, ntok, nrep = time.time(), 0, 0
    if out.exists():  # rebuild histories of finished turns for 2-turn resumes
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["conv"] in hist:
                hist[r["conv"]] = r["messages"] + [{"role": "assistant", "content": r["reply"]}]
    max_turns = max(len(c["turns"]) for c in convs[a.start:a.end])
    for t in range(max_turns):
        jobs = []
        for i in hist:
            turns = convs[i]["turns"]
            if t >= len(turns):
                continue
            msgs = (hist[i] if t > 0 else []) + [{"role": "user", "content": turns[t]}]
            if t > 0 and len(hist[i]) != 2 * t:
                continue  # turn t-1 did not finish
            if (i, len(msgs)) in done:
                continue
            ids = tok(chat_template.render(msgs), add_special_tokens=False)["input_ids"]
            jobs.append((len(ids), i, msgs, ids))
        jobs.sort()  # similar lengths per batch: less padding
        b = a.batch
        k = 0
        while k < len(jobs):
            chunk = jobs[k:k + b]
            L = max(len(j[3]) for j in chunk)
            inp = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, j in enumerate(chunk):
                inp[r, L - len(j[3]):] = torch.tensor(j[3])
                att[r, L - len(j[3]):] = 1
            try:
                with torch.no_grad():
                    gen = model.generate(input_ids=inp.cuda(), attention_mask=att.cuda(),
                                         max_new_tokens=a.max_tokens, do_sample=False,
                                         pad_token_id=pad)[:, L:]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                b = max(1, b // 2)
                print(f"OOM: batch -> {b}", flush=True)
                continue
            for r, j in enumerate(chunk):
                g = gen[r].tolist()
                n = len(g)
                for e, x in enumerate(g):
                    if x == tok.eos_token_id or x == pad:
                        n = e
                        break
                reply = tok.decode(g[:n], skip_special_tokens=True).strip()
                finish = "length" if n >= a.max_tokens else "stop"
                _, i, msgs, _ = j
                f.write(json.dumps({"conv": i, "src": convs[i]["src"], "messages": msgs,
                                    "reply": reply, "finish": finish}) + "\n")
                hist[i] = msgs + [{"role": "assistant", "content": reply}]
                ntok += n
                nrep += 1
            f.flush()
            k += len(chunk)
            el = time.time() - t0
            print(f"turn {t}: {k}/{len(jobs)} replies, {ntok} tokens, {ntok / el:.0f} tok/s, "
                  f"{el / 60:.1f} min", flush=True)
    f.close()
    print("DONE", nrep, "replies", ntok, "tokens", flush=True)


if __name__ == "__main__":
    main()
