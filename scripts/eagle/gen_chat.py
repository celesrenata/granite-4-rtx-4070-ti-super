#!/usr/bin/env python3
"""Self-distillation data for the EAGLE head: the target's own greedy replies.

Sends UltraChat / ShareGPT user turns (ds/chat_prompts.json, from
extract_prompts.py) to the live granite4 API and records each reply. Two-turn
conversations send turn 2 with the target's own turn-1 reply as history, as a
real chat would. Appends one JSON line per generated reply to --out
({"conv", "src", "messages", "reply", "finish"}) and resumes from it, so the
run can be stopped and restarted. No GPU of its own: it only uses the API.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def chat(url, messages, max_tokens):
    body = json.dumps({"model": "granite-4.2-8b", "messages": messages,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        doc = json.loads(r.read())
    c = doc["choices"][0]
    return c["message"]["content"], c["finish_reason"], doc["usage"]["completion_tokens"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=1800)
    ap.add_argument("--max-tokens", type=int, default=448)
    ap.add_argument("--url", default="http://127.0.0.1:2701/v1/chat/completions")
    a = ap.parse_args()
    convs = json.load(open(a.prompts))[: a.limit]
    out = Path(a.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["conv"], len(r["messages"])))
    t0, n_tok, n_rep = time.time(), 0, 0
    with out.open("a") as f:
        for i, c in enumerate(convs):
            msgs = []
            for t, user in enumerate(c["turns"]):
                msgs = msgs + [{"role": "user", "content": user}]
                key = (i, len(msgs))
                prev = None
                if key in done:
                    # Rebuild the history from the saved reply for the next turn.
                    for line in out.read_text().splitlines():
                        r = json.loads(line)
                        if (r["conv"], len(r["messages"])) == key:
                            prev = r["reply"]
                            break
                else:
                    try:
                        reply, finish, ntok = chat(a.url, msgs, a.max_tokens)
                    except Exception as e:  # skip a failed request, keep going
                        print(f"conv {i} turn {t}: {e}", flush=True)
                        break
                    f.write(json.dumps({"conv": i, "src": c["src"], "messages": msgs,
                                        "reply": reply, "finish": finish}) + "\n")
                    f.flush()
                    prev = reply
                    n_tok += ntok
                    n_rep += 1
                msgs = msgs + [{"role": "assistant", "content": prev}]
            if i % 25 == 0 and n_rep:
                el = time.time() - t0
                print(f"conv {i}/{len(convs)}: {n_rep} replies, {n_tok} tokens, "
                      f"{n_tok / el:.1f} tok/s, {el / 60:.1f} min", flush=True)
    print("DONE", n_rep, "replies", n_tok, "tokens", flush=True)


if __name__ == "__main__":
    main()
