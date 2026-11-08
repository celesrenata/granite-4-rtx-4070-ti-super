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
import threading
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
    ap.add_argument("--workers", type=int, default=2,
                    help="conversations in flight (the engine runs one per session)")
    a = ap.parse_args()
    convs = json.load(open(a.prompts))[: a.limit]
    out = Path(a.out)
    saved = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            saved[(r["conv"], len(r["messages"]))] = r["reply"]
    t0 = time.time()
    stats = {"tok": 0, "rep": 0}
    lock = threading.Lock()
    todo = list(enumerate(convs))
    f = out.open("a")

    def work():
        while True:
            with lock:
                if not todo:
                    return
                i, c = todo.pop(0)
            msgs = []
            for t, user in enumerate(c["turns"]):
                msgs = msgs + [{"role": "user", "content": user}]
                key = (i, len(msgs))
                if key in saved:
                    prev = saved[key]
                else:
                    reply = None
                    for attempt in range(120):  # the API may be down (GPU windows)
                        try:
                            reply, finish, ntok = chat(a.url, msgs, a.max_tokens)
                            break
                        except Exception as e:
                            if attempt == 0:
                                print(f"conv {i} turn {t}: {e}; retrying", flush=True)
                            time.sleep(30)
                    if reply is None:
                        break
                    with lock:
                        f.write(json.dumps({"conv": i, "src": c["src"], "messages": msgs,
                                            "reply": reply, "finish": finish}) + "\n")
                        f.flush()
                        stats["tok"] += ntok
                        stats["rep"] += 1
                        if stats["rep"] % 25 == 0:
                            el = time.time() - t0
                            print(f"conv {i}/{len(convs)}: {stats['rep']} new replies, "
                                  f"{stats['tok']} tokens, {stats['tok'] / el:.1f} tok/s, "
                                  f"{el / 60:.1f} min", flush=True)
                    prev = reply
                msgs = msgs + [{"role": "assistant", "content": prev}]

    ths = [threading.Thread(target=work) for _ in range(max(1, a.workers))]
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    f.close()
    n_rep, n_tok = stats["rep"], stats["tok"]
    print("DONE", n_rep, "replies", n_tok, "tokens", flush=True)


if __name__ == "__main__":
    main()
