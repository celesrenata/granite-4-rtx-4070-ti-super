#!/usr/bin/env python3
"""Records (tokens.bin / mask.bin) from gen_chat.py output, for dump-hidden.

Each record is exactly what the service runs for that turn: the frontend's
chat-template render of the history (thinking off), tokenized by the
frontend's tokenizer, then the reply tokens (+ <|im_end|> when it stopped).
The mask marks the reply tokens (the head trains on those positions).
Conversations with conv >= --eval-from go to the eval split. Records longer
than --max-len tokens are dropped (dump-hidden's --max-context).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, "/home/celes/src/paged-shared")
from frontend import chat_template, tokenizer  # noqa: E402

EOS = 100257


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--eval-from", type=int, default=1700)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    tok = tokenizer.ByteBPE(a.tokenizer)
    outs = {}
    for name, d in (("train", a.train_out), ("eval", a.eval_out)):
        Path(d).mkdir(parents=True, exist_ok=True)
        outs[name] = [open(Path(d) / "tokens.bin", "wb"), open(Path(d) / "mask.bin", "wb"),
                      {"records": 0, "tokens": 0, "loss_tokens": 0, "dropped": 0}, Path(d)]
    for line in open(a.data):
        r = json.loads(line)
        o = outs["eval" if r["conv"] >= a.eval_from else "train"]
        p = tok.encode(chat_template.render(r["messages"]))
        g = tok.encode(r["reply"], allowed_special=False) if r["reply"] else []
        if r["finish"] == "stop":
            g.append(EOS)
        ids = p + g
        if len(ids) > a.max_len or not g:
            o[2]["dropped"] += 1
            continue
        o[0].write(struct.pack("<i", len(ids)) + struct.pack(f"<{len(ids)}i", *ids))
        o[1].write(bytes([0] * len(p) + [1] * len(g)))
        o[2]["records"] += 1
        o[2]["tokens"] += len(ids)
        o[2]["loss_tokens"] += len(g)
    for name, (ft, fm, meta, d) in outs.items():
        ft.close()
        fm.close()
        (d / "meta.json").write_text(json.dumps(meta))
        print(name, meta)


if __name__ == "__main__":
    main()
