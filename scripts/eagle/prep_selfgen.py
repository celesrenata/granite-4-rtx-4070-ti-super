#!/usr/bin/env python3
"""Records (tokens.bin / mask.bin) from target-generated replies.

Input: acceptance_data.json from scripts/drafter_acceptance.py collect
([{"prompt", "reply", "finish"}], replies are the 8B's greedy output through
the live API). Each record is the service's prompt rendering + reply
(+ <|im_end|> when the reply stopped); the mask marks the reply tokens.
"""
import argparse
import json
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    data = json.load(open(args.data))
    with open(out / "tokens.bin", "wb") as ft, open(out / "mask.bin", "wb") as fm:
        for d in data:
            # frontend/chat_template.py, thinking off
            prompt = f"<|im_start|>user\n{d['prompt']}<|im_end|>\n<|im_start|>assistant\n<think></think>"
            p = tok.encode(prompt, add_special_tokens=False).ids
            r = tok.encode(d["reply"], add_special_tokens=False).ids
            if d.get("finish") == "stop":
                r.append(100257)
            ids = p + r
            ft.write(struct.pack("<i", len(ids)) + struct.pack(f"<{len(ids)}i", *ids))
            fm.write(bytes([0] * len(p) + [1] * len(r)))
            n_tok += len(ids)
    (out / "meta.json").write_text(json.dumps({"records": len(data), "tokens": n_tok}))
    print(f"{len(data)} records, {n_tok} tokens")


if __name__ == "__main__":
    main()
