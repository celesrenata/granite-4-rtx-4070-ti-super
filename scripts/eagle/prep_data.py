#!/usr/bin/env python3
"""Build EAGLE draft-head training sequences from public chat data.

Reads UltraChat 200k (train_sft parquet) and ShareGPT V4.3 (json), renders
each conversation in the Granite 4.2 chat format the service uses
(frontend/chat_template.py; assistant turns as the checkpoint template writes
them: `<|im_start|>assistant\\n<think></think>{content}<|im_end|>\\n`) and
writes:

  OUT/tokens.bin   records of (int32 n, n int32 token ids)  -> engine dump-hidden
  OUT/mask.bin     n uint8 per record: 1 where the token is assistant content
                   (or the <|im_end|> closing it), i.e. a token the draft head
                   learns to predict
  OUT/meta.json    record count, token count, sources

A held-out split goes to OUT_eval/ in the same layout.
"""
from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def ultrachat(paths):
    import pyarrow.parquet as pq

    for p in paths:
        for batch in pq.ParquetFile(p).iter_batches(columns=["messages"], batch_size=1024):
            for msgs in batch.column(0).to_pylist():
                yield "ultrachat", [(m["role"], m["content"]) for m in msgs]


def sharegpt(path):
    roles = {"human": "user", "user": "user", "gpt": "assistant", "chatgpt": "assistant",
             "system": "system"}
    for conv in json.load(open(path)):
        turns = []
        for t in conv.get("conversations", []):
            r = roles.get(t.get("from"))
            if r is None or not isinstance(t.get("value"), str):
                turns = []
                break
            turns.append((r, t["value"]))
        if turns:
            yield "sharegpt", turns


def encode_conv(tok, turns, max_len):
    ids, mask = [], []

    def add(text, m):
        e = tok.encode(text, add_special_tokens=False).ids
        ids.extend(e)
        mask.extend([m] * len(e))

    if not any(r == "assistant" and c.strip() for r, c in turns):
        return None
    for i, (role, content) in enumerate(turns):
        if role == "assistant":
            add(f"{IM_START}assistant\n<think></think>", 0)
            add(content.strip(), 1)
            add(IM_END, 1)
            add("\n", 0)
        elif role in ("user", "system"):
            if role == "system" and i != 0:
                return None
            add(f"{IM_START}{role}\n{content}{IM_END}\n", 0)
        if len(ids) >= max_len:
            break
    ids, mask = ids[:max_len], mask[:max_len]
    if sum(mask) < 16:
        return None
    return ids, mask


def write_split(out: Path, items, sources):
    out.mkdir(parents=True, exist_ok=True)
    ntok = 0
    with open(out / "tokens.bin", "wb") as ft, open(out / "mask.bin", "wb") as fm:
        for ids, mask in items:
            ft.write(struct.pack("<i", len(ids)))
            ft.write(struct.pack(f"<{len(ids)}i", *ids))
            fm.write(bytes(mask))
            ntok += len(ids)
    (out / "meta.json").write_text(json.dumps(
        {"records": len(items), "tokens": ntok, "loss_tokens": sum(sum(m) for _, m in items),
         "sources": sources}, indent=1))
    return ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ultrachat", nargs="+", required=True)
    ap.add_argument("--sharegpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--tokens", type=int, default=16_000_000, help="training token budget")
    ap.add_argument("--eval-records", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    convs = list(ultrachat(args.ultrachat)) + list(sharegpt(args.sharegpt))
    random.Random(args.seed).shuffle(convs)
    train, evals, counts, budget = [], [], {}, 0
    for src, turns in convs:
        r = encode_conv(tok, turns, args.max_len)
        if r is None:
            continue
        if len(evals) < args.eval_records:
            evals.append(r)
            continue
        train.append(r)
        counts[src] = counts.get(src, 0) + 1
        budget += len(r[0])
        if budget >= args.tokens:
            break
    out = Path(args.out)
    n = write_split(out, train, counts)
    write_split(Path(str(out) + "_eval"), evals, {})
    print(f"train: {len(train)} records, {n} tokens {counts}; eval: {len(evals)} records")


if __name__ == "__main__":
    main()
