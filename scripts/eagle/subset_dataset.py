#!/usr/bin/env python3
"""Extract a record range [start, end) from an EAGLE dataset dir into a new dir.

Used to build a tiny overfit set (e.g. the 6 held-out qa records) for a capacity
test. tokens.bin is record blocks [len, tokens...]; mask.bin is flat per token;
hidden.bin is flat int16, one 4096-row per token. hidden is read via memmap so
a large source dir can be sliced without loading it into RAM.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


def parse_raw(raw: np.ndarray):
    lengths, offsets = [], []
    i = 0
    while i < len(raw):
        lengths.append(int(raw[i]))
        offsets.append(i)
        i += 1 + int(raw[i])
    return lengths, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="exclusive")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = np.fromfile(src / "tokens.bin", dtype=np.int32)
    lengths, offsets = parse_raw(raw)
    assert 0 <= a.start < a.end <= len(lengths), \
        f"range [{a.start},{a.end}) out of {len(lengths)} records"
    s, e = a.start, a.end
    sub_raw = raw[offsets[s]:offsets[e - 1] + 1 + lengths[e - 1]]
    tok_start = sum(lengths[:s])
    tok_end = sum(lengths[:e])
    mask = np.fromfile(src / "mask.bin", dtype=np.uint8)[tok_start:tok_end]
    hid = np.memmap(src / "hidden.bin", dtype=np.int16, mode="r")
    H = hid.size // sum(lengths)  # int16 rows per token
    sub_hid = np.asarray(hid[tok_start * H: tok_end * H])
    sub_raw.astype(np.int32).tofile(out / "tokens.bin")
    mask.astype(np.uint8).tofile(out / "mask.bin")
    sub_hid.astype(np.int16).tofile(out / "hidden.bin")
    ntok = tok_end - tok_start
    json.dump({"records": e - s, "tokens": ntok}, open(out / "meta.json", "w"))
    print(f"{src}: records [{s},{e}), {ntok} tokens -> {out}", flush=True)


if __name__ == "__main__":
    main()
