#!/usr/bin/env python3
"""Concatenate EAGLE dataset dirs into one training dir (prefix subset supported).

Dataset layout (written by prep_selfgen.py + dump-hidden):
- tokens.bin:  [len, tokens...]* record blocks, int32
- mask.bin:    flat uint8, one per token (size == sum of record lengths)
- hidden.bin:  flat int16, one 4096-row per token (rows == total tokens)

Parts are "dir" (all records) or "dir:N" (first N records of dir). The output is
written record-for-record with matching mask/hidden slices, so train_eagle.py
can read it directly. hidden.bin is read via memmap so a huge source dir (e.g.
the full chat set) can be subsetted without loading it into RAM.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np


def parse_raw(raw: np.ndarray):
    """Return (lengths, offsets) — length and byte offset of each record block."""
    lengths, offsets = [], []
    i = 0
    while i < len(raw):
        lengths.append(int(raw[i]))
        offsets.append(i)
        i += 1 + int(raw[i])
    return lengths, offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True,
                    help='dir[:N] — N = first N records of that dir (all if omitted)')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    out_raw, out_mask, out_hidden, total_tokens, nrec_total = [], [], [], 0, 0
    H = None
    for spec in a.parts:
        dirpath, nrec = spec.rsplit(":", 1) if ":" in spec else (spec, None)
        dirpath = Path(dirpath)
        raw = np.fromfile(dirpath / "tokens.bin", dtype=np.int32)
        lengths, offsets = parse_raw(raw)
        nsel = len(lengths) if nrec is None else int(nrec)
        assert 0 < nsel <= len(lengths), f"{dirpath}: bad N={nsel} of {len(lengths)}"
        tok_end = sum(lengths[:nsel])
        sub_raw = raw if nsel == len(lengths) else \
            raw[offsets[0]:offsets[nsel - 1] + 1 + lengths[nsel - 1]]
        mask = np.fromfile(dirpath / "mask.bin", dtype=np.uint8)
        hid = np.memmap(dirpath / "hidden.bin", dtype=np.int16, mode="r")
        Hp = hid.size // sum(lengths)
        if H is None:
            H = Hp
        else:
            assert Hp == H, f"hidden dim mismatch in {dirpath}: {Hp} vs {H}"
        sub_hid = np.asarray(hid[:tok_end * H]).reshape(-1)  # materialize the slice
        out_raw.append(sub_raw)
        out_mask.append(mask[:tok_end])
        out_hidden.append(sub_hid)
        total_tokens += tok_end
        nrec_total += nsel
        print(f"{dirpath}: {nsel} records, {tok_end} tokens (hidden dim {H})", flush=True)

    np.concatenate(out_raw).astype(np.int32).tofile(out / "tokens.bin")
    np.concatenate(out_mask).astype(np.uint8).tofile(out / "mask.bin")
    np.concatenate([h.astype(np.int16) for h in out_hidden]).tofile(out / "hidden.bin")
    json.dump({"records": nrec_total, "tokens": total_tokens}, open(out / "meta.json", "w"))
    print(f"out {out}: {nrec_total} records, {total_tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
