#!/usr/bin/env python3
"""Concatenate Records datasets (train + serving-style) into one directory.

Each source dir has tokens.bin (length-prefixed int32 records), mask.bin (1
byte/token) and hidden.bin (one int16 row per token, H=4096). The format is a
flat stream, so concatenating the first `count` records of each source in
order is valid: copy the token record headers+bodies, then the matching
prefix of mask.bin (1 byte/token) and hidden.bin (H*2 bytes/token).

--spec takes `name:count` pairs in output order; `count` is records to take
from each source (0 or omitted = all). Repeat a source to upsample, e.g.
`--spec serve:400 train:12992`.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

H = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--base", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = Path(a.base)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    per = {}
    with open(out / "tokens.bin", "wb") as ft, \
         open(out / "mask.bin", "wb") as fm, \
         open(out / "hidden.bin", "wb") as fh:
        for spec in a.spec:
            name, _, cnt = spec.partition(":")
            cnt = 0 if cnt in ("", "0") else int(cnt)
            src = base / name
            tok = taken = 0
            with open(src / "tokens.bin", "rb") as ft_in, \
                 open(src / "mask.bin", "rb") as fm_in, \
                 open(src / "hidden.bin", "rb") as fh_in:
                while True:
                    hdr = ft_in.read(4)
                    if len(hdr) < 4:
                        break
                    n = int.from_bytes(hdr, "little")
                    if cnt and taken >= cnt:
                        break
                    ft.write(hdr)
                    body = ft_in.read(4 * n)
                    ft.write(body)
                    fm.write(fm_in.read(n))
                    fh.write(fh_in.read(n * H * 2))
                    tok += n
                    taken += 1
            meta = json.loads((src / "meta.json").read_text())
            per[spec] = {"records": taken, "tokens": tok}
            print(f"{name}: {taken}/{meta.get('records')} records, {tok} tokens")
    (out / "meta.json").write_text(json.dumps({
        "records": sum(v["records"] for v in per.values()),
        "tokens": sum(v["tokens"] for v in per.values()),
        "sources": per}))
    print(f"-> {out}: {sum(v['records'] for v in per.values())} records, "
          f"{sum(v['tokens'] for v in per.values())} tokens")


if __name__ == "__main__":
    main()
