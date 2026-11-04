#!/usr/bin/env python3
"""Split serving_data.json into a held-out eval set (N per style) and a train
set. Deterministic: records are sorted by (style, prompt) before carving, so
the same eval prompts are reproduced across runs. Writes per-style eval JSONs
too, so each style can be prepped/dumped/measured independently.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-style", type=int, default=6)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    data = json.load(open(a.data))
    styles = sorted({d["style"] for d in data})
    by_style = {s: sorted((d for d in data if d["style"] == s),
                          key=lambda d: d["prompt"]) for s in styles}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eval_all, train = [], []
    for s in styles:
        ev = by_style[s][:a.per_style]
        tr = by_style[s][a.per_style:]
        for d in ev:
            eval_all.append(d)
        for d in tr:
            train.append(d)
        # per-style eval JSON (for independent prep/dump/measure)
        (out / f"eval_{s}.json").write_text(json.dumps(ev, indent=1))
        print(f"{s}: {len(ev)} eval, {len(tr)} train")
    (out / "eval.json").write_text(json.dumps(eval_all, indent=1))
    (out / "eval_styles.json").write_text(json.dumps([d["style"] for d in eval_all], indent=0))
    (out / "train.json").write_text(json.dumps(train, indent=1))
    (out / "train_styles.json").write_text(json.dumps([d["style"] for d in train], indent=0))
    print(f"eval={len(eval_all)} train={len(train)} -> {out}")


if __name__ == "__main__":
    main()
