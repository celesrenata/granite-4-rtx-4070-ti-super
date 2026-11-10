#!/usr/bin/env python3
"""Long-context prompts for the EAGLE head (r15long): coherent multi-file "agent reads".

The deployed heads were trained on records of <= 4096 tokens (dump-hidden --max-context 4096,
RoPE tables of 8192 positions in train_eagle.py), while the service drafts at up to 98K
positions; acceptance there falls from ~0.13 to ~0.05-0.08. These prompts are what long-context
traffic looks like: several related files (one directory subtree at a time, in path order, each
under a "// File:" header the way an agent's tool reads appear) followed by one task about them.

Corpora are directory trees of text files (--corpus name=path, repeatable). Directories are split
into train / eval by a stable hash, so held-out prompts never share a file with training ones.
Writes the gen_chat.py prompt format: [{"turns": [prompt], "src": "...", "split": "train"|"eval",
"target_tokens": L}], train prompts first (conversation index < eval_from).
"""
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

EXT_LANG = {".ts": "typescript", ".tsx": "tsx", ".js": "javascript", ".py": "python", ".nix": "nix",
            ".md": "markdown", ".json": "json", ".cu": "cuda", ".cc": "cpp", ".h": "cpp", ".cuh": "cuda",
            ".sh": "bash", ".rs": "rust", ".go": "go", ".toml": "toml", ".yaml": "yaml", ".yml": "yaml"}
# Target prompt lengths (tokens) and weights: long-context traffic is mostly 8K-32K.
LENGTHS = [(3000, 10), (6000, 15), (10000, 20), (16000, 20), (24000, 15), (32000, 12), (48000, 8)]
TASKS = [
    "Explain what the code above does, file by file, and how the files fit together.",
    "Review the code above. List the most likely bugs, missing error handling and edge cases, with the file and the fix for each.",
    "Write unit tests for `{sym}`. Cover the normal path and the edge cases you can see in the code.",
    "Where is `{sym}` defined, what does it do, and where is it used? Quote the relevant lines.",
    "Propose a refactor of `{sym}` that makes it easier to test, and show the new code.",
    "Summarize the architecture shown above: the main components, their responsibilities and the data flow between them.",
    "Produce a JSON object that maps each file path above to a one-sentence summary of its purpose.",
    "Add a new option to `{sym}` that lets callers disable it, and show every change needed across the files above.",
    "What happens when `{sym}` receives invalid input? Trace the code path and point out anything unsafe.",
    "Write concise developer documentation for the module above: purpose, public API, and one usage example.",
    "List every TODO, FIXME or unfinished piece in the files above and suggest how to complete each one.",
    "Explain the control flow of `{sym}` step by step, as if onboarding a new engineer.",
]
SYM_RE = re.compile(r"\b(?:function|class|def|interface|const|export function|async function)\s+([A-Za-z_][A-Za-z0-9_]{3,})")


def split_of(d: str) -> str:
    return "eval" if int(hashlib.sha1(d.encode()).hexdigest(), 16) % 10 == 0 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--weights", default="", help="name=w,... corpus sampling weights")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--frontend", required=True, help="dir holding the frontend package")
    ap.add_argument("--train", type=int, default=480)
    ap.add_argument("--eval", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=15)
    a = ap.parse_args()
    sys.path.insert(0, a.frontend)
    from frontend import tokenizer  # noqa: E402
    tok = tokenizer.ByteBPE(a.tokenizer)
    rng = random.Random(a.seed)
    weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(",") if x)}
    corpora = {}
    for spec in a.corpus:
        name, root = spec.split("=", 1)
        root = Path(root)
        files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in EXT_LANG
                 and 200 < p.stat().st_size < 200_000]
        by_dir = {}
        for p in files:
            by_dir.setdefault(str(p.parent.relative_to(root)), []).append(p)
        # A unit = a directory subtree: its files plus those of its subdirectories, in path order.
        dirs = sorted(by_dir)
        units = {"train": [], "eval": []}
        for d in dirs:
            members = [p for dd in dirs if dd == d or dd.startswith(d + "/") for p in by_dir[dd]]
            if len(members) >= 2:
                units[split_of(name + ":" + d)].append((d, members))
        corpora[name] = (root, units)
        print(f"{name}: {len(files)} files, {len(units['train'])} train / {len(units['eval'])} eval units",
              file=sys.stderr)
    names = list(corpora)
    cw = [weights.get(n, 1.0) for n in names]

    def build(split):
        name = rng.choices(names, cw)[0]
        root, units = corpora[name]
        if not units[split]:
            return None
        target = rng.choices([l for l, _ in LENGTHS], [w for _, w in LENGTHS])[0]
        # Prefer subtrees big enough for the target; fall back to the biggest ones.
        cands = [u for u in units[split] if sum(p.stat().st_size for p in u[1]) / 3.6 >= target * 0.8]
        d, members = rng.choice(cands) if cands else max(units[split], key=lambda u: len(u[1]))
        start = rng.randrange(len(members))
        parts, ntok, used = [], 0, []
        for p in members[start:] + members[:start]:
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            block = f"// File: {p.relative_to(root)}\n```{EXT_LANG[p.suffix]}\n{text.rstrip()}\n```\n\n"
            n = len(tok.encode(block))
            if ntok + n > target:
                if ntok > target * 0.5:
                    break
                continue
            parts.append(block)
            used.append(text)
            ntok += n
        if ntok < 1500:
            return None
        syms = [s for t in used for s in SYM_RE.findall(t)]
        task = rng.choice(TASKS)
        if "{sym}" in task:
            if not syms:
                task = TASKS[0]
            else:
                task = task.format(sym=rng.choice(syms))
        return {"turns": ["".join(parts) + task], "src": f"long:{name}:{d}", "split": split,
                "target_tokens": target, "prompt_tokens_est": ntok}

    out = []
    for split, n in (("train", a.train), ("eval", a.eval)):
        got, tries = 0, 0
        while got < n and tries < n * 20:
            tries += 1
            r = build(split)
            if r is not None:
                out.append(r)
                got += 1
    Path(a.out).write_text(json.dumps(out))
    ntr = sum(r["split"] == "train" for r in out)
    tot = sum(r["prompt_tokens_est"] for r in out)
    print(json.dumps({"prompts": len(out), "train": ntr, "eval_from": ntr, "prompt_tokens": tot}))


if __name__ == "__main__":
    main()
