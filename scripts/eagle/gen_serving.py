#!/usr/bin/env python3
"""Generate target (8B) greedy replies for serving-style prompts.

Uses one persistent granite4-infer serve process. Each record is the service
chat-template render (thinking off) of a prompt; the reply is the engine's
greedy output, which is exactly what the service serves and what EAGLE must
match. Writes [{prompt, reply, finish}] for prep_selfgen.py to tokenize.

The prompt template is read verbatim from prep_selfgen.py on the same
filesystem, so this script stays byte-identical to the reference tokenizer
without re-declaring the special-token literals.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from tokenizers import Tokenizer

EOS = 100257


def load_template(path: Path):
    """Extract the prompt f-string body (between the quotes) from
    prep_selfgen.py's source. Returned as source text: escape sequences like
    \\n are still literal and must be evaluated, not string-replaced."""
    src = path.read_text()
    m = re.search(r'prompt = f"(.*)"\n', src, re.S)
    if m is None:
        raise SystemExit(f"could not find template in {path}")
    return m.group(1)


def render_prompt(tmpl_src: str, prompt_text: str) -> str:
    """Evaluate the extracted f-string so escape sequences (\\n) and the
    {d['prompt']} field behave exactly as in prep_selfgen.py."""
    fstr = 'f"' + tmpl_src + '"'
    return eval(fstr, {"__builtins__": {}}, {"d": {"prompt": prompt_text}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template-src", default="prep_selfgen.py")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-context", type=int, default=4096)
    a = ap.parse_args()

    tmpl = load_template(Path(a.template_src))
    proc = subprocess.Popen(
        [a.engine, "serve", "--checkpoint", a.checkpoint, "--gpu", "1",
         "--max-context", str(a.max_context)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1)
    assert json.loads(proc.stdout.readline())["ready"]
    tok = Tokenizer.from_file(str(Path(a.checkpoint) / "tokenizer.json"))
    data = json.load(open(a.prompts))
    out = []
    t0 = time.time()
    ntok = 0
    for i, d in enumerate(data):
        prompt = render_prompt(tmpl, d["prompt"])
        pids = tok.encode(prompt, add_special_tokens=False).ids
        t = time.time()
        proc.stdin.write(json.dumps({"tokens": pids, "max_tokens": a.max_tokens,
                                     "eos": EOS}) + "\n")
        proc.stdin.flush()
        doc = json.loads(proc.stdout.readline())
        if "error" in doc:
            print(f"[{i}] error: {doc['error']}", flush=True)
            continue
        rid = doc["generated"]
        stopped = bool(rid) and rid[-1] == EOS
        if stopped:
            rid = rid[:-1]  # drop trailing EOS; prep_selfgen re-appends it
        rtok = proc_token_to_text(tok, rid)
        finish = "stop" if stopped else "length"
        out.append({"prompt": d["prompt"], "reply": rtok, "finish": finish,
                    "style": d.get("style", "")})
        ntok += len(rid)
        if (i + 1) % 25 == 0:
            dt = time.time() - t0
            print(f"[{i+1}/{len(data)}] {dt:.0f}s {ntok/dt:.0f} tok/s", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    proc.stdin.close()
    print(f"{len(out)} records, {ntok} tokens -> {a.out}")


def proc_token_to_text(tok, ids):
    return tok.decode(ids, skip_special_tokens=True)


if __name__ == "__main__":
    main()
