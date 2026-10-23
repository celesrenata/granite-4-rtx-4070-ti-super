"""Token-for-token parity gate (P3 exit criterion).

Compares `granite4-infer generate` (greedy) against a float32 transformers
reference on fixed prompts, token-for-token.

The float32 reference loads the *same* checkpoint weights the engine loads,
so agreement proves the engine's forward pass (RoPE, GQA, SwiGLU, residual,
final norm, head) is numerically correct, while FP8 vs BF16 checkpoints show
the bounded quantization noise the spec allows.

Pass = token match rate >= --min-match (default 1.0).

The reference sequence is cached to a JSON keyed by (prompt, max_tokens,
checkpoint SHA) so the gate is fast to re-run; pass --no-cache to force
re-capture.

Usage:
    python scripts/parity_gate.py \
        --checkpoint .gt/ckpt \
        --engine ./build/engine/granite4-infer \
        --prompt "The capital of France is" \
        --max-tokens 24 --min-match 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def _ckpt_sha(ckpt: str) -> str:
    """A cheap fingerprint of the checkpoint (config.json + tokenizer.json)."""
    p = Path(ckpt)
    acc = hashlib.sha256()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        f = p / name
        if f.is_file():
            acc.update(f.read_bytes())
    return acc.hexdigest()[:16]


def cache_path(ckpt: str, prompt: str, max_tokens: int, out_dir: str) -> Path:
    key = hashlib.sha256(
        f"{_ckpt_sha(ckpt)}|{prompt}|{max_tokens}".encode()
    ).hexdigest()[:16]
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ref_{key}.json"


def reference_tokens(ckpt: str, prompt: str, max_tokens: int,
                    device: str = "cpu") -> list[int]:
    """fp32 transformers greedy generation on the prompt. Cached."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(ckpt)
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype="float32", low_cpu_mem_usage=True, device_map=device,
    )
    model.eval()
    cur = torch.tensor([ids], dtype=torch.long, device=device)
    out: list[int] = []
    for _ in range(max_tokens):
        with torch.no_grad():
            o = model(input_ids=cur[:, -512:])
            nxt = int(torch.argmax(o.logits[0, -1, :].float()).item())
        out.append(nxt)
        cur = torch.cat(
            [cur, torch.tensor([[nxt]], dtype=torch.long, device=device)],
            dim=1,
        )
    return out


def engine_tokens(engine: str, ckpt: str, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """Run granite4-infer generate (greedy) and parse the produced tokens."""
    toks = ",".join(map(str, prompt_ids))
    r = subprocess.run(
        [engine, "generate", "--checkpoint", ckpt,
         "--tokens", toks, "--max-tokens", str(max_tokens), "--seed", "0"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {r.stderr[-500:]}")
    doc = json.loads(r.stdout)
    return list(doc.get("generated", []))


def compare(mine: list[int], ref: list[int]) -> dict:
    match = sum(1 for a, b in zip(mine, ref) if a == b)
    first_diff = next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), None)
    return {
        "n_ref": len(ref),
        "n_mine": len(mine),
        "matched": match,
        "rate": (match / len(ref)) if ref else 0.0,
        "first_diff": first_diff,
        "mine": mine,
        "ref": ref,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint dir (BF16 original or FP8 export)")
    ap.add_argument("--engine", required=True, help="granite4-infer binary")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--min-match", type=float, default=1.0,
                    help="pass if token match rate >= this (default 1.0)")
    ap.add_argument("--cache-dir", default=".gt/parity_cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", help="write the full comparison report to this path")
    ap.add_argument("--gate-out", default=".gt/parity_gate.json",
                    help="write the gate-status artifact (on PASS) to this path")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt_ids = tok(args.prompt, return_tensors="pt").input_ids[0].tolist()

    # Reference (cached).
    ref = None
    if not args.no_cache:
        cp = cache_path(args.checkpoint, args.prompt, args.max_tokens,
                        args.cache_dir)
        if cp.is_file():
            ref = json.loads(cp.read_text())
    if ref is None:
        ref = reference_tokens(args.checkpoint, args.prompt, args.max_tokens,
                               args.device)
        if not args.no_cache:
            cache_path(args.checkpoint, args.prompt, args.max_tokens,
                       args.cache_dir).write_text(json.dumps(ref))

    mine = engine_tokens(args.engine, args.checkpoint, prompt_ids,
                         args.max_tokens)
    rep = compare(mine, ref)
    rep["prompt"] = args.prompt
    rep["prompt_ids"] = prompt_ids
    rep["min_match"] = args.min_match
    rep["pass"] = rep["rate"] >= args.min_match

    # Gate status artifact: the thing the serving layer keys off. Written on
    # PASS only, so the API can report `inference: available` after a passing
    # gate (docs/phases.md P3 exit criterion) and nothing else.
    if rep["pass"]:
        status = {
            "pass": True,
            "checkpoint_sha": _ckpt_sha(args.checkpoint),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "prompt": args.prompt,
            "prompt_ids": prompt_ids,
            "max_tokens": args.max_tokens,
            "min_match": args.min_match,
            "matched": rep["matched"],
            "rate": rep["rate"],
            "engine": str(Path(args.engine).resolve()),
            "utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        }
        if args.out:
            Path(args.out).write_text(json.dumps(rep, indent=2))
        gate_file = Path(args.gate_out)
        gate_file.write_text(json.dumps(status, indent=2))
        print(f"gate status written: {gate_file}")

    print(f"prompt: {args.prompt!r}")
    print(f"prompt ids: {prompt_ids}")
    print(f"token match: {rep['matched']}/{rep['n_ref']} "
          f"({rep['rate']:.1%}), first diff at step {rep['first_diff']}")
    if rep["first_diff"] is not None:
        i = rep["first_diff"]
        print(f"  step {i + 1}: engine={rep['mine'][i]} reference={rep['ref'][i]}")
        print(f"  reference text: {tok.decode(rep['ref'])!r}")
        print(f"  engine text:    {tok.decode(rep['mine'])!r}")
    print(f"PARITY {'PASS' if rep['pass'] else 'FAIL'} "
          f"(min-match {args.min_match:.1%})")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
