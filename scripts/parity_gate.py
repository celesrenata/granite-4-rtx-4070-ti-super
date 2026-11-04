"""Token-for-token parity gate (P3 exit criterion).

Compares `granite4-infer generate` (greedy) against a float32 transformers
reference on fixed prompts, token-for-token.

The float32 reference loads the *same* checkpoint weights the engine loads,
so agreement proves the engine's forward pass (RoPE, GQA, SwiGLU, residual,
final norm, head) is numerically correct, while FP8 vs BF16 checkpoints show
the bounded quantization noise the spec allows.

Two modes:

* free (default): the engine generates greedily from the prompt and its
  tokens are compared with the reference's, position by position. Pass =
  token match rate >= --min-match (default 1.0). One bf16 near-tie flips the
  rest of the sequence, so this mode suits the f32 host oracle.
* teacher-forced: the engine runs one forward over prompt + reference[:-1]
  and each position's top-1 is checked against the reference token. A
  position passes when the reference token's logit is within
  --ulp-tolerance bf16 ULPs of the engine's maximum (0 = exact top-1);
  ties and 1-2 ULP gaps are bf16 rounding, not arithmetic defects. Pass =
  passing-position rate >= --min-match. The artifact records the mode, the
  tolerance and the exact top-1 rate. This is the gate for the bf16 GPU path.

The reference sequence is cached to a JSON keyed by (prompt, max_tokens,
checkpoint SHA) so the gate is fast to re-run; pass --no-cache to force
re-capture. --reference FILE uses a shipped reference instead
({"prompt", "prompt_ids", "checkpoint_sha", "tokens"}); its checkpoint_sha
must match, and then neither torch nor transformers is needed.

Usage:
    python scripts/parity_gate.py \
        --checkpoint .gt/ckpt \
        --engine ./build/engine/granite4-infer \
        --prompt "The capital of France is" \
        --max-tokens 24 --min-match 1.0
    python scripts/parity_gate.py --checkpoint CKPT --engine granite4-infer \
        --gpu --mode teacher-forced --reference reference/granite/parity_ref.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


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


def engine_tokens(engine: str, ckpt: str, prompt_ids: list[int], max_tokens: int,
                  gpu: bool = False) -> list[int]:
    """Run granite4-infer generate (greedy) and parse the produced tokens."""
    toks = ",".join(map(str, prompt_ids))
    cmd = [engine, "generate", "--checkpoint", ckpt,
           "--tokens", toks, "--max-tokens", str(max_tokens), "--seed", "0"]
    if gpu:
        cmd += ["--gpu", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {r.stderr[-500:]}")
    doc = json.loads(r.stdout)
    return list(doc.get("generated", []))


def engine_logits(engine: str, ckpt: str, tokens: list[int], last: int,
                  gpu: bool = False) -> list[list[float]]:
    """Run granite4-infer forward and return the last `last` logits rows."""
    cmd = [engine, "forward", "--checkpoint", ckpt,
           "--tokens", ",".join(map(str, tokens)), "--last", str(last)]
    if gpu:
        cmd += ["--gpu", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"engine failed: {r.stderr[-500:]}")
    rows = json.loads(r.stdout)["logits"]
    return [row["values"] for row in rows]


def bf16_ulps(a: float, b: float) -> int:
    """Distance between a and b in bf16 ULPs (both rounded to bf16, RNE)."""
    def key(x: float) -> int:
        u = struct.unpack("<I", struct.pack("<f", x))[0]
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
        return 0x8000 - (u & 0x7FFF) if u & 0x8000 else 0x8000 + u  # monotone
    return abs(key(a) - key(b))


def compare_teacher_forced(rows: list[list[float]], ref: list[int],
                           ulp_tolerance: int) -> dict:
    """Per position: exact top-1, or ref logit within ulp_tolerance of max."""
    if len(rows) != len(ref):
        raise ValueError(f"engine returned {len(rows)} rows for {len(ref)} reference tokens")
    exact = passed = 0
    misses = []
    for i, (row, want) in enumerate(zip(rows, ref)):
        top = max(range(len(row)), key=row.__getitem__)
        gap = bf16_ulps(row[top], row[want])
        exact += top == want
        if top == want or gap <= ulp_tolerance:
            passed += 1
        if top != want:
            misses.append({"position": i, "engine": top, "reference": want,
                           "engine_logit": row[top], "reference_logit": row[want],
                           "ulps": gap})
    n = len(ref)
    return {"n_ref": n, "matched": passed, "rate": passed / n if n else 0.0,
            "exact": exact, "exact_rate": exact / n if n else 0.0,
            "misses": misses, "ref": ref}


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
    ap.add_argument("--gpu", action="store_true",
                    help="gate the CUDA path (generate --gpu 1); the artifact then "
                         "tells the server to serve with --gpu 1")
    ap.add_argument("--mode", choices=("free", "teacher-forced"), default="free")
    ap.add_argument("--ulp-tolerance", type=int, default=None,
                    help="teacher-forced: bf16 ULPs the reference token's logit may sit "
                         "below the engine's maximum (default 2; free mode: unused)")
    ap.add_argument("--reference", help="shipped reference JSON (skips transformers)")
    ap.add_argument("--out", help="write the full comparison report to this path")
    ap.add_argument("--gate-out", default=".gt/parity_gate.json",
                    help="write the gate-status artifact (on PASS) to this path")
    args = ap.parse_args()
    ulp_tolerance = (args.ulp_tolerance if args.ulp_tolerance is not None
                     else (2 if args.mode == "teacher-forced" else 0))

    from frontend import tokenizer as granite_tokenizer

    tok = granite_tokenizer.ByteBPE(Path(args.checkpoint) / "tokenizer.json")

    ref = None
    if args.reference:
        shipped = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        if shipped.get("checkpoint_sha") != _ckpt_sha(args.checkpoint):
            print("PARITY FAIL (reference was captured for a different checkpoint)")
            return 1
        if shipped.get("prompt") != args.prompt:
            print("PARITY FAIL (reference prompt differs from --prompt)")
            return 1
        prompt_ids = list(shipped["prompt_ids"])
        ref = list(shipped["tokens"])[: args.max_tokens]
    else:
        prompt_ids = tok.encode(args.prompt)
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

    if args.mode == "teacher-forced":
        rows = engine_logits(args.engine, args.checkpoint, prompt_ids + ref[:-1],
                             len(ref), gpu=args.gpu)
        rep = compare_teacher_forced(rows, ref, ulp_tolerance)
    else:
        mine = engine_tokens(args.engine, args.checkpoint, prompt_ids,
                             args.max_tokens, gpu=args.gpu)
        rep = compare(mine, ref)
    rep["prompt"] = args.prompt
    rep["prompt_ids"] = prompt_ids
    rep["min_match"] = args.min_match
    rep["pass"] = rep["rate"] >= args.min_match

    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2))

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
            "gpu": args.gpu,
            "mode": args.mode,
            "ulp_tolerance": ulp_tolerance if args.mode == "teacher-forced" else 0,
            "exact_rate": rep.get("exact_rate", rep["rate"]),
            "utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
        }
        gate_file = Path(args.gate_out)
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps(status, indent=2))
        print(f"gate status written: {gate_file}")

    print(f"prompt: {args.prompt!r}")
    print(f"prompt ids: {prompt_ids}")
    if args.mode == "teacher-forced":
        print(f"teacher-forced: {rep['matched']}/{rep['n_ref']} within {ulp_tolerance} "
              f"bf16 ULP ({rep['rate']:.1%}); exact top-1 {rep['exact']}/{rep['n_ref']}")
        for m in rep["misses"]:
            print(f"  pos {m['position']}: engine={m['engine']} ({m['engine_logit']}) "
                  f"reference={m['reference']} ({m['reference_logit']}) {m['ulps']} ULP")
    else:
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
