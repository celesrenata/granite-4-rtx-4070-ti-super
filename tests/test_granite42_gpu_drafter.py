"""Granite 4.2 GPU drafter acceptance and identity tests.

These tests drive the real `granite4-infer` binary on the Granite 4.2 8B
checkpoint using prompts rendered from the checkpoint's chat template
(system/user/assistant turns, reasoning preamble, tool-call and code
continuations). They require the CUDA build and the staged checkpoint, so
they are opt-in:

    GRANITE4_GPU_TESTS=1 \
    GRANITE4_INFER=./result/bin/granite4-infer \
    GRANITE4_CKPT=/var/lib/granite4/ckpt \
    python -m pytest -q tests/test_granite42_gpu_drafter.py

Without `GRANITE4_GPU_TESTS` the GPU cases are skipped; the tokenizer-only
case runs wherever the checkpoint tokenizer is staged.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from frontend.chat_template import render
from frontend.tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parents[1]
DRAFT_RE = re.compile(
    r"G4_DRAFT steps=(\d+) proposed=(\d+) accepted=(\d+) rate=([0-9.]+)")

CASES = [
    dict(
        name="reasoning_repeat_list",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content":
                "Repeat the following list exactly, one item per line, then "
                "continue the same pattern:\nred\nblue\nred\nblue\n"},
        ],
        enable_thinking=True,
        max_tokens=24,
        # Measured 0.727 on gremlin-1.
        min_rate=0.5,
    ),
    dict(
        name="tool_call_json",
        messages=[
            {"role": "user", "content":
                'Emit JSON tool calls with alternating keys, continuing this '
                'exact pattern: {"a": 1, "b": 2, "a": 1,'},
        ],
        enable_thinking=False,
        max_tokens=24,
        # Measured 0.583 on gremlin-1.
        min_rate=0.5,
    ),
    dict(
        name="code_chain",
        messages=[
            {"role": "user", "content":
                "Continue this Python code exactly:\n\n"
                "def f1(x):\n    return x + 1\n\n"
                "def f2(x):\n    return x + 2\n\n"
                "def f3(x):\n"},
        ],
        enable_thinking=False,
        max_tokens=24,
        # Measured 0.500 on gremlin-1; keep margin for kernel revisions.
        min_rate=0.4,
    ),
    dict(
        name="chat_control",
        messages=[{"role": "user", "content": "The capital of France is"}],
        enable_thinking=False,
        max_tokens=24,
        min_rate=None,
    ),
]


def _checkpoint() -> Path:
    return Path(os.environ.get("GRANITE4_CKPT", "/var/lib/granite4/ckpt"))


def _infer_binary() -> Path:
    explicit = os.environ.get("GRANITE4_INFER")
    if explicit:
        return Path(explicit)
    for candidate in (ROOT / "result" / "bin" / "granite4-infer",
                      ROOT / "build" / "engine" / "granite4-infer"):
        if candidate.exists():
            return candidate
    pytest.skip("granite4-infer binary is not built")


def _tokenize(text: str) -> list[int]:
    ckpt = _checkpoint() / "tokenizer.json"
    if not ckpt.exists():
        pytest.skip("Granite 4.2 checkpoint tokenizer is not staged")
    return ByteBPE(ckpt).encode(text)


def _run(ids: list[int], extra: list[str]) -> tuple[list[int], dict[str, int]]:
    binary = _infer_binary()
    ckpt = _checkpoint()
    cmd = [str(binary), "generate", "--checkpoint", str(ckpt),
           "--tokens", ",".join(map(str, ids)), "--max-tokens", "24",
           "--gpu", "1", *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    generated = json.loads(proc.stdout)["generated"]
    match = DRAFT_RE.search(proc.stderr)
    stats = {}
    if match:
        stats = dict(steps=int(match.group(1)), proposed=int(match.group(2)),
                     accepted=int(match.group(3)), rate=float(match.group(4)))
    return generated, stats


def test_granite42_prompt_contract():
    """Rendered Granite 4.2 turns must use the model's own special markers."""
    text = render(
        [{"role": "system", "content": "sys"},
         {"role": "user", "content": "hello"}],
        enable_thinking=True,
    )
    assert text.startswith("<|im_start|>system\nsys")
    assert "<|im_start|>user\nhello<|im_end|>\n" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    ids = _tokenize(text)
    assert 100256 in ids  # <|im_start|>
    assert 100274 in ids  # <think>


@pytest.mark.skipif(
    os.environ.get("GRANITE4_GPU_TESTS") != "1",
    reason="set GRANITE4_GPU_TESTS=1 to run GPU drafter acceptance tests")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_granite42_draft_identity_and_acceptance(case):
    ids = _tokenize(render(case["messages"],
                           enable_thinking=case["enable_thinking"]))
    baseline, _ = _run(ids, [])
    draft, stats = _run(ids, ["--draft-lookup", "4"])
    assert draft == baseline, f"{case['name']}: draft diverged from greedy"
    print(f"\n{case['name']}: {stats}")
    if case["min_rate"] is not None:
        assert stats.get("rate", 0.0) >= case["min_rate"], case["name"]
        assert stats.get("accepted", 0) >= 3, case["name"]
    else:
        # Ordinary chat has no user-text repetition, but the Granite chat
        # template itself repeats structural markers (<|im_start|>, newlines),
        # so chance proposals do appear (measured 4/8 on gremlin-1). Identity
        # is the guarantee here; acceptance is reported, not asserted.
        assert stats.get("steps", 1) >= 1, case["name"]
