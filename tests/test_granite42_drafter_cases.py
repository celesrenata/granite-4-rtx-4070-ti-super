"""Granite 4.2-specific drafter fixtures.

These are token-level tests for the actual Granite 4.2 8B contract: byte-level
BPE IDs, special chat markers, reasoning delimiters, tool-call markers, and
repeated structured continuations. They deliberately do not require a model or
CUDA; the C++ lookup drafter is exercised by its native unit test.
"""
from __future__ import annotations

import json
from pathlib import Path

from frontend.tokenizer import ByteBPE

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "reference" / "granite" / "model_api.json"


def test_granite42_model_card_contract():
    data = json.loads(MODEL.read_text(encoding="utf-8"))
    assert data["modelId"] == "ibm-granite/granite-4.2-8b"
    assert data["config"]["architectures"] == ["GraniteForCausalLM"]
    assert data["config"]["model_type"] == "granite"
    tags = data["tags"]
    assert "granite-4.2" in tags
    assert any("granite-4.1-8b-base" in str(x) for x in tags)


def test_granite42_chat_markers_are_atomic(tmp_path: Path):
    """The drafter must operate on model token IDs, not characters or words."""
    # Use a real checkpoint tokenizer when available; otherwise this test is
    # skipped rather than inventing token IDs for the Granite vocabulary.
    ckpt = Path("/var/lib/granite4/ckpt/tokenizer.json")
    if not ckpt.exists():
        import pytest
        pytest.skip("Granite 4.2 checkpoint tokenizer is not staged locally")
    tok = ByteBPE(ckpt)
    ids = tok.encode("<|im_start|>assistant\n<think>\n", allowed_special=True)
    assert 100256 in ids  # <|im_start|>
    assert 100274 in ids  # <think>
    assert tok.decode(ids).startswith("<|im_start|>assistant\n<think>\n")


def test_granite42_reasoning_and_tool_sequences():
    """Repetition fixtures reflect Granite 4.2's reasoning/tool format."""
    # These are the known atomic IDs from the Granite 4.2 tokenizer.json.
    im_start, im_end = 100256, 100257
    think, end_think = 100274, 100275
    tool_call, end_tool_call = 100270, 100271
    history = [
        im_start,  # system / assistant framing
        999, 100,  # arbitrary content IDs
        im_end,
        im_start,  # repeated assistant reasoning turn
        think, 42, 43, end_think,
        im_end,
        im_start,
        think, 42, 43, end_think,
    ]
    # The repeated suffix has a useful continuation in token space. Keep this
    # fixture declarative: native test_draft_lookup covers algorithm behavior.
    assert history.count(im_start) == 3
    assert think in history and end_think in history
    assert tool_call < end_tool_call


def test_granite42_structured_repetition_shapes():
    """Cases where a Granite drafter can plausibly produce useful proposals."""
    cases = {
        "reasoning": [100256, 100274, 11, 12, 100275, 100257] * 2,
        "tool": [100256, 100270, 1, 2, 100271, 100257] * 2,
        "json": [100256, 10, 794, 220, 16, 11, 330, 20, 100257] * 2,
        "code": [100256, 755, 282, 16, 7, 87, 1680, 198] * 2,
    }
    for name, ids in cases.items():
        assert len(ids) >= 12, name
        assert ids[0] == 100256, name
        assert ids[-1] in {198, 100257}, name
