"""Hermetic tests for the parity gate's comparison logic (no engine, no torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import parity_gate  # noqa: E402


def test_bf16_ulps_counts_representable_steps():
    assert parity_gate.bf16_ulps(18.0, 18.0) == 0
    # bf16 spacing in [16, 32) is 0.125.
    assert parity_gate.bf16_ulps(18.0, 17.875) == 1
    assert parity_gate.bf16_ulps(18.0, 17.75) == 2
    # Across zero and between signs the order stays monotone.
    assert parity_gate.bf16_ulps(-1.0, 1.0) == parity_gate.bf16_ulps(1.0, -1.0) > 0
    assert parity_gate.bf16_ulps(0.0, -0.0) <= 1


def _row(n: int, **logits: float) -> list[float]:
    row = [0.0] * n
    for k, v in logits.items():
        row[int(k[1:])] = v
    return row


def test_teacher_forced_exact_and_tolerance():
    rows = [
        _row(8, t3=5.0),               # exact top-1
        _row(8, t1=18.0, t2=17.875),   # ref 2 is 1 ULP below the max
        _row(8, t1=18.0, t2=17.0),     # ref 2 is 8 ULP below
    ]
    rep = parity_gate.compare_teacher_forced(rows, [3, 2, 2], ulp_tolerance=2)
    assert rep["exact"] == 1 and rep["matched"] == 2
    assert [m["position"] for m in rep["misses"]] == [1, 2]
    assert rep["misses"][0]["ulps"] == 1
    strict = parity_gate.compare_teacher_forced(rows, [3, 2, 2], ulp_tolerance=0)
    assert strict["matched"] == 1


def test_teacher_forced_rejects_row_count_mismatch():
    with pytest.raises(ValueError):
        parity_gate.compare_teacher_forced([_row(4)], [1, 2], ulp_tolerance=2)


def test_shipped_reference_is_well_formed():
    ref = json.loads((Path(__file__).resolve().parent.parent /
                      "reference/granite/parity_ref.json").read_text())
    assert ref["prompt_ids"] == [791, 6864, 315, 9822, 374]
    assert len(ref["tokens"]) == 24 and all(isinstance(t, int) for t in ref["tokens"])
    assert len(ref["checkpoint_sha"]) == 16
