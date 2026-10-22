"""Model constants and config validation. No CUDA, no GPU, no C++ build
required — the C++ side is checked by parsing the header as text."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from frontend import config as fc

ROOT = Path(__file__).resolve().parent.parent

EXPECTED_VALUES = {
    "num_layers": 40,
    "hidden_size": 4096,
    "intermediate_size": 12800,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 100352,
    "max_context_length": 131072,
}


def test_expected_constants():
    assert fc.EXPECTED == fc.ModelConfig(**EXPECTED_VALUES)


def test_architecture_dict():
    assert fc.architecture_dict() == EXPECTED_VALUES


def test_validate_accepts_expected():
    fc.validate(fc.EXPECTED)  # must not raise


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("num_layers", 32),
        ("hidden_size", 8192),
        ("intermediate_size", 10240),
        ("num_attention_heads", 28),
        ("num_key_value_heads", 4),
        ("head_dim", 64),
        ("vocab_size", 65536),
        ("max_context_length", 32768),
    ],
)
def test_validate_rejects_single_field_deviation(field: str, bad: int):
    kwargs = dict(EXPECTED_VALUES)
    kwargs[field] = bad
    with pytest.raises(fc.ConfigError, match=re.escape(field)):
        fc.validate(fc.ModelConfig(**kwargs))


def test_validate_rejects_multiple_deviations():
    kwargs = dict(EXPECTED_VALUES)
    kwargs["num_layers"] = 32
    kwargs["vocab_size"] = 1
    with pytest.raises(fc.ConfigError) as excinfo:
        fc.validate(fc.ModelConfig(**kwargs))
    assert "num_layers" in str(excinfo.value)
    assert "vocab_size" in str(excinfo.value)


def test_validate_rejects_gqa_ratio_mismatch():
    # 32 attention heads with 5 KV heads is structurally invalid (not a
    # multiple) and also off-spec.
    kwargs = dict(EXPECTED_VALUES)
    kwargs["num_key_value_heads"] = 5
    with pytest.raises(fc.ConfigError):
        fc.validate(fc.ModelConfig(**kwargs))


def test_validate_rejects_head_dim_inconsistency():
    # heads*head_dim != hidden_size, even if every field is otherwise
    # plausible.
    kwargs = dict(EXPECTED_VALUES)
    kwargs["head_dim"] = 120
    with pytest.raises(fc.ConfigError, match="head_dim"):
        fc.validate(fc.ModelConfig(**kwargs))


def test_model_id_and_gpu_target():
    assert fc.MODEL_ID == "granite-4.2-8b"
    assert fc.GPU_TARGET == "sm_89"


# ---------------------------------------------------------------------------
# Cross-language consistency: the C++ header must hold the same constants
# (parsed as text so no C++ build or GPU is needed).
# ---------------------------------------------------------------------------

def _cpp_value(header: str, name: str) -> int:
    m = re.search(rf"\b{name}\s*=\s*(\d+)", header)
    assert m, f"{name} not found in engine/src/model_config.h"
    return int(m.group(1))


def test_cpp_header_matches_python():
    header = (ROOT / "engine" / "src" / "model_config.h").read_text(encoding="utf-8")
    for field, expected in EXPECTED_VALUES.items():
        assert _cpp_value(header, field) == expected, field
    assert 'kModelId[] = "granite-4.2-8b"' in header


def test_cpp_header_targets_sm89():
    header = (ROOT / "engine" / "src" / "model_config.h").read_text(encoding="utf-8")
    assert "kCudaArchMajor = 8" in header
    assert "kCudaArchMinor = 9" in header
    assert 'kGpuTarget[] = "sm_89"' in header


def test_cmake_targets_arch_89():
    cmake = (ROOT / "engine" / "CMakeLists.txt").read_text(encoding="utf-8")
    m = re.search(r"set\(CMAKE_CUDA_ARCHITECTURES\s+(\S+)\)", cmake)
    assert m, "CMAKE_CUDA_ARCHITECTURES not set"
    # Explicit arch only: the value must be plain "89" with no 'native'
    # flag and no list of other arches - builds are never retargeted.
    assert m.group(1) == "89"
