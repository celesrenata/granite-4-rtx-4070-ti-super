"""Model constants of the target model, Granite 4.2 8B.

Single source of truth on the Python side. Must stay in sync with
engine/src/model_config.h - tests/test_config.py cross-checks the two.
These values are the target model's spec; the weight-loading phase (P1)
re-validates them against the checkpoint's config.json before doing
anything else and rejects the checkpoint on mismatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_context_length: int


EXPECTED = ModelConfig(
    num_layers=40,
    hidden_size=4096,
    intermediate_size=12800,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    vocab_size=100352,
    max_context_length=131072,
)

MODEL_ID = "granite-4.2-8b"
GPU_TARGET = "sm_89"  # NVIDIA RTX 4070 Ti Super (Ada, compute capability 8.9)


class ConfigError(ValueError):
    """A ModelConfig does not match the Granite 4.2 8B spec (or is structurally inconsistent)."""


def architecture_dict() -> dict[str, int]:
    """The constants as a plain dict, in field order (for API responses)."""
    return asdict(EXPECTED)


def validate(cfg: ModelConfig) -> None:
    """Raise ConfigError listing every deviation from the Granite 4.2 8B spec."""
    exp = asdict(EXPECTED)
    got = asdict(cfg)
    issues = [
        f"{field}: expected {exp[field]}, got {got[field]}"
        for field in exp
        if exp[field] != got[field]
    ]
    if cfg.num_key_value_heads <= 0:
        issues.append("num_key_value_heads must be positive")
    elif cfg.num_attention_heads % cfg.num_key_value_heads != 0:
        issues.append("num_attention_heads must be a multiple of num_key_value_heads")
    if cfg.num_attention_heads * cfg.head_dim != cfg.hidden_size:
        issues.append("num_attention_heads * head_dim must equal hidden_size")
    if issues:
        raise ConfigError("; ".join(issues))
