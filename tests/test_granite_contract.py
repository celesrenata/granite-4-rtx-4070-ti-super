"""Hermetic tests for the pre-kernel Granite reference contract.

No CUDA, no network, no real checkpoint: every fixture is a synthetic,
already-parsed JSON mapping. The validator under test is stdlib-only by
design and never opens files.

Covers:
  * synthetic pass - a fully consistent config/tokenizer mapping yields a
    clean ContractResult;
  * missing evidence - absent files, absent required fields, absent evidence
    keys are all reported (never silently treated as verified);
  * mismatches - wrong values, wrong types, invalid shapes, broken
    structural relations, optional-field bad shapes, cross-check conflicts.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import pytest

from reference.granite import (
    PLANNING,
    SCHEMA,
    VERIFIED_CHECKPOINT,
    ContractResult,
    FieldResult,
    validate_checkpoint,
)
from reference.granite import contract as contract_mod

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# A config that matches the planning spec and carries all evidence metadata.
def make_config() -> dict[str, Any]:
    return {
        "architectures": ["GraniteMoeModel"],
        "model_type": "granitemoe",
        "hidden_size": PLANNING["hidden_size"],
        "num_hidden_layers": PLANNING["num_hidden_layers"],
        "intermediate_size": PLANNING["intermediate_size"],
        "num_attention_heads": PLANNING["num_attention_heads"],
        "num_key_value_heads": PLANNING["num_key_value_heads"],
        "head_dim": PLANNING["head_dim"],
        "vocab_size": PLANNING["vocab_size"],
        "max_position_embeddings": PLANNING["max_position_embeddings"],
        # Optional fields, supplied with valid shapes.
        "attention_bias": False,
        "attention_dropout": 0.0,
        "sliding_window": 4096,
        "rope_theta": 10000.0,
        "rope_scaling": {"type": "linear", "factor": 4.0},
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }


def make_tokenizer() -> dict[str, Any]:
    return {
        "eos_token_id": 1,
        "bos_token_id": 2,
        "pad_token_id": 0,
        "chat_template": "{% for message in messages %}...",
    }


PASS_CONFIG = make_config()
PASS_TOKENIZER = make_tokenizer()


def field_by(result: ContractResult, field: str) -> FieldResult:
    matches = [f for f in result.fields if f.field == field]
    assert len(matches) == 1, f"expected exactly one result for {field!r}, got {len(matches)}"
    return matches[0]


def statuses_of(result: ContractResult, field: str) -> list[str]:
    return [f.status for f in result.fields if f.field == field]


# ---------------------------------------------------------------------------
# Synthetic pass
# ---------------------------------------------------------------------------

def test_pass_clean_result() -> None:
    result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    assert isinstance(result, ContractResult)
    assert result.ok, f"expected clean pass, got mismatches: {result.mismatches}"
    assert result.mismatches == ()
    assert result.missing_evidence == ()


def test_pass_required_fields_reported_verified() -> None:
    result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    for name, expected in PLANNING.items():
        fr = field_by(result, f"config.{name}")
        assert fr.status == "verified"
        assert fr.expected == expected
        assert fr.actual == expected
    for field in (
        "evidence.config.json",
        "evidence.tokenizer_config.json",
        "config.architectures",
        "config.model_type",
        "tokenizer.chat_template",
        "tokenizer.eos_token_id",
    ):
        assert field_by(result, field).status == "verified", field


def test_pass_optional_supplied_shapes() -> None:
    result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    assert field_by(result, "config.attention_bias").status == "verified"
    assert field_by(result, "config.sliding_window").status == "verified"
    assert field_by(result, "config.rope_theta").status == "verified"
    assert field_by(result, "config.rope_scaling").status == "verified"
    assert field_by(result, "config.rms_norm_eps").status == "verified"
    assert field_by(result, "config.tie_word_embeddings").status == "verified"
    assert field_by(result, "tokenizer.bos_token_id").status == "verified"
    # Consistency relations hold for the synthetic spec.
    assert field_by(result, "consistency.hidden_div_heads").status == "verified"
    assert field_by(result,
                    "consistency.heads_div_kv_heads").status == "verified"
    assert field_by(result,
                    "consistency.heads_x_head_dim_equals_hidden").status == "verified"


def test_to_dict_roundtrip_and_schema() -> None:
    result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    data = result.to_dict()
    assert data["schema"] == SCHEMA
    assert data["ok"] is True
    assert json.loads(json.dumps(data))["mismatches"] == []
    assert all(f["field"] for f in data["fields"])


# ---------------------------------------------------------------------------
# Missing evidence - always reported, never silently verified
# ---------------------------------------------------------------------------

def test_missing_config_contents() -> None:
    result = validate_checkpoint(None, PASS_TOKENIZER)
    assert not result.ok
    assert field_by(result, "evidence.config.json").status == "missing"
    assert "config.json (not supplied)" in result.missing_evidence
    # Every required architecture field is reported missing.
    for name in PLANNING:
        assert field_by(result, f"config.{name}").status == "missing"


def test_missing_tokenizer_contents() -> None:
    result = validate_checkpoint(PASS_CONFIG, None)
    assert field_by(result, "evidence.tokenizer_config.json").status == "missing"
    assert field_by(result, "tokenizer.chat_template").status == "missing"
    assert field_by(result, "tokenizer.eos_token_id").status == "missing"
    assert "chat template (tokenizer.chat_template) (missing key)" in result.missing_evidence


def test_missing_required_field_is_reported() -> None:
    cfg = make_config()
    del cfg["head_dim"]
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    fr = field_by(result, "config.head_dim")
    assert fr.status == "missing"
    assert fr.expected == PLANNING["head_dim"]
    assert "head_dim" in str(result.missing_evidence) or not result.ok
    # The relation check involving head_dim is not silently invented:
    # it is emitted only on present integer inputs, so it is absent here.
    assert "consistency.heads_x_head_dim_equals_hidden" not in [
        f.field for f in result.fields]


def test_missing_evidence_keys_not_treated_as_verified() -> None:
    cfg = make_config()
    del cfg["architectures"]
    tok = make_tokenizer()
    del tok["eos_token_id"]
    result = validate_checkpoint(cfg, tok)
    assert not result.ok
    assert field_by(result, "config.architectures").status == "missing"
    assert field_by(result, "tokenizer.eos_token_id").status == "missing"
    assert any("model implementation" in m for m in result.missing_evidence)
    assert any("eos token id" in m for m in result.missing_evidence)


def test_optional_field_absent_is_fine() -> None:
    cfg = make_config()
    for name in ("attention_bias", "rope_theta", "sliding_window",
                 "rope_scaling", "rms_norm_eps", "tie_word_embeddings",
                 "attention_dropout"):
        cfg.pop(name, None)
    tok = make_tokenizer()
    tok.pop("bos_token_id", None)
    tok.pop("pad_token_id", None)
    result = validate_checkpoint(cfg, tok)
    assert result.ok, f"optional absences must not fail: {result.mismatches}"
    assert field_by(result, "config.rope_theta").status == "absent"
    assert field_by(result, "tokenizer.bos_token_id").status == "absent"


# ---------------------------------------------------------------------------
# Mismatches
# ---------------------------------------------------------------------------

def test_mismatch_value() -> None:
    cfg = make_config()
    cfg["num_hidden_layers"] = 38
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    fr = field_by(result, "config.num_hidden_layers")
    assert fr.status == "mismatch"
    assert fr.expected == 40
    assert fr.actual == 38
    assert not result.ok


def test_mismatch_multiple_values_all_reported() -> None:
    cfg = make_config()
    cfg["hidden_size"] = 2048
    cfg["vocab_size"] = 64000
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    bad = {f.field for f in result.mismatches}
    assert "config.hidden_size" in bad
    assert "config.vocab_size" in bad
    # Other required fields remain verified - only deviations are listed.
    assert field_by(result, "config.num_hidden_layers").status == "verified"


def test_type_error_value() -> None:
    cfg = make_config()
    cfg["hidden_size"] = "4096"  # string, not integer
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    fr = field_by(result, "config.hidden_size")
    assert fr.status == "type_error"
    assert not result.ok


def test_boolean_is_not_an_integer() -> None:
    cfg = make_config()
    cfg["vocab_size"] = True
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    assert field_by(result, "config.vocab_size").status == "type_error"


def test_evidence_invalid_shape() -> None:
    cfg = make_config()
    cfg["architectures"] = ["GraniteMoeModel", 42]  # non-string entry
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    assert field_by(result, "config.architectures").status == "invalid"
    assert "model implementation (config.architectures) (present, invalid shape)" in result.missing_evidence


def test_empty_string_evidence_is_missing() -> None:
    tok = make_tokenizer()
    tok["chat_template"] = "   "
    result = validate_checkpoint(PASS_CONFIG, tok)
    assert field_by(result, "tokenizer.chat_template").status == "invalid"


def test_token_id_wrong_shape() -> None:
    tok = make_tokenizer()
    tok["eos_token_id"] = "eos"
    result = validate_checkpoint(PASS_CONFIG, tok)
    assert field_by(result, "tokenizer.eos_token_id").status == "invalid"
    assert not result.ok


def test_top_level_non_object_reported() -> None:
    result = validate_checkpoint([1, 2, 3], PASS_TOKENIZER)  # type: ignore[arg-type]  # list, not mapping
    assert not result.ok
    assert field_by(result, "evidence.config.json").status == "invalid"
    assert field_by(result, "config.num_hidden_layers").status == "missing"


def test_consistency_inconsistency_reported() -> None:
    cfg = make_config()
    cfg["num_attention_heads"] = 31  # 4096 % 31 != 0
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    bad = {f.field for f in result.mismatches}
    assert "config.num_attention_heads" in bad
    assert "consistency.hidden_div_heads" in bad
    assert field_by(result, "consistency.hidden_div_heads").status == "inconsistent"


def test_optional_field_bad_shape() -> None:
    cfg = make_config()
    cfg["rope_theta"] = -1.0  # not a positive number
    cfg["sliding_window"] = "4096"  # string
    cfg["rope_scaling"] = {"type": ""}  # empty string type
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    assert field_by(result, "config.rope_theta").status == "type_error"
    assert field_by(result, "config.sliding_window").status == "type_error"
    assert field_by(result, "config.rope_scaling").status == "verified"
    assert field_by(result, "config.rope_scaling.type").status == "type_error"
    assert not result.ok


def test_optional_field_absent_not_reported_bad() -> None:
    cfg = make_config()
    cfg.pop("attention_dropout")
    result = validate_checkpoint(cfg, PASS_TOKENIZER)
    assert field_by(result, "config.attention_dropout").status == "absent"
    assert "config.attention_dropout" not in [f.field for f in result.mismatches]


def test_inputs_not_mutated() -> None:
    cfg = make_config()
    tok = make_tokenizer()
    cfg_before = copy.deepcopy(cfg)
    tok_before = copy.deepcopy(tok)
    validate_checkpoint(cfg, tok)
    assert cfg == cfg_before
    assert tok == tok_before


def test_planning_and_verified_stay_separate() -> None:
    # Planning constants are expectations; the verified map starts empty and
    # validation never writes into it.
    assert VERIFIED_CHECKPOINT == {}
    before_planning = dict(PLANNING)
    validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    assert PLANNING == before_planning
    assert contract_mod.VERIFIED_CHECKPOINT == {}


def test_cross_check_reports_conflict_when_verified_values_recorded() -> None:
    contract_mod.VERIFIED_CHECKPOINT.clear()
    contract_mod.VERIFIED_CHECKPOINT.update({"hidden_size": 4096, "num_hidden_layers": 42})
    try:
        result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
        assert field_by(result, "cross.hidden_size").status == "verified"
        assert field_by(result, "cross.num_hidden_layers").status == "cross_conflict"
        assert not result.ok
        assert any("cross" in f.scope for f in result.mismatches)
    finally:
        contract_mod.VERIFIED_CHECKPOINT.clear()


def test_field_result_is_frozen_and_bounded() -> None:
    fr = FieldResult("config.hidden_size", "architecture", "verified",
                     4096, 4096, "matches the planning spec")
    # Frozen dataclass: attribute assignment raises at runtime.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(fr, "field", "x")
    # Expected/actual are small scalars; the report never carries raw
    # tokenizer contents.
    result = validate_checkpoint(PASS_CONFIG, PASS_TOKENIZER)
    dumped = json.dumps(result.to_dict())
    assert "{% for message in messages %}" not in dumped
