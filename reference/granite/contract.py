"""Pre-kernel reference contract for the Granite 4.2 8B checkpoint.

Stdlib-only. This module defines what must be checked against the real
checkpoint before any kernel is built (phase P1):

* required architecture fields, compared against the planning spec;
* evidence metadata - presence of ``config.json``, ``tokenizer_config.json``,
  the model implementation identifiers, the chat template, and token IDs;
* optional attention / scaling / norm fields, checked only when supplied.

Key invariants:

* The validator accepts **already-parsed** JSON mappings only. It never opens
  files, never touches the network, and never reads model weights.
* Planning constants (:data:`PLANNING`) describe the *target spec* and are
  kept strictly separate from verified checkpoint values
  (:data:`VERIFIED_CHECKPOINT`), which start empty and are to be filled later
  by an independent checkpoint inspection.
* A field is reported as ``verified`` only when evidence was actually
  supplied and checked. Missing or invalid evidence is always reported (see
  ``ContractResult.missing_evidence``) and is never silently treated as
  verified.
* Raw tokenizer contents (chat template text, vocabularies, added-token
  strings) are never echoed in the report; only structural facts and small
  scalar values (architecture integers, token IDs) are.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Callable

SCHEMA = "granite4.reference-contract.v1"

# ---------------------------------------------------------------------------
# Planning constants - the target model's documented spec (Granite 4.2 8B).
# Source: docs/architecture.md, frontend/config.py, engine/src/model_config.h.
# These are *expectations*, not observations: a checkpoint that matches them
# is consistent with the plan, but matching alone is not independently
# "verified" in the sense of :data:`VERIFIED_CHECKPOINT`.
# ---------------------------------------------------------------------------
PLANNING: dict[str, int] = {
    "num_hidden_layers": 40,
    "hidden_size": 4096,
    "intermediate_size": 12800,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 100352,
    "max_position_embeddings": 131072,
}

# ---------------------------------------------------------------------------
# Verified checkpoint values - observations from the real checkpoint, kept
# strictly separate from the planning constants above. Intentionally empty
# until an independent checkpoint inspection (phase P1) records values here.
# The validator uses this mapping only for cross-checks; an empty mapping
# means "no cross-checks", not "everything verified".
# ---------------------------------------------------------------------------
VERIFIED_CHECKPOINT: dict[str, int] = {}

# Field statuses.
STATUS_VERIFIED = "verified"      # evidence supplied and it passed its check
STATUS_MISSING = "missing"        # required evidence or value not supplied
STATUS_MISMATCH = "mismatch"     # right type, wrong value (vs planning spec)
STATUS_TYPE_ERROR = "type_error"  # supplied, wrong JSON type/shape
STATUS_INVALID = "invalid"       # evidence present but not a JSON object
STATUS_INCONSISTENT = "inconsistent"  # structural relation violated
STATUS_ABSENT = "absent"         # optional field not supplied (not a problem)
STATUS_CROSS_CONFLICT = "cross_conflict"  # conflicts with VERIFIED_CHECKPOINT

BAD_STATUSES = frozenset(
    {
        STATUS_MISSING,
        STATUS_MISMATCH,
        STATUS_TYPE_ERROR,
        STATUS_INVALID,
        STATUS_INCONSISTENT,
        STATUS_CROSS_CONFLICT,
    }
)


@dataclass(frozen=True)
class FieldResult:
    """Outcome for one contracted field.

    ``field`` is a dotted path (e.g. ``config.num_hidden_layers``), ``scope``
    is one of ``architecture`` / ``evidence`` / ``consistency`` /
    ``optional.<group>`` / ``cross-check``, and ``expected`` / ``actual`` are
    bounded values - evidence-scope results never carry raw file contents.
    """

    field: str
    scope: str
    status: str
    expected: Any
    actual: Any
    detail: str


@dataclass(frozen=True)
class ContractResult:
    """Structured outcome of :func:`validate_checkpoint`."""

    fields: tuple[FieldResult, ...]
    missing_evidence: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True only when no field is missing, mistyped, or off-spec."""
        return all(f.status not in BAD_STATUSES for f in self.fields)

    @property
    def mismatches(self) -> tuple[FieldResult, ...]:
        """Every field that failed its check (all deviations, not just the first)."""
        return tuple(f for f in self.fields if f.status in BAD_STATUSES)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (stdlib ``json`` round-trip safe)."""
        return {
            "schema": SCHEMA,
            "ok": self.ok,
            "mismatches": [asdict(f) for f in self.mismatches],
            "missing_evidence": list(self.missing_evidence),
            "fields": [asdict(f) for f in self.fields],
        }


# ---------------------------------------------------------------------------
# Shape predicates (strict: JSON booleans are *not* integers).
# ---------------------------------------------------------------------------

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


def _is_positive_number(value: Any) -> bool:
    return _is_number(value) and value > 0


def _is_non_negative_int(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_unit_interval(value: Any) -> bool:
    return _is_number(value) and 0.0 <= value <= 1.0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_architectures(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_nonempty_str(v) for v in value)
    )


def _is_token_ids(value: Any) -> bool:
    if _is_non_negative_int(value):
        return True
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(_is_non_negative_int(v) for v in value)
    )


# Optional attention / scaling / norm fields in config.json. Checked only
# when supplied; when supplied, the shape must satisfy the predicate.
_OPTIONAL_FIELDS: tuple[tuple[str, str, Callable[[Any], bool]], ...] = (
    ("attention_bias", "attention", _is_bool),
    ("attention_dropout", "attention", _is_unit_interval),
    ("sliding_window", "attention", lambda v: _is_int(v) and v > 0),
    ("rope_theta", "scaling", _is_positive_number),
    ("rope_scaling", "scaling", lambda v: isinstance(v, Mapping)),
    ("rms_norm_eps", "norm", _is_positive_number),
    ("tie_word_embeddings", "other", _is_bool),
)

# Optional token IDs in tokenizer_config.json, checked only when supplied.
_OPTIONAL_TOKEN_IDS = ("bos_token_id", "pad_token_id")


def _int_or_none(mapping: Mapping[str, Any], name: str) -> int | None:
    value = mapping.get(name)
    return value if _is_int(value) else None


def _file_evidence(
    field: str,
    label: str,
    supplied: Any,
    parsed: Mapping[str, Any] | None,
    fields: list[FieldResult],
    missing: list[str],
) -> None:
    """Evidence that a config/tokenizer file's parsed contents exist."""
    if supplied is None:
        fields.append(FieldResult(
            field, "evidence", STATUS_MISSING, "JSON object", None,
            f"{label} not supplied"))
        missing.append(f"{label} (not supplied)")
    elif parsed is None:
        fields.append(FieldResult(
            field, "evidence", STATUS_INVALID, "JSON object", None,
            f"{label} supplied but not a JSON object"))
        missing.append(f"{label} (present, invalid shape)")
    else:
        fields.append(FieldResult(
            field, "evidence", STATUS_VERIFIED, "JSON object", None,
            f"{label} supplied"))


def _evidence_key(
    mapping: Mapping[str, Any],
    field: str,
    label: str,
    predicate: Callable[[Any], bool],
    fields: list[FieldResult],
    missing: list[str],
) -> None:
    """Evidence that a key is present with a valid shape. Values are not echoed."""
    key = field.rsplit(".", 1)[1]
    if key not in mapping:
        fields.append(FieldResult(
            field, "evidence", STATUS_MISSING, None, None,
            f"{label}: key not present"))
        missing.append(f"{label} (missing key)")
    elif predicate(mapping[key]):
        fields.append(FieldResult(
            field, "evidence", STATUS_VERIFIED, None, None,
            f"{label}: present, shape ok (value not echoed)"))
    else:
        fields.append(FieldResult(
            field, "evidence", STATUS_INVALID, None, None,
            f"{label}: present, but does not satisfy the expected shape"))
        missing.append(f"{label} (present, invalid shape)")


def _required_field(
    cfg: Mapping[str, Any],
    name: str,
    expected: int,
    fields: list[FieldResult],
) -> None:
    """Required architecture field vs the planning spec (all deviations reported)."""
    field = f"config.{name}"
    if name not in cfg:
        fields.append(FieldResult(
            field, "architecture", STATUS_MISSING, expected, None,
            "not present in config.json"))
        return
    value = cfg[name]
    if not _is_int(value):
        fields.append(FieldResult(
            field, "architecture", STATUS_TYPE_ERROR, expected, value,
            f"expected an integer, got {type(value).__name__}"))
    elif value != expected:
        fields.append(FieldResult(
            field, "architecture", STATUS_MISMATCH, expected, value,
            f"expected {expected} (planning spec), got {value}"))
    else:
        fields.append(FieldResult(
            field, "architecture", STATUS_VERIFIED, expected, value,
            "matches the planning spec"))


def _optional_key(
    mapping: Mapping[str, Any],
    field: str,
    group: str,
    predicate: Callable[[Any], bool],
    fields: list[FieldResult],
) -> None:
    """Optional field: absent is fine; when supplied the shape must check out."""
    key = field.rsplit(".", 1)[1]
    scope = f"optional.{group}"
    if key not in mapping:
        fields.append(FieldResult(
            field, scope, STATUS_ABSENT, None, None,
            "not supplied; not required"))
        return
    value = mapping[key]
    if predicate(value):
        fields.append(FieldResult(
            field, scope, STATUS_VERIFIED, None, value,
            f"supplied; shape checks out ({group})"))
    else:
        fields.append(FieldResult(
            field, scope, STATUS_TYPE_ERROR, None, value,
            f"supplied; does not satisfy the expected shape ({group})"))


def _consistency(cfg: Mapping[str, Any], fields: list[FieldResult]) -> None:
    """Structural relations among the architecture integers.

    Each check is emitted only when the fields it involves are present and
    integer-typed; missing or mistyped inputs are already reported by the
    required-field checks, so nothing is double-counted.
    """
    hidden = _int_or_none(cfg, "hidden_size")
    heads = _int_or_none(cfg, "num_attention_heads")
    kv_heads = _int_or_none(cfg, "num_key_value_heads")
    head_dim = _int_or_none(cfg, "head_dim")

    if hidden is not None and heads is not None:
        ok = hidden % heads == 0
        fields.append(FieldResult(
            "consistency.hidden_div_heads", "consistency",
            STATUS_VERIFIED if ok else STATUS_INCONSISTENT,
            None, None,
            f"hidden_size {hidden} "
            + ("divides evenly by" if ok else "is not divisible by")
            + f" num_attention_heads {heads}"))
    if heads is not None and kv_heads is not None:
        ok = heads % kv_heads == 0
        fields.append(FieldResult(
            "consistency.heads_div_kv_heads", "consistency",
            STATUS_VERIFIED if ok else STATUS_INCONSISTENT,
            None, None,
            f"num_attention_heads {heads} "
            + ("divides evenly by" if ok else "is not divisible by")
            + f" num_key_value_heads {kv_heads}"))
    if hidden is not None and heads is not None and head_dim is not None:
        ok = heads * head_dim == hidden
        fields.append(FieldResult(
            "consistency.heads_x_head_dim_equals_hidden", "consistency",
            STATUS_VERIFIED if ok else STATUS_INCONSISTENT,
            None, None,
            f"num_attention_heads * head_dim "
            + ("==" if ok else "!=")
            + f" hidden_size ({heads} * {head_dim} vs {hidden})"))


def _cross_check(
    cfg: Mapping[str, Any],
    fields: list[FieldResult],
) -> None:
    """Cross-check config values against VERIFIED_CHECKPOINT, when recorded.

    A missing or mistyped config value is not double-reported here; it was
    already reported by the required-field checks.
    """
    for name, verified_value in VERIFIED_CHECKPOINT.items():
        actual = cfg.get(name)
        if not _is_int(actual):
            continue
        if actual == verified_value:
            fields.append(FieldResult(
                f"cross.{name}", "cross-check", STATUS_VERIFIED,
                verified_value, actual,
                "matches previously verified checkpoint value"))
        else:
            fields.append(FieldResult(
                f"cross.{name}", "cross-check", STATUS_CROSS_CONFLICT,
                verified_value, actual,
                f"checkpoint value {actual} conflicts with previously "
                f"verified value {verified_value}"))


def validate_checkpoint(config: Any, tokenizer: Any) -> ContractResult:
    """Validate parsed checkpoint JSON mappings against the reference contract.

    ``config`` / ``tokenizer`` are the *parsed* contents of
    ``config.json`` / ``tokenizer_config.json``. A value of ``None`` means the
    contents were not supplied; a non-mapping value means the contents were
    not a JSON object. Both are reported as evidence problems - the validator
    returns structured mismatches and never raises on bad input.

    The validator is pure: it never mutates its inputs, never touches files,
    and never records into :data:`VERIFIED_CHECKPOINT`.
    """
    fields: list[FieldResult] = []
    missing_evidence: list[str] = []

    cfg: Mapping[str, Any] | None
    tok: Mapping[str, Any] | None
    if isinstance(config, Mapping):
        cfg = config
    else:
        cfg = None
    if isinstance(tokenizer, Mapping):
        tok = tokenizer
    else:
        tok = None

    _file_evidence(
        "evidence.config.json", "config.json", config, cfg,
        fields, missing_evidence)
    _file_evidence(
        "evidence.tokenizer_config.json", "tokenizer_config.json", tokenizer, tok,
        fields, missing_evidence)

    cfg_m: Mapping[str, Any] = cfg if cfg is not None else {}
    tok_m: Mapping[str, Any] = tok if tok is not None else {}

    # Evidence metadata: model implementation, chat template, token IDs.
    _evidence_key(
        cfg_m, "config.architectures",
        "model implementation (config.architectures)", _is_architectures,
        fields, missing_evidence)
    _evidence_key(
        cfg_m, "config.model_type",
        "model type (config.model_type)", _is_nonempty_str,
        fields, missing_evidence)
    _evidence_key(
        tok_m, "tokenizer.chat_template",
        "chat template (tokenizer.chat_template)", _is_nonempty_str,
        fields, missing_evidence)
    _evidence_key(
        tok_m, "tokenizer.eos_token_id",
        "eos token id (tokenizer.eos_token_id)", _is_token_ids,
        fields, missing_evidence)
    for name in _OPTIONAL_TOKEN_IDS:
        _optional_key(tok_m, f"tokenizer.{name}", "token-ids", _is_token_ids, fields)

    # Required architecture fields vs the planning spec.
    for name, expected in PLANNING.items():
        _required_field(cfg_m, name, expected, fields)

    _consistency(cfg_m, fields)

    # Optional attention / scaling / norm fields, checked only when supplied.
    for name, group, predicate in _OPTIONAL_FIELDS:
        _optional_key(cfg_m, f"config.{name}", group, predicate, fields)
        if name == "rope_scaling" and isinstance(cfg_m.get(name), Mapping):
            scaling = cfg_m[name]
            if "type" in scaling and not _is_nonempty_str(scaling["type"]):
                fields.append(FieldResult(
                    "config.rope_scaling.type", "optional.scaling",
                    STATUS_TYPE_ERROR, "non-empty string", scaling["type"],
                    "rope_scaling.type supplied but not a non-empty string"))

    _cross_check(cfg_m, fields)

    return ContractResult(tuple(fields), tuple(missing_evidence))
