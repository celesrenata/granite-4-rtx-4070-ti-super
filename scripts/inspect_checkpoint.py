#!/usr/bin/env python
"""Phase-0 checkpoint characterization inspector (stdlib only, no network).

Reads a *local* checkpoint directory and prints a bounded structural
report on `config.json` and `tokenizer_config.json`. The required
Granite 4.2 8B planning fields are verified and every deviation is
reported field-by-field.

Safety guarantees:
  * Local files only: no network access, no downloads.
  * Model weight files are never opened.
  * Raw tokenizer contents (vocabularies, added-token strings, chat
    templates) are never echoed; only structural facts (presence, JSON
    validity, key counts, selected key names, bounded values) are
    reported.

Exit codes:
  0  ok           - all planning fields match the spec
  1  mismatch     - one or more planning fields are missing, mistyped,
                    or off-spec (listed in `issues`)
  2  invalid input - directory/file missing, unreadable, or not a JSON
                    object (takes precedence over mismatch)

Usage:
  python scripts/inspect_checkpoint.py CHECKPOINT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "granite4.checkpoint-inspection.v1"

REQUIRED_FIELDS: dict[str, int] = {
    "num_hidden_layers": 40,
    "hidden_size": 4096,
    "intermediate_size": 12800,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 100352,
}

# Fields not present in the checkpoint's config.json: derived from other
# planning fields, matching the C++ parser (checkpoint_main.cc).
DERIVED_FIELDS: dict[str, tuple[str, str]] = {
    "head_dim": ("hidden_size", "num_attention_heads"),
}

# Key names whose *presence* is reported in tokenizer_config.json.
# Values are never reported.
TOKENIZER_SELECTED_KEYS = (
    "tokenizer_class",
    "model_max_length",
    "bos_token",
    "eos_token",
    "pad_token",
    "added_tokens",
    "chat_template",
)

MAX_KEY_NAMES = 24
MAX_KEY_NAME_LEN = 64
MAX_SCALAR_LEN = 32

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_INVALID_INPUT = 2

NOTES = (
    "Bounded structural report. No model weights are read, no raw "
    "tokenizer contents are echoed, and no network access is used."
)


class InvalidInputError(Exception):
    """Checkpoint input is missing, unreadable, or not a JSON object."""


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object; raise InvalidInputError without echoing contents."""
    try:
        raw = path.read_bytes()
    except OSError:
        raise InvalidInputError(f"{path.name}: cannot read file")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise InvalidInputError(f"{path.name}: not valid JSON")
    if not isinstance(data, dict):
        raise InvalidInputError(f"{path.name}: top-level JSON value must be an object")
    return data


def _key_summary(data: dict[str, Any]) -> dict[str, Any]:
    """Bounded structural summary of a JSON object (key names, never values)."""
    keys = list(data.keys())
    return {
        "top_level_key_count": len(keys),
        "sample_key_names": [str(k)[:MAX_KEY_NAME_LEN] for k in keys[:MAX_KEY_NAMES]],
        "key_names_truncated": len(keys) > MAX_KEY_NAMES,
    }


def _safe_scalar(value: object) -> str:
    """Bounded, safe rendering of an unexpected value (no raw blobs)."""
    if isinstance(value, (dict, list)):
        text = f"<{type(value).__name__}>"
    else:
        text = str(value)
    return text[:MAX_SCALAR_LEN]


def _check_field(config: dict[str, Any], name: str) -> tuple[dict[str, Any], str | None]:
    """Compare one planning field against the required spec value."""
    expected = REQUIRED_FIELDS[name]
    if name in DERIVED_FIELDS and name not in config:
        return _check_derived_field(config, name)
    if name not in config:
        return (
            {"expected": expected, "actual": None, "ok": False},
            f"{name}: required planning field missing",
        )
    got = config[name]
    if isinstance(got, bool) or not isinstance(got, int):
        return (
            {"expected": expected, "actual": _safe_scalar(got), "ok": False},
            f"{name}: expected integer, got {_safe_scalar(got)} (type {type(got).__name__})",
        )
    if got != expected:
        return (
            {"expected": expected, "actual": got, "ok": False},
            f"{name}: expected {expected}, got {got}",
        )
    return {"expected": expected, "actual": got, "ok": True}, None


def _check_derived_field(
    config: dict[str, Any], name: str
) -> tuple[dict[str, Any], str | None]:
    """Check a field derived from other planning fields (absent in config).

    head_dim = hidden_size / num_attention_heads when the config omits it.
    """
    expected = REQUIRED_FIELDS[name]
    num_name, den_name = DERIVED_FIELDS[name]
    if num_name not in config or den_name not in config:
        return (
            {"expected": expected, "actual": None, "derived_from": [num_name, den_name], "ok": False},
            f"{name}: cannot derive ({num_name} and/or {den_name} missing)",
        )
    num = config[num_name]
    den = config[den_name]
    if (
        isinstance(num, bool)
        or not isinstance(num, int)
        or isinstance(den, bool)
        or not isinstance(den, int)
        or den == 0
    ):
        return (
            {"expected": expected, "actual": None, "derived_from": [num_name, den_name], "ok": False},
            f"{name}: cannot derive ({num_name}={_safe_scalar(num)}, {den_name}={_safe_scalar(den)})",
        )
    if num % den != 0:
        return (
            {"expected": expected, "actual": None, "derived_from": [num_name, den_name], "ok": False},
            f"{name}: {num_name} not divisible by {den_name}",
        )
    got = num // den
    if got != expected:
        return (
            {"expected": expected, "actual": got, "derived_from": [num_name, den_name], "ok": False},
            f"{name}: expected {expected}, got {got} (derived from {num_name}/{den_name})",
        )
    return (
        {"expected": expected, "actual": got, "derived_from": [num_name, den_name], "ok": True},
        None,
    )


def inspect_checkpoint(checkpoint_dir: Path) -> tuple[int, dict[str, Any]]:
    """Inspect a local checkpoint directory; return (exit_code, report)."""
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "checkpoint_dir": str(checkpoint_dir),
        "verdict": "invalid_input",
        "issues": [],
        "planning_fields": {},
        "config_json": {"present": False, "valid_json": False},
        "tokenizer_config_json": {
            "present": False,
            "valid_json": False,
            "contents_echoed": False,
            "selected_keys_present": {},
        },
        "weight_files_touched": False,
        "network_access": False,
        "notes": NOTES,
    }

    if not checkpoint_dir.is_dir():
        report["issues"].append("checkpoint directory is missing or not a directory")
        return EXIT_INVALID_INPUT, report

    config_path = checkpoint_dir / "config.json"
    tokenizer_path = checkpoint_dir / "tokenizer_config.json"
    report["config_json"]["present"] = config_path.is_file()
    report["tokenizer_config_json"]["present"] = tokenizer_path.is_file()

    config: dict[str, Any] | None = None
    if not report["config_json"]["present"]:
        report["issues"].append("config.json: missing")
    else:
        try:
            config = _load_json_object(config_path)
        except InvalidInputError as exc:
            report["issues"].append(str(exc))
        else:
            report["config_json"]["valid_json"] = True
            report["config_json"].update(_key_summary(config))

    if not report["tokenizer_config_json"]["present"]:
        report["issues"].append("tokenizer_config.json: missing")
    else:
        try:
            tokenizer = _load_json_object(tokenizer_path)
        except InvalidInputError as exc:
            report["issues"].append(str(exc))
        else:
            summary = report["tokenizer_config_json"]
            summary["valid_json"] = True
            summary.update(_key_summary(tokenizer))
            summary["selected_keys_present"] = {
                k: k in tokenizer for k in TOKENIZER_SELECTED_KEYS
            }

    if config is not None:
        fields: dict[str, dict[str, Any]] = {}
        for name in REQUIRED_FIELDS:
            entry, issue = _check_field(config, name)
            fields[name] = entry
            if issue is not None:
                report["issues"].append(issue)
        report["planning_fields"] = fields

    input_ok = (
        report["config_json"]["present"]
        and report["config_json"]["valid_json"]
        and report["tokenizer_config_json"]["present"]
        and report["tokenizer_config_json"]["valid_json"]
    )
    if not input_ok:
        return EXIT_INVALID_INPUT, report

    if report["issues"]:
        report["verdict"] = "mismatch"
        return EXIT_MISMATCH, report

    report["verdict"] = "ok"
    return EXIT_OK, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_checkpoint",
        description=(
            "Print a bounded structural report for a local checkpoint "
            "directory and verify the required Granite 4.2 8B planning "
            "fields. Stdlib only; no network access."
        ),
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Path to a local checkpoint directory containing config.json "
        "and tokenizer_config.json",
    )
    args = parser.parse_args(argv)
    code, report = inspect_checkpoint(Path(args.checkpoint_dir))
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
