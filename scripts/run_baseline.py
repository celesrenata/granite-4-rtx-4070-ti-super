#!/usr/bin/env python
"""Phase-0 baseline runner (stdlib only; no CUDA, no network).

Reads a JSON manifest that declares explicit `argv` commands to measure
and, *before running anything*, verifies:

  * the required checkpoint path points at an existing local directory
    containing `config.json` and `tokenizer_config.json`;
  * every declared prerequisite entry exists locally;
  * no required string is a placeholder (see `_is_placeholder`).

If verification fails, the runner executes nothing, writes a record
stating that measurements were not run, and exits nonzero.

When verification passes, each enabled command is executed via
`subprocess.run` (no shell) and the record stores, per command: the
declared argv, exit code, start/finish times and wall duration, a bounded
tail of combined output (total byte count and truncation flag), and any
timeout flag. Disabled commands are recorded as skipped and never run.

The runner records raw facts only. It never derives, estimates, or
fabricates performance claims about the model; the record carries an
explicit `performance_claims: "none"` marker.

Manifest schema (illustrative: benchmarks/baseline.example.json):
  {
    "schema": "granite4.baseline-manifest.v1",
    "required_checkpoint_path": "path/to/checkpoint",
    "required_entries": ["path/a", "path/b"],
    "record_path": "benchmarks/baseline.record.json",
    "timeout_seconds": 300,
    "environment_to_record": [],
    "commands": [
      {"name": "label", "enabled": true,
       "argv": ["python", "scripts/..."]}
    ]
  }

Exit codes:
  0  record written, all executed commands exited 0
  1  record written, one or more executed commands exited nonzero
  2  verification failed (placeholder / missing prerequisites) - no
     command was executed; the record states measurements were not run
  3  record could not be written
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "granite4.baseline-record.v1"
MANIFEST_SCHEMA = "granite4.baseline-manifest.v1"

MAX_TAIL_BYTES = 2048
MAX_ENV_VALUE_CHARS = 256

PLACEHOLDER_MARKERS = (
    "placeholder",
    "unset",
    "todo",
    "tbd",
    "fixme",
    "changeme",
    "xxxx",
    "replace-me",
)

EXIT_OK = 0
EXIT_COMMAND_FAILED = 1
EXIT_REFUSED = 2
EXIT_RECORD_FAILED = 3

NO_CLAIMS_NOTE = (
    "none: this record stores raw command measurements only. No "
    "performance, quality, or model behavior claims are derived, "
    "estimated, or fabricated from it."
)


class Refusal(Exception):
    """Verification failed; no command may be executed."""


def _is_placeholder(value: Any) -> tuple[bool, str]:
    """Detect placeholder/missing strings. Returns (is_placeholder, reason)."""
    if not isinstance(value, str):
        return True, "value is not a string"
    if value == "":
        return True, "value is empty"
    if value.startswith("$"):
        return True, "value starts with a shell-style $ reference"
    if "$" in value and any(c.isalpha() and c.isupper() for c in value.split("$", 1)[1][:16]):
        return True, "value contains a $-interpolated reference"
    lowered = value.lower()
    for marker in PLACEHOLDER_MARKERS:
        if marker in lowered:
            return True, f"value contains placeholder marker {marker!r}"
    return False, ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail(data: bytes | None) -> tuple[int, str, bool]:
    """Bounded output tail: (total_bytes, tail_text, truncated)."""
    if data is None:
        return 0, "", False
    total = len(data)
    window = data[-MAX_TAIL_BYTES:]
    truncated = total > MAX_TAIL_BYTES
    return total, window.decode("utf-8", errors="replace"), truncated


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise Refusal(f"manifest cannot be read: {exc}")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Refusal(f"manifest is not valid JSON: {exc.__class__.__name__}")
    if not isinstance(data, dict):
        raise Refusal("manifest top-level value must be a JSON object")
    return data


def _verify_checkpoint(checkpoint_path: Path, refusals: list[str]) -> None:
    if not checkpoint_path.is_dir():
        refusals.append(f"required checkpoint path missing or not a directory: {checkpoint_path}")
        return
    for name in ("config.json", "tokenizer_config.json"):
        if not (checkpoint_path / name).is_file():
            refusals.append(f"checkpoint directory lacks {name}: {checkpoint_path}")


def verify_manifest(manifest: dict[str, Any], base_dir: Path) -> None:
    """Verify required checkpoint path, prerequisites, and placeholders.

    Raises Refusal with all discovered problems; returns None on success.
    """
    refusals: list[str] = []

    checkpoint_raw = manifest.get("required_checkpoint_path")
    if checkpoint_raw is None:
        refusals.append("manifest lacks required_checkpoint_path")
    else:
        is_ph, why = _is_placeholder(checkpoint_raw)
        if is_ph:
            refusals.append(f"required_checkpoint_path is a placeholder/missing value: {why}")
        else:
            cp = Path(str(checkpoint_raw)).expanduser()
            if not cp.is_absolute():
                cp = (base_dir / cp).resolve()
            _verify_checkpoint(cp, refusals)

    entries = manifest.get("required_entries", [])
    if not isinstance(entries, list):
        refusals.append("required_entries must be a list of paths")
        entries = []
    for entry in entries:
        is_ph, why = _is_placeholder(entry)
        if is_ph:
            refusals.append(f"required entry is a placeholder/missing value: {why}")
            continue
        p = Path(str(entry)).expanduser()
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        if not p.exists():
            refusals.append(f"required entry does not exist: {p}")

    commands = manifest.get("commands", [])
    if not isinstance(commands, list):
        refusals.append("commands must be a list")
        commands = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            refusals.append("each command must be a JSON object")
            continue
        argv = cmd.get("argv")
        if not isinstance(argv, list) or not argv:
            refusals.append(f"command {cmd.get('name', '<unnamed>')}: argv must be a non-empty list")
            continue
        for part in argv:
            is_ph, why = _is_placeholder(part)
            if is_ph:
                refusals.append(
                    f"command {cmd.get('name', '<unnamed>')}: argv element is a "
                    f"placeholder/missing value: {why}"
                )

    if refusals:
        raise Refusal("; ".join(refusals))


def _default_record_path(base_dir: Path) -> Path:
    return base_dir / "baseline.record.json"


def _resolve_record_path(manifest: dict[str, Any], base_dir: Path) -> Path:
    """Resolve the declared record path; fall back to the default if invalid."""
    raw = manifest.get("record_path", "baseline.record.json")
    if not isinstance(raw, str):
        return _default_record_path(base_dir)
    is_ph, _why = _is_placeholder(raw)
    if is_ph:
        return _default_record_path(base_dir)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def _provenance(cwd: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cwd": str(cwd),
        "argv0": sys.executable,
        "recorded_at": _now(),
        "environment": {},
        "note": (
            "environment values are recorded only for names explicitly "
            "declared in the manifest (environment_to_record); no full "
            "environment dump is written."
        ),
    }
    names = manifest.get("environment_to_record", [])
    if isinstance(names, list):
        for name in names:
            if isinstance(name, str) and name in os.environ:
                provenance["environment"][name] = os.environ[name][:MAX_ENV_VALUE_CHARS]
    return provenance


def _run_command(cmd: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    argv = [str(x) for x in cmd["argv"]]
    record: dict[str, Any] = {
        "name": str(cmd.get("name", "")),
        "argv": argv,
        "enabled": bool(cmd.get("enabled", False)),
        "skipped": False,
        "exit_code": None,
        "timed_out": False,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "output_bytes": None,
        "output_tail": None,
        "output_tail_truncated": None,
        "note": None,
    }
    if not record["enabled"]:
        record["skipped"] = True
        record["note"] = "disabled in manifest; not executed"
        return record

    started = datetime.now(timezone.utc)
    record["started_at"] = started.isoformat()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        record["timed_out"] = True
        record["note"] = f"command timed out after {timeout_seconds}s"
        total, tail, truncated = _tail(exc.output)
        record.update(
            output_bytes=total,
            output_tail=tail,
            output_tail_truncated=truncated,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        record["duration_seconds"] = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 6
        )
        return record
    except (OSError, subprocess.SubprocessError) as exc:
        record["note"] = f"command could not be executed: {exc.__class__.__name__}"
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        record["duration_seconds"] = round(
            (datetime.now(timezone.utc) - started).total_seconds(), 6
        )
        return record

    finished = datetime.now(timezone.utc)
    record["exit_code"] = proc.returncode
    record["finished_at"] = finished.isoformat()
    record["duration_seconds"] = round((finished - started).total_seconds(), 6)
    combined = proc.stdout + proc.stderr
    total, tail, truncated = _tail(combined)
    record.update(output_bytes=total, output_tail=tail, output_tail_truncated=truncated)
    return record


def run_baseline(manifest_path: Path) -> tuple[int, dict[str, Any]]:
    """Run the baseline declared by the manifest; return (exit_code, record)."""
    base_dir = manifest_path.resolve().parent
    record_path = _default_record_path(base_dir)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "status": "not_run",
        "measurements_run": False,
        "refusals": [],
        "commands": [],
        "provenance": None,
        "performance_claims": "none",
        "performance_derived": False,
        "notes": NO_CLAIMS_NOTE,
        "generated_at": _now(),
    }

    manifest: dict[str, Any] | None = None
    try:
        manifest = _load_manifest(manifest_path)
        record_path = _resolve_record_path(manifest, base_dir)
        record["manifest_schema"] = manifest.get("schema")
        declared_schema = manifest.get("schema")
        if declared_schema is not None and declared_schema != MANIFEST_SCHEMA:
            raise Refusal(
                f"manifest schema is not {MANIFEST_SCHEMA}: {declared_schema!r}"
            )
    except Refusal as exc:
        record["refusals"].append(str(exc))
        record["provenance"] = _provenance(Path.cwd(), manifest or {})
        record["statement"] = "Measurements were not run: " + "; ".join(record["refusals"])
        _write_record(record_path, record)
        return EXIT_REFUSED, record

    try:
        verify_manifest(manifest, base_dir)
    except Refusal as exc:
        record["refusals"].append(str(exc))
        record["provenance"] = _provenance(Path.cwd(), manifest)
        record["statement"] = "Measurements were not run: " + "; ".join(record["refusals"])
        _write_record(record_path, record)
        return EXIT_REFUSED, record

    commands = manifest.get("commands", [])
    timeout = float(manifest.get("timeout_seconds", 300))
    executed = 0
    failed = 0
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue
        entry = _run_command(cmd, timeout)
        record["commands"].append(entry)
        if not entry["skipped"]:
            executed += 1
            if entry["exit_code"] != 0:
                failed += 1

    record["measurements_run"] = executed > 0
    record["commands_executed"] = executed
    record["commands_failed"] = failed
    record["status"] = "completed_with_failures" if failed else "completed"
    if executed == 0:
        record["statement"] = (
            "No enabled commands were executed; no measurements were taken."
            if record["commands"]
            else "Manifest declares no commands; no measurements were taken."
        )
    else:
        record["statement"] = (
            f"Measurements were run: {executed} command(s) executed, "
            f"{failed} non-zero exit(s). Raw facts only; no performance claims derived."
        )
    record["provenance"] = _provenance(Path.cwd(), manifest)
    _write_record(record_path, record)
    return (EXIT_COMMAND_FAILED if failed else EXIT_OK), record


def _write_record(record_path: Path, record: dict[str, Any]) -> bool:
    try:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    record["record_path"] = str(record_path)
    return True


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="run_baseline",
        description=(
            "Verify prerequisites from a JSON manifest, then execute the "
            "explicitly declared argv commands and record raw facts. "
            "Stdlib only; never derives or fabricates performance claims."
        ),
    )
    parser.add_argument("manifest", help="Path to the baseline manifest JSON")
    args = parser.parse_args(argv)
    code, record = run_baseline(Path(args.manifest))
    print(json.dumps(record, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
