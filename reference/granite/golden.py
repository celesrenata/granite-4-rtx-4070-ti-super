"""Phase-1 golden-reference capture/compare utility for Granite 4.2 8B.

Stdlib-only, evidence-first. This module captures *Golden* records from a
local reference adapter and compares two capture documents.

Design invariants:

* **No network, no default downloads.** The utility reads exactly two local
  inputs - a checkpoint directory and a prompts file - and imports exactly
  one caller-supplied local adapter module.
* **Prerequisites are rejected, never assumed.** A capture is refused
  (nothing is written) when the adapter module or the checkpoint files are
  missing, when the requested sampling mode is not supported by the adapter,
  or when the adapter returns incomplete evidence.
* **A record is written only for evidence that exists.** Each record
  contains the prompt id, exact input/output token IDs, the checksums and
  shapes of the selected intermediate tensors *supplied by the adapter*,
  adapter/library metadata, a UTC timestamp, and the SHA256 hashes of
  ``config.json`` / ``tokenizer_config.json``.
* **The comparator refuses absent evidence.** Token IDs and checksums are
  compared exactly; only optional numeric ``values`` may be compared within
  caller-supplied tolerances. Any missing record, intermediate, or field
  marks the verdict ``refused`` - never ``match``.

CLI:

    python -m reference.granite.golden capture \
        --adapter tests/fake_granite_adapter.py \
        --checkpoint-dir /local/granite-4.2-8b \
        --prompts prompts.json --out golden/reference.json

    python -m reference.granite.golden compare \
        --reference golden/reference.json \
        --candidate golden/engine.json --atol 1e-3 --rtol 1e-2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD_SCHEMA = "granite4.golden-record.v1"
CAPTURE_SCHEMA = "granite4.golden-capture.v1"
REPORT_SCHEMA = "granite4.golden-compare-report.v1"

EXIT_MATCH = 0
EXIT_DIVERGENT = 1
EXIT_REFUSED = 2

EXIT_BY_VERDICT = {
    "match": EXIT_MATCH,
    "divergent": EXIT_DIVERGENT,
    "refused": EXIT_REFUSED,
}

_CHECKPOINT_FILES = ("config.json", "tokenizer_config.json")
_OPTIONAL_CHECKPOINT_FILES = ("vocab.json", "tokenizer.json", "added_tokens.json")
_ADAPTER_REQUIRED_ATTRS = ("ADAPTER_NAME", "SUPPORTED_SAMPLING", "create_reference_adapter")
_SAMPLING_PASSTHROUGH_KEYS = ("seed", "temperature", "top_p", "max_new_tokens")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_MAX_DIVERGENCE_DETAILS = 8

_COMPARE_NOTE = (
    "Token IDs and tensor checksums are compared exactly; optional numeric "
    "values are compared within caller-supplied tolerances. Adapter/library "
    "metadata is informational and is never treated as a divergence."
)


class GoldenError(Exception):
    """Base error for the golden tooling."""


class PrerequisiteError(GoldenError):
    """A required prerequisite is missing or invalid; the run is refused."""


class CompareError(GoldenError):
    """A capture document is unusable; the comparison is refused."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Current UTC time, second precision, RFC 3339 with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str | Path) -> str:
    """SHA256 hex digest of a file's bytes (chunked)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(doc: dict[str, Any], out_path: str | Path) -> None:
    """Write a JSON document atomically (tmp file + rename)."""
    p = Path(out_path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(p.parent if p.parent else "."),
                                    prefix=".golden-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_name, p)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _is_id_list(value: Any) -> bool:
    """Non-empty list of non-negative integers (token IDs)."""
    return (isinstance(value, list) and len(value) > 0
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                    for v in value))


def _is_shape(value: Any) -> bool:
    """Tensor shape: list of >= 1 non-negative integers."""
    return (isinstance(value, list) and len(value) >= 1
            and all(isinstance(d, int) and not isinstance(d, bool) and d >= 0
                    for d in value))


def _is_checksum(value: Any) -> bool:
    """64-character lowercase hex string (SHA256)."""
    return isinstance(value, str) and bool(_HEX64_RE.fullmatch(value))


def _is_finite_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_finite_number(v) for v in value)


# ---------------------------------------------------------------------------
# Prerequisite verification
# ---------------------------------------------------------------------------

def verify_checkpoint_dir(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Verify the local checkpoint prerequisites and hash the config files.

    Requires a directory containing a readable JSON-object ``config.json``
    and ``tokenizer_config.json``. Raises :class:`PrerequisiteError`
    otherwise; model weight files are never opened.
    """
    d = Path(checkpoint_dir)
    if not d.is_dir():
        raise PrerequisiteError(
            f"checkpoint directory not found: {d} - supply a local checkpoint "
            "directory; nothing is downloaded")
    _sha256_keys = {"config.json": "config_json_sha256",
                    "tokenizer_config.json": "tokenizer_config_sha256"}
    info: dict[str, Any] = {"path": str(d)}
    for name in _CHECKPOINT_FILES:
        p = d / name
        if not p.is_file():
            raise PrerequisiteError(f"missing prerequisite file: {p}")
        try:
            data = json.loads(p.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            raise PrerequisiteError(f"{name}: not a readable JSON file")
        if not isinstance(data, dict):
            raise PrerequisiteError(f"{name}: top-level JSON value must be an object")
        info[_sha256_keys[name]] = sha256_file(p)
    info["optional_sha256"] = {
        name: sha256_file(d / name) for name in _OPTIONAL_CHECKPOINT_FILES
        if (d / name).is_file()
    }
    return info


# ---------------------------------------------------------------------------
# Adapter protocol
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdapterHandle:
    """A loaded local reference adapter module plus its instance."""
    module: str
    name: str
    supported_sampling: frozenset[str]
    adapter: Any


def _import_adapter_module(module: str) -> Any:
    """Import the caller-supplied local adapter module.

    Accepts a ``.py`` file path (loaded from disk) or an importable dotted
    module name. No network; no hidden defaults.
    """
    path = Path(module)
    if module.endswith(".py") or (path.exists() and path.is_file()):
        if not path.is_file():
            raise PrerequisiteError(f"adapter module file not found: {module}")
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(
            f"golden_ref_adapter_{digest}", str(path))
        if spec is None or spec.loader is None:
            raise PrerequisiteError(f"cannot load adapter module from {module}")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # noqa: BLE001 - reported, never claimed
            raise PrerequisiteError(
                f"adapter module {module!r} failed to load: "
                f"{type(exc).__name__}: {exc}") from exc
        return mod
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise PrerequisiteError(
            f"adapter module not found (no local file, not importable): {module!r}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise PrerequisiteError(
            f"adapter module {module!r} failed to import: {type(exc).__name__}: {exc}"
        ) from exc


def load_adapter(module: str, checkpoint_dir: str | Path) -> AdapterHandle:
    """Load the local adapter module, check the protocol, instantiate it."""
    mod = _import_adapter_module(module)
    for attr in _ADAPTER_REQUIRED_ATTRS:
        if not hasattr(mod, attr):
            raise PrerequisiteError(
                f"adapter module {module!r} lacks required attribute {attr!r}")
    name = mod.ADAPTER_NAME
    if not isinstance(name, str) or not name:
        raise PrerequisiteError(
            f"adapter module {module!r}: ADAPTER_NAME must be a non-empty string")
    supported = mod.SUPPORTED_SAMPLING
    if not isinstance(supported, (set, frozenset, list, tuple)) or not supported:
        raise PrerequisiteError(
            f"adapter module {module!r}: SUPPORTED_SAMPLING must be a non-empty "
            "collection of sampling-mode strings")
    if not all(isinstance(m, str) and m for m in supported):
        raise PrerequisiteError(
            f"adapter module {module!r}: SUPPORTED_SAMPLING entries must be "
            "non-empty strings")
    if not callable(mod.create_reference_adapter):
        raise PrerequisiteError(
            f"adapter module {module!r}: create_reference_adapter must be callable")
    try:
        adapter = mod.create_reference_adapter(checkpoint_dir)
    except Exception as exc:  # noqa: BLE001 - the adapter itself rejected it
        raise PrerequisiteError(
            f"adapter {name!r} refused the local checkpoint directory: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(adapter, "run", None)):
        raise PrerequisiteError(f"adapter {name!r}: instance must expose run(...)")
    if not callable(getattr(adapter, "adapter_metadata", None)):
        raise PrerequisiteError(
            f"adapter {name!r}: instance must expose adapter_metadata()")
    return AdapterHandle(module=module, name=name,
                         supported_sampling=frozenset(str(m) for m in supported),
                         adapter=adapter)


def normalize_sampling(spec: Any) -> dict[str, Any]:
    """Validate and normalize a sampling spec; refuse anything malformed."""
    if not isinstance(spec, dict):
        raise PrerequisiteError("sampling must be a JSON object")
    mode = spec.get("mode")
    if not isinstance(mode, str) or not mode:
        raise PrerequisiteError("sampling.mode must be a non-empty string")
    out: dict[str, Any] = {"mode": mode}
    for key in _SAMPLING_PASSTHROUGH_KEYS:
        if key in spec:
            out[key] = spec[key]
    return out


def load_prompts(prompts_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the prompts file (JSON array of prompt objects)."""
    p = Path(prompts_path)
    if not p.is_file():
        raise PrerequisiteError(f"prompts file not found: {p}")
    try:
        data = json.loads(p.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PrerequisiteError(
            f"prompts file {p.name}: not valid JSON ({type(exc).__name__})") from exc
    if not isinstance(data, list) or not data:
        raise PrerequisiteError(f"prompts file {p.name}: must be a non-empty JSON array")
    seen: set[str] = set()
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise PrerequisiteError(f"prompts[{i}]: each prompt must be an object")
        pid = item.get("prompt_id")
        if not isinstance(pid, str) or not pid:
            raise PrerequisiteError(f"prompts[{i}]: prompt_id must be a non-empty string")
        if pid in seen:
            raise PrerequisiteError(f"prompts[{i}]: duplicate prompt_id {pid!r}")
        seen.add(pid)
        has_ids, has_text = "input_ids" in item, "text" in item
        if has_ids == has_text:
            raise PrerequisiteError(
                f"prompts[{i}] ({pid!r}): exactly one of input_ids / text is required")
        if has_ids and not _is_id_list(item["input_ids"]):
            raise PrerequisiteError(
                f"prompts[{i}] ({pid!r}): input_ids must be a non-empty list of "
                "non-negative ints")
        if has_text and not isinstance(item["text"], str):
            raise PrerequisiteError(f"prompts[{i}] ({pid!r}): text must be a string")
    return data


def validate_run_result(raw: Any, prompt_id: str,
                       requested_ids: list[int] | None) -> list[str]:
    """Validate an adapter run result; returns the list of rejection issues."""
    if not isinstance(raw, dict):
        return [f"prompt {prompt_id!r}: adapter.run() must return an object"]
    issues: list[str] = []
    ids = raw.get("input_ids")
    if not _is_id_list(ids):
        issues.append("input_ids must be a non-empty list of non-negative token ids")
    elif requested_ids is not None and ids != requested_ids:
        issues.append("input_ids differ from the requested token ids")
    if not _is_id_list(raw.get("output_ids")):
        issues.append("output_ids must be a non-empty list of non-negative token ids")
    inter = raw.get("intermediates")
    if not isinstance(inter, dict) or not inter:
        issues.append("intermediates must be a non-empty mapping of tensor name -> evidence")
    else:
        for name, item in inter.items():
            if not isinstance(name, str) or not name:
                issues.append(f"intermediate name {name!r} must be a non-empty string")
                continue
            if not isinstance(item, dict):
                issues.append(f"intermediate {name!r}: evidence must be an object")
                continue
            if not _is_shape(item.get("shape")):
                issues.append(f"intermediate {name!r}: shape must be a list of non-negative ints")
            if not _is_checksum(item.get("sha256")):
                issues.append(f"intermediate {name!r}: sha256 must be a 64-char lowercase hex string")
            if "values" in item and not _is_number_list(item["values"]):
                issues.append(f"intermediate {name!r}: values must be a list of finite numbers")
            if "checksum_kind" in item and not isinstance(item["checksum_kind"], str):
                issues.append(f"intermediate {name!r}: checksum_kind must be a string")
    return issues


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def run_capture(adapter_module: str, checkpoint_dir: str | Path,
                prompts_path: str | Path, default_sampling: dict[str, Any]) -> dict[str, Any]:
    """Run the full capture; returns the capture document.

    Raises :class:`PrerequisiteError` - and therefore writes nothing - when
    any prerequisite fails, when sampling is unsupported, or when the
    adapter's evidence is incomplete.
    """
    checkpoint = verify_checkpoint_dir(checkpoint_dir)
    handle = load_adapter(adapter_module, checkpoint_dir)
    prompts = load_prompts(prompts_path)

    try:
        metadata = handle.adapter.adapter_metadata()
    except Exception as exc:  # noqa: BLE001
        raise PrerequisiteError(
            f"adapter adapter_metadata() failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise PrerequisiteError("adapter_metadata() must return a JSON object")
    try:
        json.dumps(metadata)
    except (TypeError, ValueError):
        raise PrerequisiteError("adapter_metadata() is not JSON-serializable")
    identity = {"module": handle.module, "name": handle.name, **metadata}

    records: list[dict[str, Any]] = []
    for prompt in prompts:
        sampling = normalize_sampling(prompt.get("sampling", default_sampling))
        if sampling["mode"] not in handle.supported_sampling:
            raise PrerequisiteError(
                f"unsupported sampling mode {sampling['mode']!r} for prompt "
                f"{prompt['prompt_id']!r}: adapter {handle.name!r} supports "
                f"{sorted(handle.supported_sampling)}; refusing to capture")
        requested_ids = prompt.get("input_ids")
        try:
            raw = handle.adapter.run(prompt_id=prompt["prompt_id"],
                                     text=prompt.get("text"),
                                     input_ids=requested_ids,
                                     sampling=sampling)
        except Exception as exc:  # noqa: BLE001 - e.g. unsupported sampling
            raise PrerequisiteError(
                f"adapter refused prompt {prompt['prompt_id']!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        issues = validate_run_result(raw, prompt["prompt_id"], requested_ids)
        if issues:
            raise PrerequisiteError(
                f"adapter evidence rejected for prompt {prompt['prompt_id']!r}: "
                + "; ".join(issues))
        record: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "captured_at_utc": utc_now_iso(),
            "prompt_id": prompt["prompt_id"],
            "sampling": sampling,
            "input_ids": list(raw["input_ids"]),
            "output_ids": list(raw["output_ids"]),
            "intermediates": {
                name: {
                    "shape": list(item["shape"]),
                    "sha256": item["sha256"],
                    **({"checksum_kind": item["checksum_kind"]} if "checksum_kind" in item else {}),
                    **({"values": list(item["values"])} if "values" in item else {}),
                }
                for name, item in raw["intermediates"].items()
            },
            "adapter": identity,
            "checkpoint": {key: checkpoint[key] for key in
                           ("path", "config_json_sha256", "tokenizer_config_sha256",
                            "optional_sha256")},
        }
        records.append(record)

    return {
        "schema": CAPTURE_SCHEMA,
        "captured_at_utc": utc_now_iso(),
        "adapter": identity,
        "checkpoint": checkpoint,
        "prompts": [p["prompt_id"] for p in prompts],
        "records": records,
    }


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tolerances:
    """Caller-supplied tolerances for optional numeric ``values``.

    Token IDs and checksums are always compared exactly; tolerances apply
    only to the adapter-supplied numeric ``values`` (``|a-b| <= atol + rtol*|b|``
    with ``b`` = reference).
    """
    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("atol", "rtol"):
            value = getattr(self, field_name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value < 0):
                raise CompareError(f"{field_name} must be a non-negative finite number")


def load_capture_document(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a capture document."""
    p = Path(path)
    if not p.is_file():
        raise CompareError(f"capture document not found: {p}")
    try:
        data = json.loads(p.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CompareError(
            f"{p.name}: not valid JSON ({type(exc).__name__})") from exc
    if not isinstance(data, dict) or data.get("schema") != CAPTURE_SCHEMA:
        raise CompareError(f"{p.name}: not a {CAPTURE_SCHEMA} capture document")
    if not isinstance(data.get("records"), list) or not data["records"]:
        raise CompareError(f"{p.name}: capture document has no records")
    return data


def _as_id_list(value: Any) -> list[int]:
    """Narrow a value already checked by :func:`_is_id_list` to a list."""
    return [int(v) for v in value]


def _id_diff(a: list[int], b: list[int]) -> str:
    if len(a) != len(b):
        return f"length {len(a)} != {len(b)}"
    bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if not bad:
        return "values differ"
    sample = ", ".join(f"[{i}] {a[i]} != {b[i]}" for i in bad[:_MAX_DIVERGENCE_DETAILS])
    more = f", ... +{len(bad) - _MAX_DIVERGENCE_DETAILS} more" if len(bad) > _MAX_DIVERGENCE_DETAILS else ""
    return sample + more


def _compare_record(pid: str, r: dict[str, Any], c: dict[str, Any],
                   tol: Tolerances, missing: list[str], divergences: list[str],
                   reasons: list[str]) -> None:
    """Compare two same-prompt records, appending findings to the lists."""
    rs, cs = r.get("sampling"), c.get("sampling")
    if not isinstance(rs, dict) or not isinstance(cs, dict):
        missing.append(f"{pid}: sampling absent in one record")
    elif rs != cs:
        reasons.append(
            f"{pid}: sampling differs (reference={json.dumps(rs, sort_keys=True)} "
            f"candidate={json.dumps(cs, sort_keys=True)})")
        return

    for field in ("input_ids", "output_ids"):
        a_ids, b_ids = r.get(field), c.get(field)
        if not _is_id_list(a_ids) or not _is_id_list(b_ids):
            missing.append(f"{pid}: {field} absent or malformed in one record")
        elif a_ids != b_ids:
            divergences.append(f"{pid}: {field} differ: "
                               f"{_id_diff(_as_id_list(a_ids), _as_id_list(b_ids))}")

    for key in ("config_json_sha256", "tokenizer_config_sha256"):
        rc = (r.get("checkpoint") or {}).get(key)
        cc = (c.get("checkpoint") or {}).get(key)
        if not _is_checksum(rc) or not _is_checksum(cc):
            missing.append(f"{pid}: checkpoint {key} absent in one record")
        elif rc != cc:
            reasons.append(f"{pid}: checkpoint {key} differs between records")

    a_inter, b_inter = r.get("intermediates"), c.get("intermediates")
    if (not isinstance(a_inter, dict) or not a_inter
            or not isinstance(b_inter, dict) or not b_inter):
        missing.append(f"{pid}: intermediates absent or empty in one record")
        return

    for name in sorted(set(a_inter) | set(b_inter)):
        if name not in a_inter:
            missing.append(f"{pid}: intermediate {name!r} missing from reference record")
            continue
        if name not in b_inter:
            missing.append(f"{pid}: intermediate {name!r} missing from candidate record")
            continue
        a, b = a_inter[name], b_inter[name]
        if not isinstance(a, dict) or not isinstance(b, dict):
            missing.append(f"{pid}: intermediate {name!r} malformed in one record")
            continue
        a_shape, b_shape = a.get("shape"), b.get("shape")
        if not _is_shape(a_shape) or not _is_shape(b_shape):
            missing.append(f"{pid}: intermediate {name!r}: shape absent in one record")
        elif a_shape != b_shape:
            divergences.append(f"{pid}: intermediate {name!r}: shape {a_shape} != {b_shape}")
        a_ck, b_ck = a.get("sha256"), b.get("sha256")
        if not _is_checksum(a_ck) or not _is_checksum(b_ck):
            missing.append(f"{pid}: intermediate {name!r}: checksum absent in one record")
        elif a_ck != b_ck:
            divergences.append(
                f"{pid}: intermediate {name!r}: checksum "
                f"{str(a_ck)[:12]}... != {str(b_ck)[:12]}...")
        a_vals, b_vals = a.get("values"), b.get("values")
        if a_vals is None and b_vals is None:
            continue
        if a_vals is None or b_vals is None:
            missing.append(f"{pid}: intermediate {name!r}: values present in only one record")
            continue
        if not _is_number_list(a_vals) or not _is_number_list(b_vals):
            missing.append(f"{pid}: intermediate {name!r}: values malformed in one record")
            continue
        if len(a_vals) != len(b_vals):
            divergences.append(
                f"{pid}: intermediate {name!r}: values length {len(a_vals)} != {len(b_vals)}")
            continue
        n_bad = 0
        for i, (x, y) in enumerate(zip(a_vals, b_vals)):
            if abs(x - y) > tol.atol + tol.rtol * abs(y):
                n_bad += 1
                if n_bad <= _MAX_DIVERGENCE_DETAILS:
                    divergences.append(
                        f"{pid}: intermediate {name!r}: values[{i}] = {x} vs {y} "
                        f"(atol={tol.atol}, rtol={tol.rtol})")
        if n_bad > _MAX_DIVERGENCE_DETAILS:
            divergences.append(
                f"{pid}: intermediate {name!r}: ... and "
                f"{n_bad - _MAX_DIVERGENCE_DETAILS} more value divergence(s)")


def compare_documents(reference: dict[str, Any], candidate: dict[str, Any],
                      tolerances: Tolerances) -> dict[str, Any]:
    """Compare two capture documents; returns the compare report.

    Verdict precedence: any missing evidence or unusable record makes the
    verdict ``refused``; otherwise any divergence makes it ``divergent``;
    otherwise ``match``.
    """
    doc_missing: list[str] = []
    checks: list[dict[str, Any]] = []

    def _refused_report(reasons: list[str]) -> dict[str, Any]:
        return {
            "schema": REPORT_SCHEMA,
            "generated_at_utc": utc_now_iso(),
            "verdict": "refused",
            "tolerances": {"atol": tolerances.atol, "rtol": tolerances.rtol},
            "counts": {
                "prompts": len(checks),
                "match": sum(1 for ch in checks if ch["status"] == "match"),
                "divergent": sum(1 for ch in checks if ch["status"] == "divergent"),
                "refused": sum(1 for ch in checks if ch["status"] == "refused"),
            },
            "missing_evidence": list(doc_missing),
            "reasons": reasons,
            "checks": checks,
            "adapter_metadata": {"reference": reference.get("adapter"),
                                 "candidate": candidate.get("adapter")},
            "note": _COMPARE_NOTE,
        }

    # Document-level checkpoint evidence.
    for side, doc in (("reference", reference), ("candidate", candidate)):
        ck = doc.get("checkpoint")
        if (not isinstance(ck, dict)
                or not _is_checksum(ck.get("config_json_sha256"))
                or not _is_checksum(ck.get("tokenizer_config_sha256"))):
            doc_missing.append(f"{side}: checkpoint SHA256 hashes absent")
    if not doc_missing:
        r_ck, c_ck = reference["checkpoint"], candidate["checkpoint"]
        for key in ("config_json_sha256", "tokenizer_config_sha256"):
            if r_ck[key] != c_ck[key]:
                return _refused_report([
                    f"checkpoint {key} differs between the two captures "
                    "(different checkpoints); comparison is meaningless"])

    if not isinstance(reference.get("records"), list) or not isinstance(candidate.get("records"), list):
        return _refused_report(["capture documents have no record list"])

    ref_recs: dict[Any, dict[str, Any]] = {}
    cand_recs: dict[Any, dict[str, Any]] = {}
    for rec in reference["records"]:
        if isinstance(rec, dict):
            ref_recs[rec.get("prompt_id")] = rec
        else:
            doc_missing.append("reference: record is not an object")
    for rec in candidate["records"]:
        if isinstance(rec, dict):
            cand_recs[rec.get("prompt_id")] = rec
        else:
            doc_missing.append("candidate: record is not an object")

    all_ids = sorted(set(ref_recs) | set(cand_recs), key=lambda p: (p is None, str(p)))
    for pid in all_ids:
        r, c = ref_recs.get(pid), cand_recs.get(pid)
        if r is None or c is None:
            doc_missing.append(f"prompt {pid!r} present in only one capture document")
            checks.append({
                "prompt_id": pid,
                "status": "refused",
                "sampling": None,
                "reference_adapter": (r or {}).get("adapter", {}).get("name") if isinstance((r or {}).get("adapter"), dict) else None,
                "candidate_adapter": (c or {}).get("adapter", {}).get("name") if isinstance((c or {}).get("adapter"), dict) else None,
                "divergences": [],
                "missing_evidence": ["record present in only one capture document"],
                "reasons": [],
            })
            continue
        missing: list[str] = []
        divergences: list[str] = []
        reasons: list[str] = []
        _compare_record(pid, r, c, tolerances, missing, divergences, reasons)
        if missing or reasons:
            status = "refused"
        elif divergences:
            status = "divergent"
        else:
            status = "match"
        checks.append({
            "prompt_id": pid,
            "status": status,
            "sampling": r.get("sampling"),
            "reference_adapter": (r.get("adapter") or {}).get("name")
                if isinstance(r.get("adapter"), dict) else None,
            "candidate_adapter": (c.get("adapter") or {}).get("name")
                if isinstance(c.get("adapter"), dict) else None,
            "divergences": divergences,
            "missing_evidence": missing,
            "reasons": reasons,
        })

    record_missing = [m for ch in checks for m in ch["missing_evidence"]]
    all_missing = doc_missing + record_missing
    if all_missing or any(ch["status"] == "refused" for ch in checks):
        verdict = "refused"
    elif any(ch["status"] == "divergent" for ch in checks):
        verdict = "divergent"
    else:
        verdict = "match"

    return {
        "schema": REPORT_SCHEMA,
        "generated_at_utc": utc_now_iso(),
        "verdict": verdict,
        "tolerances": {"atol": tolerances.atol, "rtol": tolerances.rtol},
        "counts": {
            "prompts": len(all_ids),
            "match": sum(1 for ch in checks if ch["status"] == "match"),
            "divergent": sum(1 for ch in checks if ch["status"] == "divergent"),
            "refused": sum(1 for ch in checks if ch["status"] == "refused"),
        },
        "missing_evidence": all_missing,
        "reasons": [],
        "checks": checks,
        "adapter_metadata": {"reference": reference.get("adapter"),
                             "candidate": candidate.get("adapter")},
        "note": _COMPARE_NOTE,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _capture_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="golden-capture",
        description="Capture golden-reference records from a local adapter "
                    "(no network, no downloads).")
    p.add_argument("--adapter", required=True, metavar="MODULE_OR_FILE",
                   help="local adapter module: a .py file path or an "
                        "importable dotted module name")
    p.add_argument("--checkpoint-dir", required=True,
                   help="local checkpoint directory with config.json and "
                        "tokenizer_config.json")
    p.add_argument("--prompts", required=True, metavar="PROMPTS_JSON",
                   help="JSON array of prompt objects (prompt_id + text or input_ids)")
    p.add_argument("--out", required=True, help="output capture document (JSON)")
    p.add_argument("--sampling-mode", default="greedy",
                   help="default sampling mode (default: greedy)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    return p


def main_capture(argv: list[str] | None = None) -> int:
    args = _capture_parser().parse_args(argv)
    default_sampling: dict[str, Any] = {"mode": args.sampling_mode}
    if args.seed is not None:
        default_sampling["seed"] = args.seed
    if args.max_new_tokens is not None:
        default_sampling["max_new_tokens"] = args.max_new_tokens
    try:
        doc = run_capture(args.adapter, args.checkpoint_dir, args.prompts,
                          default_sampling)
        write_json_atomic(doc, args.out)
    except GoldenError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        print(f"REFUSED: output write failed: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(f"captured {len(doc['records'])} record(s) -> {args.out}")
    return EXIT_MATCH


def _compare_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="golden-compare",
        description="Compare two golden capture documents; refuses absent "
                    "evidence.")
    p.add_argument("--reference", required=True, help="reference capture document")
    p.add_argument("--candidate", required=True, help="candidate capture document")
    p.add_argument("--atol", type=float, default=0.0,
                   help="absolute tolerance for numeric values (default: 0.0, exact)")
    p.add_argument("--rtol", type=float, default=0.0,
                   help="relative tolerance for numeric values (default: 0.0, exact)")
    p.add_argument("--report-out", default=None, help="optional JSON report path")
    return p


def main_compare(argv: list[str] | None = None) -> int:
    args = _compare_parser().parse_args(argv)
    try:
        reference = load_capture_document(args.reference)
        candidate = load_capture_document(args.candidate)
        tolerances = Tolerances(atol=args.atol, rtol=args.rtol)
    except CompareError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except OSError as exc:
        print(f"REFUSED: cannot read capture document: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    report = compare_documents(reference, candidate, tolerances)
    if args.report_out:
        try:
            write_json_atomic(report, args.report_out)
        except OSError as exc:
            print(f"REFUSED: report write failed: {exc}", file=sys.stderr)
            return EXIT_REFUSED
    counts = report["counts"]
    print(f"verdict={report['verdict']} prompts={counts['prompts']} "
          f"match={counts['match']} divergent={counts['divergent']} "
          f"refused={counts['refused']} missing_evidence={len(report['missing_evidence'])}")
    for ch in report["checks"]:
        if ch["status"] != "match":
            print(f"  {ch['prompt_id']}: {ch['status']}", file=sys.stderr)
            for item in ch["divergences"]:
                print(f"    divergent: {item}", file=sys.stderr)
            for item in ch["missing_evidence"]:
                print(f"    missing: {item}", file=sys.stderr)
            for item in ch["reasons"]:
                print(f"    reason: {item}", file=sys.stderr)
    return EXIT_BY_VERDICT[report["verdict"]]


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: golden capture | golden compare (see --help)", file=sys.stderr)
        return EXIT_REFUSED
    if argv[0] == "capture":
        return main_capture(argv[1:])
    if argv[0] == "compare":
        return main_compare(argv[1:])
    print(f"unknown subcommand: {argv[0]!r} (expected capture|compare)", file=sys.stderr)
    return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
