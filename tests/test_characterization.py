"""Hermetic tests for the Phase-0 checkpoint characterization tools.

No CUDA, no network: every fixture is a temporary JSON file created under
pytest's ``tmp_path``. The scripts under test are stdlib-only by design.

Covers:
  * inspector: successful synthetic inspection, planning-field mismatch
    (wrong value, missing field, mistyped value), invalid inputs
    (missing directory / missing files / non-JSON), and the guarantee
    that raw tokenizer/config values are never echoed in the report;
  * baseline runner: refusal of placeholder checkpoint paths, missing
    prerequisites, and placeholder argv (nothing executed; a record
    stating measurements were not run is written), plus raw-fact
    recording when explicit commands are actually executed and
    non-zero-exit accounting;
  * the shipped illustrative manifest stays a disabled placeholder.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

SECRET_MARKER = "SECRET-MARKER"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INSPECT = _load_module("inspect_checkpoint", SCRIPTS / "inspect_checkpoint.py")
RUN = _load_module("run_baseline", SCRIPTS / "run_baseline.py")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _write_checkpoint(
    checkpoint_dir: Path, config_overrides: dict[str, Any] | None = None
) -> Path:
    """Create a synthetic local checkpoint (JSON metadata only, no weights)."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = dict(INSPECT.REQUIRED_FIELDS)
    config["architectures"] = ["Granite4ForCausalLM"]
    config["bos_token_value"] = f"{SECRET_MARKER}-CONFIG-VALUE"
    if config_overrides:
        for key, value in config_overrides.items():
            if value is _ABSENT:
                config.pop(key, None)
            else:
                config[key] = value
    _write_json(checkpoint_dir / "config.json", config)
    tokenizer = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 128000,
        "bos_token": {"content": f"{SECRET_MARKER}-BOS", "lstrip": True},
        "eos_token": f"{SECRET_MARKER}-EOS",
        "pad_token": f"{SECRET_MARKER}-PAD",
        "added_tokens": [{"content": f"{SECRET_MARKER}-ADDED", "ids": [127]}],
        "chat_template": f"{SECRET_MARKER}-TEMPLATE",
    }
    _write_json(checkpoint_dir / "tokenizer_config.json", tokenizer)
    return checkpoint_dir


_ABSENT = object()


# ---------------------------------------------------------------------------
# Inspector: synthetic success
# ---------------------------------------------------------------------------


def test_inspector_successful_synthetic_inspection(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 0
    assert report["verdict"] == "ok"
    assert report["issues"] == []
    assert report["weight_files_touched"] is False
    assert report["network_access"] is False
    assert report["config_json"]["present"] is True
    assert report["config_json"]["valid_json"] is True
    assert report["tokenizer_config_json"]["present"] is True
    assert report["tokenizer_config_json"]["valid_json"] is True
    assert report["tokenizer_config_json"]["contents_echoed"] is False
    selected = report["tokenizer_config_json"]["selected_keys_present"]
    assert selected["tokenizer_class"] is True
    assert selected["chat_template"] is True
    for name, expected in INSPECT.REQUIRED_FIELDS.items():
        entry = report["planning_fields"][name]
        assert entry["expected"] == expected
        assert entry["actual"] == expected
        assert entry["ok"] is True


def test_inspector_report_never_echoes_raw_contents(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    _code, report = INSPECT.inspect_checkpoint(ckpt)
    dumped = json.dumps(report, sort_keys=True)
    assert SECRET_MARKER not in dumped
    # Bounded report: key names are structural, values are not.
    assert report["config_json"]["top_level_key_count"] > len(INSPECT.REQUIRED_FIELDS)


# ---------------------------------------------------------------------------
# Inspector: mismatches
# ---------------------------------------------------------------------------


def test_inspector_reports_wrong_planning_value(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt", {"num_hidden_layers": 32})
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 1
    assert report["verdict"] == "mismatch"
    assert any("num_hidden_layers" in issue for issue in report["issues"])
    entry = report["planning_fields"]["num_hidden_layers"]
    assert entry["expected"] == 40
    assert entry["actual"] == 32
    assert entry["ok"] is False
    for name in INSPECT.REQUIRED_FIELDS:
        if name != "num_hidden_layers":
            assert report["planning_fields"][name]["ok"] is True


def test_inspector_derives_missing_head_dim(tmp_path: Path) -> None:
    # head_dim is absent from the real checkpoint's config.json: it must be
    # derived as hidden_size / num_attention_heads (4096 / 32 = 128), matching
    # the C++ parser, rather than reported as a missing field.
    ckpt = _write_checkpoint(tmp_path / "ckpt", {"head_dim": _ABSENT})
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 0
    assert report["verdict"] == "ok"
    entry = report["planning_fields"]["head_dim"]
    assert entry["actual"] == 128
    assert entry["derived_from"] == ["hidden_size", "num_attention_heads"]
    assert entry["ok"] is True


def test_inspector_reports_head_dim_derivation_mismatch(tmp_path: Path) -> None:
    # hidden_size 2048 -> derived head_dim 2048/32 = 64, which is off-spec.
    ckpt = _write_checkpoint(tmp_path / "ckpt", {"head_dim": _ABSENT, "hidden_size": 2048})
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 1
    assert report["verdict"] == "mismatch"
    assert any("head_dim" in issue for issue in report["issues"])
    entry = report["planning_fields"]["head_dim"]
    assert entry["expected"] == 128
    assert entry["actual"] == 64
    assert entry["ok"] is False


def test_inspector_reports_mistyped_planning_value(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt", {"vocab_size": "100352"})
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 1
    assert report["verdict"] == "mismatch"
    entry = report["planning_fields"]["vocab_size"]
    assert entry["ok"] is False
    assert entry["actual"] == "100352"


def test_inspector_reports_multiple_mismatches(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(
        tmp_path / "ckpt",
        {"hidden_size": 2048, "num_key_value_heads": _ABSENT, "intermediate_size": 4096},
    )
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 1
    assert report["verdict"] == "mismatch"
    names_in_issues = " ".join(report["issues"])
    for name in ("hidden_size", "num_key_value_heads", "intermediate_size"):
        assert name in names_in_issues


# ---------------------------------------------------------------------------
# Inspector: invalid inputs
# ---------------------------------------------------------------------------


def test_inspector_missing_directory_is_invalid(tmp_path: Path) -> None:
    code, report = INSPECT.inspect_checkpoint(tmp_path / "does-not-exist")
    assert code == 2
    assert report["verdict"] == "invalid_input"
    assert report["config_json"]["present"] is False


def test_inspector_missing_config_is_invalid(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_json(ckpt / "tokenizer_config.json", {"tokenizer_class": "LlamaTokenizer"})
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 2
    assert report["verdict"] == "invalid_input"
    assert any("config.json" in issue for issue in report["issues"])


def test_inspector_missing_tokenizer_is_invalid(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    (ckpt / "tokenizer_config.json").unlink()
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 2
    assert report["verdict"] == "invalid_input"
    assert any("tokenizer_config.json" in issue for issue in report["issues"])


def test_inspector_non_json_is_invalid(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    (ckpt / "config.json").write_text("this is not json", encoding="utf-8")
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 2
    assert report["verdict"] == "invalid_input"
    assert any("not valid JSON" in issue for issue in report["issues"])


def test_inspector_non_object_json_is_invalid(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    (ckpt / "tokenizer_config.json").write_text("[1, 2, 3]", encoding="utf-8")
    code, report = INSPECT.inspect_checkpoint(ckpt)
    assert code == 2
    assert report["verdict"] == "invalid_input"


# ---------------------------------------------------------------------------
# Inspector: CLI end-to-end
# ---------------------------------------------------------------------------


def test_inspector_cli_stdout_is_bounded_json(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_checkpoint.py"), str(ckpt)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    report = json.loads(proc.stdout)
    assert report["verdict"] == "ok"
    assert SECRET_MARKER not in proc.stdout


def test_inspector_cli_mismatch_exit_code(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt", {"num_attention_heads": 16})
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "inspect_checkpoint.py"), str(ckpt)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["verdict"] == "mismatch"


# ---------------------------------------------------------------------------
# Baseline runner: refusals
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, **overrides: Any) -> Path:
    manifest: dict[str, Any] = {
        "schema": RUN.MANIFEST_SCHEMA,
        "required_checkpoint_path": "ckpt/PLACEHOLDER_example",
        "required_entries": ["ckpt/PLACEHOLDER_example/config.json"],
        "record_path": "record.json",
        "timeout_seconds": 30,
        "environment_to_record": ["CUDA_VISIBLE_DEVICES"],
        "commands": [
            {
                "name": "example",
                "enabled": True,
                "argv": [sys.executable, "-c", "print('must-not-run')"],
            }
        ],
    }
    manifest.update(overrides)
    _write_json(path, manifest)
    return path


def test_baseline_refuses_placeholder_checkpoint(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json")
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert record["measurements_run"] is False
    assert record["status"] == "not_run"
    assert record["performance_derived"] is False
    assert record["performance_claims"] == "none"
    assert any("placeholder" in r for r in record["refusals"])
    assert record["commands"] == []  # nothing executed
    assert "Measurements were not run" in record["statement"]
    out = tmp_path / "record.json"
    assert out.is_file()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["measurements_run"] is False
    assert "Measurements were not run" in on_disk["statement"]


def test_baseline_refuses_missing_prerequisite_entries(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(ckpt),
        required_entries=[
            str(ckpt / "config.json"),  # exists
            str(tmp_path / "no-such-entry.json"),  # missing
        ],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert record["measurements_run"] is False
    assert any("no-such-entry.json" in r for r in record["refusals"])
    assert record["commands"] == []


def test_baseline_refuses_checkpoint_dir_without_config(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(ckpt),
        required_entries=[],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert any("config.json" in r for r in record["refusals"])
    assert record["measurements_run"] is False


def test_baseline_refuses_placeholder_argv(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    marker_file = tmp_path / "ran.txt"
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(ckpt),
        required_entries=[],
        commands=[
            {
                "name": "placeholder-script",
                "enabled": True,
                "argv": [sys.executable, "PLACEHOLDER_bench.py", str(marker_file)],
            }
        ],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert any("argv element is a placeholder" in r for r in record["refusals"])
    assert record["commands"] == []
    assert not marker_file.exists()  # nothing executed before refusal


def test_baseline_refuses_nonexistent_checkpoint_path(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(tmp_path / "not-a-checkpoint"),
        required_entries=[],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert any("missing or not a directory" in r for r in record["refusals"])


def test_baseline_refuses_unknown_manifest_schema(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.json", schema="other.schema.v9")
    code, record = RUN.run_baseline(manifest)
    assert code == 2
    assert record["measurements_run"] is False
    assert any("schema" in r for r in record["refusals"])


# ---------------------------------------------------------------------------
# Baseline runner: executed commands record raw facts only
# ---------------------------------------------------------------------------


def test_baseline_records_raw_facts_for_enabled_command(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(ckpt),
        required_entries=[str(ckpt / "config.json")],
        commands=[
            {
                "name": "fact-emitter",
                "enabled": True,
                "argv": [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout-fact'); print('stderr-fact', file=sys.stderr)",
                ],
            },
            {
                "name": "disabled-noop",
                "enabled": False,
                "argv": [sys.executable, "-c", "import sys; sys.exit(3)"],
            },
        ],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 0
    assert record["measurements_run"] is True
    assert record["commands_executed"] == 1
    assert record["commands_failed"] == 0
    assert record["status"] == "completed"
    assert record["performance_derived"] is False
    assert record["performance_claims"] == "none"

    executed = record["commands"][0]
    assert executed["name"] == "fact-emitter"
    assert executed["exit_code"] == 0
    assert executed["duration_seconds"] is not None
    assert executed["duration_seconds"] >= 0
    assert executed["started_at"] is not None
    assert executed["finished_at"] is not None
    assert "stdout-fact" in executed["output_tail"]
    assert "stderr-fact" in executed["output_tail"]
    assert executed["output_tail_truncated"] is False

    skipped = record["commands"][1]
    assert skipped["skipped"] is True
    assert skipped["exit_code"] is None
    assert "disabled" in (skipped["note"] or "")

    provenance = record["provenance"]
    assert provenance["python"] == sys.version.split()[0]
    assert "cwd" in provenance
    assert "note" in provenance  # environment is opt-in only

    out = tmp_path / "record.json"
    assert out.is_file()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["measurements_run"] is True
    assert on_disk["commands"][0]["exit_code"] == 0


def test_baseline_counts_nonzero_command_exit(tmp_path: Path) -> None:
    ckpt = _write_checkpoint(tmp_path / "ckpt")
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        required_checkpoint_path=str(ckpt),
        required_entries=[],
        commands=[
            {
                "name": "failing",
                "enabled": True,
                "argv": [sys.executable, "-c", "import sys; sys.exit(2)"],
            }
        ],
    )
    code, record = RUN.run_baseline(manifest)
    assert code == 1
    assert record["measurements_run"] is True
    assert record["status"] == "completed_with_failures"
    assert record["commands_failed"] == 1
    assert record["commands"][0]["exit_code"] == 2


# ---------------------------------------------------------------------------
# Shipped illustrative manifest
# ---------------------------------------------------------------------------


def test_example_manifest_is_disabled_placeholder(tmp_path: Path) -> None:
    example = ROOT / "benchmarks" / "baseline.example.json"
    assert example.is_file()
    manifest = json.loads(example.read_text(encoding="utf-8"))
    assert manifest["schema"] == RUN.MANIFEST_SCHEMA
    assert "PLACEHOLDER" in str(manifest["required_checkpoint_path"])
    for cmd in manifest["commands"]:
        assert cmd["enabled"] is False
    # And the runner must refuse it exactly as designed.
    code, record = RUN.run_baseline(example)
    assert code == 2
    assert record["measurements_run"] is False
    assert record["commands"] == []
    assert any("placeholder" in r for r in record["refusals"])
    record_path = example.parent / manifest["record_path"]
    try:
        assert record_path.is_file()
        on_disk = json.loads(record_path.read_text(encoding="utf-8"))
        assert on_disk["measurements_run"] is False
        assert "Measurements were not run" in on_disk["statement"]
        assert on_disk["performance_derived"] is False
    finally:
        # Do not leave test-generated artifacts in the repository.
        record_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
