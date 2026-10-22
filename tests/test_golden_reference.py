"""Hermetic tests for the Phase-1 golden-reference tooling.

No torch, no CUDA, no network, no real checkpoint: the adapter under test
is the synthetic, deterministic :mod:`tests.fake_granite_adapter`; all
files live in temporary directories.

Covers:
  * capture - full record content (prompt id, exact token IDs, intermediate
    checksums/shapes, adapter metadata, UTC timestamp, config/tokenizer
    SHA256s);
  * prerequisite rejection - missing adapter, missing checkpoint files,
    non-JSON config, absent adapter attributes;
  * unsupported sampling rejection (capture refused, nothing written);
  * incomplete adapter evidence rejection;
  * compare - match / divergent (token IDs, shape, checksum, values within
    and beyond tolerances) / refused (missing evidence, absent records,
    checkpoint hash mismatch);
  * CLI exit codes for capture and compare.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from reference.granite import golden as g
from reference.granite.golden import (
    CAPTURE_SCHEMA,
    CompareError,
    PrerequisiteError,
    Tolerances,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_ADAPTER = REPO_ROOT / "tests" / "fake_granite_adapter.py"
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def checkpoint_dir(tmp_path: Path) -> Path:
    """A synthetic local checkpoint dir with the required JSON files."""
    d = tmp_path / "checkpoint"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"architectures": ["GraniteMoeModel"],
                    "model_type": "granitemoe", "vocab_size": 100352}),
        encoding="utf-8")
    (d / "tokenizer_config.json").write_text(
        json.dumps({"eos_token_id": 1}), encoding="utf-8")
    (d / "vocab.json").write_text('{"<pad>": 0}', encoding="utf-8")
    return d


@pytest.fixture()
def prompts_path(tmp_path: Path) -> Path:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([
        {"prompt_id": "p-greedy", "text": "hello world",
         "sampling": {"mode": "greedy", "max_new_tokens": 8}},
        {"prompt_id": "p-seeded", "input_ids": [101, 202, 303],
         "sampling": {"mode": "seeded", "seed": 42, "max_new_tokens": 4}},
    ]), encoding="utf-8")
    return p


def _capture(checkpoint_dir: Path, prompts_path: Path,
             adapter: Path | str = FAKE_ADAPTER,
             mode: str = "greedy") -> dict[str, Any]:
    return g.run_capture(str(adapter), checkpoint_dir, prompts_path,
                         {"mode": mode})


def _write_doc(tmp_path: Path, name: str, doc: dict[str, Any]) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Capture: content
# ---------------------------------------------------------------------------

def test_capture_record_content(checkpoint_dir: Path, prompts_path: Path) -> None:
    doc = _capture(checkpoint_dir, prompts_path)
    assert doc["schema"] == CAPTURE_SCHEMA
    assert UTC_RE.fullmatch(doc["captured_at_utc"])
    assert doc["prompts"] == ["p-greedy", "p-seeded"]
    assert doc["adapter"]["name"] == "fake-granite-reference"
    assert doc["adapter"]["library"] == "fake"
    assert doc["adapter"]["deterministic"] is True

    ck = doc["checkpoint"]
    assert ck["config_json_sha256"] == g.sha256_file(checkpoint_dir / "config.json")
    assert ck["tokenizer_config_sha256"] == g.sha256_file(
        checkpoint_dir / "tokenizer_config.json")
    assert ck["optional_sha256"]["vocab.json"] == g.sha256_file(
        checkpoint_dir / "vocab.json")

    rec = doc["records"][0]
    assert rec["schema"] == g.RECORD_SCHEMA
    assert rec["prompt_id"] == "p-greedy"
    assert rec["sampling"] == {"mode": "greedy", "max_new_tokens": 8}
    assert UTC_RE.fullmatch(rec["captured_at_utc"])
    assert isinstance(rec["input_ids"], list) and len(rec["input_ids"]) >= 8
    assert all(isinstance(t, int) and t >= 0 for t in rec["input_ids"])
    assert len(rec["output_ids"]) == 8

    inter = rec["intermediates"]
    assert set(inter) == {"layer_0.hidden_state", "logits"}
    for name, item in inter.items():
        assert g._is_shape(item["shape"])
        assert g._is_checksum(item["sha256"])
        assert g._is_number_list(item["values"])
    assert len(inter["layer_0.hidden_state"]["shape"]) == 3
    assert len(inter["logits"]["shape"]) == 2


def test_capture_is_deterministic(checkpoint_dir: Path, prompts_path: Path) -> None:
    a = _capture(checkpoint_dir, prompts_path)
    b = _capture(checkpoint_dir, prompts_path)
    for ra, rb in zip(a["records"], b["records"]):
        assert ra["input_ids"] == rb["input_ids"]
        assert ra["output_ids"] == rb["output_ids"]
        assert ra["intermediates"] == rb["intermediates"]


def test_capture_seeded_differs_from_default_seed(checkpoint_dir: Path,
                                                 tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([
        {"prompt_id": "s1", "text": "x", "sampling": {"mode": "seeded", "seed": 1,
                                                     "max_new_tokens": 4}},
        {"prompt_id": "s2", "text": "x", "sampling": {"mode": "seeded", "seed": 2,
                                                     "max_new_tokens": 4}},
    ]), encoding="utf-8")
    doc = _capture(checkpoint_dir, p)
    out = {r["prompt_id"]: r["output_ids"] for r in doc["records"]}
    assert out["s1"] != out["s2"]


# ---------------------------------------------------------------------------
# Capture: prerequisite rejection
# ---------------------------------------------------------------------------

def test_capture_refuses_missing_checkpoint(tmp_path: Path,
                                            prompts_path: Path) -> None:
    out = tmp_path / "out.json"
    with pytest.raises(PrerequisiteError, match="checkpoint directory not found"):
        g.run_capture(str(FAKE_ADAPTER), tmp_path / "nope", prompts_path,
                      {"mode": "greedy"})
    # CLI path too: refused, nothing written.
    code = g.main_capture(["--adapter", str(FAKE_ADAPTER),
                           "--checkpoint-dir", str(tmp_path / "nope"),
                           "--prompts", str(prompts_path), "--out", str(out)])
    assert code == g.EXIT_REFUSED
    assert not out.exists()


def test_capture_refuses_missing_config_json(tmp_path: Path,
                                             prompts_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="missing prerequisite file"):
        _capture(d, prompts_path)


def test_capture_refuses_non_json_config(tmp_path: Path,
                                         prompts_path: Path) -> None:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("not json", encoding="utf-8")
    (d / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="not a readable JSON"):
        _capture(d, prompts_path)


def test_capture_refuses_non_object_config(checkpoint_dir: Path,
                                          prompts_path: Path) -> None:
    (checkpoint_dir / "config.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="must be an object"):
        _capture(checkpoint_dir, prompts_path)


def test_capture_refuses_missing_adapter_file(tmp_path: Path,
                                              checkpoint_dir: Path,
                                              prompts_path: Path) -> None:
    missing = tmp_path / "does_not_exist.py"
    with pytest.raises(PrerequisiteError, match="adapter module file not found"):
        g.run_capture(str(missing), checkpoint_dir, prompts_path, {"mode": "greedy"})


def test_capture_refuses_unimportable_adapter(tmp_path: Path,
                                              checkpoint_dir: Path,
                                              prompts_path: Path) -> None:
    with pytest.raises(PrerequisiteError, match="not found"):
        g.run_capture("no.such.module.anywhere", checkpoint_dir, prompts_path,
                      {"mode": "greedy"})


def _write_adapter(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_capture_refuses_adapter_missing_attribute(tmp_path: Path,
                                                   checkpoint_dir: Path,
                                                   prompts_path: Path) -> None:
    mod = _write_adapter(tmp_path, "bad_adapter.py",
                         "ADAPTER_NAME = 'x'\n"
                         "def create_reference_adapter(d):\n"
                         "    raise NotImplementedError\n")
    # SUPPORTED_SAMPLING absent.
    with pytest.raises(PrerequisiteError, match="SUPPORTED_SAMPLING"):
        g.run_capture(str(mod), checkpoint_dir, prompts_path, {"mode": "greedy"})


def test_capture_refuses_adapter_load_failure(tmp_path: Path,
                                              checkpoint_dir: Path,
                                              prompts_path: Path) -> None:
    mod = _write_adapter(tmp_path, "boom_adapter.py",
                         "import nonexistent_module_xyz\n")
    with pytest.raises(PrerequisiteError, match="failed to load"):
        g.run_capture(str(mod), checkpoint_dir, prompts_path, {"mode": "greedy"})


def test_capture_refuses_adapter_checkpoint_rejection(tmp_path: Path,
                                                      checkpoint_dir: Path,
                                                      prompts_path: Path) -> None:
    """The checkpoint dir is structurally fine; the adapter itself refuses."""
    body = """
ADAPTER_NAME = "strict"
SUPPORTED_SAMPLING = frozenset({"greedy"})

def create_reference_adapter(checkpoint_dir):
    raise RuntimeError("no local checkpoint weights visible")
"""
    mod = _write_adapter(tmp_path, "strict_adapter.py", body)
    with pytest.raises(PrerequisiteError, match="refused the local checkpoint"):
        g.run_capture(str(mod), checkpoint_dir, prompts_path, {"mode": "greedy"})


def test_capture_refuses_unsupported_sampling(checkpoint_dir: Path,
                                              tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([
        {"prompt_id": "t1", "text": "x",
         "sampling": {"mode": "temperature", "temperature": 0.7}}]),
        encoding="utf-8")
    out = tmp_path / "out.json"
    with pytest.raises(PrerequisiteError, match="unsupported sampling mode"):
        _capture(checkpoint_dir, p)
    code = g.main_capture(["--adapter", str(FAKE_ADAPTER),
                           "--checkpoint-dir", str(checkpoint_dir),
                           "--prompts", str(p), "--out", str(out)])
    assert code == g.EXIT_REFUSED
    assert not out.exists()


def test_capture_refuses_seeded_without_seed(checkpoint_dir: Path,
                                             tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([
        {"prompt_id": "s", "text": "x", "sampling": {"mode": "seeded"}}]),
        encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="requires an integer 'seed'"):
        _capture(checkpoint_dir, p)


def test_capture_refuses_malformed_prompts(checkpoint_dir: Path,
                                           tmp_path: Path) -> None:
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps([{"prompt_id": "a", "text": "x"},
                              {"prompt_id": "a", "text": "y"}]), encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="duplicate prompt_id"):
        _capture(checkpoint_dir, p)
    p.write_text(json.dumps([{"prompt_id": "a", "input_ids": [1], "text": "x"}]),
                 encoding="utf-8")
    with pytest.raises(PrerequisiteError,
                       match="exactly one of input_ids / text"):
        _capture(checkpoint_dir, p)
    p.write_text(json.dumps("not an array"), encoding="utf-8")
    with pytest.raises(PrerequisiteError, match="non-empty JSON array"):
        _capture(checkpoint_dir, p)


def test_capture_refuses_incomplete_adapter_evidence(tmp_path: Path,
                                                     checkpoint_dir: Path,
                                                     prompts_path: Path) -> None:
    """An adapter that omits evidence must be refused, not captured."""
    body = """
ADAPTER_NAME = "incomplete"
SUPPORTED_SAMPLING = frozenset({"greedy"})

def create_reference_adapter(checkpoint_dir):
    class A:
        def run(self, prompt_id, text, input_ids, sampling):
            return {"input_ids": [1, 2, 3], "output_ids": [4],
                    "intermediates": {"t": {"shape": [1, 2],
                                            "sha256": "nothex"}}}
        def adapter_metadata(self):
            return {"library": "incomplete"}
    return A()
"""
    mod = _write_adapter(tmp_path, "incomplete_adapter.py", body)
    with pytest.raises(PrerequisiteError,
                       match="evidence rejected for prompt"):
        g.run_capture(str(mod), checkpoint_dir, prompts_path, {"mode": "greedy"})


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

def test_compare_match(checkpoint_dir: Path, prompts_path: Path,
                       tmp_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    ref_p = _write_doc(tmp_path, "ref.json", ref)
    cand = _capture(checkpoint_dir, prompts_path)
    cand_p = _write_doc(tmp_path, "cand.json", cand)
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "match"
    assert report["counts"] == {"prompts": 2, "match": 2, "divergent": 0,
                                "refused": 0}
    assert report["missing_evidence"] == []
    # CLI round-trip.
    code = g.main_compare(["--reference", str(ref_p), "--candidate", str(cand_p)])
    assert code == g.EXIT_MATCH
    assert g.load_capture_document(ref_p)["schema"] == CAPTURE_SCHEMA


def test_compare_divergent_output_ids(checkpoint_dir: Path, prompts_path: Path,
                                     tmp_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    for rec in cand["records"]:
        rec["output_ids"] = [99999, 1, 2, 3] + rec["output_ids"][4:]
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "divergent"
    assert report["counts"]["divergent"] == 2
    for ch in report["checks"]:
        assert ch["divergences"]
        assert any("output_ids" in d for d in ch["divergences"])


def test_compare_divergent_shape_and_checksum(checkpoint_dir: Path,
                                              prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    for rec in cand["records"]:
        rec["intermediates"]["logits"]["shape"] = [4, 64]
        rec["intermediates"]["logits"]["sha256"] = "0" * 64
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "divergent"
    joined = " ".join(d for ch in report["checks"] for d in ch["divergences"])
    assert "shape" in joined
    assert "checksum" in joined


def test_compare_values_within_tolerances_match(checkpoint_dir: Path,
                                                prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    for rec in cand["records"]:
        vals = rec["intermediates"]["logits"]["values"]
        rec["intermediates"]["logits"]["values"] = [
            v + (0.0001 if v >= 0 else -0.0001) for v in vals]
    strict = g.compare_documents(ref, cand, Tolerances())
    assert strict["verdict"] == "divergent"
    relaxed = g.compare_documents(ref, cand, Tolerances(atol=1e-3, rtol=0.0))
    assert relaxed["verdict"] == "match"
    assert relaxed["tolerances"] == {"atol": 1e-3, "rtol": 0.0}


def test_compare_values_beyond_tolerances_divergent(checkpoint_dir: Path,
                                                   prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    for rec in cand["records"]:
        vals = rec["intermediates"]["logits"]["values"]
        rec["intermediates"]["logits"]["values"] = [v + 0.5 for v in vals]
    report = g.compare_documents(ref, cand, Tolerances(atol=1e-3, rtol=1e-2))
    assert report["verdict"] == "divergent"
    for ch in report["checks"]:
        assert any("values[" in d for d in ch["divergences"])


def test_compare_refuses_missing_evidence(checkpoint_dir: Path,
                                          prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    # Candidate record loses output_ids and one intermediate.
    cand["records"][0]["output_ids"] = []
    del cand["records"][1]["intermediates"]["logits"]
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "refused"
    assert any("output_ids" in m for m in report["missing_evidence"])
    assert any("logits" in m and "missing from candidate" in m
               for m in report["missing_evidence"])


def test_compare_refuses_record_absent_in_candidate(checkpoint_dir: Path,
                                                    prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    cand["records"] = cand["records"][:1]
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "refused"
    assert any("present in only one capture document" in m
               for m in report["missing_evidence"])


def test_compare_refuses_checkpoint_hash_mismatch(checkpoint_dir: Path,
                                                  prompts_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    cand["checkpoint"]["config_json_sha256"] = "f" * 64
    report = g.compare_documents(ref, cand, Tolerances())
    assert report["verdict"] == "refused"
    assert any("config_json_sha256" in r for r in report["reasons"])


def test_compare_refuses_malformed_document(tmp_path: Path) -> None:
    good = {
        "schema": CAPTURE_SCHEMA,
        "captured_at_utc": "2026-01-01T00:00:00Z",
        "adapter": {"name": "a", "library": "l"},
        "checkpoint": {"config_json_sha256": "a" * 64,
                       "tokenizer_config_sha256": "b" * 64,
                       "optional_sha256": {}},
        "prompts": ["p1"],
        "records": [{
            "schema": g.RECORD_SCHEMA, "prompt_id": "p1",
            "sampling": {"mode": "greedy"},
            "input_ids": [1], "output_ids": [2],
            "intermediates": {"t": {"shape": [1, 2], "sha256": "c" * 64}},
            "adapter": {"name": "a"},
            "checkpoint": {"config_json_sha256": "a" * 64,
                           "tokenizer_config_sha256": "b" * 64,
                           "optional_sha256": {}},
        }],
    }
    ref_p = _write_doc(tmp_path, "ref.json", good)
    bad_p = tmp_path / "bad.json"
    bad_p.write_text('{"schema": "other"}', encoding="utf-8")
    with pytest.raises(CompareError, match="not a"):
        g.load_capture_document(bad_p)
    with pytest.raises(CompareError, match="not found"):
        g.load_capture_document(tmp_path / "missing.json")
    report = g.compare_documents(good, good, Tolerances())
    assert report["verdict"] == "match"
    code = g.main_compare(["--reference", str(ref_p), "--candidate", str(bad_p)])
    assert code == g.EXIT_REFUSED


def test_tolerances_validation() -> None:
    with pytest.raises(CompareError, match="atol"):
        Tolerances(atol=-1.0)
    with pytest.raises(CompareError, match="rtol"):
        Tolerances(rtol=float("nan"))
    with pytest.raises(CompareError):
        Tolerances(atol=True)  # bool is not accepted as a number
    assert Tolerances(atol=0.0, rtol=0.0)


# ---------------------------------------------------------------------------
# CLI end-to-end
# ---------------------------------------------------------------------------

def test_cli_capture_and_compare_roundtrip(checkpoint_dir: Path,
                                           prompts_path: Path,
                                           tmp_path: Path,
                                           capsys: Any) -> None:
    ref_out = tmp_path / "ref.json"
    code = g.main_capture(["--adapter", str(FAKE_ADAPTER),
                           "--checkpoint-dir", str(checkpoint_dir),
                           "--prompts", str(prompts_path),
                           "--out", str(ref_out)])
    assert code == g.EXIT_MATCH
    assert ref_out.is_file()

    # Candidate from a copy of the same checkpoint content: identical hashes
    # and identical deterministic outputs => match.
    cand_ckpt = tmp_path / "ckpt2"
    cand_ckpt.mkdir()
    shutil.copy2(checkpoint_dir / "config.json", cand_ckpt / "config.json")
    shutil.copy2(checkpoint_dir / "tokenizer_config.json",
                 cand_ckpt / "tokenizer_config.json")
    cand_out = tmp_path / "cand.json"
    code = g.main_capture(["--adapter", str(FAKE_ADAPTER),
                           "--checkpoint-dir", str(cand_ckpt),
                           "--prompts", str(prompts_path),
                           "--out", str(cand_out)])
    assert code == g.EXIT_MATCH

    report_out = tmp_path / "report.json"
    code = g.main_compare(["--reference", str(ref_out),
                           "--candidate", str(cand_out),
                           "--report-out", str(report_out)])
    assert code == g.EXIT_MATCH
    report = json.loads(report_out.read_text(encoding="utf-8"))
    assert report["verdict"] == "match"
    captured = capsys.readouterr()
    assert "verdict=match" in captured.out


def test_cli_compare_divergent_exit(checkpoint_dir: Path, prompts_path: Path,
                                   tmp_path: Path) -> None:
    ref = _capture(checkpoint_dir, prompts_path)
    cand = _capture(checkpoint_dir, prompts_path)
    cand["records"][0]["output_ids"] = [1]
    ref_p = _write_doc(tmp_path, "ref.json", ref)
    cand_p = _write_doc(tmp_path, "cand.json", cand)
    assert g.main_compare(["--reference", str(ref_p),
                           "--candidate", str(cand_p)]) == g.EXIT_DIVERGENT
    assert g.main([]) == g.EXIT_REFUSED
    assert g.main(["bogus"]) == g.EXIT_REFUSED


def test_script_wrappers_are_importable() -> None:
    """The scripts/ wrappers must expose the CLI entry points."""
    import importlib.util
    for script, attr in (("golden_capture.py", "main_capture"),
                         ("golden_compare.py", "main_compare")):
        path = REPO_ROOT / "scripts" / script
        spec = importlib.util.spec_from_file_location(f"gw_{script[:-3]}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, attr)
