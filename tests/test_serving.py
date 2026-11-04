"""P3 serving-path tests: the gate-driven flip from 501 to live inference.

The parity gate writes a status artifact on PASS; the server loads it and
only then reports `inference: available` and serves completions. These tests
pin both halves:

  * artifact validation (GateStatus.load) - the ONLY gate from "gate ran"
    to "engine serves": absent / non-PASS / malformed / unusable all mean
    "not passed";
  * the live API behaviour when an engine is installed: /health, /metrics
    and /v1/models flip to `available`, and /v1/chat/completions returns a
    200 completion (with an injected stub engine - no real weights).

They run hermetically: no real checkpoint, no GPU.
"""

from __future__ import annotations

import json
import sys
import subprocess
import threading
from collections.abc import Generator
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from frontend import api, engine_service
from frontend.config import MODEL_ID


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        # Deterministic, content-independent mapping: one id per char, +1.
        return [ord(c) + 1 for c in text[:64]]

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        return "stub-reply"


def _gate_dict(**over) -> dict:
    d = {
        "pass": True,
        "checkpoint": "/tmp/granite4_test_ckpt",
        "checkpoint_sha": "abcd1234",
        "engine": "/tmp/granite4_test_engine",
        "prompt": "The capital of France is",
        "max_tokens": 24,
        "min_match": 1.0,
        "matched": 24,
        "rate": 1.0,
        "utc": "2026-10-22T00:00:00+00:00",
    }
    d.update(over)
    return d
class _StubEngine(engine_service.EngineService):
    """EngineService with a fixed token stream (no real engine binary)."""

    def __init__(self, gate, generated: list[int], eos_id: int | None = None):
        super().__init__(gate, tok=_FakeTokenizer(), eos_id=eos_id)
        self._generated = list(generated)

    def _generate_ids(self, prompt_ids, max_tokens):
        return self._generated


@pytest.fixture
def served_server() -> Generator[ThreadingHTTPServer, None, None]:
    """A live server with a stub serving engine installed, restored after."""
    saved = api._SERVING_ENGINE
    gate = engine_service.GateStatus(
        passed=True, checkpoint="/tmp/ckpt", checkpoint_sha="x",
        engine="/tmp/bin", prompt="p", max_tokens=24, min_match=1.0,
        matched=24, rate=1.0, utc="t", raw={},
    )
    api.set_serving_engine(_StubEngine(gate, [7, 8, 9]))
    srv = api.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)
    api.set_serving_engine(saved)


def _http(server, method, path, body=None):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1],
                                      timeout=10)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def _post(server, body):
    return _http(server, "POST", "/v1/chat/completions", body)


def _valid_body(**over) -> bytes:
    payload = {"model": MODEL_ID,
               "messages": [{"role": "user", "content": "hi"}]}
    payload.update(over)
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Gate artifact validation
# ---------------------------------------------------------------------------

def test_gate_load_missing_returns_none(tmp_path):
    assert engine_service.GateStatus.load(tmp_path / "nope.json") is None


def test_gate_load_non_pass_returns_none(tmp_path):
    f = tmp_path / "g.json"
    d = _gate_dict()
    d["pass"] = False
    f.write_text(json.dumps(d))
    assert engine_service.GateStatus.load(f) is None


def test_gate_load_missing_fields_returns_none(tmp_path):
    d = _gate_dict()
    del d["engine"]
    f = tmp_path / "g.json"
    f.write_text(json.dumps(d))
    assert engine_service.GateStatus.load(f) is None


def test_gate_load_malformed_json_returns_none(tmp_path):
    f = tmp_path / "g.json"
    f.write_text("{not json")
    assert engine_service.GateStatus.load(f) is None


def test_gate_load_bad_engine_returns_none(tmp_path):
    f = tmp_path / "g.json"
    f.write_text(json.dumps(_gate_dict(engine=str(tmp_path / "missing_bin"))))
    assert engine_service.GateStatus.load(f) is None


def test_gate_load_valid(tmp_path):
    # A real executable + real dir validate.
    engine_bin = tmp_path / "engine"
    engine_bin.write_text("#!/bin/sh\n")
    engine_bin.chmod(0o755)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    f = tmp_path / "g.json"
    f.write_text(json.dumps(_gate_dict(engine=str(engine_bin),
                                        checkpoint=str(ckpt))))
    g = engine_service.GateStatus.load(f)
    assert g is not None
    assert g.passed is True
    assert g.checkpoint_sha == "abcd1234"
    assert g.max_tokens == 24
    assert g.rate == 1.0


def test_load_serving_engine_absent_returns_none(tmp_path):
    assert engine_service.load_serving_engine(str(tmp_path / "no.json")) is None


# ---------------------------------------------------------------------------
# Live API behaviour with a serving engine
# ---------------------------------------------------------------------------

def test_health_reports_available(served_server):
    status, _, data = _http(served_server, "GET", "/health")
    assert status == 200
    assert json.loads(data)["inference"] == "available"


def test_metrics_reports_available(served_server):
    status, _, data = _http(served_server, "GET", "/metrics")
    assert status == 200
    assert json.loads(data)["inference"] == "available"


def test_models_reports_available(served_server):
    status, _, data = _http(served_server, "GET", "/v1/models")
    assert status == 200
    assert json.loads(data)["data"][0]["status"] == "available"


def test_chat_completions_serves_200(served_server):
    status, _, data = _post(served_server, _valid_body())
    assert status == 200, data
    body = json.loads(data)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "stub-reply"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 3
    assert body["usage"]["prompt_tokens"] > 0


def test_chat_completions_empty_generation(served_server, monkeypatch):
    # A stub that emits nothing -> 200 with empty content, reason "length".
    import frontend.api as api_mod
    saved = api_mod._SERVING_ENGINE
    gate = engine_service.GateStatus(
        passed=True, checkpoint="/tmp/ckpt", checkpoint_sha="x",
        engine="/tmp/bin", prompt="p", max_tokens=24, min_match=1.0,
        matched=24, rate=1.0, utc="t", raw={},
    )
    api_mod.set_serving_engine(_StubEngine(gate, []))
    status, _, data = _post(served_server, _valid_body())
    api_mod.set_serving_engine(saved)
    assert status == 200, data
    body = json.loads(data)
    assert body["choices"][0]["message"]["content"] == ""
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 0


def test_chat_completions_invalid_max_tokens_400(served_server):
    # The API validator rejects non-int types; the serving backend rejects
    # non-positive ints and booleans. Both surface as 400.
    for bad in (0, -1, True):
        status, _, data = _post(served_server, _valid_body(max_tokens=bad))
        assert status == 400, (bad, data)
        assert json.loads(data)["error"]["code"] == "invalid_params"
    for bad in ("5", 1.5):
        status, _, data = _post(served_server, _valid_body(max_tokens=bad))
        assert status == 400, (bad, data)
        assert json.loads(data)["error"]["code"] == "invalid_max_tokens"


def test_chat_completions_engine_failure_500(served_server, monkeypatch):
    import frontend.api as api_mod
    saved = api_mod._SERVING_ENGINE
    gate = engine_service.GateStatus(
        passed=True, checkpoint="/tmp/ckpt", checkpoint_sha="x",
        engine="/tmp/bin", prompt="p", max_tokens=24, min_match=1.0,
        matched=24, rate=1.0, utc="t", raw={},
    )
    class _Boom(engine_service.EngineService):
        def __init__(self):
            super().__init__(gate, tok=_FakeTokenizer())
        def _generate_ids(self, prompt_ids, max_tokens):
            raise engine_service.ServingError(500, "engine_failed", "boom")
    api_mod.set_serving_engine(_Boom())
    status, _, data = _post(served_server, _valid_body())
    api_mod.set_serving_engine(saved)
    assert status == 500, data
    assert json.loads(data)["error"]["code"] == "engine_failed"


def test_eos_truncates_generation(monkeypatch, tmp_path):
    # The real _generate_ids must stop at the checkpoint's eos_token_id and
    # emit only tokens up to and including it.
    engine_bin = tmp_path / "engine"
    engine_bin.write_text("#!/bin/sh\n")
    engine_bin.chmod(0o755)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tokenizer_config.json").write_text(json.dumps(
        {"eos_token": "<|im_end|>", "bos_token": "<|start|>"}))
    gate = engine_service.GateStatus(
        passed=True, checkpoint=str(ckpt), checkpoint_sha="x",
        engine=str(engine_bin), prompt="p", max_tokens=24, min_match=1.0,
        matched=24, rate=1.0, utc="t", raw={},
    )
    eng = engine_service.EngineService(gate, tok=_FakeTokenizer(),
                                      eos_id=2)
    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            cmd, 0,
            stdout=json.dumps({"generated": [5, 2, 7, 8]}), stderr="")

    monkeypatch.setattr(engine_service.subprocess, "run", _fake_run)
    out = eng._generate_ids([1, 3, 4], 24)
    assert out == [5, 2]  # truncated at eos
    # And the command is greedy + the right token string.
    assert captured["cmd"][1] == "generate"
    assert captured["cmd"][captured["cmd"].index("--tokens") + 1] == "1,3,4"
    assert captured["cmd"][captured["cmd"].index("--seed") + 1] == "0"
    # The engine gets the stop token too, and no --gpu without a GPU gate.
    assert captured["cmd"][captured["cmd"].index("--eos") + 1] == "2"
    assert "--gpu" not in captured["cmd"]


def test_eos_id_comes_from_config_json(tmp_path):
    # tokenizer_config.json only names the eos token string; the integer id
    # the generated ids are compared against is config.json's eos_token_id.
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    (ckpt / "config.json").write_text(json.dumps({"eos_token_id": 100257}))
    assert engine_service.EngineService._eos_id(str(ckpt)) == 100257
    (ckpt / "config.json").write_text(json.dumps({"eos_token_id": "x"}))
    assert engine_service.EngineService._eos_id(str(ckpt)) is None
    assert engine_service.EngineService._eos_id(str(tmp_path / "missing")) is None


_FAKE_SERVE = """#!{python}
import json, sys
args = sys.argv[1:]
assert args[0] == "serve" and args[args.index("--gpu") + 1] == "1", args
with open({log!r}, "a") as f:
    f.write("start\\n")
print(json.dumps({{"ready": True, "max_context": 8192}}), flush=True)
eagle = args[args.index("--eagle") + 1] if "--eagle" in args else None
for line in sys.stdin:
    req = json.loads(line)
    if (req.get("eagle", 0) > 0) != (eagle is not None):
        print(json.dumps({{"error": "eagle request without a loaded head"}}), flush=True)
        continue
    if req["tokens"] == [8]:
        print(json.dumps({{"generated": [req.get("eagle", 0), 2]}}), flush=True)
        continue
    if req["tokens"] == [666]:
        sys.exit(3)  # simulate an engine crash
    if req["tokens"] == [7]:
        print(json.dumps({{"error": "prompt + max_tokens exceeds the context"}}), flush=True)
        continue
    ids = [t + 1 for t in req["tokens"]][: req["max_tokens"]] + [req.get("eos", 99), 42]
    print(json.dumps({{"generated": ids}}), flush=True)
"""


def _gpu_engine(tmp_path, monkeypatch, **kw):
    log = tmp_path / "starts.log"
    engine_bin = tmp_path / "engine"
    engine_bin.write_text(_FAKE_SERVE.format(python=sys.executable, log=str(log)))
    engine_bin.chmod(0o755)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(_gate_dict(engine=str(engine_bin), checkpoint=str(ckpt),
                                          gpu=True)))
    gate = engine_service.GateStatus.load(path)
    assert gate is not None and gate.gpu

    def _no_run(*a, **kw):
        raise AssertionError("a GPU gate must not spawn one engine per request")

    monkeypatch.setattr(engine_service.subprocess, "run", _no_run)
    kw.setdefault("eagle_head", "")
    return engine_service.EngineService(gate, tok=_FakeTokenizer(), eos_id=2, **kw), log


def test_gpu_engine_eagle_head_and_k_reach_the_engine(monkeypatch, tmp_path):
    eng, log = _gpu_engine(tmp_path, monkeypatch, eagle_head="/heads/h.safetensors", eagle_k=4)
    try:
        assert eng._generate_ids([8], 8) == [4, 2]  # the fake echoes the request's K
        assert eng._generate_ids([4], 8) == [5, 2]
    finally:
        eng.close()


def test_gpu_engine_without_head_does_not_draft(monkeypatch, tmp_path):
    eng, _ = _gpu_engine(tmp_path, monkeypatch, eagle_k=4)
    try:
        assert eng._generate_ids([8], 8) == [0, 2]
    finally:
        eng.close()


def test_gpu_gate_uses_one_persistent_engine(monkeypatch, tmp_path):
    eng, log = _gpu_engine(tmp_path, monkeypatch)
    try:
        assert eng._generate_ids([10, 20], 8) == [11, 21, 2]  # stops at eos 2
        assert eng._generate_ids([5], 8) == [6, 2]
        assert log.read_text().count("start") == 1  # loaded once, reused
    finally:
        eng.close()


def test_gpu_engine_warm_starts_once(monkeypatch, tmp_path):
    eng, log = _gpu_engine(tmp_path, monkeypatch)
    try:
        eng.warm()
        eng.warm()
        assert eng._generate_ids([4], 8) == [5, 2]
        assert log.read_text().count("start") == 1
    finally:
        eng.close()


def test_gpu_engine_error_reply_is_500_and_engine_survives(monkeypatch, tmp_path):
    eng, log = _gpu_engine(tmp_path, monkeypatch)
    try:
        with pytest.raises(engine_service.ServingError) as e:
            eng._generate_ids([7], 8)
        assert e.value.status == 500 and "exceeds the context" in e.value.message
        assert eng._generate_ids([4], 8) == [5, 2]
        assert log.read_text().count("start") == 1
    finally:
        eng.close()


def test_gpu_engine_crash_restarts(monkeypatch, tmp_path):
    eng, log = _gpu_engine(tmp_path, monkeypatch)
    try:
        with pytest.raises(engine_service.ServingError) as e:
            eng._generate_ids([666], 8)
        assert e.value.code == "engine_failed"
        assert eng._generate_ids([4], 8) == [5, 2]  # a fresh engine serves
        assert log.read_text().count("start") == 2
    finally:
        eng.close()


class _RecordingTokenizer(_FakeTokenizer):
    def __init__(self):
        self.decoded: list[list[int]] = []

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        self.decoded.append(list(ids))
        return "x" * len(list(ids))


def _completion(generated, eos_id, max_tokens):
    gate = engine_service.GateStatus(
        passed=True, checkpoint="/tmp/ckpt", checkpoint_sha="x",
        engine="/tmp/bin", prompt="p", max_tokens=24, min_match=1.0,
        matched=24, rate=1.0, utc="t", raw={},
    )
    eng = _StubEngine(gate, generated, eos_id=eos_id)
    tok = _RecordingTokenizer()
    eng._tok = tok
    body = {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": max_tokens}
    return eng.chat_completions(body, "id"), tok


def test_eos_is_not_part_of_the_text_and_finishes_with_stop():
    out, tok = _completion([5, 6, 2], eos_id=2, max_tokens=8)
    assert tok.decoded == [[5, 6]]
    assert out["choices"][0]["message"]["content"] == "xx"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["completion_tokens"] == 3


def test_hitting_max_tokens_finishes_with_length():
    out, _ = _completion([5, 6, 7], eos_id=2, max_tokens=3)
    assert out["choices"][0]["finish_reason"] == "length"
    out, _ = _completion([5, 6, 2], eos_id=2, max_tokens=3)  # eos as the last allowed
    assert out["choices"][0]["finish_reason"] == "stop"


def test_default_health_is_unavailable():
    # Without a loaded artifact, the server is in the Phase-0 state.
    saved = api._SERVING_ENGINE
    api.set_serving_engine(None)
    try:
        assert api.health_payload()["inference"] == "unavailable"
        assert api.metrics_payload()["inference"] == "unavailable"
        assert api.models_payload()["data"][0]["status"] == "unavailable"
    finally:
        api.set_serving_engine(saved)
