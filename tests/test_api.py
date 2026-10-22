"""Live HTTP tests of the Phase 0 API. Runs without CUDA/GPU: the server
is stdlib-only and the engine library is optional.

Tests hit a real socket (127.0.0.1:0) so header handling, body limits and
threading behave exactly as in production."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.server import ThreadingHTTPServer

import pytest

from frontend import api
from frontend.config import EXPECTED, MODEL_ID, architecture_dict


@pytest.fixture(scope="module")
def server() -> Generator[ThreadingHTTPServer, None, None]:
    srv = api.make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def _http(server: ThreadingHTTPServer, method: str, path: str, body: bytes | None = None):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, dict(resp.getheaders()), data


def _post(server, body: bytes):
    return _http(server, "POST", "/v1/chat/completions", body)


def _valid_body(**over) -> bytes:
    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": "hi"}]}
    payload.update(over)
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health(server):
    status, headers, data = _http(server, "GET", "/health")
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    body = json.loads(data)
    assert body["status"] == "ok"
    assert body["inference"] == "unavailable"
    assert body["model"] == MODEL_ID
    assert body["gpu_target"] == "sm_89"


def test_health_query_string_ignored(server):
    status, _, _ = _http(server, "GET", "/health?x=1")
    assert status == 200


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics(server):
    status, _, data = _http(server, "GET", "/metrics")
    assert status == 200
    body = json.loads(data)
    assert body["model"] == MODEL_ID
    assert body["gpu_target"] == "sm_89"
    assert body["inference"] == "unavailable"
    assert body["architecture"] == architecture_dict()
    assert body["architecture"] == {
        "num_layers": EXPECTED.num_layers,
        "hidden_size": EXPECTED.hidden_size,
        "intermediate_size": EXPECTED.intermediate_size,
        "num_attention_heads": EXPECTED.num_attention_heads,
        "num_key_value_heads": EXPECTED.num_key_value_heads,
        "head_dim": EXPECTED.head_dim,
        "vocab_size": EXPECTED.vocab_size,
        "max_context_length": EXPECTED.max_context_length,
    }
    # The engine library is optional in Phase 0; the report must be honest
    # about its presence either way.
    assert "present" in body["engine"]
    assert body["engine"]["status"] in ("unavailable", "unimplemented")


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

def test_models(server):
    status, _, data = _http(server, "GET", "/v1/models")
    assert status == 200
    body = json.loads(data)
    assert body["object"] == "list"
    [model] = body["data"]
    assert model["id"] == MODEL_ID
    assert model["object"] == "model"
    assert model["status"] == "unavailable"
    assert model["architecture"]["num_layers"] == 40


# ---------------------------------------------------------------------------
# 404s
# ---------------------------------------------------------------------------

def test_unknown_get_route_is_404(server):
    status, _, data = _http(server, "GET", "/nope")
    assert status == 404
    assert json.loads(data)["error"]["code"] == "not_found"


def test_unknown_post_route_is_404(server):
    status, _, data = _http(server, "POST", "/nope", b"{}")
    assert status == 404
    assert json.loads(data)["error"]["code"] == "not_found"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — validation matrix (400s)
# ---------------------------------------------------------------------------

def _code(data: bytes) -> str:
    return json.loads(data)["error"]["code"]


def test_empty_body_is_400(server):
    status, _, data = _post(server, b"")
    assert status == 400
    assert _code(data) == "empty_body"


def test_invalid_json_is_400(server):
    status, _, data = _post(server, b"{not json")
    assert status == 400
    assert _code(data) == "invalid_json"


def test_non_object_json_is_400(server):
    status, _, data = _post(server, b"[1,2,3]")
    assert status == 400
    assert _code(data) == "invalid_body"


def test_missing_model_is_400(server):
    status, _, data = _post(
        server, json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    )
    assert status == 400
    assert _code(data) == "invalid_model"


def test_empty_model_is_400(server):
    status, _, data = _post(
        server, json.dumps({"model": "", "messages": [{"role": "user", "content": "hi"}]}).encode()
    )
    assert status == 400
    assert _code(data) == "invalid_model"


def test_unknown_model_is_400(server):
    status, _, data = _post(
        server, json.dumps({"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}).encode()
    )
    assert status == 400
    assert _code(data) == "unknown_model"
    assert MODEL_ID in json.loads(data)["error"]["message"]


def test_missing_messages_is_400(server):
    status, _, data = _post(server, json.dumps({"model": MODEL_ID}).encode())
    assert status == 400
    assert _code(data) == "invalid_messages"


def test_empty_messages_is_400(server):
    status, _, data = _post(server, json.dumps({"model": MODEL_ID, "messages": []}).encode())
    assert status == 400
    assert _code(data) == "invalid_messages"


def test_non_list_messages_is_400(server):
    status, _, data = _post(server, json.dumps({"model": MODEL_ID, "messages": "hi"}).encode())
    assert status == 400
    assert _code(data) == "invalid_messages"


def test_non_object_message_is_400(server):
    status, _, data = _post(
        server, json.dumps({"model": MODEL_ID, "messages": ["hi"]}).encode()
    )
    assert status == 400
    assert _code(data) == "invalid_message"


def test_bad_role_is_400(server):
    body = _valid_body(messages=[{"role": "tool", "content": "x"}])
    status, _, data = _post(server, body)
    assert status == 400
    assert _code(data) == "invalid_role"


def test_bad_content_type_is_400(server):
    body = _valid_body(messages=[{"role": "user", "content": 42}])
    status, _, data = _post(server, body)
    assert status == 400
    assert _code(data) == "invalid_content"


def test_user_missing_content_is_400(server):
    body = _valid_body(messages=[{"role": "user"}])
    status, _, data = _post(server, body)
    assert status == 400
    assert _code(data) == "missing_content"


def test_non_int_max_tokens_is_400(server):
    status, _, data = _post(server, _valid_body(max_tokens="16"))
    assert status == 400
    assert _code(data) == "invalid_max_tokens"


def test_non_numeric_temperature_is_400(server):
    status, _, data = _post(server, _valid_body(temperature="warm"))
    assert status == 400
    assert _code(data) == "invalid_temperature"


def test_out_of_range_temperature_is_400(server):
    status, _, data = _post(server, _valid_body(temperature=3.0))
    assert status == 400
    assert _code(data) == "invalid_temperature"


def test_error_shape_is_openai_like(server):
    status, _, data = _post(server, b"garbage")
    assert status == 400
    err = json.loads(data)["error"]
    assert set(err) == {"message", "type", "code"}
    assert err["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Valid request -> 501 inference_unavailable
# ---------------------------------------------------------------------------

def test_valid_request_is_501(server):
    status, headers, data = _post(server, _valid_body())
    assert status == 501
    body = json.loads(data)
    assert body["error"]["code"] == "inference_unavailable"
    assert body["error"]["type"] == "server_error"
    assert "not implemented" in body["error"]["message"]


def test_valid_request_with_options_is_501(server):
    status, _, _ = _post(server, _valid_body(max_tokens=64, temperature=0.7))
    assert status == 501


def test_multiturn_valid_request_is_501(server):
    body = _valid_body(
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
    )
    status, _, _ = _post(server, body)
    assert status == 501


def test_unicode_content_is_valid(server):
    body = _valid_body(messages=[{"role": "user", "content": "こんにちは"}])
    status, _, _ = _post(server, body)
    assert status == 501


# ---------------------------------------------------------------------------
# Unit level: chat_completions_response shape
# ---------------------------------------------------------------------------

def test_chat_completions_response_shape():
    status, body = api.chat_completions_response()
    assert status == 501
    assert body["error"]["code"] == "inference_unavailable"
