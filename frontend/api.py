"""HTTP API of the Granite 4.2 8B engine.

Dependency-light by design: only the Python standard library
(http.server, json, time). The engine library is not loaded at import
time; the optional .so is probed per request when the status endpoint asks
for it.

Inference is gate-driven: the server loads a passing parity-gate artifact
(`scripts/parity_gate.py --gate-out <path>`; default `.gt/parity_gate.json`)
at startup and only then serves completions. Without it, the endpoints
stay in the Phase-0 state.

Endpoints:
    GET  /health                 -> {"status": "ok", "inference": "available|unavailable"}
    GET  /metrics                -> engine status (status code, device, message)
    GET  /v1/models              -> model card, status "available" after a passing gate
    POST /v1/chat/completions    -> 400 on invalid request; 200 with a completion
                                     when the gate has passed, else 501
                                     (inference has not been enabled)

The 501 body carries the same error shape OpenAI clients expect, so a
frontend that tolerates HTTP errors will surface the reason cleanly.
"""

from __future__ import annotations

import json
from typing import Any
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from frontend.config import (
    EXPECTED,
    MODEL_ID,
    GPU_TARGET,
    ModelConfig,
    architecture_dict,
    validate,
)

API_VERSION = "v1"
SERVER_NAME = "granite4-rtx-scaffold"

# The serving backend (Phase 3). Set at startup from the parity-gate artifact
# when it has passed; None means "gate has not passed" and the endpoints stay
# in the Phase-0 state (inference unavailable, 501 on chat).
_SERVING_ENGINE = None  # engine_service.EngineService | None



class ApiError(Exception):
    """A rejected request (4xx) or server condition (5xx), with an error body."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def to_body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error" if self.status == 400 else "server_error",
                "code": self.code,
            }
        }


def _not_found() -> ApiError:
    return ApiError(404, "not_found", "unknown route")


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "inference": "available" if _SERVING_ENGINE is not None else "unavailable",
        "model": MODEL_ID,
        "gpu_target": GPU_TARGET,
        "version": API_VERSION,
    }


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def _engine_status() -> dict[str, Any]:
    """Probe the optional engine library via ctypes.

    Returns {"present": bool, "status": ..., "cuda_device_present": ...}.
    The library is optional in Phase 0: a missing .so is a normal state,
    not an error.
    """
    from frontend import engine_bridge

    lib = engine_bridge.load()
    if lib is None:
        return {"present": False, "status": "unavailable", "cuda_device_present": False}
    try:
        return engine_bridge.status(lib)
    except Exception as exc:  # malformed ABI, broken build, ...
        return {
            "present": True,
            "status": "unavailable",
            "cuda_device_present": False,
            "message": f"engine status probe failed: {exc.__class__.__name__}",
        }


def metrics_payload() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "gpu_target": GPU_TARGET,
        "architecture": architecture_dict(),
        "inference": "available" if _SERVING_ENGINE is not None else "unavailable",
        "engine": _engine_status(),
    }


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

def models_payload() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "ibm",
                "status": "available" if _SERVING_ENGINE is not None else "unavailable",
                "architecture": architecture_dict(),
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

def _validate_chat_request(raw: bytes) -> dict[str, Any]:
    """Raise ApiError(400, ...) when the request body is not a valid
    minimal OpenAI chat-completions request. Returns a parsed dict otherwise."""
    if not raw:
        raise ApiError(400, "empty_body", "request body must be a JSON object")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(400, "invalid_json", f"body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_body", "request body must be a JSON object")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise ApiError(400, "invalid_model", "'model' must be a non-empty string")
    if model != MODEL_ID:
        raise ApiError(400, "unknown_model", f"model '{model}' is not served; use '{MODEL_ID}'")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "invalid_messages", "'messages' must be a non-empty array")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ApiError(400, "invalid_message", f"messages[{i}] must be an object")
        role = msg.get("role")
        if role not in ("system", "user", "assistant"):
            raise ApiError(
                400,
                "invalid_role",
                f"messages[{i}].role must be one of system|user|assistant, got {role!r}",
            )
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            raise ApiError(400, "invalid_content", f"messages[{i}].content must be a string")
        if content is None and role == "user":
            raise ApiError(400, "missing_content", f"messages[{i}].content is required for role=user")

    for key, typ in (("max_tokens", int), ("temperature", (int, float))):
        if key in body and not isinstance(body[key], typ):
            raise ApiError(400, f"invalid_{key}", f"'{key}' must be {typ}")
    if "temperature" in body:
        t = body["temperature"]
        if not (0.0 <= float(t) <= 2.0):
            raise ApiError(400, "invalid_temperature", "'temperature' must be in [0, 2]")

    return body


def chat_completions_response() -> tuple[int, dict[str, Any]]:
    """The 501 response for a valid request: inference has not been
    enabled (the parity gate has not passed for this deployment).

    Kept separate from validation so the tests can exercise both branches
    with the same handler shape.
    """
    status = HTTPStatus.NOT_IMPLEMENTED
    body = {
        "error": {
            "message": (
                f"{MODEL_ID} inference is not implemented on this deployment: "
                "the parity gate has not passed, so the engine is not enabled. "
                "Request validation succeeded; the endpoint returns 200 once "
                "a passing gate artifact is loaded."
            ),
            "type": "server_error",
            "code": "inference_unavailable",
        }
    }
    return int(status), body


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class ApiHandler(BaseHTTPRequestHandler):
    server_version = SERVER_NAME

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _handle_get(self, path: str) -> None:
        if path == "/health":
            self._send_json(200, health_payload())
        elif path == "/metrics":
            self._send_json(200, metrics_payload())
        elif path == "/v1/models":
            self._send_json(200, models_payload())
        else:
            self._send_json(404, _not_found().to_body())

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler convention)
        path = self.path.split("?", 1)[0]
        self._handle_get(path)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/v1/chat/completions":
            self._send_json(404, _not_found().to_body())
            return
        length = self.headers.get("Content-Length")
        try:
            length = int(length) if length is not None else 0
        except ValueError:
            self._send_json(400, ApiError(400, "bad_content_length", "Content-Length must be an integer").to_body())
            return
        try:
            raw = self.rfile.read(length) if length > 0 else b""
        except OSError:
            self._send_json(400, ApiError(400, "read_failed", "failed to read request body").to_body())
            return
        try:
            body = _validate_chat_request(raw)
        except ApiError as exc:
            self._send_json(exc.status, exc.to_body())
            return
        if _SERVING_ENGINE is not None:
            from frontend import engine_service

            try:
                resp = _SERVING_ENGINE.chat_completions(body, "cmpl-local")
            except engine_service.ServingError as exc:
                self._send_json(exc.status, exc.to_body())
                return
            self._send_json(200, resp)
            return
        status, resp_body = chat_completions_response()
        self._send_json(status, resp_body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - keep test output quiet
        del format, args
        pass


def set_serving_engine(engine) -> None:
    """Install the serving backend (an EngineService, or None to unset)."""
    global _SERVING_ENGINE
    _SERVING_ENGINE = engine


def make_server(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), ApiHandler)


def main() -> None:  # pragma: no cover - process entry point
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description=f"{SERVER_NAME} (P3 serving)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--gate-artifact",
        default=os.environ.get("GRANITE4_GATE", ".gt/parity_gate.json"),
        help="path to a passing parity-gate artifact; serving is off unless it exists and validates")
    args = parser.parse_args()
    from frontend import engine_service

    set_serving_engine(engine_service.load_serving_engine(args.gate_artifact))
    server = make_server(args.host, args.port)
    state = "available" if _SERVING_ENGINE is not None else "unavailable"
    print(f"{SERVER_NAME}: {args.host}:{args.port} (inference: {state})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
