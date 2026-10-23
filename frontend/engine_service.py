"""Gate-driven inference backend for the HTTP API.

Phase 3 serving path. The server only reports `inference: available` (and
serves completions) after the parity gate has passed for the configured
checkpoint: `scripts/parity_gate.py` writes a gate-status artifact on PASS,
and this module is the ONLY thing that turns that artifact into a serving
engine. A missing, failed, or stale gate artifact leaves the server in its
Phase-0 state (501, inference unavailable) - no partial serving.

Generation shells out to the `granite4-infer` binary named by the artifact
(`granite4-infer generate --tokens ... --max-tokens ... --seed 0`, greedy)
and decodes the produced ids with the checkpoint's byte-level BPE tokenizer.
Generation is bounded: `--max-tokens` (caller-requested, capped by
MAX_COMPLETION_TOKENS) plus one stop at the checkpoint's eos_token_id.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontend import chat_template, tokenizer

# Hard ceiling per request; the parity-validated engine has been verified at
# modest lengths, and the host forward is O(context) per step.
MAX_COMPLETION_TOKENS = 128

# A per-request timeout generous for host-side greedy decode.
_GENERATE_TIMEOUT_S = 600


class ServingError(Exception):
    """A rejected request (4xx) or engine failure (5xx) with an error body."""

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


@dataclass
class GateStatus:
    """A parsed, validated gate-status artifact."""

    passed: bool
    checkpoint: str
    checkpoint_sha: str
    engine: str
    prompt: str
    max_tokens: int
    min_match: float
    matched: int
    rate: float
    utc: str
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "GateStatus | None":
        """Load and validate a gate artifact. Returns None when absent or
        unusable (missing fields, non-PASS, unreadable binary). The caller must
        treat None as "gate has not passed"."""
        p = Path(path)
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or raw.get("pass") is not True:
            return None
        needed = ("checkpoint", "checkpoint_sha", "engine", "prompt",
                  "max_tokens", "min_match", "matched", "rate", "utc")
        if any(k not in raw for k in needed):
            return None
        for key in ("checkpoint", "engine"):
            if not isinstance(raw[key], str) or not raw[key]:
                return None
        g = cls(
            passed=True,
            checkpoint=str(raw["checkpoint"]),
            checkpoint_sha=str(raw["checkpoint_sha"]),
            engine=str(raw["engine"]),
            prompt=str(raw["prompt"]),
            max_tokens=int(raw["max_tokens"]),
            min_match=float(raw["min_match"]),
            matched=int(raw["matched"]),
            rate=float(raw["rate"]),
            utc=str(raw["utc"]),
            raw=raw,
        )
        # The binary must exist and be executable to serve with it.
        if not (Path(g.engine).is_file() and os.access(g.engine, os.X_OK)):
            return None
        if not Path(g.checkpoint).is_dir():
            return None
        return g


class EngineService:
    """A serving engine bound to a passed gate artifact.

    Production loads the checkpoint's tokenizer and shells out to the
    `granite4-infer` binary named by the gate; tests may inject a tokenizer
    and patch `_generate_ids` for hermetic runs.
    """

    def __init__(self, gate: GateStatus, *, timeout_s: int = _GENERATE_TIMEOUT_S,
                 tok=None, eos_id: int | None = None):
        self.gate = gate
        self.timeout_s = timeout_s
        if tok is None:
            tok_path = Path(gate.checkpoint) / "tokenizer.json"
            self._tok = tokenizer.ByteBPE(tok_path) if tok_path.is_file() else None
        else:
            self._tok = tok
        self._eos = self._eos_id(gate.checkpoint) if eos_id is None else eos_id

    @classmethod
    def from_gate_path(cls, path: str | Path) -> "EngineService | None":
        gate = GateStatus.load(path)
        if gate is None:
            return None
        return cls(gate)

    @staticmethod
    def _eos_id(checkpoint_dir: str) -> int | None:
        try:
            cfg = json.loads(
                (Path(checkpoint_dir) / "tokenizer_config.json").read_text(
                    encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            eos = cfg.get("eos_token")
            if isinstance(eos, str):
                return eos
        return None

    def is_available(self) -> bool:
        return self._tok is not None

    # -- generation ---------------------------------------------------------

    def _generate_ids(self, prompt_ids: list[int], max_tokens: int) -> list[int]:
        toks = ",".join(map(str, prompt_ids))
        cmd = [
            self.gate.engine, "generate", "--checkpoint", self.gate.checkpoint,
            "--tokens", toks, "--max-tokens", str(max_tokens), "--seed", "0",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            raise ServingError(504, "generation_timeout",
                               "engine did not finish within the deadline")
        if r.returncode != 0:
            raise ServingError(500, "engine_failed",
                               "engine generation failed: " + (r.stderr or "").strip()[:200])
        try:
            doc = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise ServingError(500, "engine_malformed", "engine emitted non-JSON output")
        ids = doc.get("generated")
        if not isinstance(ids, list):
            raise ServingError(500, "engine_malformed",
                               "engine output missing 'generated' token list")
        out: list[int] = []
        for t in ids:
            if not isinstance(t, int):
                raise ServingError(500, "engine_malformed", "engine emitted a non-int token")
            out.append(t)
            if self._eos is not None and t == self._eos:
                break
        return out

    def chat_completions(self, body: dict[str, Any], request_id: str) -> dict[str, Any]:
        messages = body.get("messages")
        try:
            text = chat_template.render(messages)
        except ValueError as e:
            raise ServingError(400, "invalid_body", str(e))
        if self._tok is None:
            raise ServingError(503, "tokenizer_unavailable",
                               "checkpoint tokenizer is not readable")
        prompt_ids = self._tok.encode(text)
        if not prompt_ids:
            raise ServingError(400, "empty_prompt", "prompt encodes to zero tokens")
        max_tokens = body.get("max_tokens", MAX_COMPLETION_TOKENS)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ServingError(400, "invalid_params",
                               f"max_tokens must be a positive integer (got {max_tokens!r})")
        max_tokens = min(max_tokens, MAX_COMPLETION_TOKENS)
        try:
            gen_ids = self._generate_ids(prompt_ids, max_tokens)
        except ServingError:
            raise
        except Exception as e:  # defensive: engine misbehaviour is a 500
            raise ServingError(500, "engine_failed", f"engine error: {e}")
        text_out = self._tok.decode(gen_ids) if gen_ids else ""
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": "stop" if gen_ids else "length",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(gen_ids),
                "total_tokens": len(prompt_ids) + len(gen_ids),
            },
        }


def load_serving_engine(gate_path: str) -> EngineService | None:
    """Load the serving engine from a gate-status artifact, or None."""
    return EngineService.from_gate_path(gate_path)
