"""Gate-driven inference backend for the HTTP API.

Phase 3 serving path. The server only reports `inference: available` (and
serves completions) after the parity gate has passed for the configured
checkpoint: `scripts/parity_gate.py` writes a gate-status artifact on PASS,
and this module is the ONLY thing that turns that artifact into a serving
engine. A missing, failed, or stale gate artifact leaves the server in its
Phase-0 state (501, inference unavailable) - no partial serving.

Generation runs the `granite4-infer` binary named by the artifact and decodes
the produced ids with the checkpoint's byte-level BPE tokenizer. When the gate
validated the GPU path, one persistent `granite4-infer serve --gpu 1` process
keeps the model resident and answers every request (no per-request load);
otherwise each request shells out to `granite4-infer generate --tokens ...
--max-tokens ... --seed 0` (greedy, host oracle). Generation is bounded:
`--max-tokens` (caller-requested, capped by MAX_COMPLETION_TOKENS) plus one
stop at the checkpoint's eos_token_id, which the engine also receives
(`--eos`) so it stops computing there.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from frontend import chat_template, tokenizer

# Hard ceiling per request on the host-oracle path (its forward replays the
# whole context every step). The GPU engine allows up to its context
# (SERVE_MAX_CONTEXT - prompt) and defaults to GPU_DEFAULT_MAX_TOKENS.
MAX_COMPLETION_TOKENS = 128
GPU_DEFAULT_MAX_TOKENS = 1024

# A per-request timeout generous for host-side greedy decode (the GPU path
# finishes MAX_COMPLETION_TOKENS in a few seconds).
_GENERATE_TIMEOUT_S = 600

# The persistent GPU engine: positions per request, and how long its one-time
# load (checkpoint pack + upload) may take.
SERVE_MAX_CONTEXT = int(os.environ.get("GRANITE4_MAX_CONTEXT", "8192"))
_SERVE_STARTUP_TIMEOUT_S = 600

# Speculative decoding with a trained EAGLE draft head (engine `--eagle`):
# the head file and the tokens drafted per round. The engine's exact greedy
# verify keeps the output identical to plain greedy decoding, so this is a
# speed setting only; unset (or K = 0) decodes without drafting.
EAGLE_HEAD = os.environ.get("GRANITE4_EAGLE_HEAD", "")
EAGLE_K = int(os.environ.get("GRANITE4_EAGLE_K", "3"))
# Resident conversations: the engine keeps SESSIONS sessions (each up to
# SERVE_MAX_CONTEXT positions, each with its own EAGLE state) on one shared KV
# pool, and routes a request to the session whose cached history is the longest
# prefix of its prompt, so interleaved chats each resume instead of
# re-prefilling. KV_POOL_TOKENS sizes the pool in positions; 0 (default) is
# SESSIONS x SERVE_MAX_CONTEXT, i.e. every session can fill its context at once.
SESSIONS = int(os.environ.get("GRANITE4_SESSIONS", "1"))
KV_POOL_TOKENS = int(os.environ.get("GRANITE4_KV_POOL_TOKENS", "0"))


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


class ServeProcess:
    """One persistent `granite4-infer serve` engine.

    Requests are serialized (the engine runs one at a time). The process is
    started on first use; a timeout, a malformed reply, or a dead process
    kills it, and the next request starts a fresh one.
    """

    def __init__(self, cmd: list[str], *, startup_timeout_s: float = _SERVE_STARTUP_TIMEOUT_S):
        self._cmd = cmd
        self._startup_timeout_s = startup_timeout_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()

    def _start(self) -> None:
        proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, bufsize=1)
        lines: queue.Queue[str | None] = queue.Queue()

        def _pump() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)  # EOF: the engine exited

        threading.Thread(target=_pump, daemon=True).start()
        self._proc, self._lines = proc, lines
        ready = self._read(self._startup_timeout_s)
        if ready.get("ready") is not True:
            self.close()
            raise ServingError(500, "engine_failed", "engine did not report ready")

    def _read(self, timeout_s: float) -> dict[str, Any]:
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise ServingError(504, "generation_timeout",
                               "engine did not finish within the deadline")
        if line is None:
            self.close()
            raise ServingError(500, "engine_failed", "engine process exited")
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            self.close()
            raise ServingError(500, "engine_malformed", "engine emitted non-JSON output")
        if not isinstance(doc, dict):
            self.close()
            raise ServingError(500, "engine_malformed", "engine emitted a non-object")
        return doc

    def warm(self) -> None:
        """Start the engine (load the model) if it is not running."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()

    def request(self, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self.close()
                raise ServingError(500, "engine_failed", "engine process exited")
            return self._read(timeout_s)

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


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
    gpu: bool = False  # the gate validated `generate --gpu 1`

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
            gpu=raw.get("gpu") is True,
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
                 tok=None, eos_id: int | None = None, eagle_head: str | None = None,
                 eagle_k: int | None = None):
        self.gate = gate
        self.timeout_s = timeout_s
        if tok is None:
            tok_path = Path(gate.checkpoint) / "tokenizer.json"
            self._tok = tokenizer.ByteBPE(tok_path) if tok_path.is_file() else None
        else:
            self._tok = tok
        self._eos = self._eos_id(gate.checkpoint) if eos_id is None else eos_id
        head = EAGLE_HEAD if eagle_head is None else eagle_head
        k = EAGLE_K if eagle_k is None else eagle_k
        self._eagle_k = k if (gate.gpu and head and k > 0) else 0
        cmd = [gate.engine, "serve", "--checkpoint", gate.checkpoint, "--gpu", "1",
               "--max-context", str(SERVE_MAX_CONTEXT), "--sessions", str(max(1, SESSIONS))]
        if KV_POOL_TOKENS > 0:
            cmd += ["--kv-pool-tokens", str(KV_POOL_TOKENS)]
        if self._eagle_k:
            cmd += ["--eagle", head, "--eagle-k", str(self._eagle_k)]
        self._serve = ServeProcess(cmd) if gate.gpu else None

    @classmethod
    def from_gate_path(cls, path: str | Path) -> "EngineService | None":
        gate = GateStatus.load(path)
        if gate is None:
            return None
        return cls(gate)

    @staticmethod
    def _eos_id(checkpoint_dir: str) -> int | None:
        """The checkpoint's integer eos_token_id (config.json), or None.

        tokenizer_config.json only names the eos *token* (a string such as
        "<|im_end|>"), which never equals a generated id; the id itself lives
        in config.json."""
        try:
            cfg = json.loads(
                (Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        eos = cfg.get("eos_token_id")
        if isinstance(eos, int) and not isinstance(eos, bool) and eos >= 0:
            return eos
        return None

    def is_available(self) -> bool:
        return self._tok is not None

    # -- generation ---------------------------------------------------------

    def _generate_ids(self, prompt_ids: list[int], max_tokens: int) -> list[int]:
        if self._serve is not None:
            payload: dict[str, Any] = {"tokens": prompt_ids, "max_tokens": max_tokens,
                                       "seed": 0}
            if self._eos is not None:
                payload["eos"] = self._eos
            if self._eagle_k:
                payload["eagle"] = self._eagle_k
            doc = self._serve.request(payload, self.timeout_s)
            if "error" in doc:
                raise ServingError(500, "engine_failed",
                                   "engine generation failed: " + str(doc["error"])[:200])
            return self._parse_ids(doc)
        toks = ",".join(map(str, prompt_ids))
        cmd = [
            self.gate.engine, "generate", "--checkpoint", self.gate.checkpoint,
            "--tokens", toks, "--max-tokens", str(max_tokens), "--seed", "0",
        ]
        if self._eos is not None:
            cmd += ["--eos", str(self._eos)]
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
        if not isinstance(doc, dict):
            raise ServingError(500, "engine_malformed", "engine emitted a non-object")
        return self._parse_ids(doc)

    def _parse_ids(self, doc: dict[str, Any]) -> list[int]:
        ids = doc.get("generated")
        if not isinstance(ids, list):
            raise ServingError(500, "engine_malformed",
                               "engine output missing 'generated' token list")
        out: list[int] = []
        for t in ids:
            if not isinstance(t, int) or isinstance(t, bool):
                raise ServingError(500, "engine_malformed", "engine emitted a non-int token")
            out.append(t)
            if self._eos is not None and t == self._eos:
                break
        return out

    def warm(self) -> None:
        """Load the persistent engine ahead of the first request (no-op for
        the per-request host path). Failures are logged, not raised: the
        next request retries."""
        if self._serve is None:
            return
        try:
            self._serve.warm()
            print("granite4: GPU engine ready", flush=True)
        except ServingError as e:
            print(f"granite4: GPU engine warm-up failed: {e.message}", flush=True)

    def close(self) -> None:
        """Stop the persistent engine, if one is running."""
        if self._serve is not None:
            self._serve.close()

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
        if self._serve is not None:
            # A draft round reserves positions for the anchor and K drafts.
            reserve = self._eagle_k + 1 if self._eagle_k else 0
            room = SERVE_MAX_CONTEXT - len(prompt_ids) - reserve
            if room < 1:
                raise ServingError(400, "context_length_exceeded",
                                   f"prompt is {len(prompt_ids)} tokens; the context is "
                                   f"{SERVE_MAX_CONTEXT}")
            ceiling, default = room, min(GPU_DEFAULT_MAX_TOKENS, room)
        else:
            ceiling = default = MAX_COMPLETION_TOKENS
        max_tokens = body.get("max_tokens", default)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ServingError(400, "invalid_params",
                               f"max_tokens must be a positive integer (got {max_tokens!r})")
        max_tokens = min(max_tokens, ceiling)
        try:
            gen_ids = self._generate_ids(prompt_ids, max_tokens)
        except ServingError:
            raise
        except Exception as e:  # defensive: engine misbehaviour is a 500
            raise ServingError(500, "engine_failed", f"engine error: {e}")
        # The stop token ends the turn; it is counted but not part of the text.
        stopped = bool(gen_ids) and self._eos is not None and gen_ids[-1] == self._eos
        text_ids = gen_ids[:-1] if stopped else gen_ids
        text_out = self._tok.decode(text_ids) if text_ids else ""
        finish = "stop" if stopped or (gen_ids and len(gen_ids) < max_tokens) else "length"
        return {
            "id": request_id,
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": finish,
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
