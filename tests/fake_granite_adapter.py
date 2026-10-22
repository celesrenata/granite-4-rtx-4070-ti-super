"""Synthetic deterministic fake reference adapter for the golden tooling.

Stdlib only. Simulates a local Granite 4.2 8B reference adapter **without**
torch, CUDA, model weights, or network access. The golden tooling under test
only needs the adapter *protocol*; this module provides it with fully
deterministic, hash-derived "model outputs" so tests can assert exact
token-ID / checksum / shape behavior.

Protocol (what a real local Granite reference adapter must supply):

* ``ADAPTER_NAME`` - non-empty string;
* ``SUPPORTED_SAMPLING`` - non-empty collection of supported mode strings;
* ``create_reference_adapter(checkpoint_dir)`` - returns an adapter
  instance or raises if the *local* checkpoint prerequisites are missing
  (nothing is downloaded);
* instance ``run(prompt_id, text, input_ids, sampling)`` - returns
  ``{"input_ids": [...], "output_ids": [...],
     "intermediates": {name: {"shape": [...], "sha256": "...",
                              "checksum_kind": "...", "values": [...]}}}``;
* instance ``adapter_metadata()`` - JSON-serializable metadata dict.

Only deterministic modes are supported: ``greedy`` and ``seeded`` (the
latter requires an integer seed). Stochastic modes (temperature, top-p,
...) are rejected with :class:`UnsupportedSamplingMode` - the tooling must
refuse them rather than claim a result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ADAPTER_NAME = "fake-granite-reference"
SUPPORTED_SAMPLING = frozenset({"greedy", "seeded"})
LIBRARY = "fake"
VERSION = "0.0.0"

VOCAB = 100_000
MAX_NEW_DEFAULT = 16
MAX_NEW_CAP = 64
_N_VALUES = 8


class UnsupportedSamplingMode(RuntimeError):
    """The adapter does not implement the requested sampling mode."""


class InvalidCheckpointError(RuntimeError):
    """Local checkpoint prerequisites are missing or unreadable."""


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _token_stream(key: str, count: int, input_ids: list[int] | None) -> list[int]:
    """Deterministic LCG stream seeded by a SHA256 digest of the key."""
    seed = int.from_bytes(
        hashlib.sha256(_canonical_json(["tokens", key, input_ids])).digest()[:8],
        "big") or 1
    out: list[int] = []
    for _ in range(count):
        seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**64)
        out.append((seed >> 32) % VOCAB)
    return out


def _checksum(key: str, payload: Any) -> str:
    return hashlib.sha256(_canonical_json([key, payload])).hexdigest()


def _values(key: str, count: int) -> list[float]:
    """Deterministic finite floats in (-1, 1), derived from a SHA256 chain."""
    base = hashlib.sha256(_canonical_json(["values", key, count])).digest()
    vals: list[float] = []
    for i in range(count):
        chunk = hashlib.sha256(base + i.to_bytes(4, "big")).digest()
        mantissa = int.from_bytes(chunk[:6], "big") / 2**48
        sign = -1.0 if chunk[6] & 1 else 1.0
        vals.append(sign * mantissa)
    return vals


class FakeAdapter:
    """Deterministic stand-in for a local Granite reference model run."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self._check_local_prerequisites()

    def _check_local_prerequisites(self) -> None:
        d = self.checkpoint_dir
        if not d.is_dir():
            raise InvalidCheckpointError(
                f"checkpoint directory not found: {d} - supply a local "
                "checkpoint directory; nothing is downloaded")
        for name in ("config.json", "tokenizer_config.json"):
            p = d / name
            if not p.is_file():
                raise InvalidCheckpointError(f"missing prerequisite file: {p}")
            try:
                data = json.loads(p.read_bytes().decode("utf-8"))
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise InvalidCheckpointError(
                    f"{name}: not readable JSON ({type(exc).__name__})") from exc
            if not isinstance(data, dict):
                raise InvalidCheckpointError(
                    f"{name}: top-level JSON value must be an object")

    def _text_ids(self, text: str) -> list[int]:
        """Deterministic token ids for a text prompt (fake tokenization)."""
        return _token_stream(f"{ADAPTER_NAME}|text|{text}", max(len(text), 8), None)

    def run(self, prompt_id: str, text: str | None,
            input_ids: list[int] | None, sampling: dict[str, Any]) -> dict[str, Any]:
        """Run the deterministic fake model; refuses unsupported sampling."""
        mode = sampling.get("mode")
        if mode not in SUPPORTED_SAMPLING:
            raise UnsupportedSamplingMode(
                f"sampling mode {mode!r} not supported "
                f"(supported: {sorted(SUPPORTED_SAMPLING)})")
        seed = sampling.get("seed")
        if mode == "seeded" and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise UnsupportedSamplingMode(
                "sampling mode 'seeded' requires an integer 'seed'")
        n = sampling.get("max_new_tokens", MAX_NEW_DEFAULT)
        if (not isinstance(n, int) or isinstance(n, bool)
                or not 1 <= n <= MAX_NEW_CAP):
            raise ValueError("max_new_tokens must be an int in [1, 64]")
        ids = list(input_ids) if input_ids is not None else self._text_ids(text or "")
        key = f"{ADAPTER_NAME}|{prompt_id}|{mode}|{seed}"
        out_ids = _token_stream(key, n, ids)
        intermediates: dict[str, Any] = {}
        for name, shape, payload in (
            ("layer_0.hidden_state", [1, len(ids), 256],
             ("hidden", key, len(ids))),
            ("logits", [len(out_ids), 64], ("logits", key, len(out_ids))),
        ):
            intermediates[name] = {
                "shape": shape,
                "checksum_kind": "sha256-of-deterministic-payload",
                "sha256": _checksum(name, payload),
                "values": _values(name + ":" + key, _N_VALUES),
            }
        return {"input_ids": ids, "output_ids": out_ids,
                "intermediates": intermediates}

    def adapter_metadata(self) -> dict[str, Any]:
        return {"adapter": ADAPTER_NAME, "library": LIBRARY,
                "version": VERSION, "deterministic": True}


def create_reference_adapter(checkpoint_dir: str | Path) -> FakeAdapter:
    """Instantiate the fake adapter, rejecting missing local checkpoints."""
    return FakeAdapter(checkpoint_dir)
