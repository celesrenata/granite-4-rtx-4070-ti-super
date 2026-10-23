"""Small stdlib GPT-2 byte-level BPE tokenizer for Granite 4.2.

The checkpoint's tokenizer is a byte-level BPE with no normalizer.  This
module deliberately reads tokenizer.json rather than embedding a generated
vocabulary, so the runtime tokenizer stays tied to the checkpoint artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

try:
    import regex as _regex  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - packaging error
    raise RuntimeError("Granite tokenizer requires the 'regex' package") from exc

# The checkpoint was generated with the `tokenizers` library, whose
# ByteLevel pre-tokenizer (src/pre_tokenizers/byte_level.rs) uses this
# default pattern when no `pattern` key is present in tokenizer.json.
# It differs from the original OpenAI GPT-2 regex: digit runs are not
# capped at three, and a leading space glues onto the following word.
# `regex` supports the \p{} Unicode property syntax that stdlib `re` lacks.
_PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
_PAT = _regex.compile(_PATTERN)


def _bytes_to_unicode() -> dict[int, str]:
    # OpenAI GPT-2's reversible byte-to-unicode map.
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ByteBPE:
    def __init__(self, tokenizer_json: str | Path) -> None:
        data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
        model = data["model"]
        if model.get("type") != "BPE":
            raise ValueError("tokenizer model is not BPE")
        self.vocab: dict[str, int] = {str(k): int(v) for k, v in model["vocab"].items()}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(model.get("merges", [])):
            if isinstance(merge, str):
                pair = tuple(merge.split(" ", 1))
            else:
                pair = tuple(merge)
            if len(pair) == 2:
                self.merges[(pair[0], pair[1])] = rank
        self.encoder = _bytes_to_unicode()
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.added = sorted(
            ((str(x["content"]), int(x["id"])) for x in data.get("added_tokens", [])),
            key=lambda x: len(x[0]), reverse=True,
        )
        self.special_ids = {token: sid for token, sid in self.added}
        self.unk_id = self.vocab.get(model.get("unk_token", ""), None)

    @staticmethod
    def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
        return set(zip(word, word[1:]))

    def _bpe(self, token: str) -> list[str]:
        word = tuple(token)
        if len(word) < 2:
            return list(word)
        while True:
            pairs = self._pairs(word)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.merges.get(p, 10**12))
            if best not in self.merges:
                break
            first, second = best
            merged: list[str] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == first and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
        return list(word)

    def _encode_piece(self, piece: str) -> list[int]:
        return [self.vocab.get(x, self.unk_id if self.unk_id is not None else 0) for x in self._bpe(piece)]

    def encode(self, text: str, *, allowed_special: bool = True) -> list[int]:
        if not text:
            return []
        if not allowed_special or not self.special_ids:
            chunks = [(text, False)]
        else:
            # Split special tokens before applying the byte-level regex.
            # Unknown angle bracket text remains ordinary user text.
            chunks: list[tuple[str, bool]] = []
            pos = 0
            while pos < len(text):
                found = None
                found_at = len(text)
                for token, sid in self.added:
                    at = text.find(token, pos)
                    if at >= 0 and at < found_at:
                        found, found_at = (token, sid), at
                if found is None:
                    chunks.append((text[pos:], False))
                    break
                if found_at > pos:
                    chunks.append((text[pos:found_at], False))
                chunks.append((found[0], True))
                pos = found_at + len(found[0])
        out: list[int] = []
        for chunk, special in chunks:
            if special:
                out.append(self.special_ids[chunk])
            else:
                # The pre-tokenizer splits Unicode text first, then maps
                # each UTF-8 byte to GPT-2's reversible schema character.
                for piece in _PAT.findall(chunk):
                    mapped = "".join(self.encoder[b] for b in piece.encode("utf-8"))
                    out.extend(self._encode_piece(mapped))
        return out

    def decode(self, ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        pieces: list[str] = []
        for token_id in ids:
            token = self.id_to_token.get(int(token_id), "")
            if token in self.special_ids:
                if not skip_special_tokens:
                    pieces.append(token)
                continue
            pieces.append(bytes(self.decoder.get(ch, ord(ch)) for ch in token).decode("latin-1"))
        return "".join(pieces).encode("latin-1", "replace").decode("utf-8", "replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
