"""Minimal Granite chat-template renderer for ordinary system/user turns."""
from __future__ import annotations

from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def render(messages: list[dict[str, Any]], *, enable_thinking: bool = False) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    out: list[str] = []
    index = 0
    if messages[0].get("role") == "system":
        out.append(f"{IM_START}system\n{messages[0].get('content', '')}")
        index = 1
    for message in messages[index:]:
        role = message.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"unsupported message role: {role!r}")
        content = str(message.get("content", ""))
        out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    out.append(
        f"{IM_START}assistant\n<think>\n"
        if enable_thinking
        else f"{IM_START}assistant\n<think></think>"
    )
    return "".join(out)
