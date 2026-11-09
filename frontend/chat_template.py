"""Granite 4.2 chat-template renderer (stdlib port of the checkpoint's chat_template.jinja).

Covers system/user/assistant/tool turns, tool definitions, assistant tool calls, tool responses
(grouped into one user turn), reasoning content and the history-thinking truncation. Output is
meant to match transformers' apply_chat_template(..., tokenize=False, add_generation_prompt=True).
"""
from __future__ import annotations

import json
from typing import Any

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

_TOOLS_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call in natural language BEFORE "
    "the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)
_TOOL_SKIP_KEYS = {"defer_loading", "strict"}


def _tojson(value: Any) -> str:
    """The `tojson(ensure_ascii=False)` of transformers' template environment (plain json.dumps)."""
    return json.dumps(value, ensure_ascii=False)


def _text(content: Any) -> str:
    """Message content as text (OpenAI content parts are joined; None is empty)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            if not isinstance(p, dict) or p.get("type", "text") == "text"
        )
    return str(content)


def _tool_to_json(tool: dict[str, Any]) -> str:
    return "{" + ", ".join(
        f'"{k}": {_tojson(v)}' for k, v in tool.items() if k not in _TOOL_SKIP_KEYS
    ) + "}"


def _arguments(call: dict[str, Any]) -> dict[str, Any]:
    """A call's arguments as a dict (OpenAI sends them as a JSON string)."""
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool call arguments are not valid JSON: {exc}") from exc
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise ValueError("tool call arguments must be a JSON object")
    return args


def _tool_call_text(call: dict[str, Any]) -> str:
    call = call.get("function", call)
    out = [f"<tool_call>\n<function={call.get('name')}>\n"]
    for name, value in _arguments(call).items():
        if isinstance(value, (dict, list)):
            value = _tojson(value)
        else:
            value = str(value)
        out.append(f"<parameter={name}>\n{value}\n</parameter>\n")
    out.append("</function>\n</tool_call>\n")
    return "".join(out)


def _system_block(system: str, tools: list[dict[str, Any]]) -> str:
    out = [f"{IM_START}system\n{system}"]
    if tools:
        if system:
            out.append("\n\n")
        out.append("# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for tool in tools:
            tool = tool.get("function", tool)
            if not tool.get("defer_loading"):
                out.append("\n" + _tool_to_json(tool))
        out.append("\n</tools>" + _TOOLS_INSTRUCTIONS)
    out.append(f"{IM_END}\n")
    return "".join(out)


def _assistant_content(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content")
    content = _text(message.get("content"))
    if isinstance(reasoning, str) and reasoning.strip():
        return f"<think>\n{reasoning}\n</think>\n{content}"
    if "<think>" not in content and "</think>" not in content:
        return "<think></think>" + content
    return content


def _assistant_turn(message: dict[str, Any], truncated: bool) -> str:
    content = _assistant_content(message)
    calls = message.get("tool_calls")
    if calls:
        out = [f"{IM_START}assistant\n"]
        if content.strip():
            if not truncated:
                out.append(content.strip() + "\n")
            else:
                c = content
                if "</think>" in c:
                    c = c.split("</think>")[-1]
                elif "<think>" in c:
                    c = c.split("<think>")[0]
                out.append("<think></think>" + c.strip() + "\n")
        else:
            out.append("<think></think>")
        out.extend(_tool_call_text(call) for call in calls)
        out.append(f"{IM_END}\n")
        return "".join(out)
    if not truncated:
        return f"{IM_START}assistant\n{content.strip()}{IM_END}\n"
    c = content
    if "<think>" in c and "</think>" in c:
        c = "<think></think>" + c.split("</think>")[-1]
    c = c.strip()
    return f"{IM_START}assistant\n{c}{IM_END}\n" if c else f"{IM_START}assistant\n{IM_END}\n"


def render(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    low_effort: bool = False,
    truncate_history_thinking: bool = True,
) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant", "tool", "system"}:
            raise ValueError(f"unsupported message role: {role!r}")
    tools = tools or []
    if messages[0].get("role") == "system":
        system, rest = _text(messages[0].get("content")), messages[1:]
    else:
        system, rest = "", messages
    last_user = max((i for i, m in enumerate(rest) if m["role"] == "user"), default=-1)

    out = [_system_block(system, tools)]
    for i, message in enumerate(rest):
        role = message["role"]
        if role == "assistant":
            out.append(_assistant_turn(message, truncate_history_thinking and i < last_user))
        elif role in ("user", "system"):
            content = _text(message.get("content"))
            if role == "user" and i == last_user and low_effort:
                content += "\n\n{reasoning effort: low}"
            out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
        else:  # tool: consecutive responses share one user turn
            if i > 0 and rest[i - 1]["role"] != "tool":
                out.append(f"{IM_START}user\n")
            out.append(f"<tool_response>\n{_text(message.get('content'))}\n</tool_response>\n")
            if i == len(rest) - 1 or rest[i + 1]["role"] != "tool":
                out.append(f"{IM_END}\n")
    out.append(
        f"{IM_START}assistant\n<think>\n" if enable_thinking else f"{IM_START}assistant\n<think></think>"
    )
    return "".join(out)
