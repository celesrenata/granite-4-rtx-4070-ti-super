"""Tool-call and reasoning parsing of Granite 4.2 output.

Granite 4.2's chat template asks for the same call format as Qwen3-Coder:
`<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>`, with optional
reasoning in `<think>...</think>`. The parser is a port of the one in qwen3-27b-5090/frontend/chat.py
(itself modelled on vLLM's qwen3 reasoning and qwen3_coder tool parsers), trimmed to what this
server needs: it turns a whole generated text into reasoning, content and OpenAI tool calls.
"""

import ast
import json
import re
import uuid

THINK_END = "</think>"
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"


def new_id(prefix, n=24):
    return f"{prefix}{uuid.uuid4().hex[:n]}"


# ---------------------------------------------------------------- tool calls (qwen3_coder XML)


def _types(schema):
    if not isinstance(schema, dict):
        return set()
    t = schema.get("type")
    if isinstance(t, str):
        return {t}
    if isinstance(t, list):
        return {x for x in t if isinstance(x, str)}
    out = set()
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key) or []:
            out |= _types(sub)
    return out


def _convert(value, schema):
    """A parameter's text by its JSON schema: strings stay as they are, other types are parsed (JSON, then
    Python literals such as True / None, as vLLM's qwen3_coder parser does); unparsable text stays a string."""
    types = _types(schema)
    if types == {"string"}:
        return value
    v = value.strip()
    if "boolean" in types and v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none") and ("null" in types or not types):
        return None
    try:
        out = json.loads(v)
    except (json.JSONDecodeError, ValueError):
        try:
            out = ast.literal_eval(v)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return value
    if "string" in types and not isinstance(out, (dict, list, bool, int, float)) and out is not None:
        return value
    if types and "string" in types and isinstance(out, (int, float)) and not ({"integer", "number"} & types):
        return value  # e.g. a path "123" of a string parameter
    return out


def tool_schemas(tools):
    """name -> {parameter: schema} of OpenAI-style tool definitions."""
    out = {}
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            out[fn["name"]] = ((fn.get("parameters") or {}).get("properties") or {})
    return out


def parse_call_block(block, schemas):
    """The calls of one `<tool_call>...</tool_call>` block: [{"name", "arguments" (dict)}]."""
    calls = []
    for m in re.finditer(r"<function=([^>\n]+)>(.*?)</function>", block, re.S):
        name, body = m.group(1).strip(), m.group(2)
        props = schemas.get(name, {})
        args = {}
        for p in re.finditer(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", body, re.S):
            pname = p.group(1).strip()
            schema = props.get(pname)
            args[pname] = _convert(p.group(2), schema) if schema is not None else _convert(p.group(2), {})
        calls.append({"name": name, "arguments": args})
    return calls


def openai_call(call, index=None):
    out = {"id": new_id("call_"), "type": "function",
           "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
    if index is not None:
        out["index"] = index
    return out


def _partial_suffix(text, marker):
    """Length of the longest suffix of text that is a proper prefix of marker."""
    for n in range(min(len(marker) - 1, len(text)), 0, -1):
        if text.endswith(marker[:n]):
            return n
    return 0


class OutputParser:
    """The generated text, fed piece by piece, as events: ("reasoning", text), ("content", text) and
    ("tool_call", {"name", "arguments"}). Text that may still turn out to be part of a marker, and trailing
    whitespace (dropped before `</think>` and `<tool_call>`), is held back until it is known.

    thinking: the prompt ended in `<think>` (the output starts with reasoning). tools: tool definitions
    (None: tool calls are not parsed and stay in the content)."""

    def __init__(self, thinking, tools=None):
        self.mode = "reasoning" if thinking else "content"
        self.tools = tools is not None
        self.schemas = tool_schemas(tools)
        self.buf = ""
        self.lead = True  # at the start of a reasoning / content part (leading whitespace is dropped)
        self.ncalls = 0

    def _text(self, kind, upto, events, hold_ws=True):
        """Emits buf[:upto] as `kind` (keeping trailing whitespace back when hold_ws)."""
        seg = self.buf[:upto]
        if self.lead:
            seg = seg.lstrip()
            if seg:
                self.lead = False
        keep = seg[len(seg.rstrip()):] if hold_ws else ""
        emit = seg[:len(seg) - len(keep)]
        if emit:
            events.append((kind, emit))
        self.buf = keep + self.buf[upto:]

    def feed(self, text):
        self.buf += text
        events = []
        while True:
            if self.mode == "reasoning":
                i = self.buf.find(THINK_END)
                if i < 0:
                    self._text("reasoning", len(self.buf) - _partial_suffix(self.buf, THINK_END), events)
                    return events
                self._text("reasoning", i, events)
                self.buf = self.buf.lstrip()[len(THINK_END):] if self.buf.lstrip().startswith(THINK_END) else ""
                self.mode, self.lead = "content", True
            elif self.mode == "content":
                i = self.buf.find(TOOL_START) if self.tools else -1
                if i < 0:
                    hold = _partial_suffix(self.buf, TOOL_START) if self.tools else 0
                    self._text("content", len(self.buf) - hold, events)
                    return events
                self._text("content", i, events)
                self.buf = self.buf[self.buf.find(TOOL_START):]  # (held whitespace before the call is dropped)
                self.mode = "tool"
            elif self.mode == "tool":
                i = self.buf.find(TOOL_END)
                if i < 0:
                    return events
                block, self.buf = self.buf[:i + len(TOOL_END)], self.buf[i + len(TOOL_END):]
                calls = parse_call_block(block, self.schemas)
                if not calls:  # (not a call after all: keep the text)
                    events.append(("content", block))
                for c in calls:
                    self.ncalls += 1
                    events.append(("tool_call", c))
                self.mode = "after"
            else:  # "after": between / after calls only another call is expected
                rest = self.buf.lstrip()
                if not rest:
                    return events
                if rest.startswith(TOOL_START):
                    self.buf, self.mode = rest, "tool"
                elif TOOL_START.startswith(rest):
                    return events
                else:  # text after the calls: content
                    self.buf, self.mode, self.lead = rest, "content", False

    def finish(self):
        """Events for what is left at the end of the generation."""
        events = []
        if self.mode == "reasoning":
            self._text("reasoning", len(self.buf), events)
        elif self.mode == "content":
            self._text("content", len(self.buf), events, hold_ws=False)
        elif self.mode == "tool":
            # Cut off inside a call (max_tokens): a complete function still counts, otherwise it is text.
            calls = parse_call_block(self.buf, self.schemas) if "</function>" in self.buf else []
            for c in calls:
                self.ncalls += 1
                events.append(("tool_call", c))
            if not calls and self.buf.strip():
                events.append(("content", self.buf))
        self.buf = ""
        return events


def parse_output(text, thinking, tools=None):
    """(reasoning or None, content, [calls]) of a whole output."""
    p = OutputParser(thinking, tools)
    reasoning, content, calls = [], [], []
    for kind, v in p.feed(text) + p.finish():
        if kind == "reasoning":
            reasoning.append(v)
        elif kind == "content":
            content.append(v)
        else:
            calls.append(v)
    return ("".join(reasoning) if thinking else None), "".join(content), calls

