"""Tool calling: prompt rendering, output parsing, the completion message and SSE streaming."""

from __future__ import annotations

import json

import pytest

from frontend import api, chat_template, engine_service, tool_calls
from tests import test_serving as ts
from tests.test_serving import _post, _valid_body, served_server  # noqa: F401 (fixture)

TOOLS = [{"type": "function", "function": {
    "name": "search", "description": "Search files.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                   "required": ["query"]}}}]
CALL_TEXT = ("I'll search.\n<tool_call>\n<function=search>\n<parameter=query>\nREADME\n</parameter>\n"
             "<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>")


# --- rendering -------------------------------------------------------------

def test_tools_are_listed_in_the_system_turn():
    text = chat_template.render([{"role": "user", "content": "hi"}], tools=TOOLS)
    assert text.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n")
    assert '"name": "search"' in text
    assert "<function=example_function_name>" in text
    assert text.endswith("<|im_start|>assistant\n<think></think>")


def test_system_message_precedes_tools():
    text = chat_template.render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}],
                                tools=TOOLS)
    assert text.startswith("<|im_start|>system\nBe brief.\n\n# Tools")


def test_tool_calls_and_responses_round_trip_into_the_prompt():
    call = {"id": "c", "type": "function",
            "function": {"name": "search", "arguments": json.dumps({"query": "x", "limit": 2})}}
    text = chat_template.render([
        {"role": "user", "content": "find"},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c", "content": "result"},
    ], tools=TOOLS)
    assert ("<|im_start|>assistant\n<think></think>\n<tool_call>\n<function=search>\n<parameter=query>\nx\n"
            "</parameter>\n<parameter=limit>\n2\n</parameter>\n</function>\n</tool_call>\n<|im_end|>\n"
            "<|im_start|>user\n<tool_response>\nresult\n</tool_response>\n<|im_end|>\n") in text


def test_invalid_tool_call_arguments_are_rejected():
    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}]}]
    with pytest.raises(ValueError):
        chat_template.render(msgs)


# --- parsing ---------------------------------------------------------------

def test_parse_call_converts_by_schema():
    reasoning, content, calls = tool_calls.parse_output(CALL_TEXT, False, TOOLS)
    assert reasoning is None
    assert content == "I'll search."
    assert calls == [{"name": "search", "arguments": {"query": "README", "limit": 5}}]


def test_without_tools_the_call_text_stays_content():
    _, content, calls = tool_calls.parse_output(CALL_TEXT, False, None)
    assert calls == [] and "<tool_call>" in content


def test_thinking_output_is_split():
    reasoning, content, _ = tool_calls.parse_output("plan it\n</think>\nanswer", True, None)
    assert (reasoning, content) == ("plan it", "answer")


# --- API -------------------------------------------------------------------

class _TextEngine(engine_service.EngineService):
    def __init__(self, gate, text):
        class Tok(ts._FakeTokenizer):
            def decode(self, ids, *, skip_special_tokens=False):
                return text
        super().__init__(gate, tok=Tok(), eos_id=2)

    def _generate_ids(self, prompt_ids, max_tokens):
        return [7, 8, 2]


@pytest.fixture
def tool_server(served_server):
    api.set_serving_engine(_TextEngine(api._SERVING_ENGINE.gate, CALL_TEXT))
    return served_server


def test_completion_returns_tool_calls(tool_server):
    status, _, data = _post(tool_server, _valid_body(tools=TOOLS))
    assert status == 200, data
    choice = json.loads(data)["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "I'll search."
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "search"
    assert json.loads(call["function"]["arguments"]) == {"query": "README", "limit": 5}


def test_stream_returns_sse_with_tool_calls(tool_server):
    status, headers, data = _post(tool_server, _valid_body(tools=TOOLS, stream=True))
    assert status == 200 and headers["Content-Type"] == "text/event-stream"
    lines = [ln[6:] for ln in data.decode().split("\n\n") if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    first, last = json.loads(lines[0]), json.loads(lines[-2])
    delta = first["choices"][0]["delta"]
    assert delta["role"] == "assistant" and delta["tool_calls"][0]["index"] == 0
    assert last["choices"][0]["finish_reason"] == "tool_calls"


def test_tool_choice_none_ignores_tools(tool_server):
    status, _, data = _post(tool_server, _valid_body(tools=TOOLS, tool_choice="none"))
    choice = json.loads(data)["choices"][0]
    assert status == 200 and "tool_calls" not in choice["message"]
    assert choice["finish_reason"] == "stop"


def test_malformed_tools_are_400(tool_server):
    status, _, data = _post(tool_server, _valid_body(tools=["nope"]))
    assert status == 400 and json.loads(data)["error"]["code"] == "invalid_tools"
