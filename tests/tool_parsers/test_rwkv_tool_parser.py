# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers import ToolParser, ToolParserManager


class FakeTokenizer:
    def get_vocab(self) -> dict[str, int]:
        return {}


def _tool(name: str) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "description": f"{name} description",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    )


def _request(
    tools: list[ChatCompletionToolsParam] | None = None,
    tool_choice: Any = "auto",
) -> ChatCompletionRequest:
    kwargs: dict[str, Any] = {}
    if tools is not None:
        kwargs.update(
            tools=[tool.model_dump() for tool in tools],
            tool_choice=tool_choice,
        )
    return ChatCompletionRequest(
        model="rwkv-test",
        messages=[{"role": "user", "content": "hi"}],
        **kwargs,
    )


def _parser(tools: list[ChatCompletionToolsParam] | None = None) -> ToolParser:
    return ToolParserManager.get_tool_parser("rwkv")(FakeTokenizer(), tools=tools)


def _tool_call(name: str, arguments: dict[str, Any] | str) -> str:
    return (
        "**Tool Call:**\n"
        "```json\n"
        f"{json.dumps({'name': name, 'arguments': arguments}, indent=2)}\n"
        "```"
    )


def _stream(
    parser: ToolParser,
    text: str,
    request: ChatCompletionRequest,
    chunk_size: int,
) -> list[DeltaMessage]:
    previous_text = ""
    deltas: list[DeltaMessage] = []
    for index in range(0, len(text), chunk_size):
        delta_text = text[index : index + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


@pytest.mark.parametrize(
    "text",
    [
        "There is no tool call here.",
        '```json\n{"name": "get_weather", "arguments": {}}\n```',
        '**Tool Call:**\n```json\n{"name": "get_weather"}\n```',
        _tool_call("unknown", {"city": "Paris"}),
    ],
)
def test_non_tool_output_remains_content(text: str) -> None:
    tools = [_tool("get_weather")]
    result = _parser(tools).extract_tool_calls(text, _request(tools))

    assert (result.tools_called, result.tool_calls, result.content) == (False, [], text)


@pytest.mark.parametrize(
    ("tool_choice", "name"),
    [
        pytest.param("auto", "get_weather", id="auto"),
        pytest.param("required", "get_weather", id="required"),
        pytest.param(
            {"type": "function", "function": {"name": "get_weather"}},
            "get_weather",
            id="named",
        ),
    ],
)
def test_extracts_authoritative_fenced_json(tool_choice: Any, name: str) -> None:
    tools = [_tool("get_weather")]
    request = _request(tools, tool_choice)
    text = "Checking.\n" + _tool_call(name, {"city": "Paris"})

    result = _parser(tools).extract_tool_calls(text, request)

    assert result.tools_called
    assert result.content == "Checking.\n"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == name
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}


def test_named_choice_rejects_a_different_tool() -> None:
    tools = [_tool("get_weather"), _tool("get_forecast")]
    request = _request(
        tools,
        {"type": "function", "function": {"name": "get_weather"}},
    )
    text = _tool_call("get_forecast", {"city": "Paris"})

    result = _parser(tools).extract_tool_calls(text, request)

    assert (result.tools_called, result.tool_calls, result.content) == (False, [], text)


def test_extracts_parallel_calls_and_json_string_arguments() -> None:
    tools = [_tool("get_weather"), _tool("get_forecast")]
    request = _request(tools)
    text = "\n".join(
        [
            _tool_call("get_weather", '{"city": "Paris"}'),
            _tool_call("get_forecast", {"city": "Berlin"}),
        ]
    )

    result = _parser(tools).extract_tool_calls(text, request)

    assert result.tools_called
    assert [call.function.name for call in result.tool_calls] == [
        "get_weather",
        "get_forecast",
    ]
    assert [json.loads(call.function.arguments) for call in result.tool_calls] == [
        {"city": "Paris"},
        {"city": "Berlin"},
    ]


def test_keeps_complete_calls_before_trailing_incomplete_call() -> None:
    tools = [_tool("get_weather")]
    complete = _tool_call("get_weather", {"city": "Paris"})
    trailing = '**Tool Call:**\n```json\n{"name": "get_weather"'

    result = _parser(tools).extract_tool_calls(
        complete + "\n" + trailing,
        _request(tools, "required"),
    )

    assert result.tools_called
    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}


@pytest.mark.parametrize("chunk_size", [1, 5, 17])
def test_streams_content_and_complete_tool_calls(chunk_size: int) -> None:
    tools = [_tool("get_weather")]
    request = _request(tools)
    text = "Checking.\n" + _tool_call("get_weather", {"city": "Paris"})

    deltas = _stream(_parser(tools), text, request, chunk_size)

    content = "".join(delta.content or "" for delta in deltas)
    tool_calls = [call for delta in deltas for call in (delta.tool_calls or [])]
    assert content == "Checking.\n"
    assert len(tool_calls) == 1
    assert tool_calls[0].function is not None
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments or "") == {"city": "Paris"}


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "get_weather"}},
    ],
)
def test_required_and_named_keep_native_format(tool_choice: Any) -> None:
    tools = [_tool("get_weather")]
    request = _request(tools, tool_choice)

    adjusted = _parser(tools).adjust_request(request)

    assert adjusted.structured_outputs is None
    assert not type(_parser(tools)).supports_required_and_named
