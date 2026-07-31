# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser import ParserManager
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.tool_parsers.rwkv_tool_parser import RWKVToolParser


def _request_and_parser() -> tuple[ChatCompletionRequest, RWKVToolParser]:
    tool = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object"},
            },
        }
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Read it"}],
        tools=[tool],
        tool_choice="auto",
    )
    return request, RWKVToolParser(RWKVTokenizer(), [tool])


def test_rwkv_parser_extracts_canonical_tool_call():
    request, parser = _request_and_parser()
    output = (
        "I'll read it.\n**Tool Call:**\n```json\n"
        '{"name":"read","arguments":{"path":"a.txt"}}\n```'
    )

    result = parser.extract_tool_calls(output, request)

    assert result.tools_called
    assert result.content == "I'll read it.\n"
    assert result.tool_calls[0].function.name == "read"
    assert json.loads(result.tool_calls[0].function.arguments) == {"path": "a.txt"}


def test_rwkv_parser_preserves_unknown_tool_call_as_content():
    request, parser = _request_and_parser()
    output = (
        '**Tool Call:**\n```json\n{"name":"write","arguments":{"path":"a.txt"}}\n```'
    )

    result = parser.extract_tool_calls(output, request)

    assert not result.tools_called
    assert result.content == output


def test_rwkv_parser_accepts_legacy_call_and_normalizes_arguments():
    request, parser = _request_and_parser()
    output = '<tool_call>{"name":"read","arguments":{"filePath":"a.txt","limit":4}}\n'

    result = parser.extract_tool_calls(output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "path": "a.txt",
        "limit": 4,
    }


def test_rwkv_parser_extracts_parallel_and_standalone_calls():
    request, parser = _request_and_parser()
    parallel = (
        '**Tool Call:**\n```json\n{"name":"read","arguments":{"path":"a.txt"}}\n```'
        '\n**Tool Call:**\n```json\n{"name":"read","arguments":{"path":"b.txt"}}\n```'
    )

    result = parser.extract_tool_calls(parallel, request)

    assert result.tools_called
    assert result.content is None
    assert [
        json.loads(tool_call.function.arguments)["path"]
        for tool_call in result.tool_calls
    ] == ["a.txt", "b.txt"]

    standalone = parser.extract_tool_calls(
        '```json\n{"name":"read","arguments":{"path":"c.txt"}}\n```',
        request,
    )
    assert standalone.tools_called
    assert json.loads(standalone.tool_calls[0].function.arguments) == {"path": "c.txt"}


def test_rwkv_parser_normalizes_legacy_edit_arguments():
    request, _ = _request_and_parser()
    edit = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "Edit a file",
                "parameters": {"type": "object"},
            },
        }
    )
    parser = RWKVToolParser(RWKVTokenizer(), [edit])
    output = (
        '<tool_call>{"name":"edit","arguments":{"edits":'
        '[{"filePath":"a.txt","oldText":"old","newText":"new"}]}}\n'
    )

    result = parser.extract_tool_calls(output, request)

    assert result.tools_called
    assert json.loads(result.tool_calls[0].function.arguments) == {
        "path": "a.txt",
        "edits": [{"oldText": "old", "newText": "new"}],
    }


def test_rwkv_parser_preserves_malformed_and_non_tool_json():
    request, parser = _request_and_parser()
    malformed = '**Tool Call:**\n```json\n{"name":"read","arguments":\n```'
    bash = "```bash\ncat a.txt\n```"

    assert parser.extract_tool_calls(malformed, request).content == malformed
    assert parser.extract_tool_calls(bash, request).content == bash


@pytest.mark.parametrize(
    "output",
    [
        '**Tool Call:**\n```json\n{"name":"read","arguments":\n```',
        '**Tool Call:**\n```json\n{"name":"write","arguments":{}}\n```',
        "answer\n**Tool Ca",
    ],
)
def test_rwkv_streaming_flushes_unparsed_tool_text_at_finish(output: str):
    request, _ = _request_and_parser()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="rwkv",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(RWKVTokenizer(), request.tools)

    delta = parser.parse_delta(
        delta_text=output,
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=True,
    )

    assert delta is not None
    assert delta.content == output
    assert not delta.tool_calls


def test_rwkv_streaming_does_not_flush_valid_tool_text_at_finish():
    request, _ = _request_and_parser()
    parser_cls = ParserManager.get_parser(
        tool_parser_name="rwkv",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    parser = parser_cls(RWKVTokenizer(), request.tools)
    output = (
        "answer\n**Tool Call:**\n```json\n"
        '{"name":"read","arguments":{"path":"a.txt"}}\n```'
    )

    delta = parser.parse_delta(
        delta_text=output,
        delta_token_ids=[],
        request=request,
        prompt_token_ids=[],
        finished=True,
    )

    assert delta is not None
    assert delta.content == "answer\n"
    assert delta.tool_calls is not None
    assert delta.tool_calls[0].function is not None
    assert delta.tool_calls[0].function.name == "read"
    assert "**Tool Call:**" not in delta.content


def test_rwkv_streaming_holds_partial_tool_marker():
    request, parser = _request_and_parser()
    delta = parser.extract_tool_calls_streaming(
        previous_text="",
        current_text="answer\n**Tool Ca",
        delta_text="answer\n**Tool Ca",
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta is not None
    assert delta.content == "answer\n"
    assert not delta.tool_calls


def test_rwkv_streaming_emits_complete_tool_call_after_partial_marker():
    request, parser = _request_and_parser()
    partial = "answer\n**Tool Ca"
    parser.extract_tool_calls_streaming(
        previous_text="",
        current_text=partial,
        delta_text=partial,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )
    complete = (
        "answer\n**Tool Call:**\n```json\n"
        '{"name":"read","arguments":{"path":"a.txt"}}\n```'
    )

    delta = parser.extract_tool_calls_streaming(
        previous_text=partial,
        current_text=complete,
        delta_text=complete[len(partial) :],
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=request,
    )

    assert delta is not None
    assert delta.content is None
    assert delta.tool_calls is not None
    assert delta.tool_calls[0].function is not None
    assert delta.tool_calls[0].function.name == "read"
    assert json.loads(delta.tool_calls[0].function.arguments) == {"path": "a.txt"}
