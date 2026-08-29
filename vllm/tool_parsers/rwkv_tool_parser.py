# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool, ToolParser
from vllm.tool_parsers.utils import partial_tag_overlap

_TOOL_CALL_MARKER = "**Tool Call:**"
_JSON_FENCE_START = "```json"
_JSON_FENCE_END = "```"


@dataclass(frozen=True)
class _ToolCallMatch:
    start: int
    payload: dict[str, Any]


class RwkvToolParser(ToolParser):
    structural_tag_model = "rwkv"
    supports_required_and_named = False

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tool_choice = request.tool_choice
            if tool_choice == "required" or isinstance(
                tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            ):
                return request
        return super().adjust_request(request)

    def _allowed_tool_names(self) -> set[str]:
        names: set[str] = set()
        for tool in self.tools:
            function = getattr(tool, "function", tool)
            name = getattr(function, "name", None)
            if isinstance(name, str):
                names.add(name)
        return names

    @staticmethod
    def _named_tool(request: ChatCompletionRequest) -> str | None:
        tool_choice = request.tool_choice
        if isinstance(tool_choice, ChatCompletionNamedToolChoiceParam):
            return tool_choice.function.name
        if isinstance(tool_choice, ToolChoiceFunction):
            return tool_choice.name
        return None

    def _parse_payload(
        self,
        payload: str,
        request: ChatCompletionRequest,
    ) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
            return None

        arguments = parsed.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None

        name = parsed["name"]
        allowed_names = self._allowed_tool_names()
        if allowed_names and name not in allowed_names:
            return None
        named_tool = self._named_tool(request)
        if named_tool is not None and name != named_tool:
            return None
        return {"name": name, "arguments": arguments}

    def _tool_call_matches(
        self,
        text: str,
        request: ChatCompletionRequest,
    ) -> list[_ToolCallMatch]:
        matches: list[_ToolCallMatch] = []
        cursor = 0
        while (marker_start := text.find(_TOOL_CALL_MARKER, cursor)) != -1:
            marker_end = marker_start + len(_TOOL_CALL_MARKER)
            fence_start = text.find(_JSON_FENCE_START, marker_end)
            if fence_start == -1 or text[marker_end:fence_start].strip():
                return matches
            payload_start = fence_start + len(_JSON_FENCE_START)
            fence_end = text.find(_JSON_FENCE_END, payload_start)
            if fence_end == -1:
                return matches
            payload = self._parse_payload(
                text[payload_start:fence_end].strip(),
                request,
            )
            if payload is None:
                return matches
            matches.append(_ToolCallMatch(marker_start, payload))
            cursor = fence_end + len(_JSON_FENCE_END)
        return matches

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        matches = self._tool_call_matches(model_output, request)
        if not matches:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        tool_calls = [
            ToolCall(
                type="function",
                function=FunctionCall(
                    name=match.payload["name"],
                    arguments=json.dumps(
                        match.payload["arguments"],
                        ensure_ascii=False,
                    ),
                ),
            )
            for match in matches
        ]
        content = model_output[: matches[0].start]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content or None,
        )

    def _content_delta(self, current_text: str) -> str | None:
        marker_start = current_text.find(_TOOL_CALL_MARKER)
        if marker_start == -1:
            sendable_idx = len(current_text) - partial_tag_overlap(
                current_text,
                _TOOL_CALL_MARKER,
            )
        else:
            sendable_idx = marker_start
        if sendable_idx <= self._sent_content_idx:
            return None
        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        del previous_text, delta_text, previous_token_ids, current_token_ids
        del delta_token_ids

        content = self._content_delta(current_text)
        tool_deltas: list[DeltaToolCall] = []
        for index, match in enumerate(self._tool_call_matches(current_text, request)):
            if index < len(self.prev_tool_call_arr):
                continue
            arguments = json.dumps(match.payload["arguments"], ensure_ascii=False)
            self.prev_tool_call_arr.append(
                {"name": match.payload["name"], "arguments": arguments}
            )
            self.streamed_args_for_tool.append(arguments)
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=match.payload["name"],
                        arguments=arguments,
                    ),
                )
            )

        if content or tool_deltas:
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        return None
