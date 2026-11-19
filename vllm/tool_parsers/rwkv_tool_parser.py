# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.generate.base.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.renderers.chat_utils import make_tool_call_id
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
    engine_based_streaming = True

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
    ) -> None:
        super().__init__(tokenizer, tools)
        self._buffer = ""
        self._tool_prefix: str | None = None
        self._payload_parts: list[str] = []
        self._saw_tool_marker = False
        self._tool_parsing_blocked = False
        self._finished = False

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
            if request.parallel_tool_calls is False:
                return matches
            cursor = fence_end + len(_JSON_FENCE_END)
        return matches

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        if request.tool_choice == "none":
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output or None,
            )
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

    def _block_further_tool_calls(self, content_parts: list[str]) -> None:
        self._tool_parsing_blocked = True
        if content := self._flush_pending_tool_text():
            content_parts.append(content)

    def _flush_pending_tool_text(self) -> str:
        content = ""
        if not self.prev_tool_call_arr:
            content = "".join(
                (self._tool_prefix or "", *self._payload_parts, self._buffer)
            )
        self._tool_prefix = None
        self._payload_parts.clear()
        self._buffer = ""
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
        del previous_text, current_text, previous_token_ids, current_token_ids
        del delta_token_ids

        if request.tool_choice == "none":
            return DeltaMessage(content=delta_text) if delta_text else None
        if not delta_text:
            return None
        if self._tool_parsing_blocked:
            if not self.prev_tool_call_arr:
                return DeltaMessage(content=delta_text)
            return None

        self._buffer += delta_text
        content_parts: list[str] = []
        tool_deltas: list[DeltaToolCall] = []

        while self._buffer:
            if self._tool_prefix is None:
                marker_start = self._buffer.find(_TOOL_CALL_MARKER)
                if marker_start == -1:
                    sendable_end = len(self._buffer) - partial_tag_overlap(
                        self._buffer,
                        _TOOL_CALL_MARKER,
                    )
                    if sendable_end > 0:
                        if not self._saw_tool_marker:
                            content_parts.append(self._buffer[:sendable_end])
                        self._buffer = self._buffer[sendable_end:]
                    break

                if marker_start > 0:
                    if not self._saw_tool_marker:
                        content_parts.append(self._buffer[:marker_start])
                    self._buffer = self._buffer[marker_start:]

                self._saw_tool_marker = True
                rest = self._buffer[len(_TOOL_CALL_MARKER) :]
                stripped = rest.lstrip()
                if not stripped:
                    break
                if not stripped.startswith(_JSON_FENCE_START):
                    if _JSON_FENCE_START.startswith(stripped):
                        break
                    self._block_further_tool_calls(content_parts)
                    break

                payload_start = (
                    len(self._buffer) - len(stripped) + len(_JSON_FENCE_START)
                )
                self._tool_prefix = self._buffer[:payload_start]
                self._buffer = self._buffer[payload_start:]

            fence_end = self._buffer.find(_JSON_FENCE_END)
            if fence_end == -1:
                payload_end = len(self._buffer) - partial_tag_overlap(
                    self._buffer, _JSON_FENCE_END
                )
                if payload_end:
                    self._payload_parts.append(self._buffer[:payload_end])
                    self._buffer = self._buffer[payload_end:]
                break

            payload = self._parse_payload(
                "".join((*self._payload_parts, self._buffer[:fence_end])).strip(),
                request,
            )
            if payload is None:
                self._block_further_tool_calls(content_parts)
                break

            index = len(self.prev_tool_call_arr)
            arguments = json.dumps(payload["arguments"], ensure_ascii=False)
            self.prev_tool_call_arr.append(
                {"name": payload["name"], "arguments": arguments}
            )
            self.streamed_args_for_tool.append(arguments)
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    type="function",
                    id=make_tool_call_id(),
                    function=DeltaFunctionCall(
                        name=payload["name"],
                        arguments=arguments,
                    ),
                )
            )
            self._tool_prefix = None
            self._payload_parts.clear()
            self._buffer = self._buffer[fence_end + len(_JSON_FENCE_END) :]
            if request.parallel_tool_calls is False:
                self._tool_parsing_blocked = True
                self._buffer = ""
                break

        content = "".join(content_parts) or None
        if content or tool_deltas:
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        return None

    def finish_streaming(self) -> DeltaMessage | None:
        if self._finished:
            return None
        self._finished = True
        if content := self._flush_pending_tool_text():
            return DeltaMessage(content=content)
        return None
