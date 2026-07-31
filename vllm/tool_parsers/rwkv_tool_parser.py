# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import regex as re
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
_LEGACY_TOOL_CALL_MARKER = "<tool_call>"
_JSON_FENCE_START = "```json"
_FENCED_JSON_RE = re.compile(
    r"\*\*Tool Call:\*\*\s*```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)
_STANDALONE_FENCED_JSON_RE = re.compile(
    r"```json\s*(?P<json>.*?)\s*```",
    re.DOTALL,
)


@dataclass(frozen=True)
class _ToolCallMatch:
    start: int
    payload: dict[str, Any]


class RWKVToolParser(ToolParser):
    supports_required_and_named = False

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self._sent_content_idx = 0

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        if request.tools:
            tool_choice = request.tool_choice
            if tool_choice == "required" or isinstance(
                tool_choice,
                (ChatCompletionNamedToolChoiceParam, ToolChoiceFunction),
            ):
                request.skip_special_tokens = False
                return request
        return super().adjust_request(request)

    def _allowed_tool_names(self) -> set[str]:
        names: set[str] = set()
        for tool in self.tools:
            function = getattr(tool, "function", tool)
            if (name := getattr(function, "name", None)) is not None:
                names.add(name)
        return names

    @staticmethod
    def _loads_mapping(payload: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _normalize_legacy_arguments(
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if "path" not in arguments and isinstance(arguments.get("filePath"), str):
            arguments = {**arguments, "path": arguments["filePath"]}

        if name == "read" and "path" in arguments:
            return {
                key: arguments[key]
                for key in ("path", "offset", "limit")
                if key in arguments
            }

        if name != "edit":
            return arguments

        path = arguments.get("path")
        edits = arguments.get("edits")
        if not isinstance(path, str) and isinstance(edits, list):
            for edit in edits:
                if isinstance(edit, dict):
                    nested_path = edit.get("path", edit.get("filePath"))
                    if isinstance(nested_path, str):
                        path = nested_path
                        break
        if not isinstance(path, str):
            return arguments

        normalized: dict[str, Any] = {"path": path}
        if isinstance(edits, list):
            normalized_edits = [
                {"oldText": edit["oldText"], "newText": edit["newText"]}
                for edit in edits
                if isinstance(edit, dict) and {"oldText", "newText"} <= edit.keys()
            ]
            if normalized_edits:
                normalized["edits"] = normalized_edits
        elif {"oldText", "newText"} <= arguments.keys():
            normalized["edits"] = [
                {
                    "oldText": arguments["oldText"],
                    "newText": arguments["newText"],
                }
            ]
        return normalized

    def _parse_tool_call_payload(self, payload: str) -> dict[str, Any] | None:
        parsed = self._loads_mapping(payload)
        if (
            parsed is None
            or not isinstance(parsed.get("name"), str)
            or "arguments" not in parsed
        ):
            return None
        if isinstance(parsed["arguments"], str):
            parsed["arguments"] = self._loads_mapping(parsed["arguments"])
        if not isinstance(parsed["arguments"], dict):
            return None
        parsed["arguments"] = self._normalize_legacy_arguments(
            parsed["name"], parsed["arguments"]
        )
        allowed_names = self._allowed_tool_names()
        if allowed_names and parsed["name"] not in allowed_names:
            return None
        return parsed

    def _iter_tool_call_matches(self, text: str) -> Sequence[_ToolCallMatch]:
        if _TOOL_CALL_MARKER in text:
            matches: list[_ToolCallMatch] = []
            for match in _FENCED_JSON_RE.finditer(text):
                payload = self._parse_tool_call_payload(match.group("json"))
                if payload is None:
                    return []
                matches.append(_ToolCallMatch(match.start(), payload))
            return matches

        if not self.tools:
            return []

        matches = []
        marker_end = 0
        while (marker_start := text.find(_LEGACY_TOOL_CALL_MARKER, marker_end)) != -1:
            payload_start = marker_start + len(_LEGACY_TOOL_CALL_MARKER)
            line_start = text.find("{", payload_start)
            if line_start == -1:
                break
            line_end = text.find("\n", line_start)
            if line_end == -1:
                break
            payload = self._parse_tool_call_payload(text[line_start:line_end].strip())
            if payload is not None:
                matches.append(_ToolCallMatch(marker_start, payload))
            marker_end = line_end

        for match in _STANDALONE_FENCED_JSON_RE.finditer(text):
            payload = self._parse_tool_call_payload(match.group("json"))
            if payload is not None:
                matches.append(_ToolCallMatch(match.start(), payload))
        matches.sort(key=lambda match: match.start)
        return matches

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        del request
        matches = self._iter_tool_call_matches(model_output)
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
                        match.payload["arguments"], ensure_ascii=False
                    ),
                ),
            )
            for match in matches
        ]
        content = model_output[: matches[0].start]
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=tool_calls,
            content=content if content else None,
        )

    def _extract_content_delta(self, current_text: str) -> str | None:
        marker_starts = [
            current_text.find(_TOOL_CALL_MARKER),
            current_text.find(_LEGACY_TOOL_CALL_MARKER),
        ]
        if self.tools:
            marker_starts.append(current_text.find(_JSON_FENCE_START))
        marker_start = min(
            (start for start in marker_starts if start != -1),
            default=-1,
        )
        if marker_start == -1:
            overlap = max(
                partial_tag_overlap(current_text, _TOOL_CALL_MARKER),
                partial_tag_overlap(current_text, _LEGACY_TOOL_CALL_MARKER),
            )
            if self.tools:
                overlap = max(
                    overlap,
                    partial_tag_overlap(current_text, _JSON_FENCE_START),
                )
            sendable_idx = len(current_text) - overlap
        else:
            sendable_idx = marker_start

        if sendable_idx <= self._sent_content_idx:
            return None
        content = current_text[self._sent_content_idx : sendable_idx]
        self._sent_content_idx = sendable_idx
        return content

    def get_streaming_fallback_content(
        self,
        current_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        del request
        if self.prev_tool_call_arr or self._sent_content_idx >= len(current_text):
            return None
        content = current_text[self._sent_content_idx :]
        self._sent_content_idx = len(current_text)
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
        del delta_token_ids, request

        content = self._extract_content_delta(current_text)
        tool_deltas: list[DeltaToolCall] = []
        for index, match in enumerate(self._iter_tool_call_matches(current_text)):
            if index < len(self.prev_tool_call_arr):
                continue
            payload = match.payload
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
        if content or tool_deltas:
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        return None
