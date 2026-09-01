# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from openai.types.responses import ToolChoiceFunction

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.engine.adapters import ParserEngineToolAdapter
from vllm.parser.rwkv import RwkvParser


class RwkvToolParser(ParserEngineToolAdapter):
    """Expose the declarative RWKV parser through the ToolParser API."""

    _parser_engine_cls = RwkvParser
    structural_tag_model = "rwkv"
    supports_required_and_named = False

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
                return self._parser_engine.adjust_request(request)
        return super().adjust_request(request)
