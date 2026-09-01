# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""RWKV native fenced-JSON tool-call parser.

RWKV emits one OpenAI-style JSON envelope per fenced tool block::

    **Tool Call:**
    ```json
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    ```

The declarative parser engine consumes streaming deltas exactly once. The
generic JSON-envelope mode buffers and validates each payload at the closing
fence, then emits one complete tool-call delta.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

TOOL_CALL_MARKER = "**Tool Call:**"
JSON_FENCE_START = "\n```json"
JSON_FENCE_END = "\n```"


@functools.cache
def rwkv_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="rwkv",
        initial_state=ParserState.CONTENT,
        terminals={
            "TOOL_MARKER": TOOL_CALL_MARKER,
            "JSON_FENCE_START": JSON_FENCE_START,
            "TOOL_END": JSON_FENCE_END,
        },
        transitions={
            (ParserState.CONTENT, "TOOL_MARKER"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.TOOL_PREAMBLE, "JSON_FENCE_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_MARKER"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
        },
        # The complete envelope is validated at TOOL_END, so incremental JSON
        # brace holdback would only add per-character events here.
        tool_args_json=False,
        json_tool_call_envelope=True,
        validate_tool_names=True,
        drop_whitespace_only_content_before_tools=False,
        strip_content_whitespace_with_tools=False,
    )


class RwkvParser(ParserEngine):
    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("parser_engine_config", rwkv_config())
        super().__init__(tokenizer, tools, **kwargs)
