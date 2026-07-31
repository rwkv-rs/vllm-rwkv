# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from typing_extensions import override

from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.tokenizers.rwkv_chat import (
    RWKV_BOS_EOS_TOKEN_ID,
    RWKV_NATIVE_CHAT_TEMPLATE,
    ensure_rwkv_prompt_bos,
    resolve_rwkv_prompt_template,
)

from .hf import HfRenderer
from .params import ChatParams


class RwkvRenderer(HfRenderer):
    """HF-compatible rendering with RWKV's prompt and token contracts."""

    default_tool_parser = "rwkv"

    @override
    def get_bos_token_id(self) -> int:
        return RWKV_BOS_EOS_TOKEN_ID

    @override
    def get_eos_token_id(self) -> int:
        return RWKV_BOS_EOS_TOKEN_ID

    @override
    def resolve_ignore_eos(self, ignore_eos: bool) -> bool:
        return False

    @override
    def normalize_prompt_token_ids(
        self,
        token_ids: Sequence[int],
        *,
        max_length: int | None = None,
    ) -> list[int]:
        return ensure_rwkv_prompt_bos(
            token_ids,
            max_length=max_length,
            truncate_from_left=True,
        )

    @override
    def get_chat_stop(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> str | None:
        template = params.chat_template
        if template is not None and template.strip() != RWKV_NATIVE_CHAT_TEMPLATE:
            return None
        kwargs = params.chat_template_kwargs
        prompt_template = resolve_rwkv_prompt_template(
            prompt_template=kwargs.get("rwkv_prompt_template"),
            messages=messages,
            tools=kwargs.get("tools") or (),
        )
        return prompt_template.stop
