# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.tokenizers.rwkv_defaults import (
    RWKV_PROMPT_TEMPLATE_ASSISTANT,
    RWKV_PROMPT_TEMPLATE_BOT,
)


@pytest.fixture(scope="module")
def tokenizer() -> RWKVTokenizer:
    # Uses the bundled canonical RWKV World vocabulary; no network artifact.
    return RWKVTokenizer()


def test_rwkv_plain_completion_token_sequence(tokenizer: RWKVTokenizer) -> None:
    assert tokenizer.encode("Hello") == [0, 33155]
    assert tokenizer.bos_token_id == tokenizer.eos_token_id == 0


@pytest.mark.parametrize(
    ("prompt_template", "expected_text", "expected_token_ids"),
    [
        pytest.param(
            RWKV_PROMPT_TEMPLATE_BOT,
            "User✿hi✿\nBot✿<think",
            [0, 24281, 10060, 1922, 10060, 11, 5645, 10060, 61, 35762],
            id="bot",
        ),
        pytest.param(
            RWKV_PROMPT_TEMPLATE_ASSISTANT,
            "User: hi\n\nAssistant: <think",
            [0, 24281, 59, 4571, 261, 5585, 41693, 59, 295, 35762],
            id="assistant",
        ),
    ],
)
def test_rwkv_chat_completion_token_sequence(
    tokenizer: RWKVTokenizer,
    prompt_template: str,
    expected_text: str,
    expected_token_ids: list[int],
) -> None:
    messages = [{"role": "user", "content": "hi"}]

    assert (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            rwkv_prompt_template=prompt_template,
        )
        == expected_text
    )
    assert (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            rwkv_prompt_template=prompt_template,
        )
        == expected_token_ids
    )
