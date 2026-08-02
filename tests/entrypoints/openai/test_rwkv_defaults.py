# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field

import pytest

from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers.rwkv_defaults import (
    RWKV_BOS_EOS_TOKEN_ID,
    RWKV_PROMPT_TEMPLATE_BOT,
    resolve_rwkv_prompt_template,
    resolve_rwkv_sampling_params,
    resolve_rwkv_tool_parser,
)


@dataclass
class _HFConfig:
    model_type: str = "rwkv7"


@dataclass
class _ModelConfig:
    tokenizer_mode: str = "rwkv"
    hf_config: _HFConfig = field(default_factory=_HFConfig)


def test_rwkv_template_and_user_stops_are_merged_in_contract_order() -> None:
    prompt_template = resolve_rwkv_prompt_template(
        prompt_template=RWKV_PROMPT_TEMPLATE_BOT
    )
    caller = SamplingParams(
        stop=["END", "✿"],
        stop_token_ids=[7, RWKV_BOS_EOS_TOKEN_ID],
        ignore_eos=True,
    )

    resolved = resolve_rwkv_sampling_params(
        caller,
        _ModelConfig(),
        prompt_template=prompt_template,
    )

    assert resolved.stop == ["✿", "END"]
    assert resolved.stop_token_ids == [RWKV_BOS_EOS_TOKEN_ID, 7]
    assert resolved.ignore_eos is False
    assert caller.stop == ["END", "✿"]
    assert caller.stop_token_ids == [7, RWKV_BOS_EOS_TOKEN_ID]


def test_rwkv_defaults_do_not_apply_to_other_models() -> None:
    model_config = _ModelConfig(tokenizer_mode="auto", hf_config=_HFConfig("llama"))
    sampling_params = SamplingParams(stop=["END"], stop_token_ids=[7])

    resolved = resolve_rwkv_sampling_params(sampling_params, model_config)

    assert resolved is sampling_params


def test_rwkv_native_chat_rejects_beam_search_without_text_stops() -> None:
    with pytest.raises(ValueError, match="beam search does not support"):
        resolve_rwkv_sampling_params(
            BeamSearchParams(beam_width=2, max_tokens=8),
            _ModelConfig(),
            prompt_template=resolve_rwkv_prompt_template(),
        )


def test_raw_pth_filename_is_not_rwkv_serving_identity() -> None:
    assert (
        resolve_rwkv_tool_parser(
            tool_parser=None,
            enable_auto_tools=True,
            model="rwkv7-g1g-7.2b-20260523-ctx8192.pth",
        )
        is None
    )

    assert (
        resolve_rwkv_tool_parser(
            tool_parser=None,
            enable_auto_tools=True,
            tokenizer_mode="rwkv",
            model="weights.pth",
        )
        == "rwkv"
    )
