# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This test file includes some cases where it is inappropriate to
only get the `eos_token_id` from the tokenizer as defined by
`BaseRenderer.get_eos_token_id`.
"""

from vllm.tokenizers import get_tokenizer
from vllm.transformers_utils.config import try_get_generation_config


def test_get_llama3_eos_token():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 128009

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == [128001, 128008, 128009]


def test_get_blip2_eos_token():
    model_name = "Salesforce/blip2-opt-2.7b"

    tokenizer = get_tokenizer(model_name)
    assert tokenizer.eos_token_id == 2

    generation_config = try_get_generation_config(model_name, trust_remote_code=False)
    assert generation_config is not None
    assert generation_config.eos_token_id == 50118


def test_rwkv7_hf_artifact_generation_config_is_loaded(tmp_path):
    from transformers import GenerationConfig

    from vllm.transformers_utils.configs.rwkv7 import RWKV7Config

    RWKV7Config(
        hidden_size=64,
        head_size=64,
        num_hidden_layers=1,
        eos_token_id=0,
    ).save_pretrained(tmp_path)
    GenerationConfig(eos_token_id=0, max_new_tokens=23).save_pretrained(tmp_path)

    generation_config = try_get_generation_config(
        str(tmp_path),
        trust_remote_code=False,
    )

    assert generation_config is not None
    assert generation_config.eos_token_id == 0
    assert generation_config.max_new_tokens == 23
