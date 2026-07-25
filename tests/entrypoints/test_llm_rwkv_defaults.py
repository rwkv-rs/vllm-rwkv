# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, E702, I001
from types import MethodType, SimpleNamespace
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.rwkv_defaults import (
    RWKV_DEFAULT_STOP_TOKEN_IDS,
    RWKV_DEFAULT_STOPS,
)


def _offline_llm(*, rwkv=True):
    llm = object.__new__(LLM)
    captured = {}
    llm.model_config = SimpleNamespace(
        runner_type="generate",
        tokenizer_mode="rwkv" if rwkv else "auto",
        hf_config=SimpleNamespace(model_type="rwkv7" if rwkv else "llama"),
    )
    llm._run_completion = MethodType(
        lambda self, **kwargs: captured.update(kwargs) or [], llm
    )
    return llm, captured


def test_offline_rwkv_boundaries_merge_without_mutating_callers():
    llm, captured = _offline_llm()
    caller = SamplingParams(stop=[], stop_token_ids=[])
    llm.generate("prompt", caller, use_tqdm=False)
    resolved = captured["params"]
    assert (resolved.stop, resolved.stop_token_ids, resolved.all_stop_token_ids) == (
        list(RWKV_DEFAULT_STOPS),
        list(RWKV_DEFAULT_STOP_TOKEN_IDS),
        set(RWKV_DEFAULT_STOP_TOKEN_IDS),
    )
    assert (
        caller.all_stop_token_ids == set()
        and resolved.all_stop_token_ids is not caller.all_stop_token_ids
        and resolved.ignore_eos is False
        and (caller.stop, caller.stop_token_ids) == ([], [])
    )
    task = SamplingParams(stop=["END"], stop_token_ids=[7])
    llm.generate("prompt", task, use_tqdm=False)
    resolved = captured["params"]
    assert (resolved.stop, resolved.stop_token_ids, resolved.all_stop_token_ids) == (
        ["END", *RWKV_DEFAULT_STOPS],
        [7, *RWKV_DEFAULT_STOP_TOKEN_IDS],
        {7, *RWKV_DEFAULT_STOP_TOKEN_IDS},
    )
    llm.generate("first", caller, use_tqdm=False)
    first = captured["params"]
    llm.generate(["second", "third"], (caller, task), use_tqdm=False)
    assert (
        captured["params"] is not first
        and isinstance(captured["params"], tuple)
        and (task.stop, task.stop_token_ids) == (["END"], [7])
        and (caller.stop, caller.stop_token_ids) == ([], [])
    )


def test_offline_defaults_exclude_logprobs_and_other_models():
    llm, captured = _offline_llm()
    params = SamplingParams(
        detokenize=False, prompt_logprobs=1, max_tokens=1, stop_token_ids=[]
    )
    llm.generate("prompt", params, use_tqdm=False)
    assert captured["params"] is params and (params.stop, params.stop_token_ids) == (
        [],
        [],
    )
    llm, captured = _offline_llm(rwkv=False)
    llm.generate("prompt", SamplingParams(stop=[]), use_tqdm=False)
    assert (captured["params"].stop, captured["params"].stop_token_ids) == ([], [])
