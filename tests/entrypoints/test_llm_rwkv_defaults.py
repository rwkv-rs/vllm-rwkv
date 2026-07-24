# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.rwkv_defaults import RWKV_DEFAULT_STOP_TOKEN_IDS, RWKV_DEFAULT_STOPS

def _offline_llm(*, rwkv: bool = True):
    llm = object.__new__(LLM)
    llm.model_config = SimpleNamespace(
        runner_type="generate", tokenizer_mode="rwkv" if rwkv else "auto",
        hf_config=SimpleNamespace(model_type="rwkv7" if rwkv else "llama"))
    captured = {}
    llm._run_completion = MethodType(
        lambda self, **kwargs: captured.update(kwargs) or [], llm)
    return llm, captured

def test_offline_rwkv_boundaries_merge_without_mutating_callers() -> None:
    llm, captured = _offline_llm()
    caller = SamplingParams(stop=[], stop_token_ids=[])
    llm.generate("prompt", caller, use_tqdm=False)
    resolved = captured["params"]
    assert (resolved.stop, resolved.stop_token_ids) == (
        list(RWKV_DEFAULT_STOPS), list(RWKV_DEFAULT_STOP_TOKEN_IDS))
    assert resolved.all_stop_token_ids == set(RWKV_DEFAULT_STOP_TOKEN_IDS) and caller.all_stop_token_ids == set() and resolved.all_stop_token_ids is not caller.all_stop_token_ids
    assert resolved.ignore_eos is False and (caller.stop, caller.stop_token_ids) == ([], [])
    task = SamplingParams(stop=["END"], stop_token_ids=[7])
    llm.generate("prompt", task, use_tqdm=False)
    resolved = captured["params"]
    assert (resolved.stop, resolved.stop_token_ids) == (
        ["END", *RWKV_DEFAULT_STOPS], [7, *RWKV_DEFAULT_STOP_TOKEN_IDS])
    assert resolved.all_stop_token_ids == {7, *RWKV_DEFAULT_STOP_TOKEN_IDS}
    assert (task.stop, task.stop_token_ids) == (["END"], [7])
    llm.generate("first", caller, use_tqdm=False)
    first = captured["params"]
    llm.generate(["second", "third"], (caller, task), use_tqdm=False)
    assert captured["params"] is not first and isinstance(captured["params"], tuple)
    assert (caller.stop, caller.stop_token_ids) == ([], [])

def test_offline_defaults_exclude_logprobs_and_other_models() -> None:
    llm, captured = _offline_llm()
    params = SamplingParams(detokenize=False, prompt_logprobs=1,
                            max_tokens=1, stop_token_ids=[])
    llm.generate("prompt", params, use_tqdm=False)
    assert captured["params"] is params
    assert (params.stop, params.stop_token_ids) == ([], [])
    llm, captured = _offline_llm(rwkv=False)
    llm.generate("prompt", SamplingParams(stop=[]), use_tqdm=False)
    assert (captured["params"].stop, captured["params"].stop_token_ids) == ([], [])
