# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501, E702, I001
from types import MethodType, SimpleNamespace
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from vllm.tokenizers.rwkv_defaults import (
    RWKV_BOS_EOS_TOKEN_ID,
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
    llm._run_chat = MethodType(
        lambda self, **kwargs: captured.update(kwargs) or [], llm
    )
    return llm, captured


def test_offline_rwkv_boundaries_merge_without_mutating_callers():
    llm, captured = _offline_llm()
    caller = SamplingParams(stop=[], stop_token_ids=[])
    llm.generate("prompt", caller, use_tqdm=False)
    resolved = captured["params"]
    assert (resolved.stop, resolved.stop_token_ids, resolved.all_stop_token_ids) == (
        [],
        [RWKV_BOS_EOS_TOKEN_ID],
        {RWKV_BOS_EOS_TOKEN_ID},
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
        ["END"],
        [RWKV_BOS_EOS_TOKEN_ID, 7],
        {RWKV_BOS_EOS_TOKEN_ID, 7},
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


def test_offline_rwkv_chat_prepends_template_and_token_stops():
    llm, captured = _offline_llm()
    params = SamplingParams(
        stop=["END"],
        stop_token_ids=[7],
        ignore_eos=True,
    )

    llm.chat(
        [{"role": "user", "content": "hi"}],
        params,
        use_tqdm=False,
    )

    resolved = captured["params"]
    assert resolved.stop == ["✿", "END"]
    assert resolved.stop_token_ids == [RWKV_BOS_EOS_TOKEN_ID, 7]
    assert resolved.ignore_eos is False


def test_offline_rwkv_chat_resolves_each_conversation_template():
    llm, captured = _offline_llm()

    llm.chat(
        [
            [{"role": "user", "content": "plain"}],
            [
                {"role": "user", "content": "call"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "noop", "arguments": {}}}],
                },
            ],
        ],
        SamplingParams(),
        use_tqdm=False,
    )

    resolved = captured["params"]
    assert [params.stop for params in resolved] == [["✿"], ["\n### User"]]
    assert all(params.stop_token_ids == [RWKV_BOS_EOS_TOKEN_ID] for params in resolved)


def test_offline_defaults_preserve_no_detokenize_and_other_models():
    llm, captured = _offline_llm()
    params = SamplingParams(
        detokenize=False, prompt_logprobs=1, max_tokens=1, stop_token_ids=[]
    )
    llm.generate("prompt", params, use_tqdm=False)
    resolved = captured["params"]
    assert resolved is not params
    assert resolved.stop == []
    assert resolved.stop_token_ids == [RWKV_BOS_EOS_TOKEN_ID]
    assert params.stop_token_ids == []
    llm, captured = _offline_llm(rwkv=False)
    llm.generate("prompt", SamplingParams(stop=[]), use_tqdm=False)
    assert (captured["params"].stop, captured["params"].stop_token_ids) == ([], [])
