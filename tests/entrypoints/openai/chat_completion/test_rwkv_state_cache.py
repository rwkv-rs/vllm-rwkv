# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from vllm.entrypoints.openai.chat_completion.api_router import _rwkv_state_action
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
)
from vllm.entrypoints.openai.chat_completion.serving import (
    OpenAIServingChat,
    _promote_rwkv_state_stop_strings,
    _rwkv_state_turn_boundary_token_ids,
)
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.tokenizers.rwkv_defaults import (
    RWKV_PROMPT_TEMPLATE_ASSISTANT,
    RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING,
)


class _BoundaryTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return {
            "\n": [10],
            "\n\n": [12],
            "✿": [10060],
            "✿\n": [10060, 10],
            "two tokens": [8, 9],
        }[text]


def _minimal_serving() -> OpenAIServingChat:
    serving = object.__new__(OpenAIServingChat)
    serving.response_role = "assistant"
    serving.parser_cls = None
    serving.enable_auto_tools = False
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.enable_log_deltas = False
    serving.enable_force_include_usage = False
    serving.request_logger = None
    serving.system_fingerprint = None
    serving.enable_per_request_metrics = False
    return serving


def _request_output() -> RequestOutput:
    return RequestOutput(
        request_id="test-id",
        prompt="User question",
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=None,
        outputs=[
            CompletionOutput(
                index=0,
                text="Answer",
                token_ids=[4],
                cumulative_logprob=None,
                logprobs=None,
                finish_reason="stop",
            )
        ],
        finished=True,
        metrics=None,
    )


async def _single_output():
    yield _request_output()


def test_state_ref_is_omitted_from_unrelated_stream_chunks() -> None:
    chunk = ChatCompletionStreamResponse(model="test-model", choices=[])

    assert "rwkv_state_ref" not in chunk.model_dump(exclude_unset=True)

    chunk.rwkv_state_ref = "goal-a"
    assert chunk.model_dump(exclude_unset=True)["rwkv_state_ref"] == "goal-a"


def test_state_write_promotes_single_token_template_stop() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "start"}],
        stop_token_ids=[6],
    )

    _promote_rwkv_state_stop_strings(
        request,
        {"stop": ["✿", "two tokens"]},
        _BoundaryTokenizer(),
    )

    assert request.stop_token_ids == [6, 10060]


@pytest.mark.parametrize(
    ("chat_template_kwargs", "finish_reason", "expected"),
    [
        ({}, "stop", [10]),
        ({}, "length", [10060, 10]),
        ({"rwkv_prompt_template": RWKV_PROMPT_TEMPLATE_ASSISTANT}, "length", [12]),
        (
            {"rwkv_prompt_template": RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING},
            "length",
            [10],
        ),
    ],
)
def test_state_turn_boundary_follows_native_template(
    chat_template_kwargs,
    finish_reason,
    expected,
) -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "delta"}],
    )

    token_ids = _rwkv_state_turn_boundary_token_ids(
        request,
        chat_template_kwargs,
        _BoundaryTokenizer(),
        finish_reason=finish_reason,
    )

    assert token_ids == expected


@pytest.mark.asyncio
async def test_serving_state_action_uses_worker_collective_rpc() -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            return [{"state_ref": "goal-b", "nbytes": 123}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    result = await serving.rwkv_state_cache_action("clone", "goal-a", "goal-b")

    assert calls == [
        ("rwkv_state_cache_action", ("prepare_clone", "goal-a", "goal-b")),
        ("rwkv_state_cache_action", ("clone", "goal-a", "goal-b")),
    ]
    assert result["workers"][0]["state_ref"] == "goal-b"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "state_ref", "target_ref", "rollback"),
    [
        ("clone", "goal-a", "goal-b", ("drop_if_exists", "goal-b", "")),
        ("prepare_write", "goal-a", "", ("cancel_write", "goal-a", "")),
        (
            "prepare_read",
            "goal-a",
            "lease-a",
            ("cancel_read", "goal-a", "lease-a"),
        ),
    ],
)
async def test_serving_state_action_rolls_back_partial_worker_change(
    action,
    state_ref,
    target_ref,
    rollback,
) -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            if args[0] == action:
                raise RuntimeError("rank failed")
            return [{"ready": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    with pytest.raises(RuntimeError, match="rank failed"):
        await serving.rwkv_state_cache_action(action, state_ref, target_ref)

    assert calls[-1] == ("rwkv_state_cache_action", rollback)


@pytest.mark.asyncio
async def test_state_request_acquires_and_releases_read_write_guards() -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            return [{"ready": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    await serving._acquire_rwkv_state_request("goal-parent", "lease-a", "goal-child")
    await serving._cleanup_rwkv_state_request("goal-parent", "lease-a", "goal-child")

    assert calls == [
        ("rwkv_state_cache_action", ("prepare_read", "goal-parent", "lease-a")),
        ("rwkv_state_cache_action", ("prepare_write", "goal-child", "")),
        ("rwkv_state_cache_action", ("cancel_read", "goal-parent", "lease-a")),
        ("rwkv_state_cache_action", ("cancel_write", "goal-child", "")),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (KeyError("Unknown RWKV State ref: missing"), 404),
        (RuntimeError("RWKV State ref is in use: active"), 409),
    ],
)
async def test_http_state_action_maps_worker_errors(error, expected_status) -> None:
    class _Handler:
        async def rwkv_state_cache_action(self, *_args):
            raise error

    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=_Handler()))
    )

    with pytest.raises(HTTPException) as captured:
        await _rwkv_state_action(raw_request, "inspect", "missing")

    assert captured.value.status_code == expected_status


@pytest.mark.asyncio
async def test_state_restore_prepends_tail_boundary_and_strips_delta_bos() -> None:
    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            assert method == "rwkv_state_cache_action"
            assert args == ("restore_metadata", "goal-parent", "")
            return [
                {
                    "state_ref": "goal-parent",
                    "pending_tail_token_ids": [41, 42],
                    "finish_reason": "stop",
                }
            ]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "delta"}],
        vllm_xargs={"rwkv_state_read_ref": " goal-parent "},
    )
    engine_inputs = [
        {
            "type": "token",
            "prompt_token_ids": [0, 50, 51],
            "assistant_tokens_mask": [0, 0, 0],
            "prompt_token_offsets": [(0, 1), (1, 2), (2, 3)],
        }
    ]

    await serving._prepend_rwkv_state_tail(
        request, engine_inputs, {}, _BoundaryTokenizer()
    )

    assert engine_inputs[0]["prompt_token_ids"] == [41, 42, 10, 50, 51]
    assert engine_inputs[0]["assistant_tokens_mask"] == [1, 1, 1, 0, 0]
    assert "prompt_token_offsets" not in engine_inputs[0]
    assert request.vllm_xargs is not None
    assert request.vllm_xargs["rwkv_state_read_ref"] == "goal-parent"
    assert str(request.vllm_xargs["rwkv_state_read_lease"]).startswith("rwkv-read-")


@pytest.mark.asyncio
async def test_state_request_rejects_multiple_sampled_sequences() -> None:
    serving = object.__new__(OpenAIServingChat)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "delta"}],
        n=2,
        vllm_xargs={"rwkv_state_write_ref": "goal-child"},
    )

    with pytest.raises(ValueError, match="one sampled sequence"):
        await serving._prepend_rwkv_state_tail(
            request,
            [{"type": "token", "prompt_token_ids": [0, 50]}],
            {},
            _BoundaryTokenizer(),
        )


@pytest.mark.asyncio
async def test_state_write_auto_generates_server_ref() -> None:
    serving = object.__new__(OpenAIServingChat)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "start"}],
        vllm_xargs={
            "rwkv_state_write_ref": "auto",
            "rwkv_state_read_lease": "client-forged",
        },
    )

    state_ref = await serving._prepend_rwkv_state_tail(
        request,
        [{"type": "token", "prompt_token_ids": [0, 50]}],
        {},
        _BoundaryTokenizer(),
    )

    assert state_ref is not None and state_ref.startswith("rwkv-")
    assert request.vllm_xargs == {"rwkv_state_write_ref": state_ref}


@pytest.mark.asyncio
async def test_state_receipt_waits_until_workers_confirm_commit() -> None:
    calls = 0

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [{"state_ref": "goal-child", "exists": False}]
            return [{"state_ref": "goal-child", "exists": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    await serving._confirm_rwkv_state_ref("goal-child")

    assert calls == 2


@pytest.mark.asyncio
async def test_state_ref_is_returned_by_full_and_streaming_responses() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "question"}],
        return_token_ids=True,
    )
    serving = _minimal_serving()
    serving._confirm_rwkv_state_ref = AsyncMock()

    response = await serving.chat_completion_full_generator(
        request,
        _single_output(),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "question"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
        rwkv_state_write_ref="goal-child",
    )

    assert response.rwkv_state_ref == "goal-child"
    chunks = []
    async for line in serving.chat_completion_stream_generator(
        request,
        _single_output(),
        "chatcmpl-test-id",
        "test-model",
        conversation=[{"role": "user", "content": "question"}],
        tokenizer=MagicMock(),
        request_metadata=RequestResponseMetadata(request_id="chatcmpl-test-id"),
        rwkv_state_write_ref="goal-child",
    ):
        if line.startswith("data: ") and "[DONE]" not in line:
            chunks.append(json.loads(line[6:]))
    assert chunks[-1]["rwkv_state_ref"] == "goal-child"
    assert serving._confirm_rwkv_state_ref.await_count == 2
