# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

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


class _BoundaryTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return {"\n": [10], "\n\n": [12], "✿\n": [10060, 10]}[text]


def test_state_ref_is_omitted_from_unrelated_stream_chunks() -> None:
    chunk = ChatCompletionStreamResponse(model="test-model", choices=[])

    assert "rwkv_state_ref" not in chunk.model_dump(exclude_unset=True)

    chunk.rwkv_state_ref = "goal-a"
    assert chunk.model_dump(exclude_unset=True)["rwkv_state_ref"] == "goal-a"


def test_state_write_promotes_single_token_stop_strings() -> None:
    class _Tokenizer:
        def encode(self, text, add_special_tokens=True):
            assert not add_special_tokens
            return {"stop": [7], "two tokens": [8, 9]}[text]

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "start"}],
        stop_token_ids=[6],
    )

    _promote_rwkv_state_stop_strings(
        request,
        {"stop": ["stop", "two tokens"]},
        _Tokenizer(),
    )

    assert request.stop_token_ids == [6, 7]


@pytest.mark.parametrize(
    ("chat_template_kwargs", "finish_reason", "expected"),
    [
        ({"rwkv_prompt_template": "assistant"}, "length", [12]),
        ({"rwkv_prompt_template": "function_calling"}, "length", [10]),
    ],
)
def test_state_turn_boundary_follows_template(
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
async def test_serving_clone_rolls_back_partial_worker_commit() -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            if args[0] == "clone":
                raise RuntimeError("rank failed")
            return [{"ready": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    with pytest.raises(RuntimeError, match="rank failed"):
        await serving.rwkv_state_cache_action("clone", "goal-a", "goal-b")

    assert calls[-1] == (
        "rwkv_state_cache_action",
        ("drop_if_exists", "goal-b", ""),
    )


@pytest.mark.asyncio
async def test_serving_prepare_write_rolls_back_partial_reservation() -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            if args[0] == "prepare_write":
                raise RuntimeError("rank full")
            return [{"canceled": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    with pytest.raises(RuntimeError, match="rank full"):
        await serving.rwkv_state_cache_action("prepare_write", "goal-a")

    assert calls[-1] == (
        "rwkv_state_cache_action",
        ("cancel_write", "goal-a", ""),
    )


@pytest.mark.asyncio
async def test_serving_prepare_read_rolls_back_partial_lease() -> None:
    calls = []

    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            calls.append((method, args))
            if args[0] == "prepare_read":
                raise RuntimeError("rank missing")
            return [{"canceled": True}]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()

    with pytest.raises(RuntimeError, match="rank missing"):
        await serving.rwkv_state_cache_action("prepare_read", "goal-a", "lease-a")

    assert calls[-1] == (
        "rwkv_state_cache_action",
        ("cancel_read", "goal-a", "lease-a"),
    )


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
async def test_http_state_action_maps_unknown_ref_to_404() -> None:
    class _Handler:
        async def rwkv_state_cache_action(self, *_args):
            raise KeyError("Unknown RWKV State ref: missing")

    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=_Handler()))
    )

    with pytest.raises(HTTPException) as captured:
        await _rwkv_state_action(raw_request, "inspect", "missing")

    assert captured.value.status_code == 404
    assert "missing" in captured.value.detail


@pytest.mark.asyncio
async def test_http_state_action_maps_wrapped_in_use_error_to_409() -> None:
    class _Handler:
        async def rwkv_state_cache_action(self, *_args):
            raise RuntimeError(
                "Call to collective_rpc method failed: RWKV State ref is in use: active"
            )

    raw_request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(openai_serving_chat=_Handler()))
    )

    with pytest.raises(HTTPException) as captured:
        await _rwkv_state_action(raw_request, "drop", "active")

    assert captured.value.status_code == 409


@pytest.mark.asyncio
async def test_state_restore_prepends_unprocessed_tail_before_scheduling() -> None:
    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            assert method == "rwkv_state_cache_action"
            assert args == ("restore_metadata", "goal-parent", "")
            return [
                {
                    "state_ref": "goal-parent",
                    "pending_tail_token_ids": [41, 42],
                    "finish_reason": "stop",
                },
                {
                    "state_ref": "goal-parent",
                    "pending_tail_token_ids": [41, 42],
                    "finish_reason": "stop",
                },
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
            "prompt_token_ids": [50, 51],
            "prompt_token_offsets": [(0, 1), (1, 2)],
        }
    ]

    await serving._prepend_rwkv_state_tail(
        request, engine_inputs, {}, _BoundaryTokenizer()
    )

    assert engine_inputs[0]["prompt_token_ids"] == [41, 42, 10, 50, 51]
    assert "prompt_token_offsets" not in engine_inputs[0]
    assert request.vllm_xargs is not None
    assert request.vllm_xargs["rwkv_state_read_ref"] == "goal-parent"
    assert str(request.vllm_xargs["rwkv_state_read_lease"]).startswith("rwkv-read-")


@pytest.mark.asyncio
async def test_state_restore_closes_length_limited_bot_turn() -> None:
    class _EngineClient:
        async def collective_rpc(self, method, args=()):
            assert method == "rwkv_state_cache_action"
            assert args == ("restore_metadata", "goal-parent", "")
            return [
                {
                    "state_ref": "goal-parent",
                    "pending_tail_token_ids": [49],
                    "finish_reason": "length",
                }
            ]

    serving = object.__new__(OpenAIServingChat)
    serving.engine_client = _EngineClient()
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "delta"}],
        vllm_xargs={"rwkv_state_read_ref": "goal-parent"},
    )
    engine_inputs = [{"type": "token", "prompt_token_ids": [50]}]

    await serving._prepend_rwkv_state_tail(
        request, engine_inputs, {}, _BoundaryTokenizer()
    )

    assert engine_inputs[0]["prompt_token_ids"] == [49, 10060, 10, 50]


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
            [{"type": "token", "prompt_token_ids": [50]}],
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
        [{"type": "token", "prompt_token_ids": [50]}],
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
