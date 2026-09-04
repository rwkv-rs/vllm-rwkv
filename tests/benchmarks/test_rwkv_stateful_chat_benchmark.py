# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from benchmarks.rwkv7 import benchmark_stateful_chat as benchmark
from benchmarks.rwkv7.benchmark_stateful_chat import (
    ChatResult,
    TurnComparison,
    build_replay_payload,
    build_state_payload,
    canonical_bot_turn_boundary,
    parse_conversations,
    summarize,
)


def test_payloads_replay_tokens_or_send_only_state_delta() -> None:
    full = build_replay_payload(
        model="model",
        prompt_token_ids=[1, 2, 3],
        max_tokens=32,
    )
    state = build_state_payload(
        model="model",
        user_content="second",
        state_ref="parent",
        max_tokens=32,
    )

    assert full["prompt"] == [1, 2, 3]
    assert full["temperature"] == state["temperature"] == 1.0
    assert full["top_k"] == state["top_k"] == 1
    assert "vllm_xargs" not in full
    assert state["messages"] == [{"role": "user", "content": "second"}]
    assert state["vllm_xargs"] == {
        "rwkv_state_read_ref": "parent",
        "rwkv_state_write_ref": "auto",
    }


def test_summary_reports_first_divergence_and_prompt_savings() -> None:
    def result(content: str, prompt_tokens: int, latency_ms: float) -> ChatResult:
        return ChatResult(content, [1], None, prompt_tokens, 4, latency_ms, "stop")

    comparisons = [
        TurnComparison("a", 1, True, result("x", 10, 5), result("x", 10, 4)),
        TurnComparison("a", 2, False, result("y", 30, 7), result("z", 12, 5)),
    ]

    summary = summarize(comparisons)

    assert summary["exact_match_rate"] == 0.5
    assert summary["first_divergence"] == {"conversation_id": "a", "turn": 2}
    assert summary["prompt_token_savings"] == 18
    assert summary["prompt_token_savings_ratio"] == pytest.approx(0.45)
    assert summary["full_mean_latency_ms"] == 6
    assert summary["state_mean_latency_ms"] == 4.5


def test_parse_conversations_rejects_empty_turns() -> None:
    assert parse_conversations([{"id": "a", "turns": ["one", "two"]}]) == [
        ("a", ["one", "two"])
    ]

    with pytest.raises(ValueError, match="non-empty string turns"):
        parse_conversations([{"id": "a", "turns": [""]}])


def test_canonical_bot_turn_boundary_closes_only_truncated_outputs() -> None:
    assert canonical_bot_turn_boundary("stop") == "\n"
    assert canonical_bot_turn_boundary("length") == "✿\n"
    assert canonical_bot_turn_boundary("repetition") == "✿\n"

    with pytest.raises(RuntimeError, match="unsupported finish reason"):
        canonical_bot_turn_boundary(None)


@pytest.mark.asyncio
async def test_conversation_strips_repeated_bos_and_drops_state_refs(
    monkeypatch,
) -> None:
    responses = iter(
        [
            ChatResult("a1", [1], None, 10, 2, 1, "stop"),
            ChatResult("a1", [1], "state-1", 10, 2, 1, "stop"),
            ChatResult("a2", [2], None, 20, 2, 1, "stop"),
            ChatResult("a2", [2], "state-2", 5, 2, 1, "stop"),
        ]
    )
    payloads = []
    dropped = []

    async def fake_call(_session, _base_url, payload, *, chat):
        payloads.append((payload, chat))
        return next(responses)

    async def fake_tokenize(_session, _base_url, _model, content):
        return {"q1": [0, 10], "q2": [0, 20]}[content]

    async def fake_tokenize_text(_session, _base_url, _model, text):
        assert text == "\n"
        return [30]

    async def fake_drop_state_ref(_session, _base_url, state_ref):
        dropped.append(state_ref)

    monkeypatch.setattr(benchmark, "call_generation", fake_call)
    monkeypatch.setattr(benchmark, "tokenize_chat_delta", fake_tokenize)
    monkeypatch.setattr(benchmark, "tokenize_text", fake_tokenize_text)
    monkeypatch.setattr(benchmark, "drop_state_ref", fake_drop_state_ref)

    comparisons = await benchmark.run_conversation(
        object(),
        base_url="http://server",
        model="model",
        conversation_id="conversation",
        turns=["q1", "q2"],
        max_tokens=8,
    )

    assert all(item.exact_match for item in comparisons)
    assert payloads[2][0]["prompt"] == [0, 10, 1, 30, 20]
    assert payloads[3][0]["messages"] == [{"role": "user", "content": "q2"}]
    assert payloads[3][0]["vllm_xargs"]["rwkv_state_read_ref"] == "state-1"
    assert dropped == ["state-1", "state-2"]


@pytest.mark.asyncio
async def test_run_warms_one_turn_without_including_it_in_results(
    monkeypatch,
    tmp_path,
) -> None:
    input_file = tmp_path / "input.json"
    input_file.write_text(
        json.dumps([{"id": "chat", "turns": ["one", "two"]}]),
        encoding="utf-8",
    )
    calls = []
    result = ChatResult("a", [1], "state", 10, 1, 1, "stop")
    comparison = TurnComparison("chat", 1, True, result, result)

    async def fake_request_json(_session, method, url, *, payload=None):
        if url.endswith("/v1/models"):
            return {"data": [{"id": "model"}]}
        assert url.endswith("/v1/rwkv/state/capabilities")
        return {"workers": [{"enabled": True}]}

    async def fake_run_conversation(_session, **kwargs):
        calls.append(kwargs)
        return [comparison]

    monkeypatch.setattr(benchmark, "request_json", fake_request_json)
    monkeypatch.setattr(benchmark, "run_conversation", fake_run_conversation)

    benchmark_result = await benchmark.run(
        SimpleNamespace(
            url="http://server",
            input_file=str(input_file),
            served_model_name=None,
            api_key=None,
            timeout=10,
            max_tokens=8,
            output=None,
            require_exact_match=True,
        )
    )

    assert [call["turns"] for call in calls] == [["one"], ["one", "two"]]
    assert benchmark_result["summary"]["turns"] == 1
