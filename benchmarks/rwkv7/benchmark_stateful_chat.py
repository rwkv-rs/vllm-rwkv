#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare RWKV State-chain chat against exact full-token replay over HTTP."""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

RWKV_BOS_EOS_TOKEN_ID = 0


@dataclass(frozen=True)
class ChatResult:
    content: str
    token_ids: list[int]
    state_ref: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    finish_reason: str | None


@dataclass(frozen=True)
class TurnComparison:
    conversation_id: str
    turn: int
    exact_match: bool
    full: ChatResult
    state: ChatResult


def build_state_payload(
    *,
    model: str,
    user_content: str,
    state_ref: str | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Build one deterministic State-chain request containing only the delta."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 1.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "vllm_xargs": {
            **({"rwkv_state_read_ref": state_ref} if state_ref else {}),
            "rwkv_state_write_ref": "auto",
        },
    }


def build_replay_payload(
    *,
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
) -> dict[str, Any]:
    """Build one deterministic completion request replaying exact tokens."""
    return {
        "model": model,
        "prompt": prompt_token_ids,
        "temperature": 1.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "stop": ["✿"],
    }


def parse_conversations(data: object) -> list[tuple[str, list[str]]]:
    """Validate the compact benchmark input format."""
    if not isinstance(data, list) or not data:
        raise ValueError("input must be a non-empty JSON list")
    conversations = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"conversation {index} must be an object")
        conversation_id = str(item.get("id", index))
        turns = item.get("turns")
        if (
            not isinstance(turns, list)
            or not turns
            or any(not isinstance(turn, str) or not turn for turn in turns)
        ):
            raise ValueError(
                f"conversation {conversation_id} needs non-empty string turns"
            )
        conversations.append((conversation_id, turns))
    return conversations


def canonical_bot_turn_boundary(finish_reason: str | None) -> str:
    """Return the bot-template boundary following an exact output token list."""
    if finish_reason == "stop":
        return "\n"
    if finish_reason in {"length", "repetition"}:
        return "✿\n"
    raise RuntimeError(f"RWKV State ref has unsupported finish reason: {finish_reason}")


def summarize(comparisons: list[TurnComparison]) -> dict[str, Any]:
    """Aggregate correctness, replay savings, and latency."""
    if not comparisons:
        raise ValueError("no turn comparisons were produced")
    full_prompt_tokens = sum(item.full.prompt_tokens for item in comparisons)
    state_prompt_tokens = sum(item.state.prompt_tokens for item in comparisons)
    exact_matches = sum(item.exact_match for item in comparisons)
    first_divergence = next(
        (
            {"conversation_id": item.conversation_id, "turn": item.turn}
            for item in comparisons
            if not item.exact_match
        ),
        None,
    )
    return {
        "turns": len(comparisons),
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / len(comparisons),
        "first_divergence": first_divergence,
        "full_prompt_tokens": full_prompt_tokens,
        "state_prompt_tokens": state_prompt_tokens,
        "prompt_token_savings": full_prompt_tokens - state_prompt_tokens,
        "prompt_token_savings_ratio": (
            1 - state_prompt_tokens / full_prompt_tokens if full_prompt_tokens else 0.0
        ),
        "full_mean_latency_ms": sum(item.full.latency_ms for item in comparisons)
        / len(comparisons),
        "state_mean_latency_ms": sum(item.state.latency_ms for item in comparisons)
        / len(comparisons),
    }


async def request_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with session.request(method, url, json=payload) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"{method} {url} failed ({response.status}): {body}")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise RuntimeError(f"{method} {url} returned a non-object response")
        return result


def _parse_generation_result(
    result: dict[str, Any],
    *,
    chat: bool,
    latency_ms: float,
) -> ChatResult:
    try:
        choice = result["choices"][0]
        content = (
            choice["message"]["content"] if chat else choice.get("text", "")
        ) or ""
        token_ids = choice["token_ids"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("generation response is missing one output") from error
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError("generation response omitted output token IDs")
    usage = result.get("usage") or {}
    return ChatResult(
        content=content,
        token_ids=token_ids,
        state_ref=result.get("rwkv_state_ref"),
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        latency_ms=latency_ms,
        finish_reason=choice.get("finish_reason"),
    )


async def call_generation(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: dict[str, Any],
    *,
    chat: bool,
) -> ChatResult:
    start = time.perf_counter()
    route = "chat/completions" if chat else "completions"
    result = await request_json(
        session,
        "POST",
        f"{base_url}/v1/{route}",
        payload=payload,
    )
    return _parse_generation_result(
        result,
        chat=chat,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def tokenize_chat_delta(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    user_content: str,
) -> list[int]:
    result = await request_json(
        session,
        "POST",
        f"{base_url}/tokenize",
        payload={
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "add_generation_prompt": True,
        },
    )
    token_ids = result.get("tokens")
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError("tokenize response omitted token IDs")
    return token_ids


async def tokenize_text(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    text: str,
) -> list[int]:
    result = await request_json(
        session,
        "POST",
        f"{base_url}/tokenize",
        payload={"model": model, "prompt": text, "add_special_tokens": False},
    )
    token_ids = result.get("tokens")
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise RuntimeError("tokenize response omitted token IDs")
    return token_ids


async def drop_state_ref(
    session: aiohttp.ClientSession,
    base_url: str,
    state_ref: str,
) -> None:
    await request_json(
        session,
        "DELETE",
        f"{base_url}/v1/rwkv/state/{state_ref}",
    )


async def run_conversation(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    conversation_id: str,
    turns: list[str],
    max_tokens: int,
) -> list[TurnComparison]:
    """Run one conversation through replay and State paths at every turn."""
    replay_history: list[int] = []
    replay_finish_reason: str | None = None
    state_ref: str | None = None
    comparisons = []
    try:
        for turn_index, user_content in enumerate(turns, start=1):
            delta_token_ids = await tokenize_chat_delta(
                session, base_url, model, user_content
            )
            if replay_history and delta_token_ids[:1] == [RWKV_BOS_EOS_TOKEN_ID]:
                delta_token_ids = delta_token_ids[1:]
            boundary_token_ids = []
            if replay_history:
                boundary_token_ids = await tokenize_text(
                    session,
                    base_url,
                    model,
                    canonical_bot_turn_boundary(replay_finish_reason),
                )
            replay_prompt_token_ids = [
                *replay_history,
                *boundary_token_ids,
                *delta_token_ids,
            ]
            full_result = await call_generation(
                session,
                base_url,
                build_replay_payload(
                    model=model,
                    prompt_token_ids=replay_prompt_token_ids,
                    max_tokens=max_tokens,
                ),
                chat=False,
            )
            state_result = await call_generation(
                session,
                base_url,
                build_state_payload(
                    model=model,
                    user_content=user_content,
                    state_ref=state_ref,
                    max_tokens=max_tokens,
                ),
                chat=True,
            )
            next_state_ref = state_result.state_ref
            if next_state_ref is None:
                raise RuntimeError("State-chain response omitted rwkv_state_ref")
            if state_ref is not None:
                await drop_state_ref(session, base_url, state_ref)
            state_ref = next_state_ref
            comparisons.append(
                TurnComparison(
                    conversation_id=conversation_id,
                    turn=turn_index,
                    exact_match=full_result.token_ids == state_result.token_ids,
                    full=full_result,
                    state=state_result,
                )
            )
            replay_history = [*replay_prompt_token_ids, *full_result.token_ids]
            replay_finish_reason = full_result.finish_reason
    finally:
        if state_ref is not None:
            await drop_state_ref(session, base_url, state_ref)
    return comparisons


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base_url = args.url.rstrip("/")
    input_data = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    conversations = parse_conversations(input_data)
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        models = await request_json(session, "GET", f"{base_url}/v1/models")
        model = args.served_model_name or models["data"][0]["id"]
        capabilities = await request_json(
            session, "GET", f"{base_url}/v1/rwkv/state/capabilities"
        )
        workers = capabilities.get("workers") or []
        if not workers or not all(worker.get("enabled") for worker in workers):
            raise RuntimeError("RWKV State cache is not enabled on every worker")

        warmup_id, warmup_turns = conversations[0]
        await run_conversation(
            session,
            base_url=base_url,
            model=model,
            conversation_id=f"{warmup_id}-warmup",
            turns=warmup_turns[:1],
            max_tokens=args.max_tokens,
        )

        comparisons = []
        for conversation_id, turns in conversations:
            comparisons.extend(
                await run_conversation(
                    session,
                    base_url=base_url,
                    model=model,
                    conversation_id=conversation_id,
                    turns=turns,
                    max_tokens=args.max_tokens,
                )
            )

    result = {
        "model": model,
        "summary": summarize(comparisons),
        "turns": [asdict(item) for item in comparisons],
    }
    if args.output:
        Path(args.output).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.require_exact_match and result["summary"]["exact_match_rate"] != 1.0:
        first = result["summary"]["first_divergence"]
        raise RuntimeError(f"RWKV State-chain diverged from full replay: {first}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--served-model-name")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output")
    parser.add_argument("--require-exact-match", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = asyncio.run(run(parse_args()))
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
