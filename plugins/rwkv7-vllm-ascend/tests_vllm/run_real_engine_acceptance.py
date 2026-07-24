#!/usr/bin/env python3
"""Real vLLM V1 dynamic/chunked/recurrent-cache acceptance on Ascend."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/models/fla-hub-rwkv7-7.2B-g0a",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/real_engine_acceptance.json"),
    )
    parser.add_argument("--trace", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    trace_path = (
        args.trace.resolve()
        if args.trace
        else output_path.with_name(output_path.stem + "_scheduler_trace.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("", encoding="utf-8")
    os.environ["RWKV7_TRACE_PATH"] = str(trace_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        add_special_tokens=False,
    )
    hello = tokenizer.encode("Hello", add_special_tokens=False)
    assert hello == [33155], hello
    assert seed

    def repeated(count: int) -> list[int]:
        return (seed * ((count + len(seed) - 1) // len(seed)))[:count]

    requests = [("short", hello), ("medium", repeated(47)), ("long", repeated(180))]
    print(
        "ACCEPTANCE_INPUT_LENGTHS",
        {key: len(value) for key, value in requests},
        flush=True,
    )
    config = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=256,
        block_size=128,
        max_num_batched_tokens=32,
        max_num_seqs=4,
        gpu_memory_utilization=0.45,
        num_gpu_blocks_override=4,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    started = time.time()
    llm = LLM(**config)
    engine_load_s = time.time() - started
    print("ACCEPTANCE_ENGINE_LOADED_SEC", engine_load_s, flush=True)
    params = SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)

    def run_batch(batch):
        prompts = [{"prompt_token_ids": ids} for _, ids in batch]
        outputs = llm.generate(prompts, params, use_tqdm=True)
        result = {}
        for (label, ids), output in zip(batch, outputs):
            sample = output.outputs[0]
            result[label] = {
                "input_tokens": len(ids),
                "output_token_ids": list(sample.token_ids),
                "finish_reason": sample.finish_reason,
                "finished": bool(output.finished),
            }
        return result

    first = run_batch(requests)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"marker": "first_batch_complete"}) + "\n")
    second = run_batch(list(reversed(requests)))
    assert first["short"]["output_token_ids"] == [45, 308, 459], first["short"]
    assert all(
        first[key]["output_token_ids"] == second[key]["output_token_ids"]
        for key, _ in requests
    )
    assert all(
        first[key]["finished"] and second[key]["finished"] for key, _ in requests
    )

    raw = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    boundary = next(
        index
        for index, event in enumerate(raw)
        if event.get("marker") == "first_batch_complete"
    )
    first_events = raw[:boundary]
    second_events = raw[boundary + 1 :]
    events = first_events + second_events
    scheduler_events = [event for event in events if "num_prefills" in event]
    zero_events = [
        event for event in events if event.get("event") == "fresh_state_zero"
    ]
    prefill_events = [event for event in scheduler_events if event["num_prefills"] > 0]
    actual_prefill_tokens = sum(event["num_prefill_tokens"] for event in prefill_events)
    # A singleton prompt can be represented as a decode by vLLM Mamba metadata.
    assert actual_prefill_tokens >= 2 * sum(len(ids) for _, ids in requests) - 2
    assert any(
        event["num_decodes"] > 0 and event["num_prefills"] > 0
        for event in scheduler_events
    )
    assert any(event["num_prefills"] > 1 for event in scheduler_events)

    def prefill_slots(batch_events, want_initial):
        slots = set()
        for event in batch_events:
            if "num_prefills" not in event:
                continue
            flags = event.get("has_initial_states_p") or []
            state_slots = event.get("prefill_state_slots") or []
            for slot, initial in zip(state_slots, flags):
                if bool(initial) == want_initial:
                    slots.add(int(slot))
        return slots

    first_fresh = prefill_slots(first_events, False)
    first_cont = prefill_slots(first_events, True)
    second_fresh = prefill_slots(second_events, False)
    second_cont = prefill_slots(second_events, True)
    assert first_fresh & first_cont
    assert second_fresh & second_cont
    reused_slots = first_fresh & second_fresh
    assert reused_slots, (first_fresh, second_fresh)
    second_zero_events = [
        event for event in second_events if event.get("event") == "fresh_state_zero"
    ]
    assert all(not event["post_zero_nonzero"] for event in zero_events)
    assert any(
        event["slot"] in reused_slots and event["pre_zero_had_nonzero"]
        for event in second_zero_events
    )
    initial_flags = [
        flag
        for event in prefill_events
        for flag in (event["has_initial_states_p"] or [])
    ]
    trace_summary = {
        "forward_steps": len(scheduler_events),
        "prefill_steps": len(prefill_events),
        "actual_prefill_tokens": actual_prefill_tokens,
        "max_prefill_tokens_in_step": max(
            event["num_prefill_tokens"] for event in prefill_events
        ),
        "mixed_decode_prefill_steps": sum(
            event["num_decodes"] > 0 and event["num_prefills"] > 0
            for event in scheduler_events
        ),
        "multi_prefill_request_steps": sum(
            event["num_prefills"] > 1 for event in scheduler_events
        ),
        "fresh_prefill_segments": initial_flags.count(False),
        "continuation_prefill_segments": initial_flags.count(True),
        "first_batch_fresh_slots": sorted(first_fresh),
        "second_batch_fresh_slots": sorted(second_fresh),
        "physically_reused_fresh_slots": sorted(reused_slots),
        "fresh_zero_events": len(zero_events),
        "reused_slots_with_nonzero_state_before_clear": sorted(
            {
                event["slot"]
                for event in second_zero_events
                if event["slot"] in reused_slots and event["pre_zero_had_nonzero"]
            }
        ),
        "trace_path": trace_path.name,
    }
    record = {
        "axis": "vllm_ascend_real_engine_acceptance",
        "status": "ACCEPTANCE_OK",
        "model": args.model,
        "engine": "vllm-v1-ascend",
        "hardware": "Ascend910B3 64GB",
        "runtime": {
            "vllm": importlib.metadata.version("vllm"),
            "vllm_ascend": importlib.metadata.version("vllm-ascend"),
            "plugin": importlib.metadata.version("rwkv7-vllm-ascend"),
        },
        "engine_load_s": engine_load_s,
        "config": {key: value for key, value in config.items() if key != "model"},
        "first_batch": first,
        "slot_reuse_reverse_batch": second,
        "hello_greedy_exact": True,
        "reuse_outputs_identical": True,
        "physical_slot_reuse_proven": True,
        "scheduler_trace": trace_summary,
        "chunk_gate": {
            "long_input_tokens": 180,
            "max_num_batched_tokens": 32,
            "minimum_prefill_steps": 6,
        },
    }
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("ACCEPTANCE_JSON", json.dumps(record, sort_keys=True), flush=True)
    print("ACCEPTANCE_OK", flush=True)


if __name__ == "__main__":
    main()
