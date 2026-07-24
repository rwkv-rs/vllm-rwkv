#!/usr/bin/env python3
"""Load the real checkpoint through vLLM V1/Ascend and generate two tokens."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

from vllm import LLM, SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/data/models/fla-hub-rwkv7-7.2B-g0a",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dict(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        gpu_memory_utilization=0.45,
        block_size=128,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    started = time.time()
    llm = LLM(**config)
    load_seconds = time.time() - started
    print("ENGINE_LOADED_SEC", load_seconds, flush=True)
    outputs = llm.generate(["Hello"], SamplingParams(temperature=0, max_tokens=2))
    sample = outputs[0].outputs[0]
    record = {
        "axis": "vllm_ascend_real_engine_smoke",
        "status": "SMOKE_OK",
        "model": args.model,
        "engine_load_s": load_seconds,
        "output_text": sample.text,
        "output_token_ids": list(sample.token_ids),
        "config": {key: value for key, value in config.items() if key != "model"},
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("SMOKE_JSON", json.dumps(record, sort_keys=True), flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
