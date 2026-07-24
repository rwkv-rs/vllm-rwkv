# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throughput benchmark for the canonical packed-varlen RWKV7 WKV op."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

HEAD_SIZE = 64
PROFILE_LENGTHS: dict[str, tuple[int, ...]] = {
    "decode_b1": (1,),
    "decode_b16": (1,) * 16,
    "decode_b32": (1,) * 32,
    "decode_b64": (1,) * 64,
    "decode_b128": (1,) * 128,
    "decode_b320": (1,) * 320,
    "equal_chunk16_b320": (16,) * 320,
    "ragged_chunk16_b320": tuple(range(1, 17)) * 20,
    "ragged_long_b32": (1, 4, 8, 16, 32, 64, 96, 128) * 4,
    "ragged_skew_b32": (128,) + (1,) * 31,
}


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    profiles: tuple[str, ...]
    warmup_iters: int
    samples: int
    sample_iters: int
    seed: int
    output: Path | None


def _percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measure_samples(
    fn: Callable[[], None],
    *,
    warmup_iters: int,
    samples: int,
    sample_iters: int,
) -> list[float]:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    latencies_ms: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(sample_iters):
            fn()
        end.record()
        end.synchronize()
        latencies_ms.append(start.elapsed_time(end) / sample_iters)
    return latencies_ms


def _source_sha256() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cpp",
        "csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cu",
    )
    return {
        relative: hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _payload(
    total_tokens: int,
    hidden_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, ...]:
    shape = (total_tokens, hidden_size)
    return (
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
        -2.0
        + 0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        ),
    )


def _run_profile(
    name: str,
    lengths: tuple[int, ...],
    config: BenchmarkConfig,
    generator: torch.Generator,
) -> dict[str, object]:
    batch_size = len(lengths)
    total_tokens = sum(lengths)
    head_count = config.hidden_size // HEAD_SIZE
    slot_count = batch_size + 7
    offsets = [0]
    for length in lengths:
        if length <= 0:
            raise ValueError(f"{name} contains a non-positive length: {length}")
        offsets.append(offsets[-1] + length)

    query_start_loc = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    slot_indices = torch.arange(
        batch_size - 1,
        -1,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    payload = _payload(total_tokens, config.hidden_size, generator)
    r, w, k, v, a, b = payload
    w0 = -2.0 + 0.02 * torch.randn(
        (config.hidden_size,),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    initial_state = 0.02 * torch.randn(
        (slot_count, head_count, HEAD_SIZE, HEAD_SIZE),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    elapsed = torch.arange(slot_count, dtype=torch.int32, device="cuda")

    benchmark_state = initial_state.clone()
    benchmark_output = torch.empty_like(r)

    def call() -> None:
        torch.ops.rwkv7_wkv_fp16_v2.wkv(
            query_start_loc,
            slot_indices,
            benchmark_state,
            r,
            w,
            w0,
            k,
            v,
            a,
            b,
            benchmark_output,
            elapsed,
        )

    latencies_ms = _measure_samples(
        call,
        warmup_iters=config.warmup_iters,
        samples=config.samples,
        sample_iters=config.sample_iters,
    )

    first_state = initial_state.clone()
    first_output = torch.empty_like(r)
    second_state = initial_state.clone()
    second_output = torch.empty_like(r)
    for state, output in (
        (first_state, first_output),
        (second_state, second_output),
    ):
        torch.ops.rwkv7_wkv_fp16_v2.wkv(
            query_start_loc,
            slot_indices,
            state,
            r,
            w,
            w0,
            k,
            v,
            a,
            b,
            output,
            elapsed,
        )
    torch.accelerator.synchronize()

    median_ms = statistics.median(latencies_ms)
    active_slots = slot_indices.long()
    untouched = torch.ones(slot_count, dtype=torch.bool, device="cuda")
    untouched[active_slots] = False
    return {
        "profile": name,
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "padding_ratio": 1.0 - total_tokens / (batch_size * max(lengths)),
        "kernel_launches": 1,
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
            "p10": _percentile(latencies_ms, 0.10),
            "p50": median_ms,
            "p90": _percentile(latencies_ms, 0.90),
        },
        "useful_tokens_per_s_p50": total_tokens * 1000.0 / median_ms,
        "deterministic_output": torch.equal(first_output, second_output),
        "deterministic_state": torch.equal(first_state, second_state),
        "untouched_slots_preserved": torch.equal(
            first_state[untouched],
            initial_state[untouched],
        ),
        "finite_output": bool(torch.isfinite(first_output).all().item()),
        "finite_state": bool(torch.isfinite(first_state[active_slots]).all().item()),
        "latency_samples_ms": latencies_ms,
    }


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE != 0:
        raise ValueError("hidden_size must be a positive multiple of 64")

    import vllm.rwkv7_ops  # noqa: F401

    import vllm

    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.seed)
    results = [
        _run_profile(name, PROFILE_LENGTHS[name], config, generator)
        for name in config.profiles
    ]
    payload: dict[str, object] = {
        "benchmark": "rwkv7_canonical_varlen_wkv",
        "provider": "rwkv7_wkv_fp16_v2::wkv",
        "device": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "hidden_size": config.hidden_size,
        "head_count": config.hidden_size // HEAD_SIZE,
        "wkv_mode": "fp16-state-fp16-io",
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "sample_iters": config.sample_iters,
        "seed": config.seed,
        "source_sha256": _source_sha256(),
        "results": results,
    }
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=tuple(PROFILE_LENGTHS),
        default=[
            "decode_b1",
            "decode_b16",
            "decode_b32",
            "decode_b64",
            "decode_b128",
            "decode_b320",
            "ragged_chunk16_b320",
            "ragged_long_b32",
            "ragged_skew_b32",
            "equal_chunk16_b320",
        ],
    )
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--sample-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        BenchmarkConfig(
            hidden_size=args.hidden_size,
            profiles=tuple(args.profiles),
            warmup_iters=args.warmup_iters,
            samples=args.samples,
            sample_iters=args.sample_iters,
            seed=args.seed,
            output=args.output,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
