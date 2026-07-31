# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark RWKV7 lm_head paths that feed FP32 sampling logits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchmarkConfig:
    hidden_size: int
    vocab_size: int
    batch_sizes: tuple[int, ...]
    warmup_iters: int
    benchmark_iters: int
    seed: int


@dataclass(frozen=True)
class Provider:
    project: Callable[[], torch.Tensor]
    convert: Callable[[torch.Tensor], torch.Tensor] | None = None


def _measure(
    fn: Callable[[], torch.Tensor],
    *,
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[float, int]:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    torch.accelerator.reset_peak_memory_stats()
    baseline = torch.accelerator.memory_allocated()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(benchmark_iters):
        output = fn()
        del output
    end.record()
    end.synchronize()
    peak_delta = max(0, torch.accelerator.max_memory_allocated() - baseline)
    return start.elapsed_time(end) / benchmark_iters, peak_delta


def _identity(output: torch.Tensor) -> torch.Tensor:
    return output


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import vllm.rwkv7_ops  # noqa: F401

    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.seed)
    weight = torch.randn(
        (config.hidden_size, config.vocab_size),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    rows: list[dict[str, object]] = []

    for batch_size in config.batch_sizes:
        hidden_states = torch.randn(
            (batch_size, config.hidden_size),
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        if batch_size == 1:
            legacy_project = lambda hidden_states=hidden_states: (
                torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(hidden_states, weight)
            )
            direct_project = lambda hidden_states=hidden_states: (
                torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32(hidden_states, weight)
            )
        else:
            legacy_project = lambda hidden_states=hidden_states: (
                torch.ops.rwkv7_v3a_ops.linear_f16(
                    hidden_states,
                    weight,
                    True,
                )
            )
            direct_project = lambda hidden_states=hidden_states: (
                torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt(
                    hidden_states,
                    weight,
                )
            )

        providers = {
            "legacy_fp16_plus_float": Provider(
                project=legacy_project,
                convert=lambda output: output.to(torch.float32),
            ),
            "direct_fp32": Provider(project=direct_project),
            "torch_mm_fp32": Provider(
                project=lambda hidden_states=hidden_states: torch.mm(
                    hidden_states,
                    weight,
                    out_dtype=torch.float32,
                )
            ),
        }

        for provider_name, provider in providers.items():
            convert = provider.convert or _identity
            projection_latency_ms, projection_peak_memory_bytes = _measure(
                provider.project,
                warmup_iters=config.warmup_iters,
                benchmark_iters=config.benchmark_iters,
            )
            projected = provider.project()
            if provider.convert is None:
                conversion_latency_ms = 0.0
                conversion_peak_memory_bytes = 0
            else:
                conversion_latency_ms, conversion_peak_memory_bytes = _measure(
                    lambda convert=convert, projected=projected: convert(projected),
                    warmup_iters=config.warmup_iters,
                    benchmark_iters=config.benchmark_iters,
                )
            total_latency_ms, peak_memory_bytes = _measure(
                lambda convert=convert, provider=provider: convert(provider.project()),
                warmup_iters=config.warmup_iters,
                benchmark_iters=config.benchmark_iters,
            )
            output = convert(projected)
            if output.dtype != torch.float32:
                raise RuntimeError(
                    f"{provider_name} returned {output.dtype}, expected torch.float32"
                )
            rows.append(
                {
                    "batch_size": batch_size,
                    "provider": provider_name,
                    "projection_latency_ms": projection_latency_ms,
                    "conversion_latency_ms": conversion_latency_ms,
                    "total_latency_ms": total_latency_ms,
                    "projection_peak_memory_bytes": projection_peak_memory_bytes,
                    "conversion_peak_memory_bytes": conversion_peak_memory_bytes,
                    "peak_memory_bytes": peak_memory_bytes,
                    "output_bytes": output.numel() * output.element_size(),
                    "output_dtype": str(output.dtype),
                    "output_shape": list(output.shape),
                }
            )
            del output, projected

    return {
        "benchmark": "rwkv7_lm_head_fp32",
        "device": torch.cuda.get_device_name(),
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "warmup_iters": config.warmup_iters,
        "benchmark_iters": config.benchmark_iters,
        "results": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 320])
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        BenchmarkConfig(
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            batch_sizes=tuple(args.batch_sizes),
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            seed=args.seed,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
