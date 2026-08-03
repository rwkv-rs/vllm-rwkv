# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare legacy native and fused FLA packed-varlen RWKV7 WKV paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.model_executor.models.rwkv7_wkv_backend import (
    prepare_fla_rwkv7_recurrent_metadata,
    run_fla_rwkv7_recurrent_from_decay_logits,
)

HEAD_SIZE = 64
PROVIDERS = ("legacy_native", "fla_fused")
WKV_MODES = ("fp16", "fp32io16")
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
    providers: tuple[str, ...]
    wkv_modes: tuple[str, ...]
    warmup_iters: int
    samples: int
    sample_iters: int
    seed: int
    output: Path | None


@dataclass(frozen=True)
class ProfileCase:
    name: str
    lengths: tuple[int, ...]
    query_start_loc: torch.Tensor
    slot_indices: torch.Tensor
    initial_state: torch.Tensor
    r: torch.Tensor
    w: torch.Tensor
    w0: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    elapsed: torch.Tensor


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
        "csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cpp",
        "csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cu",
        "vllm/model_executor/models/rwkv7_wkv_backend.py",
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
    return tuple(
        0.02
        * torch.randn(
            shape,
            dtype=torch.float16,
            device="cuda",
            generator=generator,
        )
        for _ in range(6)
    )


def _build_case(
    name: str,
    lengths: tuple[int, ...],
    *,
    hidden_size: int,
    mode: str,
    generator: torch.Generator,
) -> ProfileCase:
    if any(length <= 0 for length in lengths):
        raise ValueError(f"{name} contains a non-positive length")
    batch_size = len(lengths)
    total_tokens = sum(lengths)
    head_count = hidden_size // HEAD_SIZE
    slot_count = batch_size + 7
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    query_start_loc = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    slot_indices = torch.arange(
        batch_size - 1,
        -1,
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    r, w, k, v, a, b = _payload(total_tokens, hidden_size, generator)
    w0 = -2.0 + 0.02 * torch.randn(
        (hidden_size,),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    state_dtype = torch.float16 if mode == "fp16" else torch.float32
    initial_state = 0.02 * torch.randn(
        (slot_count, head_count, HEAD_SIZE, HEAD_SIZE),
        dtype=state_dtype,
        device="cuda",
        generator=generator,
    )
    elapsed = torch.arange(slot_count, dtype=torch.int32, device="cuda")
    return ProfileCase(
        name=name,
        lengths=lengths,
        query_start_loc=query_start_loc,
        slot_indices=slot_indices,
        initial_state=initial_state,
        r=r,
        w=w,
        w0=w0,
        k=k,
        v=v,
        a=a,
        b=b,
        elapsed=elapsed,
    )


def _legacy_state_from_canonical(state: torch.Tensor) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous()


def _canonical_state_from_legacy(state: torch.Tensor) -> torch.Tensor:
    return state.transpose(-1, -2).contiguous()


def _execute(
    provider: str,
    mode: str,
    case: ProfileCase,
    state: torch.Tensor,
    legacy_output: torch.Tensor | None,
    validated_metadata: object | None,
) -> torch.Tensor:
    hidden_size = case.r.shape[-1]
    if provider == "legacy_native":
        if legacy_output is None:
            raise ValueError("legacy native execution requires an output buffer")
        if mode == "fp16":
            torch.ops.rwkv7_wkv_fp16_v2.wkv(
                case.query_start_loc,
                case.slot_indices,
                state,
                case.r,
                case.w,
                case.w0,
                case.k,
                case.v,
                case.a,
                case.b,
                legacy_output,
                case.elapsed,
            )
        else:
            decay_logits = torch.ops.rwkv7_fast_ops_fp16.add_vec(
                hidden_size,
                case.w,
                case.w0,
            )
            torch.ops.rwkv7_wkv_fp32_v2.wkv(
                case.query_start_loc,
                case.slot_indices,
                state,
                case.r,
                decay_logits,
                case.k,
                case.v,
                case.a,
                case.b,
                legacy_output,
            )
        return legacy_output

    if provider != "fla_fused":
        raise ValueError(f"unknown provider: {provider}")
    if validated_metadata is None:
        raise ValueError("fla_fused execution requires prevalidated metadata")
    total_tokens = case.r.shape[0]
    head_count = hidden_size // HEAD_SIZE
    packed_shape = (1, total_tokens, head_count, HEAD_SIZE)
    return run_fla_rwkv7_recurrent_from_decay_logits(
        case.r.view(packed_shape),
        case.w.view(packed_shape),
        case.k.view(packed_shape),
        case.v.view(packed_shape),
        case.a.view(packed_shape),
        case.b.view(packed_shape),
        decay_bias=case.w0.view(head_count, HEAD_SIZE),
        elapsed_t=case.elapsed if mode == "fp16" else None,
        validated_metadata=validated_metadata,
        state_pool=state,
        cu_seqlens=case.query_start_loc,
        state_indices=case.slot_indices,
        mode=mode,
    ).view_as(case.r)


def _provider_state(provider: str, canonical_state: torch.Tensor) -> torch.Tensor:
    if provider == "legacy_native":
        return _legacy_state_from_canonical(canonical_state)
    return canonical_state.clone()


def _run_provider(
    provider: str,
    mode: str,
    case: ProfileCase,
    config: BenchmarkConfig,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    benchmark_state = _provider_state(provider, case.initial_state)
    benchmark_output = torch.empty_like(case.r) if provider == "legacy_native" else None
    validated_metadata = None
    if provider == "fla_fused":
        validated_metadata = prepare_fla_rwkv7_recurrent_metadata(
            case.query_start_loc,
            case.slot_indices,
            total_tokens=case.r.shape[0],
            state_pool_size=case.initial_state.shape[0],
        )

    def call() -> None:
        _execute(
            provider,
            mode,
            case,
            benchmark_state,
            benchmark_output,
            validated_metadata,
        )

    latencies_ms = _measure_samples(
        call,
        warmup_iters=config.warmup_iters,
        samples=config.samples,
        sample_iters=config.sample_iters,
    )

    outputs = []
    states = []
    for _ in range(2):
        state = _provider_state(provider, case.initial_state)
        output_buffer = (
            torch.empty_like(case.r) if provider == "legacy_native" else None
        )
        output = _execute(
            provider,
            mode,
            case,
            state,
            output_buffer,
            validated_metadata,
        )
        outputs.append(output.clone())
        states.append(
            _canonical_state_from_legacy(state)
            if provider == "legacy_native"
            else state.clone()
        )
    torch.accelerator.synchronize()

    median_ms = statistics.median(latencies_ms)
    active_slots = case.slot_indices.long()
    untouched = torch.ones(case.initial_state.size(0), dtype=torch.bool, device="cuda")
    untouched[active_slots] = False
    result: dict[str, object] = {
        "profile": case.name,
        "provider": provider,
        "wkv_mode": mode,
        "prevalidated_metadata": validated_metadata is not None,
        "batch_size": len(case.lengths),
        "total_tokens": sum(case.lengths),
        "min_length": min(case.lengths),
        "max_length": max(case.lengths),
        "padding_ratio": 1.0
        - sum(case.lengths) / (len(case.lengths) * max(case.lengths)),
        "latency_ms": {
            "min": min(latencies_ms),
            "mean": statistics.fmean(latencies_ms),
            "p10": _percentile(latencies_ms, 0.10),
            "p50": median_ms,
            "p90": _percentile(latencies_ms, 0.90),
        },
        "useful_tokens_per_s_p50": sum(case.lengths) * 1000.0 / median_ms,
        "deterministic_output": torch.equal(outputs[0], outputs[1]),
        "deterministic_state": torch.equal(states[0], states[1]),
        "untouched_slots_preserved": torch.equal(
            states[0][untouched],
            case.initial_state[untouched],
        ),
        "finite_output": bool(torch.isfinite(outputs[0]).all().item()),
        "finite_state": bool(torch.isfinite(states[0][active_slots]).all().item()),
        "latency_samples_ms": latencies_ms,
    }
    return result, outputs[0], states[0]


def _rrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    numerator = difference.square().mean().sqrt()
    denominator = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((numerator / denominator).item())


def run(config: BenchmarkConfig) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if config.hidden_size <= 0 or config.hidden_size % HEAD_SIZE != 0:
        raise ValueError("hidden_size must be a positive multiple of 64")

    import vllm.rwkv7_ops  # noqa: F401

    import vllm

    generator = torch.Generator(device="cuda")
    generator.manual_seed(config.seed)
    results: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for mode in config.wkv_modes:
        for name in config.profiles:
            case = _build_case(
                name,
                PROFILE_LENGTHS[name],
                hidden_size=config.hidden_size,
                mode=mode,
                generator=generator,
            )
            provider_results = {}
            provider_outputs = {}
            provider_states = {}
            for provider in config.providers:
                result, output, state = _run_provider(provider, mode, case, config)
                results.append(result)
                provider_results[provider] = result
                provider_outputs[provider] = output
                provider_states[provider] = state
            if {"legacy_native", "fla_fused"} <= provider_results.keys():
                legacy_latency = provider_results["legacy_native"]["latency_ms"]
                fused_latency = provider_results["fla_fused"]["latency_ms"]
                assert isinstance(legacy_latency, dict)
                assert isinstance(fused_latency, dict)
                comparisons.append(
                    {
                        "profile": name,
                        "wkv_mode": mode,
                        "fla_speedup_over_legacy_p50": (
                            float(legacy_latency["p50"]) / float(fused_latency["p50"])
                        ),
                        "output_rrmse": _rrmse(
                            provider_outputs["fla_fused"],
                            provider_outputs["legacy_native"],
                        ),
                        "state_rrmse": _rrmse(
                            provider_states["fla_fused"],
                            provider_states["legacy_native"],
                        ),
                    }
                )
    payload: dict[str, object] = {
        "benchmark": "rwkv7_varlen_wkv_provider_comparison",
        "device": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "vllm_version": vllm.__version__,
        "hidden_size": config.hidden_size,
        "head_count": config.hidden_size // HEAD_SIZE,
        "providers": config.providers,
        "wkv_modes": config.wkv_modes,
        "warmup_iters": config.warmup_iters,
        "samples": config.samples,
        "sample_iters": config.sample_iters,
        "seed": config.seed,
        "source_sha256": _source_sha256(),
        "results": results,
        "comparisons": comparisons,
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
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=PROVIDERS,
        default=list(PROVIDERS),
    )
    parser.add_argument(
        "--wkv-modes",
        nargs="+",
        choices=WKV_MODES,
        default=list(WKV_MODES),
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
            providers=tuple(args.providers),
            wkv_modes=tuple(args.wkv_modes),
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
