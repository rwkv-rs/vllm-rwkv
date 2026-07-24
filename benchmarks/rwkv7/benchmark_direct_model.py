#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure RWKV model and lm_head at the Albatross CUDA-graph boundary."""

from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .benchmark_faster3a import (
        BenchmarkConfig,
        _create_vllm_runner_llm,
        _shutdown_vllm_runner_llm,
    )
else:
    from benchmark_faster3a import (
        BenchmarkConfig,
        _create_vllm_runner_llm,
        _shutdown_vllm_runner_llm,
    )


DEFAULT_CASES = ((1, 1), (128, 1), (256, 1), (320, 1), (1, 128))
PROVIDERS = (
    "model_hidden_fullgraph",
    "model_graph_then_fp32_head",
    "model_plus_fp32_head_fullgraph",
)
FP16_LT_FAMILIES = (
    "attention-c2c",
    "ffn-down",
    "lowrank-in",
    "lowrank-out",
)


def _parse_cases(value: str) -> tuple[tuple[int, int], ...]:
    cases: list[tuple[int, int]] = []
    for item in value.replace(",", " ").split():
        parts = item.lower().split("x", 1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid case {item!r}; expected BxT")
        batch_size, token_count = (int(part) for part in parts)
        if batch_size <= 0 or token_count <= 0:
            raise argparse.ArgumentTypeError("B and T must both be positive")
        cases.append((batch_size, token_count))
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    return tuple(cases)


def _disabled_fp16_lt_families(tuning_family: str) -> tuple[str, ...]:
    if tuning_family == "all":
        return FP16_LT_FAMILIES
    if tuning_family == "lowrank":
        return ("lowrank-in", "lowrank-out")
    if tuning_family in FP16_LT_FAMILIES:
        return (tuning_family,)
    raise ValueError(f"unknown FP16 Lt tuning family: {tuning_family}")


def _ln1_tmix_fuse_for_variant(
    variant: str,
    batch_size: int,
    token_count: int,
    production_enabled: bool,
) -> bool:
    if variant == "baseline":
        return production_enabled
    if variant == "tuned":
        return production_enabled and batch_size == 1 and token_count == 1
    raise ValueError(f"unknown model comparison variant: {variant}")


def _percentile(values: list[float], quantile: float) -> float:
    import torch

    return float(
        torch.quantile(
            torch.tensor(values, dtype=torch.float64),
            quantile,
        ).item()
    )


def _capture_graph(
    fn: Callable[[], Any],
) -> tuple[Any, Any]:
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    torch.accelerator.synchronize()
    return graph, output


def _measure_replay(
    replay: Callable[[], None],
    output: Callable[[], Any],
    batch_size: int,
    token_count: int,
    provider: str,
    warmup: int,
    iters: int,
    profile_range: bool = False,
) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        replay()
    torch.accelerator.synchronize()

    durations_ms: list[float] = []
    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        replay()
        end.record()
        end.synchronize()
        durations_ms.append(float(start.elapsed_time(end)))
    if profile_range:
        torch.cuda.cudart().cudaProfilerStop()

    final_output = output()
    return _measurement_result(
        durations_ms,
        final_output,
        batch_size,
        token_count,
        provider,
    )


def _measurement_result(
    durations_ms: list[float],
    final_output: Any,
    batch_size: int,
    token_count: int,
    provider: str,
) -> dict[str, Any]:
    result = {
        "provider": provider,
        "batch_size": batch_size,
        "token_count": token_count,
        "tokens": batch_size * token_count,
        "p10_ms": _percentile(durations_ms, 0.10),
        "p50_ms": _percentile(durations_ms, 0.50),
        "p90_ms": _percentile(durations_ms, 0.90),
        "output_shape": list(final_output.shape),
        "output_dtype": str(final_output.dtype),
    }
    result["tokens_per_s_p50"] = result["tokens"] * 1000.0 / result["p50_ms"]

    return result


def _measure_replay_pair(
    replays: dict[str, Callable[[], None]],
    outputs: dict[str, Callable[[], Any]],
    batch_size: int,
    token_count: int,
    warmup: int,
    iters: int,
) -> dict[str, dict[str, Any]]:
    """Measure two graphs in alternating order to limit thermal/order drift."""
    import torch

    variants = ("baseline", "tuned")
    for iteration in range(warmup):
        order = variants if iteration % 2 == 0 else variants[::-1]
        for variant in order:
            replays[variant]()
    torch.accelerator.synchronize()

    durations_ms = {variant: [] for variant in variants}
    for iteration in range(iters):
        order = variants if iteration % 2 == 0 else variants[::-1]
        for variant in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            replays[variant]()
            end.record()
            end.synchronize()
            durations_ms[variant].append(float(start.elapsed_time(end)))

    return {
        variant: _measurement_result(
            durations_ms[variant],
            outputs[variant](),
            batch_size,
            token_count,
            f"{variant}_model_plus_fp32_head_fullgraph",
        )
        for variant in variants
    }


def _profile_graph_kernels(graph: Any, replays: int = 3) -> list[dict[str, Any]]:
    import torch

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(replays):
            graph.replay()
        torch.accelerator.synchronize()
    events = []
    for event in profiler.key_averages():
        self_device_us = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        if self_device_us <= 0:
            continue
        events.append(
            {
                "name": event.key,
                "count": int(event.count),
                "self_device_time_total_us": self_device_us,
            }
        )
    events.sort(
        key=lambda item: item["self_device_time_total_us"],
        reverse=True,
    )
    return events


def _run_worker_direct_model_benchmark(
    worker: Any,
    cases: tuple[tuple[int, int], ...],
    warmup: int,
    iters: int,
    profile_range: bool,
    profile_kernels: bool,
    joint_only: bool,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.models.rwkv7 import select_path

    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("worker.model_runner.model is unavailable")
    required = ("embed", "zero_state", "forward_from_x", "project_logits_fp32")
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise RuntimeError(f"loaded model lacks RWKV direct APIs: {missing}")

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    kernel_profiles: list[dict[str, Any]] = []
    for batch_size, token_count in cases:
        tokens = torch.arange(
            batch_size * token_count,
            dtype=torch.long,
            device="cuda",
        ).view(batch_size, token_count)
        tokens = (tokens * 1103515245 + 12345) % model.vocab_size
        embedded = model.embed(tokens).contiguous()
        torch.accelerator.synchronize()
        path = select_path(batch_size, token_count)
        query_start_loc = torch.arange(
            0,
            (batch_size + 1) * token_count,
            token_count,
            dtype=torch.int32,
            device="cuda",
        )
        wkv_slot_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
            device="cuda",
        )

        if joint_only:
            joint_state = model.zero_state(batch_size)

            def joint_call_only(
                embedded: Any = embedded,
                joint_state: Any = joint_state,
                path: Any = path,
                query_start_loc: Any = query_start_loc,
                wkv_slot_indices: Any = wkv_slot_indices,
            ) -> Any:
                joint_hidden = model.forward_from_x(
                    embedded,
                    joint_state,
                    path,
                    query_start_loc=query_start_loc,
                    wkv_slot_indices=wkv_slot_indices,
                )
                return model.project_logits_fp32(joint_hidden)

            joint_graph, joint_output = _capture_graph(joint_call_only)
            joint_result = _measure_replay(
                joint_graph.replay,
                lambda joint_output=joint_output: joint_output,
                batch_size,
                token_count,
                "model_plus_fp32_head_fullgraph",
                warmup,
                iters,
                profile_range=profile_range,
            )
            results.append(joint_result)
            if profile_kernels:
                kernel_profiles.append(
                    {
                        "batch_size": batch_size,
                        "token_count": token_count,
                        "replays": 3,
                        "events": _profile_graph_kernels(joint_graph),
                    }
                )
            del joint_call_only, joint_output, joint_graph, joint_state
            gc.collect()
            del embedded, tokens
            torch.accelerator.empty_cache()
            continue

        hidden_state = model.zero_state(batch_size)

        def hidden_call(
            embedded: Any = embedded,
            hidden_state: Any = hidden_state,
            path: Any = path,
            query_start_loc: Any = query_start_loc,
            wkv_slot_indices: Any = wkv_slot_indices,
        ) -> Any:
            return model.forward_from_x(
                embedded,
                hidden_state,
                path,
                query_start_loc=query_start_loc,
                wkv_slot_indices=wkv_slot_indices,
            )

        hidden_graph, hidden = _capture_graph(hidden_call)
        hidden_result = _measure_replay(
            hidden_graph.replay,
            lambda hidden=hidden: hidden,
            batch_size,
            token_count,
            "model_hidden_fullgraph",
            warmup,
            iters,
        )
        results.append(hidden_result)

        external_output = [hidden]

        def replay_with_external_head(
            hidden_graph: Any = hidden_graph,
            hidden: Any = hidden,
            external_output: list[Any] = external_output,
        ) -> None:
            hidden_graph.replay()
            external_output[0] = model.project_logits_fp32(hidden)

        external_result = _measure_replay(
            replay_with_external_head,
            lambda external_output=external_output: external_output[0],
            batch_size,
            token_count,
            "model_graph_then_fp32_head",
            warmup,
            iters,
        )
        results.append(external_result)

        del replay_with_external_head, hidden_call
        del external_output, hidden, hidden_graph, hidden_state
        gc.collect()
        torch.accelerator.empty_cache()

        joint_state = model.zero_state(batch_size)

        def joint_call(
            embedded: Any = embedded,
            joint_state: Any = joint_state,
            path: Any = path,
            query_start_loc: Any = query_start_loc,
            wkv_slot_indices: Any = wkv_slot_indices,
        ) -> Any:
            joint_hidden = model.forward_from_x(
                embedded,
                joint_state,
                path,
                query_start_loc=query_start_loc,
                wkv_slot_indices=wkv_slot_indices,
            )
            return model.project_logits_fp32(joint_hidden)

        joint_graph, joint_output = _capture_graph(joint_call)
        joint_result = _measure_replay(
            joint_graph.replay,
            lambda joint_output=joint_output: joint_output,
            batch_size,
            token_count,
            "model_plus_fp32_head_fullgraph",
            warmup,
            iters,
            profile_range=profile_range,
        )
        results.append(joint_result)
        if profile_kernels:
            kernel_profiles.append(
                {
                    "batch_size": batch_size,
                    "token_count": token_count,
                    "replays": 3,
                    "events": _profile_graph_kernels(joint_graph),
                }
            )

        case_results = {
            hidden_result["provider"]: hidden_result,
            external_result["provider"]: external_result,
            joint_result["provider"]: joint_result,
        }
        hidden_ms = case_results["model_hidden_fullgraph"]["p50_ms"]
        external_ms = case_results["model_graph_then_fp32_head"]["p50_ms"]
        joint_ms = case_results["model_plus_fp32_head_fullgraph"]["p50_ms"]
        comparisons.append(
            {
                "batch_size": batch_size,
                "token_count": token_count,
                "external_fp32_head_cost_p50_ms": external_ms - hidden_ms,
                "joint_capture_gain_p50_ms": external_ms - joint_ms,
            }
        )
        del joint_call, joint_output, joint_graph, joint_state
        gc.collect()
        del embedded, tokens
        torch.accelerator.empty_cache()

    return {
        "device": torch.cuda.get_device_name(),
        "warmup": warmup,
        "iters": iters,
        "providers": list(PROVIDERS),
        "results": results,
        "comparisons": comparisons,
        "kernel_profiles": kernel_profiles,
    }


def _run_worker_variant_comparison(
    worker: Any,
    cases: tuple[tuple[int, int], ...],
    warmup: int,
    iters: int,
    capture_order: str,
    comparison_kind: str,
    tuning_family: str,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.models import rwkv7

    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("worker.model_runner.model is unavailable")
    variants = (
        ("baseline", "tuned")
        if capture_order == "baseline-first"
        else ("tuned", "baseline")
    )
    att_tuned = dict(rwkv7.ATT_C2C_FP16_LT_4096)
    ffn_tuned = dict(rwkv7.FFN_DOWN_FP16_LT_4096)
    lowrank_in_tuned = dict(rwkv7.LOWRANK_IN_FP16_LT_4096)
    lowrank_out_tuned = dict(rwkv7.LOWRANK_OUT_FP16_LT_4096)
    m1_rkv_tuned = set(rwkv7.M1_RKV_GROUPED_ROWS)
    m1_cmix_prezero_tuned = set(rwkv7.M1_CMIX_PREZERO_ROWS)
    production_allow_fp16_accumulation = model.allow_fp16_accumulation
    production_ln1_tmix_fuse = rwkv7.LN1_TMIX_FUSE
    tables = {
        "attention-c2c": (
            rwkv7.ATT_C2C_FP16_LT_4096,
            att_tuned,
        ),
        "ffn-down": (
            rwkv7.FFN_DOWN_FP16_LT_4096,
            ffn_tuned,
        ),
        "lowrank-in": (
            rwkv7.LOWRANK_IN_FP16_LT_4096,
            lowrank_in_tuned,
        ),
        "lowrank-out": (
            rwkv7.LOWRANK_OUT_FP16_LT_4096,
            lowrank_out_tuned,
        ),
    }
    if comparison_kind == "fp16-lt":
        disabled_families = _disabled_fp16_lt_families(tuning_family)
    elif comparison_kind in (
        "m1-rkv",
        "m1-cmix-prezero",
        "gemm-accumulation",
        "ln1-tmix-fusion",
    ):
        disabled_families = ()
    else:
        raise ValueError(f"unknown model comparison kind: {comparison_kind}")

    def restore_production() -> None:
        for table, production_values in tables.values():
            table.clear()
            table.update(production_values)
        rwkv7.M1_RKV_GROUPED_ROWS.clear()
        rwkv7.M1_RKV_GROUPED_ROWS.update(m1_rkv_tuned)
        rwkv7.M1_CMIX_PREZERO_ROWS.clear()
        rwkv7.M1_CMIX_PREZERO_ROWS.update(m1_cmix_prezero_tuned)
        model.allow_fp16_accumulation = production_allow_fp16_accumulation
        rwkv7.LN1_TMIX_FUSE = production_ln1_tmix_fuse

    def select_variant(
        variant: str,
        batch_size: int,
        token_count: int,
    ) -> None:
        restore_production()
        if variant == "baseline":
            if comparison_kind == "m1-rkv":
                rwkv7.M1_RKV_GROUPED_ROWS.clear()
            elif comparison_kind == "m1-cmix-prezero":
                rwkv7.M1_CMIX_PREZERO_ROWS.clear()
            elif comparison_kind == "fp16-lt":
                for family in disabled_families:
                    tables[family][0].clear()
        elif variant == "tuned":
            if comparison_kind == "gemm-accumulation":
                model.allow_fp16_accumulation = False
            elif comparison_kind == "ln1-tmix-fusion":
                rwkv7.LN1_TMIX_FUSE = _ln1_tmix_fuse_for_variant(
                    variant,
                    batch_size,
                    token_count,
                    production_ln1_tmix_fuse,
                )
        else:
            raise ValueError(f"unknown model comparison variant: {variant}")

    results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    try:
        for batch_size, token_count in cases:
            tokens = torch.arange(
                batch_size * token_count,
                dtype=torch.long,
                device="cuda",
            ).view(batch_size, token_count)
            tokens = (tokens * 1103515245 + 12345) % model.vocab_size
            embedded = model.embed(tokens).contiguous()
            torch.accelerator.synchronize()
            path = rwkv7.select_path(batch_size, token_count)
            query_start_loc = torch.arange(
                0,
                (batch_size + 1) * token_count,
                token_count,
                dtype=torch.int32,
                device="cuda",
            )
            wkv_slot_indices = torch.arange(
                batch_size,
                dtype=torch.int32,
                device="cuda",
            )
            captures: dict[str, tuple[Any, Any, Any, Any]] = {}

            for variant in variants:
                select_variant(variant, batch_size, token_count)
                state = model.zero_state(batch_size)

                def joint_call(
                    state: Any = state,
                    path: Any = path,
                    embedded: Any = embedded,
                    query_start_loc: Any = query_start_loc,
                    wkv_slot_indices: Any = wkv_slot_indices,
                ) -> Any:
                    hidden = model.forward_from_x(
                        embedded,
                        state,
                        path,
                        query_start_loc=query_start_loc,
                        wkv_slot_indices=wkv_slot_indices,
                    )
                    return model.project_logits_fp32(hidden)

                graph, output = _capture_graph(joint_call)
                captures[variant] = (graph, output, state, joint_call)

            case_results = _measure_replay_pair(
                {
                    variant: captures[variant][0].replay
                    for variant in ("baseline", "tuned")
                },
                {
                    variant: (lambda output=captures[variant][1]: output)
                    for variant in ("baseline", "tuned")
                },
                batch_size,
                token_count,
                warmup,
                iters,
            )
            for variant in ("baseline", "tuned"):
                result = case_results[variant]
                results.append(result)

            baseline_output = captures["baseline"][1]
            tuned_output = captures["tuned"][1]
            difference = tuned_output.float() - baseline_output.float()
            baseline_norm = baseline_output.float().norm()
            cosine = torch.nn.functional.cosine_similarity(
                tuned_output.float().flatten(),
                baseline_output.float().flatten(),
                dim=0,
            )
            torch.accelerator.synchronize()
            comparisons.append(
                {
                    "batch_size": batch_size,
                    "token_count": token_count,
                    "tuned_gain_p50_ms": (
                        case_results["baseline"]["p50_ms"]
                        - case_results["tuned"]["p50_ms"]
                    ),
                    "tuned_gain_percent": (
                        (
                            case_results["baseline"]["p50_ms"]
                            - case_results["tuned"]["p50_ms"]
                        )
                        / case_results["baseline"]["p50_ms"]
                        * 100.0
                    ),
                    "max_abs_logit_difference": float(difference.abs().max().item()),
                    "relative_l2_logit_difference": float(
                        (difference.norm() / baseline_norm).item()
                    ),
                    "logit_cosine_similarity": float(cosine.item()),
                    "argmax_equal": bool(
                        torch.equal(
                            tuned_output.argmax(dim=-1),
                            baseline_output.argmax(dim=-1),
                        )
                    ),
                }
            )
            del captures, embedded, tokens
            gc.collect()
            torch.accelerator.empty_cache()
    finally:
        restore_production()

    return {
        "device": torch.cuda.get_device_name(),
        "warmup": warmup,
        "iters": iters,
        "capture_order": capture_order,
        "comparison_kind": comparison_kind,
        "tuning_family": (tuning_family if comparison_kind == "fp16-lt" else None),
        "measurement_order": "alternating-paired",
        "providers": [
            "baseline_model_plus_fp32_head_fullgraph",
            "tuned_model_plus_fp32_head_fullgraph",
        ],
        "tuning": {
            "attention_c2c": att_tuned,
            "ffn_down": ffn_tuned,
            "lowrank_in": {
                f"{rows}x{rank}": cfg for (rows, rank), cfg in lowrank_in_tuned.items()
            },
            "lowrank_out": {
                f"{rows}x{rank}": cfg for (rows, rank), cfg in lowrank_out_tuned.items()
            },
            "m1_rkv_grouped_rows": sorted(m1_rkv_tuned),
            "m1_cmix_prezero_rows": sorted(m1_cmix_prezero_tuned),
            "production_allow_fp16_accumulation": (production_allow_fp16_accumulation),
            "production_ln1_tmix_fuse": production_ln1_tmix_fuse,
            "tuned_allow_fp16_accumulation": (
                False
                if comparison_kind == "gemm-accumulation"
                else production_allow_fp16_accumulation
            ),
            "tuned_ln1_tmix_policy": (
                "B == 1 and T == 1"
                if comparison_kind == "ln1-tmix-fusion"
                else "production"
            ),
        },
        "results": results,
        "comparisons": comparisons,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--cases",
        type=_parse_cases,
        default=DEFAULT_CASES,
        help="comma- or space-separated BxT cases",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--profile-range",
        action="store_true",
        help="Bracket the joint full-graph timed loop with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="Collect a three-replay PyTorch profiler kernel summary.",
    )
    parser.add_argument(
        "--joint-only",
        action="store_true",
        help="Capture and measure only the full model-plus-head graph.",
    )
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--compare-fp16-lt-tuning",
        action="store_true",
        help="Compare exact FP16 Lt tables against the existing GemmEx path.",
    )
    comparison_group.add_argument(
        "--compare-m1-rkv",
        action="store_true",
        help="Compare grouped exact-M1 R/K/V split-K against separate launches.",
    )
    comparison_group.add_argument(
        "--compare-m1-cmix-prezero",
        action="store_true",
        help="Compare M1 FFN key reduction+prezero against a separate zero launch.",
    )
    comparison_group.add_argument(
        "--compare-gemm-accumulation",
        action="store_true",
        help=(
            "Compare production FP16 GEMM accumulation against an internal "
            "FP32-accumulation diagnostic."
        ),
    )
    comparison_group.add_argument(
        "--compare-ln1-tmix-fusion",
        action="store_true",
        help=(
            "Compare the production T=1 fused LN1/TMix path against the "
            "Albatross B=1,T=1-only fusion policy."
        ),
    )
    parser.add_argument(
        "--tuning-capture-order",
        choices=("baseline-first", "tuned-first"),
        default="baseline-first",
    )
    parser.add_argument(
        "--fp16-lt-family",
        choices=(
            "all",
            "lowrank",
            *FP16_LT_FAMILIES,
        ),
        default="all",
        help="Select which exact-shape Lt table family the A/B comparison toggles.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.iters <= 0:
        raise ValueError("iters must be positive")

    cases = tuple(args.cases)
    max_batch_size = max(batch_size for batch_size, _ in cases)
    max_token_count = max(token_count for _, token_count in cases)
    repo_root = Path(__file__).resolve().parents[2]
    config = BenchmarkConfig(
        repo_root=repo_root,
        model=args.model,
        batch_size=max_batch_size,
        prompt_len=max_token_count,
        warmup_tokens=1,
        decode_tokens=1,
        runner_enforce_eager=True,
    )
    llm = _create_vllm_runner_llm(config)
    try:
        if (
            args.compare_fp16_lt_tuning
            or args.compare_m1_rkv
            or args.compare_m1_cmix_prezero
            or args.compare_gemm_accumulation
            or args.compare_ln1_tmix_fusion
        ):
            comparison_kind = (
                "m1-rkv"
                if args.compare_m1_rkv
                else (
                    "m1-cmix-prezero"
                    if args.compare_m1_cmix_prezero
                    else (
                        "gemm-accumulation"
                        if args.compare_gemm_accumulation
                        else (
                            "ln1-tmix-fusion"
                            if args.compare_ln1_tmix_fusion
                            else "fp16-lt"
                        )
                    )
                )
            )
            worker_results = llm.collective_rpc(
                _run_worker_variant_comparison,
                args=(
                    cases,
                    args.warmup,
                    args.iters,
                    args.tuning_capture_order,
                    comparison_kind,
                    args.fp16_lt_family,
                ),
            )
        else:
            worker_results = llm.collective_rpc(
                _run_worker_direct_model_benchmark,
                args=(
                    cases,
                    args.warmup,
                    args.iters,
                    args.profile_range,
                    args.profile_kernels,
                    args.joint_only,
                ),
            )
    finally:
        _shutdown_vllm_runner_llm(llm)

    if len(worker_results) != 1:
        raise RuntimeError(f"expected one worker result, got {len(worker_results)}")
    payload = {
        "benchmark": "rwkv7_direct_model",
        "model": args.model,
        **worker_results[0],
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if not args.quiet:
        print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
