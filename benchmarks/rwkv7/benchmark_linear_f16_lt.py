# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune FP16-accumulation cuBLASLt plans for the RWKV7 runtime GEMM layout.

The configured index is a position in the heuristic list returned by the
current cuBLASLt environment, not a stable algorithm identifier.  This tool
therefore always uses strict lookup and treats its output as evidence for the
exact recorded GPU/CUDA/PyTorch environment only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

LOWRANK_KN = (
    (4096, 128),
    (4096, 480),
    (4096, 96),
    (128, 4096),
    (480, 4096),
    (96, 4096),
)
DENSE_KN = (
    (4096, 4096),
    (4096, 16384),
)
LAGGING_ROWS = (8, 16, 64)
FULL_LOWRANK_ROWS = (8, 16, 32, 64, 128, 256, 320)
FULL_FFN_ROWS = (32, 64, 128, 256, 320)


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int


@dataclass(frozen=True)
class Metrics:
    p10_ms: float
    p50_ms: float
    p90_ms: float
    mean_ms: float


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _measure(
    fn: Callable[[], torch.Tensor | None],
    *,
    warmup_iters: int,
    benchmark_iters: int,
) -> Metrics:
    for _ in range(warmup_iters):
        fn()
    torch.accelerator.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(benchmark_iters)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    ends[-1].synchronize()
    latencies = [
        start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
    ]
    return Metrics(
        p10_ms=_percentile(latencies, 0.10),
        p50_ms=statistics.median(latencies),
        p90_ms=_percentile(latencies, 0.90),
        mean_ms=statistics.fmean(latencies),
    )


def _measure_peak_bytes(fn: Callable[[], torch.Tensor]) -> int:
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    baseline = torch.accelerator.memory_allocated()
    output = fn()
    torch.accelerator.synchronize()
    peak = max(0, torch.accelerator.max_memory_allocated() - baseline)
    del output
    return peak


def _measure_graph(
    fn: Callable[[], torch.Tensor],
    *,
    warmup_iters: int,
    benchmark_iters: int,
) -> tuple[Metrics, int]:
    # Warm both the allocator and the shape-specific Lt plan before capture.
    fn()
    torch.accelerator.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    baseline = torch.accelerator.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = fn()
    torch.accelerator.synchronize()
    capture_peak = max(0, torch.accelerator.max_memory_allocated() - baseline)
    metrics = _measure(
        graph.replay,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )
    del graph_output, graph
    return metrics, capture_peak


def _accuracy(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | bool]:
    difference = output.float() - reference
    error_norm = difference.norm()
    reference_norm = reference.norm()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        reference.flatten(),
        dim=0,
    )
    return {
        "finite": bool(torch.isfinite(output).all().item()),
        "max_abs_error": float(difference.abs().max().item()),
        "relative_l2": float((error_norm / reference_norm).item()),
        "cosine_similarity": float(cosine.item()),
        "error_l2": float(error_norm.item()),
    }


def _candidate_shapes(profile: str) -> tuple[Shape, ...]:
    if profile == "lagging":
        shapes = [
            Shape(m, k, n) for m in LAGGING_ROWS for k, n in (*LOWRANK_KN, *DENSE_KN)
        ]
        shapes.append(Shape(64, 16384, 4096))
        return tuple(shapes)
    if profile == "full":
        lowrank = [Shape(m, k, n) for m in FULL_LOWRANK_ROWS for k, n in LOWRANK_KN]
        dense = [Shape(m, k, n) for m in FULL_LOWRANK_ROWS for k, n in DENSE_KN]
        ffn = [Shape(m, 16384, 4096) for m in FULL_FFN_ROWS]
        return tuple((*lowrank, *dense, *ffn))
    raise ValueError(f"unknown profile: {profile}")


def _environment() -> dict[str, Any]:
    device = torch.accelerator.current_device_index()
    return {
        "device_index": device,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def _run_shape(
    shape: Shape,
    *,
    workspace_limits_mb: Sequence[int],
    heuristic_indices: Sequence[int],
    warmup_iters: int,
    benchmark_iters: int,
    graph_finalists: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed + shape.m * 1009 + shape.k * 17 + shape.n)
    x = torch.randn(
        (shape.m, shape.k),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    weight = torch.randn(
        (shape.k, shape.n),
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    if shape.m == 1 and shape.n % 64 == 0:
        baseline_provider = "linear_f16_m1_splitk"
        baseline_fn = lambda: torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(
            x,
            weight,
        )
    else:
        baseline_provider = "linear_f16"
        baseline_fn = lambda: torch.ops.rwkv7_v3a_ops.linear_f16(
            x,
            weight,
            True,
        )
    baseline_output = baseline_fn()
    repeated_baseline = baseline_fn()
    reference = torch.mm(x, weight, out_dtype=torch.float32)
    torch.accelerator.synchronize()
    baseline_accuracy = _accuracy(baseline_output, reference)
    baseline = {
        "provider": baseline_provider,
        "metrics": asdict(
            _measure(
                baseline_fn,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
        ),
        "graph_metrics": None,
        "peak_bytes": _measure_peak_bytes(baseline_fn),
        "accuracy": baseline_accuracy,
        "deterministic": bool(torch.equal(baseline_output, repeated_baseline)),
    }
    baseline_graph, baseline_graph_peak = _measure_graph(
        baseline_fn,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )
    baseline["graph_metrics"] = asdict(baseline_graph)
    baseline["graph_capture_peak_bytes"] = baseline_graph_peak

    candidates: list[dict[str, Any]] = []
    for workspace_mb in workspace_limits_mb:
        for heuristic_index in heuristic_indices:
            fn = lambda workspace_mb=workspace_mb, heuristic_index=heuristic_index: (
                torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                    x,
                    weight,
                    workspace_mb,
                    heuristic_index,
                    True,
                )
            )
            try:
                output = fn()
                repeated = fn()
                torch.accelerator.synchronize()
            except RuntimeError as exc:
                candidates.append(
                    {
                        "workspace_limit_mb": workspace_mb,
                        "heuristic_index": heuristic_index,
                        "available": False,
                        "error": str(exc).splitlines()[0],
                    }
                )
                continue

            accuracy = _accuracy(output, reference)
            error_limit = float(baseline_accuracy["error_l2"]) * 1.02 + 1e-6
            accuracy_ok = (
                bool(accuracy["finite"]) and float(accuracy["error_l2"]) <= error_limit
            )
            metrics = _measure(
                fn,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
            )
            candidates.append(
                {
                    "workspace_limit_mb": workspace_mb,
                    "heuristic_index": heuristic_index,
                    "available": True,
                    "metrics": asdict(metrics),
                    "peak_bytes": _measure_peak_bytes(fn),
                    "accuracy": accuracy,
                    "accuracy_ok": accuracy_ok,
                    "deterministic": bool(torch.equal(output, repeated)),
                    "graph_metrics": None,
                }
            )
            del output, repeated

    finalists = sorted(
        (
            candidate
            for candidate in candidates
            if candidate.get("available")
            and candidate.get("accuracy_ok")
            and candidate.get("deterministic")
        ),
        key=lambda candidate: (
            candidate["metrics"]["p50_ms"],
            candidate["metrics"]["p90_ms"],
            candidate["workspace_limit_mb"],
        ),
    )[:graph_finalists]
    for finalist in finalists:
        workspace_mb = int(finalist["workspace_limit_mb"])
        heuristic_index = int(finalist["heuristic_index"])
        fn = lambda workspace_mb=workspace_mb, heuristic_index=heuristic_index: (
            torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x,
                weight,
                workspace_mb,
                heuristic_index,
                True,
            )
        )
        graph_metrics, graph_peak = _measure_graph(
            fn,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
        )
        finalist["graph_metrics"] = asdict(graph_metrics)
        finalist["graph_capture_peak_bytes"] = graph_peak

    return {
        "shape": asdict(shape),
        "baseline": baseline,
        "candidates": candidates,
        "finalists": [
            {
                "workspace_limit_mb": candidate["workspace_limit_mb"],
                "heuristic_index": candidate["heuristic_index"],
            }
            for candidate in finalists
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("lagging", "full"), default="lagging")
    parser.add_argument(
        "--shape",
        type=int,
        nargs=3,
        action="append",
        metavar=("M", "K", "N"),
        help="Override the profile with one or more exact M K N shapes.",
    )
    parser.add_argument(
        "--workspace-limits-mb",
        type=int,
        nargs="+",
        default=[0, 32, 128],
    )
    parser.add_argument(
        "--heuristic-indices",
        type=int,
        nargs="+",
        default=list(range(64)),
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument("--graph-finalists", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    import vllm.rwkv7_ops  # noqa: F401

    shapes = (
        tuple(Shape(*values) for values in args.shape)
        if args.shape
        else _candidate_shapes(args.profile)
    )
    results = [
        _run_shape(
            shape,
            workspace_limits_mb=args.workspace_limits_mb,
            heuristic_indices=args.heuristic_indices,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            graph_finalists=args.graph_finalists,
            seed=args.seed,
        )
        for shape in shapes
    ]
    payload = {
        "benchmark": "rwkv7_linear_f16_lt",
        "environment": _environment(),
        "profile": args.profile,
        "warmup_iters": args.warmup_iters,
        "benchmark_iters": args.benchmark_iters,
        "workspace_limits_mb": args.workspace_limits_mb,
        "heuristic_indices": args.heuristic_indices,
        "results": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{text}\n", encoding="utf-8")
    if not args.quiet:
        print(text)


if __name__ == "__main__":
    main()
