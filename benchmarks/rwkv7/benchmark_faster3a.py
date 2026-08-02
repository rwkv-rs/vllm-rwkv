# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 faster3a performance acceptance harness.

This harness records RWKV7 runner provenance and evaluates the canonical
faster3a throughput contract as structured JSON.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK_NAME = "rwkv7_faster3a"
ALBATROSS_REPO = "https://github.com/BlinkDL/Albatross"
ALBATROSS_COMMIT = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"
ALBATROSS_RACE_FIX = "ff144b6b11e01ac984ed05ca7f7af4dfdca97180"
ALBATROSS_2607_COMMIT = "63c53f4abf2cd891dd3a18c8f44f5b2cccc8c64b"
ALBATROSS_IMPL = "faster3a_2607"
RUNNER_RUNTIME_ENV_REQUIREMENTS = {
    "VLLM_RWKV7_WKV_MODE": "fp16",
    "VLLM_USE_RAPID_SAMPLER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
}
# These are fixed implementation semantics, not user-selectable vLLM
# environment variables. Keep them in benchmark provenance so the measured
# model contract remains explicit after the legacy runtime branches are gone.
RUNNER_FIXED_MODEL_CONTRACT = {
    "allow_fp16_accumulation": True,
    "embedding_device": "gpu",
    "rkv_mode": "off",
    "cmix_sparse": "no-fc",
    "low_rank_weight": "both",
    "orig_linear_groups": "none",
    "slot_mapped_state": True,
    "v2_kernel_warmup_policy": "skip_rwkv7",
}
RUNNER_BASELINE_TOKENS_PER_S = 9712.562
RUNNER_MAX_REGRESSION_PCT = 1.0
RUNNER_MIN_TOKENS_PER_S = RUNNER_BASELINE_TOKENS_PER_S * (
    1.0 - RUNNER_MAX_REGRESSION_PCT / 100.0
)
VLLM_RUNNER_MODE = "worker_execute_model"
VLLM_RUNNER_TIMING_TARGET = "worker.execute_model + worker.sample_tokens"
VLLM_RUNNER_TIMING_CLOCK = "cuda_event"
DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS = 16
RUNNER_INPUT_SEED = 0
VLLM_RUNNER_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "ignore_eos": True,
    "detokenize": False,
    "seed": RUNNER_INPUT_SEED,
}
# The outer benchmark launcher can still provide these historical variables,
# but vLLM deliberately rejects them because their model behavior is fixed.
LEGACY_FIXED_MODEL_ENV_VARS = (
    "VLLM_RWKV7_ALLOW_FP16_ACCUMULATION",
    "VLLM_RWKV7_EMB_DEVICE",
    "VLLM_RWKV7_RKV_MODE",
    "VLLM_RWKV7_CMIX_SPARSE",
    "VLLM_RWKV7_LOW_RANK_WEIGHT",
    "VLLM_RWKV7_ORIG_LINEAR_GROUPS",
    "VLLM_RWKV7_SLOT_MAPPED_STATE",
    "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP",
)
# The model path is a benchmark input rather than a vLLM runtime option. Strip
# it together with the legacy fixed-model variables before importing vLLM.
UNREGISTERED_BENCHMARK_ENV_VARS = (
    "VLLM_RWKV7_MODEL",
    *LEGACY_FIXED_MODEL_ENV_VARS,
)
PROVENANCE_ENV_VARS = tuple(RUNNER_RUNTIME_ENV_REQUIREMENTS)
PROVENANCE_ENV_DEFAULTS = dict(RUNNER_RUNTIME_ENV_REQUIREMENTS)
ACCEPTANCE_THRESHOLDS = {
    "runner_steady_decode": {
        "min_runner_tokens_per_s": RUNNER_MIN_TOKENS_PER_S,
        "max_regression_pct": RUNNER_MAX_REGRESSION_PCT,
    },
}


@dataclass(frozen=True)
class SourceProvenanceEntry:
    source_path: str
    target_path: str
    correspondence: str


SOURCE_PROVENANCE = (
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/rwkv7_fast_v3a.py",
        target_path="vllm/model_executor/models/rwkv7.py",
        correspondence="model-core-adapted",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_fast_ops_fp16.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_fast_ops_fp16.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_fast_ops_fp16.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_fast_ops_fp16.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_v3a_ops.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_v3a_ops.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_v3a_ops.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_v3a_ops.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp16_v2.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cpp",
        correspondence="packed-varlen-derived",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp16_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cu",
        correspondence="packed-varlen-derived",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cpp",
        correspondence="packed-varlen-derived",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cu",
        correspondence="packed-varlen-derived",
    ),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    model: str | None
    batch_size: int
    prompt_len: int
    warmup_tokens: int
    decode_tokens: int
    runner_prefill_chunk_tokens: int = DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS
    runner_enforce_eager: bool = False
    runner_cudagraph_capture_sizes: tuple[int, ...] | None = None


class RunnerModelContractError(ValueError):
    """Raised before runner startup when ``--model`` is not an HF artifact."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        artifact: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.code = code
        self.artifact = artifact


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    blocker = {"code": code, "message": message}
    blocker.update({k: v for k, v in details.items() if v is not None})
    return blocker


def _measurement_blockers(
    runtime_blockers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if runtime_blockers:
        return runtime_blockers
    return [
        _blocker(
            "missing_measurement_json",
            "Provide --measurement-json with RWKV7 faster3a benchmark metrics, "
            "or run the measurement lane first.",
        )
    ]


def _source_metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "albatross_repo": ALBATROSS_REPO,
        "albatross_commit": ALBATROSS_COMMIT,
        "albatross_changes": {
            "wkv_fp16_race_fix": ALBATROSS_RACE_FIX,
            "faster3a_2607": ALBATROSS_2607_COMMIT,
        },
        "contracts": [
            {
                "source_path": entry.source_path,
                "target_path": entry.target_path,
                "correspondence": entry.correspondence,
            }
            for entry in SOURCE_PROVENANCE
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_metadata(model: str | None) -> dict[str, Any] | None:
    if model is None:
        return None
    if _is_url(model):
        return {"kind": "url", "input_reference": model}
    path = Path(model).expanduser()
    if not path.exists():
        return {"kind": "hub_repo_id", "input_reference": model}
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "checkpoint_file",
            "input_reference": model,
            "resolved_reference": str(resolved),
            "size_bytes": resolved.stat().st_size,
        }

    config_path = resolved / "config.json"
    if not config_path.is_file():
        return {
            "kind": "local_directory",
            "input_reference": model,
            "resolved_reference": str(resolved),
            "config_sha256": None,
            "weight_files": [],
        }
    try:
        config_values = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid model artifact config: {config_path}") from error
    source_sha256 = config_values.get("rwkv_source_sha256")
    if source_sha256 is not None and (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256.lower())
    ):
        raise ValueError("RWKV7 artifact rwkv_source_sha256 is not a SHA-256 digest")
    weight_paths = sorted(resolved.glob("*.safetensors"))
    weights = [
        {
            "name": weight_path.name,
            "size_bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
        }
        for weight_path in weight_paths
    ]
    aggregate = hashlib.sha256()
    for weight in weights:
        aggregate.update(weight["name"].encode())
        aggregate.update(weight["sha256"].encode())
    return {
        "kind": "hf_artifact",
        "input_reference": model,
        "resolved_reference": str(resolved),
        "config_sha256": _sha256(config_path),
        "weights_sha256": aggregate.hexdigest() if weights else None,
        "weight_files": weights,
        "source_checkpoint_sha256": source_sha256,
    }


def _source_revision_file(repo_root: Path) -> Path | None:
    for path in (repo_root, *repo_root.parents):
        marker = path / ".helicopter-source-revision"
        if marker.is_file():
            return marker
    return None


def _synced_revision(repo_root: Path) -> str | None:
    repo_root = repo_root.resolve()
    for workspace in repo_root.parents:
        manifest = workspace / ".helicopter-dev/source-revisions.json"
        if not manifest.is_file():
            continue
        try:
            relative = repo_root.relative_to(workspace).as_posix()
            revisions = json.loads(manifest.read_text(encoding="utf-8"))
            revision = revisions["submodules"].get(relative)
        except (KeyError, OSError, ValueError):
            continue
        if revision:
            return str(revision)
    return None


def _git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        revision = _synced_revision(repo_root)
        if revision:
            return revision
        marker = _source_revision_file(repo_root)
        if marker is None:
            return None
        revision = marker.read_text(encoding="utf-8").strip()
        return revision or None
    lines = result.stdout.splitlines()
    if len(lines) != 2:
        return None
    git_root, revision = Path(lines[0]).resolve(), lines[1].strip()
    if git_root != repo_root.resolve():
        return _synced_revision(repo_root)
    try:
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return f"{revision}-dirty" if status.stdout.strip() else revision


def _cuda_device_metadata() -> dict[str, Any]:
    if not _cuda_available():
        return {"available": False}
    cuda = _cuda_module()
    device_index = cuda.current_device()
    props = cuda.get_device_properties(device_index)
    device_uuid = getattr(props, "uuid", None)
    return {
        "available": True,
        "device_index": int(device_index),
        "device_uuid": str(device_uuid) if device_uuid is not None else None,
        "device_name": cuda.get_device_name(device_index),
        "capability": list(cuda.get_device_capability(device_index)),
        "total_memory": int(props.total_memory),
    }


def _rwkv_environment_raw_metadata() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in PROVENANCE_ENV_VARS}


def _rwkv_environment_metadata() -> dict[str, str | None]:
    raw = _rwkv_environment_raw_metadata()
    resolved = {
        name: raw[name] if raw[name] is not None else PROVENANCE_ENV_DEFAULTS.get(name)
        for name in PROVENANCE_ENV_VARS
    }
    return resolved


@contextmanager
def _without_unregistered_benchmark_env_vars():
    saved = {
        name: os.environ.pop(name)
        for name in UNREGISTERED_BENCHMARK_ENV_VARS
        if name in os.environ
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def _benchmark_provenance(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "git_revision": _git_revision(config.repo_root),
        "artifact": _artifact_metadata(config.model),
        "cuda": _cuda_device_metadata(),
        "env": _rwkv_environment_metadata(),
        "raw_env": _rwkv_environment_raw_metadata(),
        "fixed_model_contract": dict(RUNNER_FIXED_MODEL_CONTRACT),
        "workload": {
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "runner_prefill_chunk_tokens": config.runner_prefill_chunk_tokens,
            "runner_enforce_eager": config.runner_enforce_eager,
            "runner_cudagraph_capture_sizes": (
                list(config.runner_cudagraph_capture_sizes)
                if config.runner_cudagraph_capture_sizes is not None
                else None
            ),
        },
        "sampling": dict(VLLM_RUNNER_SAMPLING),
    }


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(
        torch.accelerator.is_available()
        and torch.accelerator.current_accelerator().type == "cuda"
    )


def _cuda_module() -> Any:
    import torch

    return torch.cuda


def _runtime_blockers(
    config: BenchmarkConfig,
    *,
    cuda_available: bool,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if not cuda_available:
        blockers.append(
            _blocker(
                "cuda_unavailable",
                "CUDA is required for RWKV7 faster3a steady decode benchmarking.",
            )
        )

    if not config.model:
        blockers.append(
            _blocker(
                "missing_vllm_model",
                "Set --model or VLLM_RWKV7_MODEL to a standard RWKV7 HF artifact.",
            )
        )
    elif not _is_url(config.model) and not Path(config.model).expanduser().exists():
        blockers.append(
            _blocker(
                "missing_vllm_model_path",
                "The configured vLLM model path does not exist.",
                path=config.model,
            )
        )

    return blockers


def _get_number(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _duration_ms_summary(durations_s: list[float]) -> dict[str, float | None]:
    if not durations_s:
        return {f"p{p}_ms": None for p in (10, 50, 90)}
    return {
        f"p{p}_ms": _percentile(durations_s, p / 100) * 1000.0 for p in (10, 50, 90)
    }


def _tokens_per_second(tokens: int, duration_s: float) -> float:
    return float("inf") if duration_s <= 0 else tokens / duration_s


def _phase_throughput_summary(
    *,
    total_tokens: int,
    iteration_durations_s: list[float],
    unit_durations_s: list[float],
    unit_tokens: list[int],
) -> dict[str, Any]:
    total_duration_s = sum(iteration_durations_s)
    iteration_tokens = (
        total_tokens // len(iteration_durations_s) if iteration_durations_s else 0
    )
    peak_iteration = (
        max(_tokens_per_second(iteration_tokens, d) for d in iteration_durations_s)
        if iteration_durations_s
        else None
    )
    peak_unit = (
        max(
            _tokens_per_second(tokens, d)
            for tokens, d in zip(unit_tokens, unit_durations_s)
        )
        if unit_durations_s
        else None
    )
    summary = _duration_ms_summary(iteration_durations_s)
    unit = _duration_ms_summary(unit_durations_s)
    return {
        "avg_tokens_per_s": _tokens_per_second(total_tokens, total_duration_s),
        "peak_tokens_per_s": peak_iteration,
        "peak_iteration_tokens_per_s": peak_iteration,
        "peak_unit_tokens_per_s": peak_unit,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_s * 1000,
        **summary,
        "unit_p10_ms": unit["p10_ms"],
        "unit_p50_ms": unit["p50_ms"],
        "unit_p90_ms": unit["p90_ms"],
    }


def _required_vllm_runner_model(config: BenchmarkConfig) -> str:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if _is_url(config.model):
        return config.model
    model_path = Path(config.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"missing vLLM model path: {config.model}")
    resolved = model_path.resolve()
    if resolved.is_file():
        raise RunnerModelContractError(
            "unsupported_runner_checkpoint_file",
            "RWKV7 runner benchmark requires a converted HF artifact directory; "
            f"raw checkpoint files are not runtime-loadable: {resolved}. Convert "
            "once with vllm.transformers_utils.rwkv7_artifact."
            "convert_rwkv7_pth_to_hf_artifact().",
            artifact={
                "kind": "checkpoint_file",
                "input_reference": config.model,
                "resolved_reference": str(resolved),
                "size_bytes": resolved.stat().st_size,
            },
        )
    config_path = resolved / "config.json"
    weight_paths = sorted(resolved.glob("*.safetensors"))
    if not config_path.is_file() or not weight_paths:
        raise RunnerModelContractError(
            "invalid_runner_hf_artifact",
            "RWKV7 runner benchmark requires a local HF artifact directory with "
            f"config.json and safetensors weights: {resolved}",
            artifact={
                "kind": "local_directory",
                "input_reference": config.model,
                "resolved_reference": str(resolved),
                "has_config_json": config_path.is_file(),
                "safetensors_files": [path.name for path in weight_paths],
            },
        )
    return config.model


def _runner_prefill_chunk_tokens(config: BenchmarkConfig) -> int:
    if config.runner_prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    return min(config.prompt_len, config.runner_prefill_chunk_tokens)


def _create_vllm_runner_llm(config: BenchmarkConfig) -> Any:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

    max_model_len = max(1, config.prompt_len + config.decode_tokens)
    max_num_seqs = max(1, config.batch_size)
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(config)
    max_num_batched_tokens = max(
        config.batch_size * prefill_chunk_tokens,
        config.batch_size,
    )
    model = _required_vllm_runner_model(config)
    llm_kwargs: dict[str, Any] = {}
    if (
        not config.runner_enforce_eager
        and config.runner_cudagraph_capture_sizes is not None
    ):
        capture_sizes = list(config.runner_cudagraph_capture_sizes)
        if not capture_sizes or any(size <= 0 for size in capture_sizes):
            raise ValueError("runner cudagraph capture sizes must be positive")
        llm_kwargs["compilation_config"] = {
            "cudagraph_capture_sizes": capture_sizes,
        }
    with _without_unregistered_benchmark_env_vars():
        import vllm.rwkv7_ops  # noqa: F401

        from vllm import LLM

        return LLM(
            model=model,
            skip_tokenizer_init=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=prefill_chunk_tokens,
            enable_chunked_prefill=True,
            enforce_eager=config.runner_enforce_eager,
            disable_log_stats=True,
            **llm_kwargs,
        )


def _synchronize_cuda_if_available() -> None:
    try:
        import torch
    except Exception:
        return
    if _cuda_available():
        torch.accelerator.synchronize()


def _worker_cuda_synchronize() -> None:
    import torch

    if _cuda_available():
        torch.accelerator.synchronize()


def _worker_cuda_event_pair() -> tuple[Any, Any] | None:
    if not _cuda_available():
        return None
    cuda = _cuda_module()
    return (
        cuda.Event(enable_timing=True),
        cuda.Event(enable_timing=True),
    )


def _worker_time_call(
    fn: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    if cuda_events is None:
        import time

        _worker_cuda_synchronize()
        start_s = time.perf_counter()
        fn()
        _worker_cuda_synchronize()
        return time.perf_counter() - start_s

    start_event, end_event = cuda_events
    _worker_cuda_synchronize()
    start_event.record()
    fn()
    end_event.record()
    end_event.synchronize()
    return start_event.elapsed_time(end_event) / 1000.0


def _worker_time_execute_model(
    worker: Any,
    scheduler_output: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    return _worker_time_call(
        lambda: worker.execute_model(scheduler_output),
        cuda_events,
    )


def _worker_time_sample_tokens(
    worker: Any,
    grammar_output: Any,
    cuda_events: tuple[Any, Any] | None,
) -> float:
    return _worker_time_call(
        lambda: worker.sample_tokens(grammar_output),
        cuda_events,
    )


def _worker_empty_scheduler_output(finished_req_ids: set[str] | None = None) -> Any:
    from vllm.v1.core.sched.output import SchedulerOutput

    output = SchedulerOutput.make_empty()
    if finished_req_ids:
        output.finished_req_ids = finished_req_ids
    return output


def _worker_new_request_scheduler_output(
    *,
    req_ids: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
    num_scheduled_tokens: int,
) -> Any:
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_ids,
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=(),
                num_computed_tokens=0,
                lora_request=None,
                prefill_token_ids=prompt_ids,
            )
            for req_id, prompt_ids in zip(req_ids, prompt_token_ids)
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: num_scheduled_tokens for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids) * num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_add_decode_requests_scheduler_output(
    *,
    req_ids: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
    prompt_len: int,
) -> Any:
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_ids,
                mm_features=[],
                sampling_params=sampling_params,
                pooling_params=None,
                block_ids=(),
                num_computed_tokens=prompt_len,
                lora_request=None,
                prefill_token_ids=prompt_ids,
            )
            for req_id, prompt_ids in zip(req_ids, prompt_token_ids)
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_cached_prefill_scheduler_output(
    *,
    req_ids: list[str],
    num_computed_tokens: int,
    num_scheduled_tokens: int,
) -> Any:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[num_computed_tokens] * len(req_ids),
            num_output_tokens=[0] * len(req_ids),
        ),
        num_scheduled_tokens={req_id: num_scheduled_tokens for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids) * num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_cached_decode_scheduler_output(
    *,
    req_ids: list[str],
    prompt_len: int,
    step: int,
) -> Any:
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None] * len(req_ids),
            num_computed_tokens=[prompt_len + step] * len(req_ids),
            num_output_tokens=[step + 1] * len(req_ids),
        ),
        num_scheduled_tokens={req_id: 1 for req_id in req_ids},
        total_num_scheduled_tokens=len(req_ids),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _worker_internal_runner_blocker(code: str, message: str) -> dict[str, Any]:
    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "timing_clock": VLLM_RUNNER_TIMING_CLOCK,
        "blockers": [_blocker(code, message)],
    }


def _worker_prompt_inputs(
    worker: Any,
    batch_size: int,
    prompt_len: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    import torch

    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    vocab_size = int(getattr(model, "vocab_size", 0))
    if vocab_size <= 0:
        raise RuntimeError("RWKV7 runner benchmark cannot determine model vocab_size")
    generator = torch.Generator().manual_seed(RUNNER_INPUT_SEED)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, prompt_len),
        generator=generator,
    )
    return input_ids.tolist(), {
        "generator": "torch.Generator(device=cpu).manual_seed",
        "seed": RUNNER_INPUT_SEED,
        "dtype": str(input_ids.dtype),
        "shape": list(input_ids.shape),
        "prompt_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "vocab_size": vocab_size,
    }


def _worker_prefill_state_contract(
    worker: Any,
    req_ids: list[str],
    prompt_len: int,
) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    model_state = getattr(model_runner, "model_state", None)
    req_id_to_index = getattr(model_state, "req_id_to_index", None)
    elapsed = getattr(model_state, "elapsed", None)
    if not isinstance(req_id_to_index, dict) or elapsed is None:
        raise RuntimeError(
            "RWKV7 runner cannot prove recurrent state ownership after prefill"
        )
    missing = [req_id for req_id in req_ids if req_id not in req_id_to_index]
    if missing:
        raise RuntimeError(
            "RWKV7 runner prefill did not allocate state for every request: "
            f"missing={missing[:8]}"
        )
    state_indices = [int(req_id_to_index[req_id]) for req_id in req_ids]
    if len(set(state_indices)) != len(state_indices):
        raise RuntimeError("RWKV7 runner prefill state ownership is not unique")
    processed_tokens = [int(elapsed[index].item()) for index in state_indices]
    if any(tokens < prompt_len for tokens in processed_tokens):
        raise RuntimeError(
            "RWKV7 runner decode would start before prompt prefill completed: "
            f"minimum_elapsed={min(processed_tokens)} prompt_len={prompt_len}"
        )
    return {
        "validated": True,
        "request_count": len(req_ids),
        "unique_state_indices": len(set(state_indices)),
        "minimum_prefill_elapsed": min(processed_tokens),
        "maximum_prefill_elapsed": max(processed_tokens),
        "decode_state_origin": "same-request real chunked prefill",
    }


def _worker_runtime_provenance(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.models import rwkv7

    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    profile = getattr(model, "execution_profile", None)
    if model is None or profile is None:
        raise RuntimeError("RWKV7 runner cannot inspect loaded model runtime profile")

    def stage(token_count: int) -> dict[str, Any]:
        rows = batch_size * token_count
        path = rwkv7.select_path(batch_size, token_count)
        allow_lt = profile.allow_fp16_accumulation and model.hidden_size == 4096
        return {
            "batch_size": batch_size,
            "token_count": token_count,
            "rows": rows,
            "cmix_mode": path.cmix_mode,
            "resolved_fp16_lt": {
                "attention_c2c": (
                    rwkv7.ATT_C2C_FP16_LT_4096.get(rows) if allow_lt else None
                ),
                "ffn_down": (
                    rwkv7.FFN_DOWN_FP16_LT_4096.get(rows) if allow_lt else None
                ),
            },
            "m1_rkv_grouped": rows in rwkv7.M1_RKV_GROUPED_ROWS,
            "m1_cmix_prezero": rows in rwkv7.M1_CMIX_PREZERO_ROWS,
        }

    wkv_operator = (
        "torch.ops.rwkv7_wkv_fp32_v2.wkv"
        if profile.wkv_mode == "fp32io16"
        else "torch.ops.rwkv7_wkv_fp16_v2.wkv"
    )
    return {
        "precision": {
            "activation_dtype": str(next(iter(model.z.values())).dtype),
            "logits_dtype": str(torch.float32),
            "wkv_state_dtype": str(profile.wkv_state_dtype),
        },
        "wkv": {
            "mode": profile.wkv_mode,
            "provider": "vllm RWKV7 native CUDA ops",
            "operator": wkv_operator,
            "metadata_contract": "packed-varlen slot-mapped recurrent state",
        },
        "gemm": {
            "provider": "torch.ops.rwkv7_v3a_ops",
            "accumulation_policy": profile.gemm_accumulation_policy,
            "allow_fp16_accumulation": profile.allow_fp16_accumulation,
        },
        "kernels_by_stage": {
            "prefill_chunk": stage(min(prompt_len, prefill_chunk_tokens)),
            "decode": stage(1),
        },
    }


def _worker_execute_prefill_chunks(
    worker: Any,
    *,
    req_ids: list[str],
    prompt_token_ids: list[list[int]],
    sampling_params: Any,
    prompt_len: int,
    prefill_chunk_tokens: int,
    cuda_events: tuple[Any, Any] | None = None,
    measure: bool = False,
) -> tuple[list[float], list[int]]:
    first_chunk_tokens = min(prefill_chunk_tokens, prompt_len)
    prefill_output = _worker_new_request_scheduler_output(
        req_ids=req_ids,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        num_scheduled_tokens=first_chunk_tokens,
    )
    chunk_durations_s: list[float] = []
    chunk_token_counts: list[int] = []
    if measure:
        chunk_durations_s.append(
            _worker_time_execute_model(worker, prefill_output, cuda_events)
        )
        chunk_token_counts.append(len(req_ids) * first_chunk_tokens)
    else:
        worker.execute_model(prefill_output)
    num_prefill_tokens = first_chunk_tokens
    while num_prefill_tokens < prompt_len:
        chunk_tokens = min(prefill_chunk_tokens, prompt_len - num_prefill_tokens)
        prefill_output = _worker_cached_prefill_scheduler_output(
            req_ids=req_ids,
            num_computed_tokens=num_prefill_tokens,
            num_scheduled_tokens=chunk_tokens,
        )
        if measure:
            chunk_durations_s.append(
                _worker_time_execute_model(worker, prefill_output, cuda_events)
            )
            chunk_token_counts.append(len(req_ids) * chunk_tokens)
        else:
            worker.execute_model(prefill_output)
        num_prefill_tokens += chunk_tokens
    return chunk_durations_s, chunk_token_counts


def _worker_finish_execute_without_sampling(worker: Any) -> None:
    model_runner = getattr(worker, "model_runner", None)
    execute_model_state = getattr(model_runner, "execute_model_state", None)
    if execute_model_state is None:
        return
    input_batch = getattr(execute_model_state, "input_batch", None)
    model_state = getattr(model_runner, "model_state", None)
    postprocess_state = getattr(model_state, "postprocess_state", None)
    if input_batch is not None and callable(postprocess_state):
        req_states = getattr(model_runner, "req_states", None)
        num_computed_tokens = getattr(req_states, "num_computed_tokens", None)
        num_computed_gpu = getattr(num_computed_tokens, "gpu", None)
        postprocess_state(input_batch.idx_mapping, 0, num_computed_gpu)
    model_runner.execute_model_state = None


def _run_vllm_worker_internal_prefill(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not callable(getattr(worker, "execute_model", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_execute_model",
            "The vLLM worker does not expose execute_model().",
        )
    if not callable(getattr(worker, "sample_tokens", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_sample_tokens",
            "The vLLM worker does not expose sample_tokens().",
        )
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    prefill_chunk_tokens = min(prefill_chunk_tokens, prompt_len)

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
        seed=VLLM_RUNNER_SAMPLING["seed"],
    )
    prompt_token_ids, input_provenance = _worker_prompt_inputs(
        worker, batch_size, prompt_len
    )
    try:
        runtime_provenance = _worker_runtime_provenance(
            worker,
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
        )
    except RuntimeError as error:
        return _worker_internal_runner_blocker(
            "missing_runner_runtime_provenance",
            str(error),
        )
    prefix = f"rwkv7-prefill-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair()
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    iteration_durations_s: list[float] = []
    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []

    for warmup_iteration in range(warmup):
        req_ids = [
            f"{prefix}-warmup-{warmup_iteration}-{idx}" for idx in range(batch_size)
        ]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            cuda_events=cuda_events,
            measure=False,
        )
        _worker_finish_execute_without_sampling(worker)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))
    if warmup:
        _worker_cuda_synchronize()

    for iteration in range(iters):
        req_ids = [f"{prefix}-measure-{iteration}-{idx}" for idx in range(batch_size)]
        chunk_durations_s, chunk_token_counts = _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            cuda_events=cuda_events,
            measure=True,
        )
        iteration_durations_s.append(sum(chunk_durations_s))
        unit_durations_s.extend(chunk_durations_s)
        unit_tokens.extend(chunk_token_counts)
        _worker_finish_execute_without_sampling(worker)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": "worker.execute_model.prefill",
        "timing_clock": timing_clock,
        "iteration_durations_s": iteration_durations_s,
        "unit_durations_s": unit_durations_s,
        "unit_tokens": unit_tokens,
        "tokens": batch_size * prompt_len * iters,
        "warmup_iterations": warmup,
        "worker_count": 1,
        "input": input_provenance,
        "runtime": runtime_provenance,
    }


def _run_vllm_worker_internal_decode_only(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup_decode_tokens: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    if not callable(getattr(worker, "execute_model", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_execute_model",
            "The vLLM worker does not expose execute_model().",
        )
    if not callable(getattr(worker, "sample_tokens", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_sample_tokens",
            "The vLLM worker does not expose sample_tokens().",
        )
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if prompt_len <= 0:
        raise ValueError("runner decode prompt len must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup_decode_tokens < 0:
        raise ValueError("runner decode warmup tokens must be non-negative")

    from vllm import SamplingParams

    scheduled_decode_tokens = decode_tokens + warmup_decode_tokens
    sampling_params = SamplingParams(
        max_tokens=scheduled_decode_tokens,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
        seed=VLLM_RUNNER_SAMPLING["seed"],
    )
    prompt_token_ids, input_provenance = _worker_prompt_inputs(
        worker, batch_size, prompt_len
    )
    prefix = f"rwkv7-decode-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair()
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    iteration_durations_s: list[float] = []
    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []
    sample_durations_s: list[float] = []

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        worker.execute_model(
            _worker_add_decode_requests_scheduler_output(
                req_ids=req_ids,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                prompt_len=prompt_len,
            )
        )
        _worker_finish_execute_without_sampling(worker)
        _worker_cuda_synchronize()

        for step in range(warmup_decode_tokens):
            worker.execute_model(
                _worker_cached_decode_scheduler_output(
                    req_ids=req_ids,
                    prompt_len=prompt_len,
                    step=step,
                )
            )
            _worker_finish_execute_without_sampling(worker)
        if warmup_decode_tokens:
            _worker_cuda_synchronize()

        iteration_duration_s = 0.0
        for step in range(warmup_decode_tokens, scheduled_decode_tokens):
            decode_output = _worker_cached_decode_scheduler_output(
                req_ids=req_ids,
                prompt_len=prompt_len,
                step=step,
            )
            execute_duration_s = _worker_time_execute_model(
                worker,
                decode_output,
                cuda_events,
            )
            iteration_duration_s += execute_duration_s
            unit_durations_s.append(execute_duration_s)
            unit_tokens.append(batch_size)
            if include_sampling:
                sample_duration_s = _worker_time_sample_tokens(
                    worker,
                    None,
                    cuda_events,
                )
                iteration_duration_s += sample_duration_s
                sample_durations_s.append(sample_duration_s)
            else:
                _worker_finish_execute_without_sampling(worker)
        iteration_durations_s.append(iteration_duration_s)
        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": (
            "worker.execute_model+sample_tokens.decode"
            if include_sampling
            else "worker.execute_model.decode"
        ),
        "timing_clock": timing_clock,
        "iteration_durations_s": iteration_durations_s,
        "unit_durations_s": unit_durations_s,
        "unit_tokens": unit_tokens,
        "sample_durations_s": sample_durations_s,
        "tokens": batch_size * decode_tokens * iters,
        "worker_count": 1,
        "input": input_provenance,
    }


def _run_vllm_worker_internal_steady_decode(
    worker: Any,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    iters: int,
    measure: bool,
    warmup_decode_tokens: int = 0,
) -> dict[str, Any]:
    if not callable(getattr(worker, "execute_model", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_execute_model",
            "The vLLM worker does not expose execute_model().",
        )
    if not callable(getattr(worker, "sample_tokens", None)):
        return _worker_internal_runner_blocker(
            "missing_worker_sample_tokens",
            "The vLLM worker does not expose sample_tokens().",
        )

    from vllm import SamplingParams

    if warmup_decode_tokens < 0:
        raise ValueError("runner decode warmup tokens must be non-negative")
    scheduled_decode_tokens = decode_tokens + warmup_decode_tokens

    sampling_params = SamplingParams(
        max_tokens=scheduled_decode_tokens,
        temperature=VLLM_RUNNER_SAMPLING["temperature"],
        top_p=VLLM_RUNNER_SAMPLING["top_p"],
        ignore_eos=VLLM_RUNNER_SAMPLING["ignore_eos"],
        detokenize=VLLM_RUNNER_SAMPLING["detokenize"],
        seed=VLLM_RUNNER_SAMPLING["seed"],
    )
    prompt_token_ids, input_provenance = _worker_prompt_inputs(
        worker, batch_size, prompt_len
    )
    iteration_durations_s: list[float] = []
    execute_durations_s: list[float] = []
    sample_durations_s: list[float] = []
    decode_step_durations_s: list[float] = []
    postprocess_durations_s: list[float] = []
    prefix = f"rwkv7-runner-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair() if measure else None
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    prefill_chunk_tokens = min(prefill_chunk_tokens, prompt_len)
    try:
        runtime_provenance = _worker_runtime_provenance(
            worker,
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
        )
    except RuntimeError as error:
        return _worker_internal_runner_blocker(
            "missing_runner_runtime_provenance",
            str(error),
        )
    prefill_state_contracts: list[dict[str, Any]] = []

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            measure=False,
        )
        worker.sample_tokens(None)
        _worker_cuda_synchronize()
        try:
            prefill_state_contracts.append(
                _worker_prefill_state_contract(worker, req_ids, prompt_len)
            )
        except RuntimeError as error:
            return _worker_internal_runner_blocker(
                "invalid_runner_prefill_state",
                str(error),
            )

        for step in range(warmup_decode_tokens):
            worker.execute_model(
                _worker_cached_decode_scheduler_output(
                    req_ids=req_ids,
                    prompt_len=prompt_len,
                    step=step,
                )
            )
            worker.sample_tokens(None)
        if warmup_decode_tokens:
            _worker_cuda_synchronize()

        if measure:
            iteration_duration_s = 0.0
        for step in range(warmup_decode_tokens, scheduled_decode_tokens):
            decode_output = _worker_cached_decode_scheduler_output(
                req_ids=req_ids,
                prompt_len=prompt_len,
                step=step,
            )
            if measure:
                execute_duration_s = _worker_time_execute_model(
                    worker,
                    decode_output,
                    cuda_events,
                )
                sample_duration_s = _worker_time_sample_tokens(
                    worker,
                    None,
                    cuda_events,
                )
                step_duration_s = execute_duration_s + sample_duration_s
                execute_durations_s.append(execute_duration_s)
                sample_durations_s.append(sample_duration_s)
                decode_step_durations_s.append(step_duration_s)
                iteration_duration_s += step_duration_s
            else:
                worker.execute_model(decode_output)
                worker.sample_tokens(None)

        if measure:
            iteration_durations_s.append(iteration_duration_s)

        worker.execute_model(_worker_empty_scheduler_output(set(req_ids)))

    return {
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "iteration_durations_s": iteration_durations_s,
        "execute_durations_s": execute_durations_s,
        "sample_durations_s": sample_durations_s,
        "decode_step_durations_s": decode_step_durations_s,
        "postprocess_durations_s": postprocess_durations_s,
        "postprocess_timing_available": False,
        "decode_steps": decode_tokens * iters if measure else 0,
        "warmup_decode_steps": warmup_decode_tokens * iters,
        "tokens": batch_size * decode_tokens * iters if measure else 0,
        "timing_clock": timing_clock,
        "input": input_provenance,
        "runtime": runtime_provenance,
        "prefill_state": {
            "validated": True,
            "iterations": len(prefill_state_contracts),
            "request_count": batch_size,
            "minimum_prefill_elapsed": min(
                contract["minimum_prefill_elapsed"]
                for contract in prefill_state_contracts
            ),
            "maximum_prefill_elapsed": max(
                contract["maximum_prefill_elapsed"]
                for contract in prefill_state_contracts
            ),
            "unique_slot_ownership": all(
                contract["unique_state_indices"] == batch_size
                for contract in prefill_state_contracts
            ),
            "decode_state_origin": "same-request real chunked prefill",
        },
    }


def _merge_worker_internal_runner_results(
    worker_results: list[Any],
    *,
    batch_size: int,
    decode_tokens: int,
    iters: int,
) -> dict[str, Any]:
    if not worker_results:
        return _worker_internal_runner_blocker(
            "missing_internal_runner_worker_results",
            "No vLLM worker returned internal runner timing results.",
        )
    blockers = [
        blocker
        for result in worker_results
        if isinstance(result, dict)
        for blocker in result.get("blockers", [])
    ]
    if blockers:
        return {
            "measurement_mode": VLLM_RUNNER_MODE,
            "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
            "blockers": blockers,
        }

    normalized_results = [
        result for result in worker_results if isinstance(result, dict)
    ]
    if len(normalized_results) != len(worker_results):
        return _worker_internal_runner_blocker(
            "invalid_internal_runner_worker_result",
            "A vLLM worker returned a non-dict internal runner timing result.",
        )

    shared_metadata: dict[str, Any] = {}
    for key in ("input", "runtime"):
        values = [result.get(key) for result in normalized_results]
        if any(not isinstance(value, dict) for value in values) or any(
            value != values[0] for value in values[1:]
        ):
            return _worker_internal_runner_blocker(
                "inconsistent_internal_runner_provenance",
                f"Workers did not return one consistent {key} contract.",
            )
        shared_metadata[key] = values[0]
    state_contracts = [result.get("prefill_state") for result in normalized_results]
    if any(not isinstance(contract, dict) for contract in state_contracts):
        return _worker_internal_runner_blocker(
            "missing_runner_prefill_state_contract",
            "Workers did not prove decode state came from completed prefill.",
        )
    shared_metadata["prefill_state"] = {
        "validated": all(contract.get("validated") for contract in state_contracts),
        "iterations": min(
            int(contract.get("iterations", 0)) for contract in state_contracts
        ),
        "request_count": batch_size,
        "minimum_prefill_elapsed": min(
            int(contract["minimum_prefill_elapsed"]) for contract in state_contracts
        ),
        "maximum_prefill_elapsed": max(
            int(contract["maximum_prefill_elapsed"]) for contract in state_contracts
        ),
        "unique_slot_ownership": all(
            contract.get("unique_slot_ownership") for contract in state_contracts
        ),
        "decode_state_origin": "same-request real chunked prefill",
    }

    expected_decode_steps = decode_tokens * iters
    duration_specs = (
        ("iteration_durations_s", iters),
        ("execute_durations_s", expected_decode_steps),
        ("sample_durations_s", expected_decode_steps),
        ("decode_step_durations_s", expected_decode_steps),
    )
    merged_durations: dict[str, list[float]] = {}
    for key, expected_count in duration_specs:
        per_worker_values = [
            [float(value) for value in result.get(key, [])]
            for result in normalized_results
        ]
        if any(len(values) != expected_count for values in per_worker_values):
            return _worker_internal_runner_blocker(
                "missing_internal_runner_decode_samples",
                "No complete internal worker decode timing samples were recorded.",
            )
        merged_durations[key] = [
            max(worker_values[index] for worker_values in per_worker_values)
            for index in range(expected_count)
        ]

    decode_steps = sum(
        int(result.get("decode_steps", 0)) for result in normalized_results
    )
    if decode_steps < expected_decode_steps:
        return _worker_internal_runner_blocker(
            "missing_internal_runner_decode_samples",
            "Internal worker decode timing did not record every requested decode step.",
        )
    if iters == 0:
        return {
            "measurement_mode": VLLM_RUNNER_MODE,
            "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
            "decode_steps": decode_steps,
            "worker_count": len(normalized_results),
        }

    iteration_durations_s = merged_durations["iteration_durations_s"]
    execute_durations_s = merged_durations["execute_durations_s"]
    sample_durations_s = merged_durations["sample_durations_s"]
    decode_step_durations_s = merged_durations["decode_step_durations_s"]
    p50_s = _percentile(iteration_durations_s, 0.5)
    execute_p50_s = _percentile(execute_durations_s, 0.5)
    sample_p50_s = _percentile(sample_durations_s, 0.5)
    decode_step_p50_s = _percentile(decode_step_durations_s, 0.5)
    return {
        "tokens_per_s": (batch_size * decode_tokens) / p50_s,
        "p10_ms": _percentile(iteration_durations_s, 0.1) * 1000.0,
        "p50_ms": p50_s * 1000.0,
        "p90_ms": _percentile(iteration_durations_s, 0.9) * 1000.0,
        "execute_model_p50_ms": execute_p50_s * 1000.0,
        "execute_model_p50_tokens_per_s": batch_size / execute_p50_s,
        "sample_tokens_p50_ms": sample_p50_s * 1000.0,
        "sample_tokens_p50_tokens_per_s": batch_size / sample_p50_s,
        "decode_step_p50_ms": decode_step_p50_s * 1000.0,
        "decode_step_p50_tokens_per_s": batch_size / decode_step_p50_s,
        "postprocess_p50_ms": None,
        "postprocess_timing_available": False,
        "measurement_mode": VLLM_RUNNER_MODE,
        "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
        "decode_steps": decode_steps,
        "worker_count": len(normalized_results),
        **shared_metadata,
    }


def _merge_worker_internal_phase_results(
    worker_results: list[Any],
    *,
    total_tokens: int,
    expected_iterations: int,
) -> dict[str, Any]:
    if not worker_results:
        return _worker_internal_runner_blocker(
            "missing_internal_runner_worker_results",
            "No vLLM worker returned internal runner timing results.",
        )
    blockers = [
        blocker
        for result in worker_results
        if isinstance(result, dict)
        for blocker in result.get("blockers", [])
    ]
    if blockers:
        return {
            "measurement_mode": VLLM_RUNNER_MODE,
            "internal_timing_target": VLLM_RUNNER_TIMING_TARGET,
            "blockers": blockers,
        }
    normalized_results = [
        result for result in worker_results if isinstance(result, dict)
    ]
    if len(normalized_results) != len(worker_results):
        return _worker_internal_runner_blocker(
            "invalid_internal_runner_worker_result",
            "A vLLM worker returned a non-dict internal runner timing result.",
        )

    shared_metadata: dict[str, Any] = {}
    for key in ("input", "runtime"):
        values = [result.get(key) for result in normalized_results]
        if any(not isinstance(value, dict) for value in values) or any(
            value != values[0] for value in values[1:]
        ):
            return _worker_internal_runner_blocker(
                "inconsistent_internal_runner_provenance",
                f"Workers did not return one consistent {key} contract.",
            )
        shared_metadata[key] = values[0]

    per_worker_iterations = [
        [float(value) for value in result.get("iteration_durations_s", [])]
        for result in normalized_results
    ]
    if any(len(values) != expected_iterations for values in per_worker_iterations):
        return _worker_internal_runner_blocker(
            "missing_internal_runner_phase_samples",
            "No complete internal worker phase timing samples were recorded.",
        )
    iteration_durations_s = [
        max(worker_values[index] for worker_values in per_worker_iterations)
        for index in range(expected_iterations)
    ]

    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []
    worker_count = len(normalized_results)
    max_unit_count = max(
        len(result.get("unit_durations_s", [])) for result in normalized_results
    )
    for index in range(max_unit_count):
        worker_unit_durations: list[float] = []
        worker_unit_tokens: list[int] = []
        for result in normalized_results:
            durations = result.get("unit_durations_s", [])
            tokens = result.get("unit_tokens", [])
            if index < len(durations):
                worker_unit_durations.append(float(durations[index]))
                worker_unit_tokens.append(int(tokens[index]))
        if worker_unit_durations:
            unit_durations_s.append(max(worker_unit_durations))
            unit_tokens.append(sum(worker_unit_tokens) // max(1, worker_count))

    summary = _phase_throughput_summary(
        total_tokens=total_tokens,
        iteration_durations_s=iteration_durations_s,
        unit_durations_s=unit_durations_s,
        unit_tokens=unit_tokens,
    )
    first_result = normalized_results[0]
    summary.update(
        {
            "measurement_mode": first_result.get("measurement_mode", VLLM_RUNNER_MODE),
            "internal_timing_target": first_result.get(
                "internal_timing_target", VLLM_RUNNER_TIMING_TARGET
            ),
            "timing_clock": first_result.get("timing_clock", VLLM_RUNNER_TIMING_CLOCK),
            "worker_count": worker_count,
            **shared_metadata,
        }
    )
    if "warmup_iterations" in first_result:
        summary["warmup_iterations"] = first_result["warmup_iterations"]
    sample_durations = [
        float(value)
        for result in normalized_results
        for value in result.get("sample_durations_s", [])
    ]
    if sample_durations:
        sample_summary = _duration_ms_summary(sample_durations)
        summary["sample_tokens_p50_ms"] = sample_summary["p50_ms"]
    return summary


def _time_vllm_runner_steady_decode(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("runner batch size must be positive")
    if prompt_len <= 0:
        raise ValueError("runner prompt len must be positive")
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    if iters <= 0:
        raise ValueError("runner iters must be positive")

    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )

    timed_results = collective_rpc(
        _run_vllm_worker_internal_steady_decode,
        args=(
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
            decode_tokens,
            iters,
            True,
            warmup,
        ),
    )
    return _merge_worker_internal_runner_results(
        list(timed_results),
        batch_size=batch_size,
        decode_tokens=decode_tokens,
        iters=iters,
    )


def _time_vllm_runner_prefill_phase(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )
    results = collective_rpc(
        _run_vllm_worker_internal_prefill,
        args=(batch_size, prompt_len, prefill_chunk_tokens, warmup, iters),
    )
    return _merge_worker_internal_phase_results(
        list(results),
        total_tokens=batch_size * prompt_len * iters,
        expected_iterations=iters,
    )


def _time_vllm_runner_decode_phase(
    llm: Any,
    *,
    batch_size: int,
    prompt_len: int,
    prefill_chunk_tokens: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        return _worker_internal_runner_blocker(
            "missing_collective_rpc",
            "The vLLM LLM object does not expose collective_rpc().",
        )
    results = collective_rpc(
        _run_vllm_worker_internal_decode_only,
        args=(
            batch_size,
            prompt_len,
            prefill_chunk_tokens,
            decode_tokens,
            warmup,
            iters,
            include_sampling,
        ),
    )
    return _merge_worker_internal_phase_results(
        list(results),
        total_tokens=batch_size * decode_tokens * iters,
        expected_iterations=iters,
    )


def _shutdown_vllm_runner_llm(llm: Any) -> None:
    engine = getattr(llm, "llm_engine", None)
    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        engine_core = getattr(engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(timeout=30)
        except TypeError:
            shutdown()
    with suppress(Exception):
        llm.llm_engine = None
    gc.collect()
    if _cuda_available():
        cuda = _cuda_module()
        cuda.empty_cache()


def generate_vllm_runner_measurement(
    config: BenchmarkConfig,
    *,
    batch_size: int,
    prompt_len: int,
    decode_tokens: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("runner batch size must be positive")
    if prompt_len <= 0:
        raise ValueError("runner prompt len must be positive")
    if decode_tokens <= 0:
        raise ValueError("runner decode tokens must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    if iters <= 0:
        raise ValueError("runner iters must be positive")

    runner_config = replace(
        config,
        batch_size=batch_size,
        prompt_len=prompt_len,
        warmup_tokens=warmup,
        decode_tokens=decode_tokens,
    )
    _required_vllm_runner_model(runner_config)
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(runner_config)
    capacity_config = replace(
        runner_config,
        decode_tokens=decode_tokens + warmup,
    )
    llm = _create_vllm_runner_llm(capacity_config)
    try:
        parsed_prefill = _time_vllm_runner_prefill_phase(
            llm,
            batch_size=batch_size,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            warmup=warmup,
            iters=iters,
        )
        parsed_decode = _time_vllm_runner_steady_decode(
            llm,
            batch_size=batch_size,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_tokens=decode_tokens,
            warmup=warmup,
            iters=iters,
        )
    finally:
        _shutdown_vllm_runner_llm(llm)

    prefill_metrics: dict[str, Any] = {
        "runner_batch_size": batch_size,
        "runner_prompt_len": prompt_len,
        "runner_prefill_chunk_tokens": prefill_chunk_tokens,
        "runner_warmup": warmup,
        "runner_warmup_mode": "whole_prefill_iterations",
        "runner_iters": iters,
        "runner_measurement_mode": parsed_prefill.get(
            "measurement_mode", VLLM_RUNNER_MODE
        ),
        "runner_internal_timing_target": parsed_prefill.get(
            "internal_timing_target", "worker.execute_model.prefill"
        ),
        "runner_timing_clock": parsed_prefill.get(
            "timing_clock", VLLM_RUNNER_TIMING_CLOCK
        ),
        "measurement_boundary": {
            "included": [
                "prompt token embedding",
                "chunked RWKV recurrent model execution",
                "recurrent state update",
            ],
            "excluded": [
                "request construction",
                "request cleanup",
                "sampling",
                "JSON serialization",
            ],
            "warmup": "whole prefill iterations on the same prompt workload",
            "synchronization": "before each CUDA event and at event completion",
        },
    }
    if "blockers" in parsed_prefill:
        prefill_metrics["blockers"] = parsed_prefill["blockers"]
    else:
        prefill_metrics.update(
            {
                "runner_avg_tokens_per_s": parsed_prefill["avg_tokens_per_s"],
                "runner_peak_iteration_tokens_per_s": parsed_prefill[
                    "peak_iteration_tokens_per_s"
                ],
                "runner_peak_chunk_tokens_per_s": parsed_prefill[
                    "peak_unit_tokens_per_s"
                ],
                "runner_p10_ms": parsed_prefill["p10_ms"],
                "runner_p50_ms": parsed_prefill["p50_ms"],
                "runner_p90_ms": parsed_prefill["p90_ms"],
                "runner_total_tokens": parsed_prefill["total_tokens"],
                "runner_worker_count": parsed_prefill["worker_count"],
                "input": parsed_prefill["input"],
                "runtime": parsed_prefill["runtime"],
            }
        )

    runner_metrics: dict[str, Any] = {
        "runner_batch_size": batch_size,
        "runner_prompt_len": prompt_len,
        "runner_prefill_chunk_tokens": prefill_chunk_tokens,
        "runner_decode_tokens": decode_tokens,
        "runner_warmup": warmup,
        "runner_warmup_mode": "same_request_decode_steps",
        "runner_warmup_decode_tokens": warmup,
        "runner_iters": iters,
        "runner_measurement_mode": parsed_decode.get(
            "measurement_mode", VLLM_RUNNER_MODE
        ),
        "runner_internal_timing_target": parsed_decode.get(
            "internal_timing_target",
            VLLM_RUNNER_TIMING_TARGET,
        ),
        "runner_timing_clock": parsed_decode.get(
            "timing_clock",
            VLLM_RUNNER_TIMING_CLOCK,
        ),
        "runner_collective_rpc_serialization": (
            "pickle_enabled"
            if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") == "1"
            else "msgpack_only"
        ),
        "measurement_boundary": {
            "included": [
                "decode token embedding",
                "RWKV recurrent model execution from the completed prompt state",
                "FP32 logits projection",
                "rapid sampler token selection",
            ],
            "excluded": [
                "prompt prefill",
                "request construction",
                "request cleanup",
                "detokenization",
                "JSON serialization",
            ],
            "state_origin": "same-request real chunked prefill",
            "warmup": "same-request decode steps after completed prefill",
            "synchronization": "before each CUDA event and at event completion",
        },
    }
    if "blockers" in parsed_decode:
        runner_metrics["blockers"] = parsed_decode["blockers"]
    else:
        runner_metrics.update(
            {
                "runner_tokens_per_s": parsed_decode["tokens_per_s"],
                "runner_p10_ms": parsed_decode["p10_ms"],
                "runner_p50_ms": parsed_decode["p50_ms"],
                "runner_p90_ms": parsed_decode["p90_ms"],
                "runner_execute_model_p50_ms": parsed_decode.get(
                    "execute_model_p50_ms"
                ),
                "runner_execute_model_p50_tokens_per_s": parsed_decode.get(
                    "execute_model_p50_tokens_per_s"
                ),
                "runner_sample_tokens_p50_ms": parsed_decode.get(
                    "sample_tokens_p50_ms"
                ),
                "runner_sample_tokens_p50_tokens_per_s": parsed_decode.get(
                    "sample_tokens_p50_tokens_per_s"
                ),
                "runner_decode_step_p50_ms": parsed_decode.get("decode_step_p50_ms"),
                "runner_decode_step_p50_tokens_per_s": parsed_decode.get(
                    "decode_step_p50_tokens_per_s"
                ),
                "runner_postprocess_p50_ms": parsed_decode.get("postprocess_p50_ms"),
                "runner_postprocess_timing_available": parsed_decode.get(
                    "postprocess_timing_available",
                    False,
                ),
                "runner_decode_steps": parsed_decode["decode_steps"],
                "runner_worker_count": parsed_decode["worker_count"],
                "input": parsed_decode["input"],
                "runtime": parsed_decode["runtime"],
                "prefill_state": parsed_decode["prefill_state"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "source": _source_metadata(runner_config),
        "runner_prefill": prefill_metrics,
        "runner_steady_decode": runner_metrics,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": f"vllm_runner_{VLLM_RUNNER_MODE}",
            "provenance": _benchmark_provenance(runner_config),
        },
    }


def _runner_model_contract_failure_measurement(
    config: BenchmarkConfig,
    error: RunnerModelContractError,
) -> dict[str, Any]:
    blocker = _blocker(error.code, str(error), artifact=error.artifact)
    provenance_config = replace(config, model=None)
    provenance = _benchmark_provenance(provenance_config)
    provenance["artifact"] = error.artifact
    phase = {
        "runner_batch_size": config.batch_size,
        "runner_prompt_len": config.prompt_len,
        "runner_prefill_chunk_tokens": _runner_prefill_chunk_tokens(config),
        "runner_decode_tokens": config.decode_tokens,
        "runner_measurement_mode": VLLM_RUNNER_MODE,
        "blockers": [blocker],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "source": _source_metadata(config),
        "runner_prefill": dict(phase),
        "runner_steady_decode": dict(phase),
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": f"vllm_runner_{VLLM_RUNNER_MODE}",
            "provenance": provenance,
        },
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value.lower())
    )


def _runner_throughput_contract_blocker(
    measurements: dict[str, Any],
) -> dict[str, Any] | None:
    config = measurements.get("config")
    provenance = config.get("provenance") if isinstance(config, dict) else None
    raw_env = provenance.get("raw_env") if isinstance(provenance, dict) else None
    fixed_model_contract = (
        provenance.get("fixed_model_contract") if isinstance(provenance, dict) else None
    )
    if (
        not isinstance(raw_env, dict)
        or any(name not in raw_env for name in RUNNER_RUNTIME_ENV_REQUIREMENTS)
        or not isinstance(fixed_model_contract, dict)
        or any(name not in fixed_model_contract for name in RUNNER_FIXED_MODEL_CONTRACT)
    ):
        return _blocker(
            "missing_runner_throughput_provenance",
            "Runner performance acceptance requires runtime environment and "
            "fixed model contract provenance.",
        )

    violations = {
        f"runtime_env.{name}": {
            "required": required,
            "actual": raw_env[name],
        }
        for name, required in RUNNER_RUNTIME_ENV_REQUIREMENTS.items()
        if raw_env[name] != required
    }
    violations.update(
        {
            f"fixed_model_contract.{name}": {
                "required": required,
                "actual": fixed_model_contract[name],
            }
            for name, required in RUNNER_FIXED_MODEL_CONTRACT.items()
            if fixed_model_contract[name] != required
        }
    )
    if violations:
        return _blocker(
            "invalid_runner_throughput_contract",
            "Runner performance acceptance requires the FP16 throughput contract.",
            violations=violations,
        )

    artifact = provenance.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("kind") != "hf_artifact"
        or not _is_sha256(artifact.get("config_sha256"))
        or not _is_sha256(artifact.get("weights_sha256"))
        or not _is_sha256(artifact.get("source_checkpoint_sha256"))
    ):
        return _blocker(
            "missing_runner_hf_artifact_identity",
            "Runner performance acceptance requires config, safetensors, and "
            "source checkpoint SHA-256 provenance from a converted HF artifact.",
        )

    prefill = measurements.get("runner_prefill")
    decode = measurements.get("runner_steady_decode")
    required_prefill = ("input", "runtime", "measurement_boundary")
    required_decode = ("input", "runtime", "prefill_state", "measurement_boundary")
    if (
        not isinstance(prefill, dict)
        or not isinstance(decode, dict)
        or any(not isinstance(prefill.get(name), dict) for name in required_prefill)
        or any(not isinstance(decode.get(name), dict) for name in required_decode)
    ):
        return _blocker(
            "missing_runner_canonical_workload_provenance",
            "Runner acceptance requires measured prefill, input digest, actual "
            "runtime provenance, and completed-prefill decode state provenance.",
        )
    if prefill["input"] != decode["input"] or prefill["runtime"] != decode["runtime"]:
        return _blocker(
            "inconsistent_runner_canonical_workload",
            "Prefill and decode did not use one input and runtime contract.",
        )

    input_shape = decode["input"].get("shape")
    state = decode["prefill_state"]
    runtime = decode["runtime"]
    wkv = runtime.get("wkv") if isinstance(runtime, dict) else None
    gemm = runtime.get("gemm") if isinstance(runtime, dict) else None
    expected_shape = [
        int(decode.get("runner_batch_size", 0)),
        int(decode.get("runner_prompt_len", 0)),
    ]
    canonical_violations: dict[str, Any] = {}
    if input_shape != expected_shape or not _is_sha256(
        decode["input"].get("prompt_sha256")
    ):
        canonical_violations["input"] = {
            "required_shape": expected_shape,
            "actual_shape": input_shape,
            "has_sha256": _is_sha256(decode["input"].get("prompt_sha256")),
        }
    prompt_len = expected_shape[1]
    if (
        state.get("validated") is not True
        or state.get("unique_slot_ownership") is not True
        or state.get("decode_state_origin") != "same-request real chunked prefill"
        or int(state.get("minimum_prefill_elapsed", -1)) < prompt_len
    ):
        canonical_violations["prefill_state"] = state
    if (
        not isinstance(wkv, dict)
        or wkv.get("mode") != "fp16"
        or not isinstance(gemm, dict)
        or gemm.get("accumulation_policy") != "fp16"
        or gemm.get("allow_fp16_accumulation") is not True
    ):
        canonical_violations["runtime"] = runtime
    if canonical_violations:
        return _blocker(
            "invalid_runner_canonical_workload",
            "Runner measurement did not execute the canonical input, recurrent "
            "state, WKV, and GEMM contract.",
            violations=canonical_violations,
        )
    return None


def _evaluate_runner(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "runner_tokens_per_s": None,
        "runner_measurement_mode": None,
        "runner_internal_timing_target": None,
        "runner_timing_clock": None,
        "runner_execute_model_p50_ms": None,
        "runner_execute_model_p50_tokens_per_s": None,
        "runner_sample_tokens_p50_ms": None,
        "runner_sample_tokens_p50_tokens_per_s": None,
        "runner_decode_step_p50_ms": None,
        "runner_decode_step_p50_tokens_per_s": None,
        "runner_postprocess_p50_ms": None,
        "runner_postprocess_timing_available": None,
    }
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["runner_steady_decode"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("runner_steady_decode", {})
    runner_tps = _get_number(raw_metrics, "runner_tokens_per_s")
    if runner_tps is None:
        runner_blockers = raw_metrics.get("blockers")
        if runner_blockers:
            check["blockers"] = runner_blockers
            return check
        check["blockers"] = [
            _blocker(
                "missing_runner_measurement",
                "Measurement JSON must include runner_tokens_per_s for "
                "runner_steady_decode.",
            )
        ]
        return check
    if runner_tps <= 0:
        check["status"] = "failed"
        check["errors"] = ["runner_steady_decode token throughput must be positive"]
        return check

    metrics["runner_tokens_per_s"] = runner_tps
    contract_blocker = _runner_throughput_contract_blocker(measurements)
    if contract_blocker is not None:
        check["blockers"] = [contract_blocker]
        return check

    metrics.update(
        {
            "runner_tokens_per_s": runner_tps,
            "runner_measurement_mode": raw_metrics.get("runner_measurement_mode"),
            "runner_internal_timing_target": raw_metrics.get(
                "runner_internal_timing_target"
            ),
            "runner_timing_clock": raw_metrics.get("runner_timing_clock"),
            "runner_execute_model_p50_ms": raw_metrics.get(
                "runner_execute_model_p50_ms"
            ),
            "runner_execute_model_p50_tokens_per_s": raw_metrics.get(
                "runner_execute_model_p50_tokens_per_s"
            ),
            "runner_sample_tokens_p50_ms": raw_metrics.get(
                "runner_sample_tokens_p50_ms"
            ),
            "runner_sample_tokens_p50_tokens_per_s": raw_metrics.get(
                "runner_sample_tokens_p50_tokens_per_s"
            ),
            "runner_decode_step_p50_ms": raw_metrics.get("runner_decode_step_p50_ms"),
            "runner_decode_step_p50_tokens_per_s": raw_metrics.get(
                "runner_decode_step_p50_tokens_per_s"
            ),
            "runner_postprocess_p50_ms": raw_metrics.get("runner_postprocess_p50_ms"),
            "runner_postprocess_timing_available": raw_metrics.get(
                "runner_postprocess_timing_available"
            ),
        }
    )
    passed = (
        runner_tps
        >= ACCEPTANCE_THRESHOLDS["runner_steady_decode"]["min_runner_tokens_per_s"]
    )
    check["status"] = "passed" if passed else "failed"
    if not passed:
        check["errors"] = [
            "vLLM runner steady decode did not produce positive steady throughput"
        ]
    check["blockers"] = []
    return check


def build_report(
    config: BenchmarkConfig,
    *,
    measurements: dict[str, Any] | None = None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    cuda = _cuda_available() if cuda_available is None else cuda_available
    runtime_blockers = _runtime_blockers(
        config,
        cuda_available=cuda,
    )
    runner_check = _evaluate_runner(measurements, runtime_blockers)
    status = runner_check["status"]
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "overall_status": status,
        "source": _source_metadata(config),
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "cuda_available": cuda,
            "measurement_source": "json" if measurements is not None else None,
            "provenance": _benchmark_provenance(config),
        },
        "acceptance": ACCEPTANCE_THRESHOLDS,
        "checks": {"runner_steady_decode": runner_check},
    }


def _default_repo_root() -> Path:
    return REPO_ROOT


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    model = args.model or os.environ.get("VLLM_RWKV7_MODEL") or None
    return BenchmarkConfig(
        repo_root=args.repo_root.resolve(),
        model=model,
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        warmup_tokens=args.warmup_tokens,
        decode_tokens=args.decode_tokens,
        runner_prefill_chunk_tokens=args.runner_prefill_chunk_tokens,
        runner_enforce_eager=args.runner_enforce_eager,
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RWKV7 faster3a performance acceptance harness."
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument(
        "--model",
        help="Standard RWKV7 HF artifact directory or supported remote reference.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--measurement-json", type=Path)
    parser.add_argument(
        "--measure-vllm-runner",
        action="store_true",
        help="Run the canonical vLLM RWKV7 runner throughput benchmark.",
    )
    parser.add_argument(
        "--runner-batch-size",
        type=int,
        default=16,
        help="Batch size for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-prompt-len",
        type=int,
        default=128,
        help="Prompt token count for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-prefill-chunk-tokens",
        type=int,
        default=DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS,
        help=(
            "Maximum prompt tokens scheduled per request for synthetic vLLM "
            "runner prefill."
        ),
    )
    parser.add_argument(
        "--runner-enforce-eager",
        action="store_true",
        help="Disable vLLM CUDA graph capture for runner measurements.",
    )
    parser.add_argument(
        "--runner-decode-tokens",
        type=int,
        default=128,
        help="Generated token count for vLLM runner steady decode measurement.",
    )
    parser.add_argument(
        "--runner-warmup",
        type=int,
        default=1,
        help=(
            "Unmeasured same-request decode steps before timing each vLLM "
            "runner steady decode iteration."
        ),
    )
    parser.add_argument(
        "--runner-iters",
        type=int,
        default=3,
        help="Timed iterations for vLLM runner steady decode.",
    )
    parser.add_argument(
        "--measurement-output",
        type=Path,
        help="Write generated measurement JSON for a measurement mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write structured JSON to this file instead of stdout.",
    )
    args = parser.parse_args(argv)
    return args


def _load_measurements(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as measurement_file:
        data = json.load(measurement_file)
    if not isinstance(data, dict):
        raise ValueError("measurement JSON must contain an object")
    return data


def _write_report(report: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    output.write_text(text, encoding="utf-8")


def _measurement_exit_code(measurement: dict[str, Any]) -> int:
    metrics = measurement.get("runner_steady_decode")
    if not isinstance(metrics, dict):
        return 2
    if metrics.get("blockers"):
        return 2
    if _runner_throughput_contract_blocker(measurement) is not None:
        return 2
    try:
        tokens_per_s = float(metrics["runner_tokens_per_s"])
    except (KeyError, TypeError, ValueError):
        return 2
    return int(
        not math.isfinite(tokens_per_s)
        or tokens_per_s
        < ACCEPTANCE_THRESHOLDS["runner_steady_decode"]["min_runner_tokens_per_s"]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _config_from_args(args)
    if args.measure_vllm_runner:
        runner_config = replace(
            config,
            batch_size=args.runner_batch_size,
            prompt_len=args.runner_prompt_len,
            warmup_tokens=args.runner_warmup,
            decode_tokens=args.runner_decode_tokens,
        )
        try:
            measurement = generate_vllm_runner_measurement(
                config,
                batch_size=args.runner_batch_size,
                prompt_len=args.runner_prompt_len,
                decode_tokens=args.runner_decode_tokens,
                warmup=args.runner_warmup,
                iters=args.runner_iters,
            )
        except RunnerModelContractError as error:
            measurement = _runner_model_contract_failure_measurement(
                runner_config,
                error,
            )
        _write_report(measurement, args.measurement_output)
        return _measurement_exit_code(measurement)
    report = build_report(
        config,
        measurements=_load_measurements(args.measurement_json),
    )
    _write_report(report, args.output)
    if report["overall_status"] == "passed":
        return 0
    if report["overall_status"] == "blocked":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
