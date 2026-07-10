# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 faster3a performance acceptance harness.

This harness does not claim performance by default. It records Albatross
provenance, reports runtime blockers as structured JSON, and can evaluate an
external measurement JSON against the faster3a acceptance thresholds.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
BENCHMARK_NAME = "rwkv7_faster3a"
ALBATROSS_BENCH_SCRIPT = "rwkv7_fast_v3a.py"
ALBATROSS_REPO = "https://github.com/BlinkDL/Albatross"
ALBATROSS_COMMIT = "5e941fb1eeb7f735a562fb5bbb30fad19adc825b"
ALBATROSS_IMPL = "faster3a_2605"
VLLM_MODEL_ONLY_LABEL = "RWKV7ForCausalLM.forward_logits"
VLLM_RUNNER_MODE = "worker_execute_model"
VLLM_RUNNER_TIMING_TARGET = "worker.execute_model"
VLLM_RUNNER_TIMING_CLOCK = "cuda_event"
DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS = 16
DEFAULT_RUNNER_PD_PREFILL_CASES = "1x1024,32x32"
DEFAULT_RUNNER_PD_DECODE_CASES = "1024x1"
VLLM_RUNNER_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "ignore_eos": True,
    "detokenize": False,
}
STATE_MOVEMENT_COUNTERS = (
    "resident_to_decode_copies",
    "decode_compactions",
    "decode_compaction_rows",
    "decode_compaction_bytes",
    "decode_compaction_time_ns",
    "decode_full_row_copies",
    "decode_full_row_copy_bytes",
    "decode_hole_fills",
    "prefill_batches",
    "prefill_ranges",
    "prefill_groups",
    "prefill_group_model_calls",
    "prefill_varlen_batches",
    "prefill_varlen_tokens",
    "prefill_varlen_model_calls",
    "prefill_fallback_ranges",
    "prefill_fallback_model_calls",
)
PROVENANCE_ENV_VARS = (
    "VLLM_RWKV7_MODEL",
    "VLLM_RWKV7_WKV_MODE",
    "VLLM_RWKV7_EMB_DEVICE",
    "VLLM_RWKV7_RKV_MODE",
    "VLLM_RWKV7_CMIX_SPARSE",
    "VLLM_RWKV7_LOW_RANK_WEIGHT",
    "VLLM_RWKV7_ORIG_LINEAR_GROUPS",
    "VLLM_USE_RAPID_SAMPLER",
    "VLLM_USE_V2_MODEL_RUNNER",
    "VLLM_ALLOW_INSECURE_SERIALIZATION",
    "VLLM_RWKV7_SLOT_MAPPED_STATE",
    "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP",
)
PROVENANCE_ENV_DEFAULTS = {
    "VLLM_RWKV7_WKV_MODE": "fp16",
    "VLLM_RWKV7_EMB_DEVICE": "gpu",
    "VLLM_RWKV7_RKV_MODE": "off",
    "VLLM_RWKV7_CMIX_SPARSE": "no-fc",
    "VLLM_RWKV7_LOW_RANK_WEIGHT": "both",
    "VLLM_RWKV7_ORIG_LINEAR_GROUPS": "att_c2c,ffn_key,head",
    "VLLM_USE_RAPID_SAMPLER": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_RWKV7_SLOT_MAPPED_STATE": "1",
    "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP": "1",
}
BENCHMARK_ONLY_VLLM_ENV_VARS = ("VLLM_RWKV7_MODEL",)
ALBATROSS_DEFAULTS = {
    "wkv": "fp16",
    "emb": "cpu",
    "batched_rkv": "off",
    "cmix_sparse": "no-fc",
    "lowrank_weight": "both",
    "orig_linear_groups": "att_c2c,ffn_key,head",
}
ACCEPTANCE_THRESHOLDS = {
    "model_only_steady_decode": {
        "min_vllm_to_albatross_ratio": 0.95,
        "max_latency_slowdown_pct": 5.0,
    },
    "runner_steady_decode": {
        "min_runner_tokens_per_s": 1.0,
    },
    "state_movement": {
        "max_resident_to_decode_copies": 0,
        "max_decode_compactions": 0,
        "max_decode_full_row_copies": 0,
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
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp16_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp16_v2.cu",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cpp",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cpp",
        correspondence="cuda-source-port",
    ),
    SourceProvenanceEntry(
        source_path=f"{ALBATROSS_IMPL}/cuda/rwkv7_wkv_fp32_v2.cu",
        target_path="csrc/libtorch_stable/rwkv7/rwkv7_wkv_fp32_v2.cu",
        correspondence="cuda-source-port",
    ),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    repo_root: Path
    model: str | None
    albatross_root: Path | None
    albatross_impl: str
    albatross_checkpoint: Path | None
    batch_size: int
    prompt_len: int
    warmup_tokens: int
    decode_tokens: int
    runner_prefill_chunk_tokens: int = DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS
    runner_enforce_eager: bool = False
    runner_disable_rapid_sampler: bool = False
    runner_cudagraph_capture_sizes: tuple[int, ...] | None = None
    albatross_wkv: str = ALBATROSS_DEFAULTS["wkv"]
    albatross_emb: str = ALBATROSS_DEFAULTS["emb"]
    albatross_batched_rkv: str = ALBATROSS_DEFAULTS["batched_rkv"]
    albatross_cmix_sparse: str = ALBATROSS_DEFAULTS["cmix_sparse"]
    albatross_lowrank_weight: str = ALBATROSS_DEFAULTS["lowrank_weight"]
    albatross_orig_linear_groups: str = ALBATROSS_DEFAULTS["orig_linear_groups"]

    @property
    def albatross_impl_dir(self) -> Path | None:
        if self.albatross_root is None:
            return None
        return self.albatross_root / self.albatross_impl


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in ("http", "https")


def _blocker(code: str, message: str, **details: Any) -> dict[str, Any]:
    blocker = {"code": code, "message": message}
    blocker.update({k: v for k, v in details.items() if v is not None})
    return blocker


def _empty_state_movement_metrics() -> dict[str, int | None]:
    return {name: None for name in STATE_MOVEMENT_COUNTERS}


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
    impl_dir = config.albatross_impl_dir
    return {
        "albatross_repo": ALBATROSS_REPO,
        "albatross_commit": ALBATROSS_COMMIT,
        "albatross_impl": config.albatross_impl,
        "albatross_path": str(impl_dir) if impl_dir is not None else None,
        "albatross_wkv": config.albatross_wkv,
        "albatross_emb": config.albatross_emb,
        "albatross_batched_rkv": config.albatross_batched_rkv,
        "albatross_cmix_sparse": config.albatross_cmix_sparse,
        "albatross_lowrank_weight": config.albatross_lowrank_weight,
        "albatross_orig_linear_groups": config.albatross_orig_linear_groups,
        "contracts": [
            {
                "source_path": entry.source_path,
                "target_path": entry.target_path,
                "correspondence": entry.correspondence,
            }
            for entry in SOURCE_PROVENANCE
        ],
    }


def _source_revision_file(repo_root: Path) -> Path | None:
    for path in (repo_root, *repo_root.parents):
        marker = path / ".helicopter-source-revision"
        if marker.is_file():
            return marker
    return None


def _git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        marker = _source_revision_file(repo_root)
        if marker is None:
            return None
        revision = marker.read_text(encoding="utf-8").strip()
        return revision or None
    revision = result.stdout.strip()
    return revision or None


def _cuda_device_metadata() -> dict[str, Any]:
    if not _cuda_available():
        return {"available": False}
    cuda = _cuda_module()
    device_index = cuda.current_device()
    props = cuda.get_device_properties(device_index)
    return {
        "available": True,
        "device_index": int(device_index),
        "device_name": cuda.get_device_name(device_index),
        "capability": list(cuda.get_device_capability(device_index)),
        "total_memory": int(props.total_memory),
    }


def _rwkv_environment_raw_metadata() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in PROVENANCE_ENV_VARS}


def _rwkv_environment_metadata() -> dict[str, str | None]:
    raw = _rwkv_environment_raw_metadata()
    return {
        name: raw[name] if raw[name] is not None else PROVENANCE_ENV_DEFAULTS.get(name)
        for name in PROVENANCE_ENV_VARS
    }


@contextmanager
def _without_benchmark_only_vllm_env_vars():
    saved = {
        name: os.environ.pop(name)
        for name in BENCHMARK_ONLY_VLLM_ENV_VARS
        if name in os.environ
    }
    try:
        yield
    finally:
        os.environ.update(saved)


def _benchmark_provenance(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "git_revision": _git_revision(config.repo_root),
        "cuda": _cuda_device_metadata(),
        "env": _rwkv_environment_metadata(),
        "raw_env": _rwkv_environment_raw_metadata(),
        "workload": {
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "runner_prefill_chunk_tokens": config.runner_prefill_chunk_tokens,
            "runner_enforce_eager": config.runner_enforce_eager,
            "runner_disable_rapid_sampler": config.runner_disable_rapid_sampler,
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
    require_albatross_checkpoint: bool,
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
                "Set --model or VLLM_RWKV7_MODEL to a vLLM-loadable RWKV7 model.",
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

    impl_dir = config.albatross_impl_dir
    if impl_dir is None:
        blockers.append(
            _blocker(
                "missing_albatross_root",
                "Set --albatross-root or ALBATROSS_ROOT.",
            )
        )
    elif not impl_dir.is_dir():
        blockers.append(
            _blocker(
                "missing_albatross_impl_path",
                "The configured Albatross implementation directory does not exist.",
                path=str(impl_dir),
            )
        )

    if require_albatross_checkpoint:
        checkpoint = config.albatross_checkpoint
        if checkpoint is None:
            blockers.append(
                _blocker(
                    "missing_albatross_checkpoint",
                    "Set --albatross-checkpoint or ALBATROSS_PTH.",
                )
            )
        elif not checkpoint.expanduser().is_file():
            blockers.append(
                _blocker(
                    "missing_albatross_checkpoint_path",
                    "The configured Albatross checkpoint path does not exist.",
                    path=str(checkpoint),
                )
            )
    return blockers


def _get_number(metrics: dict[str, Any], name: str) -> float | None:
    value = metrics.get(name)
    if value is None:
        return None
    return float(value)


def _evaluate_model_only(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "albatross_tokens_per_s": None,
        "vllm_tokens_per_s": None,
        "vllm_to_albatross_ratio": None,
        "latency_slowdown_pct": None,
    }
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["model_only_steady_decode"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("model_only_steady_decode", {})
    albatross_tps = _get_number(raw_metrics, "albatross_tokens_per_s")
    vllm_tps = _get_number(raw_metrics, "vllm_tokens_per_s")
    metrics["albatross_tokens_per_s"] = albatross_tps
    metrics["vllm_tokens_per_s"] = vllm_tps
    if albatross_tps is None and vllm_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_model_only_measurement",
                "Measurement JSON must include albatross_tokens_per_s and "
                "vllm_tokens_per_s for model_only_steady_decode.",
            )
        ]
        return check
    if albatross_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_albatross_model_only_measurement",
                "Measurement JSON must include albatross_tokens_per_s for "
                "model_only_steady_decode.",
            )
        ]
        return check
    if vllm_tps is None:
        check["blockers"] = [
            _blocker(
                "missing_vllm_model_only_measurement",
                "Measurement JSON must include vllm_tokens_per_s for "
                "model_only_steady_decode. Albatross model-only measurement is "
                "present; generate vLLM model-only metrics with "
                "--measure-vllm-model-only.",
            )
        ]
        return check
    if albatross_tps <= 0 or vllm_tps <= 0:
        check["status"] = "failed"
        check["errors"] = [
            "model_only_steady_decode token throughput values must be positive"
        ]
        return check

    ratio = vllm_tps / albatross_tps
    slowdown_pct = (albatross_tps / vllm_tps - 1.0) * 100.0
    metrics.update(
        {
            "albatross_tokens_per_s": albatross_tps,
            "vllm_tokens_per_s": vllm_tps,
            "vllm_to_albatross_ratio": ratio,
            "latency_slowdown_pct": slowdown_pct,
        }
    )
    passed = (
        ratio
        >= ACCEPTANCE_THRESHOLDS["model_only_steady_decode"][
            "min_vllm_to_albatross_ratio"
        ]
        or slowdown_pct
        <= ACCEPTANCE_THRESHOLDS["model_only_steady_decode"]["max_latency_slowdown_pct"]
    )
    check["status"] = "passed" if passed else "failed"
    if not passed:
        check["errors"] = [
            "vLLM model-only steady decode is below the albatross threshold"
        ]
    check["blockers"] = []
    return check


def _parse_bxt_case(case: str, label: str) -> tuple[int, int]:
    try:
        batch_text, seq_text = case.lower().split("x", 1)
        batch_size = int(batch_text)
        seq_len = int(seq_text)
    except ValueError as exc:
        raise ValueError(f"{label} must use BxT format, for example 2x4") from exc
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError(f"{label} values must be positive")
    return batch_size, seq_len


def _parse_albatross_case(case: str) -> tuple[int, int]:
    return _parse_bxt_case(case, "albatross case")


def _parse_bxt_cases(
    cases: str, label: str, *, allow_empty: bool = False
) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for item in cases.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(_parse_bxt_case(item, label))
    if not parsed and not allow_empty:
        raise ValueError(f"{label} must include at least one BxT case")
    return parsed


def _format_bxt_case(batch_size: int, seq_len: int) -> str:
    return f"{batch_size}x{seq_len}"


def _parse_albatross_csv(
    output: str,
    *,
    expected_batch_size: int,
    expected_seq_len: int,
) -> dict[str, Any]:
    csv_rows = []
    for row in csv.reader(output.splitlines()):
        if row and row[0] == "csv":
            csv_rows.append(row)

    for row in csv_rows:
        if len(row) != 9:
            raise ValueError(f"unexpected albatross csv row shape: {row}")
        (
            _,
            label,
            batch_size,
            seq_len,
            iters,
            p10_ms,
            p50_ms,
            p90_ms,
            tok_s,
        ) = row
        parsed_batch_size = int(batch_size)
        parsed_seq_len = int(seq_len)
        if (
            parsed_batch_size != expected_batch_size
            or parsed_seq_len != expected_seq_len
        ):
            continue
        return {
            "label": label,
            "batch_size": parsed_batch_size,
            "seq_len": parsed_seq_len,
            "iters": int(iters),
            "p10_ms": float(p10_ms),
            "p50_ms": float(p50_ms),
            "p90_ms": float(p90_ms),
            "tokens_per_s": float(tok_s),
        }

    if csv_rows:
        raise ValueError(
            "albatross subprocess did not emit the requested BxT csv row "
            f"({expected_batch_size}x{expected_seq_len})"
        )
    raise ValueError("albatross subprocess did not emit a csv measurement row")


def _required_albatross_script(config: BenchmarkConfig) -> Path:
    impl_dir = config.albatross_impl_dir
    if impl_dir is None:
        raise ValueError("Set --albatross-root or ALBATROSS_ROOT.")
    script = impl_dir / ALBATROSS_BENCH_SCRIPT
    if not script.is_file():
        raise FileNotFoundError(f"missing albatross benchmark script: {script}")
    return script


def _required_albatross_checkpoint(config: BenchmarkConfig) -> Path:
    checkpoint = config.albatross_checkpoint
    if checkpoint is None:
        raise ValueError("Set --albatross-checkpoint or ALBATROSS_PTH.")
    checkpoint = checkpoint.expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing albatross checkpoint: {checkpoint}")
    return checkpoint


def generate_albatross_model_only_measurement(
    config: BenchmarkConfig,
    *,
    case: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    batch_size, seq_len = _parse_albatross_case(case)
    if warmup < 0:
        raise ValueError("albatross warmup must be non-negative")
    if iters <= 0:
        raise ValueError("albatross iters must be positive")

    script = _required_albatross_script(config)
    checkpoint = _required_albatross_checkpoint(config)
    command = [
        sys.executable,
        str(script),
        "--model",
        str(checkpoint),
        "--wkv",
        config.albatross_wkv,
        "--emb",
        config.albatross_emb,
        "--batched-rkv",
        config.albatross_batched_rkv,
        "--cmix-sparse",
        config.albatross_cmix_sparse,
        "--lowrank-weight",
        config.albatross_lowrank_weight,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--cases",
        f"{batch_size}x{seq_len}",
    ]
    command.extend(
        [
            "--orig-linear-groups",
            config.albatross_orig_linear_groups,
        ]
    )
    result = subprocess.run(
        command,
        cwd=script.parent,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "albatross model-only subprocess failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )

    parsed = _parse_albatross_csv(
        result.stdout,
        expected_batch_size=batch_size,
        expected_seq_len=seq_len,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "model_only_steady_decode": {
            "albatross_tokens_per_s": parsed["tokens_per_s"],
            "albatross_label": parsed["label"],
            "albatross_batch_size": parsed["batch_size"],
            "albatross_seq_len": parsed["seq_len"],
            "albatross_warmup": warmup,
            "albatross_iters": parsed["iters"],
            "albatross_p10_ms": parsed["p10_ms"],
            "albatross_p50_ms": parsed["p50_ms"],
            "albatross_p90_ms": parsed["p90_ms"],
        },
        "config": {
            "repo_root": str(config.repo_root),
            "albatross_root": (
                str(config.albatross_root)
                if config.albatross_root is not None
                else None
            ),
            "albatross_impl": config.albatross_impl,
            "albatross_checkpoint": str(checkpoint),
            "albatross_wkv": config.albatross_wkv,
            "albatross_emb": config.albatross_emb,
            "albatross_batched_rkv": config.albatross_batched_rkv,
            "albatross_cmix_sparse": config.albatross_cmix_sparse,
            "albatross_lowrank_weight": config.albatross_lowrank_weight,
            "albatross_orig_linear_groups": config.albatross_orig_linear_groups,
            "albatross_command": command,
            "measurement_source": "albatross_subprocess",
        },
    }


def _required_vllm_model_path(config: BenchmarkConfig) -> Path:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if _is_url(config.model):
        raise ValueError(
            "vLLM model-only measurement currently requires a local RWKV7 "
            f"raw .pth checkpoint, got URL: {config.model}"
        )
    model_path = Path(config.model).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f"missing vLLM model checkpoint: {model_path}")
    return model_path


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _initialize_vllm_single_process_distributed() -> str:
    import torch

    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import (
        VllmConfig,
        get_current_vllm_config_or_none,
        set_current_vllm_config,
    )

    preferred_backend = "nccl" if _cuda_available() else "gloo"
    backends = [preferred_backend]
    if preferred_backend == "nccl":
        backends.append("gloo")

    if parallel_state.model_parallel_is_initialized():
        return preferred_backend

    def initialize(backend: str) -> None:
        if not torch.distributed.is_initialized():
            parallel_state.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{_free_tcp_port()}",
                local_rank=0,
                backend=backend,
            )
        parallel_state.ensure_model_parallel_initialized(1, 1, backend=backend)

    last_error: Exception | None = None
    for backend in backends:
        try:
            if get_current_vllm_config_or_none() is None:
                with set_current_vllm_config(VllmConfig()):
                    initialize(backend)
            else:
                initialize(backend)
            return backend
        except Exception as exc:
            last_error = exc
            can_retry = (
                backend == "nccl"
                and not torch.distributed.is_initialized()
                and not parallel_state.model_parallel_is_initialized()
            )
            if can_retry:
                continue
            raise

    assert last_error is not None
    raise last_error


def _load_vllm_rwkv7_model(config: BenchmarkConfig) -> Any:
    model_path = _required_vllm_model_path(config)

    import torch
    import vllm.rwkv7_ops  # noqa: F401

    from vllm.config.compilation import CompilationConfig, CompilationMode
    from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
    from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

    hf_config = build_rwkv7_config_from_pth(model_path)
    if hf_config is None:
        raise ValueError(
            "vLLM model-only measurement currently supports RWKV7 raw .pth "
            f"checkpoints only: {model_path}"
        )
    vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
        model_config=SimpleNamespace(enforce_eager=True, hf_config=hf_config),
    )
    distributed_backend = _initialize_vllm_single_process_distributed()
    model = RWKV7ForCausalLM(vllm_config=vllm_config)
    model._benchmark_distributed_backend = distributed_backend
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"RWKV7 checkpoint must contain a state dict: {model_path}")
    model.load_weights(checkpoint.items())
    model.eval()
    return model


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def _duration_ms_summary(durations_s: list[float]) -> dict[str, float | None]:
    if not durations_s:
        return {
            "p10_ms": None,
            "p50_ms": None,
            "p90_ms": None,
        }
    return {
        "p10_ms": _percentile(durations_s, 0.1) * 1000.0,
        "p50_ms": _percentile(durations_s, 0.5) * 1000.0,
        "p90_ms": _percentile(durations_s, 0.9) * 1000.0,
    }


def _tokens_per_second(tokens: int, duration_s: float) -> float:
    if duration_s <= 0.0:
        return float("inf")
    return float(tokens) / duration_s


def _phase_throughput_summary(
    *,
    total_tokens: int,
    iteration_durations_s: list[float],
    unit_durations_s: list[float],
    unit_tokens: list[int],
) -> dict[str, Any]:
    total_duration_s = sum(iteration_durations_s)
    peak_iteration_tokens_per_s = None
    if iteration_durations_s:
        iteration_tokens = total_tokens // len(iteration_durations_s)
        peak_iteration_tokens_per_s = max(
            _tokens_per_second(iteration_tokens, duration_s)
            for duration_s in iteration_durations_s
        )
    peak_unit_tokens_per_s = None
    if unit_durations_s:
        peak_unit_tokens_per_s = max(
            _tokens_per_second(tokens, duration_s)
            for tokens, duration_s in zip(unit_tokens, unit_durations_s)
        )
    summary = _duration_ms_summary(iteration_durations_s)
    unit_summary = _duration_ms_summary(unit_durations_s)
    return {
        "avg_tokens_per_s": _tokens_per_second(total_tokens, total_duration_s),
        "peak_tokens_per_s": peak_iteration_tokens_per_s,
        "peak_iteration_tokens_per_s": peak_iteration_tokens_per_s,
        "peak_unit_tokens_per_s": peak_unit_tokens_per_s,
        "total_tokens": total_tokens,
        "total_duration_ms": total_duration_s * 1000.0,
        **summary,
        "unit_p10_ms": unit_summary["p10_ms"],
        "unit_p50_ms": unit_summary["p50_ms"],
        "unit_p90_ms": unit_summary["p90_ms"],
    }


def _time_vllm_model_only_steady_decode(
    model: Any,
    *,
    batch_size: int,
    seq_len: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.models.rwkv7 import select_path

    if not _cuda_available():
        raise RuntimeError(
            "CUDA is required for vLLM RWKV7 model-only steady decode measurement."
        )
    cuda = _cuda_module()
    if warmup < 0:
        raise ValueError("vLLM warmup must be non-negative")
    if iters <= 0:
        raise ValueError("vLLM iters must be positive")

    vocab_size = max(1, int(getattr(model, "vocab_size", 0)))
    tokens = torch.arange(
        batch_size * seq_len,
        dtype=torch.long,
        device="cuda",
    ).remainder(vocab_size)
    tokens = tokens.view(batch_size, seq_len)
    state = model.zero_state(batch_size)
    path = select_path(batch_size, seq_len)
    durations_ms: list[float] = []

    if getattr(model, "emb_cpu", False):
        x = model.embed(tokens)

        def run_model() -> Any:
            hidden_states = model.forward_from_x(x, state, path)
            return model.compute_logits(hidden_states)

    else:

        def run_model() -> Any:
            hidden_states = model.forward_tokens(tokens, state)
            return model.compute_logits(hidden_states)

    graph = cuda.CUDAGraph()
    with torch.inference_mode():
        for _ in range(warmup):
            run_model()
        torch.accelerator.synchronize()
        with cuda.graph(graph):
            run_model()
        for _ in range(warmup):
            graph.replay()
        torch.accelerator.synchronize()
        start_event = cuda.Event(enable_timing=True)
        end_event = cuda.Event(enable_timing=True)
        for _ in range(iters):
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            durations_ms.append(start_event.elapsed_time(end_event))

    p50_ms = _percentile(durations_ms, 0.5)
    return {
        "tokens_per_s": (batch_size * seq_len) / (p50_ms / 1000.0),
        "p10_ms": _percentile(durations_ms, 0.1),
        "p50_ms": p50_ms,
        "p90_ms": _percentile(durations_ms, 0.9),
        "graph": True,
        "measurement_mode": "cuda_graph_replay",
        "output": "logits",
        "logits_included": True,
        "distributed_backend": getattr(model, "_benchmark_distributed_backend", None),
    }


def generate_vllm_model_only_measurement(
    config: BenchmarkConfig,
    *,
    case: str,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    batch_size, seq_len = _parse_bxt_case(case, "vLLM case")
    if warmup < 0:
        raise ValueError("vLLM warmup must be non-negative")
    if iters <= 0:
        raise ValueError("vLLM iters must be positive")

    model = _load_vllm_rwkv7_model(config)
    parsed = _time_vllm_model_only_steady_decode(
        model,
        batch_size=batch_size,
        seq_len=seq_len,
        warmup=warmup,
        iters=iters,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "model_only_steady_decode": {
            "vllm_tokens_per_s": parsed["tokens_per_s"],
            "vllm_label": VLLM_MODEL_ONLY_LABEL,
            "vllm_batch_size": batch_size,
            "vllm_seq_len": seq_len,
            "vllm_warmup": warmup,
            "vllm_iters": iters,
            "vllm_p10_ms": parsed["p10_ms"],
            "vllm_p50_ms": parsed["p50_ms"],
            "vllm_p90_ms": parsed["p90_ms"],
            "vllm_graph": parsed["graph"],
            "vllm_measurement_mode": parsed["measurement_mode"],
            "vllm_output": "logits",
            "vllm_logits_included": True,
            "vllm_distributed_backend": parsed["distributed_backend"],
        },
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "vllm_distributed_backend": parsed["distributed_backend"],
            "measurement_source": "vllm_model_direct",
        },
    }


def _model_only_case_from_measurements(
    measurements: dict[str, Any] | None,
) -> str | None:
    if measurements is None:
        return None
    metrics = measurements.get("model_only_steady_decode", {})
    batch_size = metrics.get("albatross_batch_size")
    seq_len = metrics.get("albatross_seq_len")
    if batch_size is None or seq_len is None:
        return None
    return f"{int(batch_size)}x{int(seq_len)}"


def _merge_vllm_model_only_measurement(
    measurements: dict[str, Any],
    vllm_measurement: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(measurements)
    merged_metrics = merged.setdefault("model_only_steady_decode", {})
    vllm_metrics = vllm_measurement.get("model_only_steady_decode", {})
    albatross_batch_size = merged_metrics.get("albatross_batch_size")
    albatross_seq_len = merged_metrics.get("albatross_seq_len")
    vllm_batch_size = vllm_metrics.get("vllm_batch_size")
    vllm_seq_len = vllm_metrics.get("vllm_seq_len")
    if (
        albatross_batch_size is not None
        and albatross_seq_len is not None
        and vllm_batch_size is not None
        and vllm_seq_len is not None
        and (
            int(albatross_batch_size) != int(vllm_batch_size)
            or int(albatross_seq_len) != int(vllm_seq_len)
        )
    ):
        raise ValueError(
            "vLLM model-only case must match Albatross model-only case: "
            f"albatross={albatross_batch_size}x{albatross_seq_len}, "
            f"vllm={vllm_batch_size}x{vllm_seq_len}"
        )

    merged_metrics.update(vllm_metrics)
    merged["schema_version"] = merged.get("schema_version", SCHEMA_VERSION)
    merged["benchmark"] = merged.get("benchmark", BENCHMARK_NAME)
    merged_config = dict(merged.get("config", {}))
    merged_config.update(
        {
            "model": vllm_measurement.get("config", {}).get("model"),
            "measurement_source": "merged_vllm_model_direct",
        }
    )
    merged["config"] = merged_config
    return merged


def _required_vllm_runner_model(config: BenchmarkConfig) -> str:
    if not config.model:
        raise ValueError("Set --model or VLLM_RWKV7_MODEL.")
    if not _is_url(config.model) and not Path(config.model).expanduser().exists():
        raise FileNotFoundError(f"missing vLLM model path: {config.model}")
    return config.model


def _runner_prefill_chunk_tokens(config: BenchmarkConfig) -> int:
    if config.runner_prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    return min(config.prompt_len, config.runner_prefill_chunk_tokens)


def _create_vllm_runner_llm(config: BenchmarkConfig) -> Any:
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP", "1")
    if config.runner_disable_rapid_sampler:
        os.environ["VLLM_USE_RAPID_SAMPLER"] = "0"

    import vllm.rwkv7_ops  # noqa: F401

    from vllm import LLM

    max_model_len = max(1, config.prompt_len + config.decode_tokens)
    max_num_seqs = max(1, config.batch_size)
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(config)
    max_num_batched_tokens = max(
        config.batch_size * prefill_chunk_tokens,
        config.batch_size,
    )
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
    with _without_benchmark_only_vllm_env_vars():
        return LLM(
            model=_required_vllm_runner_model(config),
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


_RUNNER_STATE_SEARCH_ATTRS = (
    "model_runner",
    "model_state",
    "model",
    "worker",
    "workers",
    "driver_worker",
    "executor",
    "llm_engine",
    "engine_core",
    "engine_core_client",
)


def _normalize_state_movement_stats(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError("RWKV7 state movement stats must be a dict")
    missing = [name for name in STATE_MOVEMENT_COUNTERS if name not in raw]
    if missing:
        raise ValueError(
            "RWKV7 state movement stats are missing counters: " + ", ".join(missing)
        )
    return {name: int(raw[name]) for name in STATE_MOVEMENT_COUNTERS}


def _merge_state_movement_stat_dicts(raw_stats: list[Any]) -> dict[str, int]:
    stats = [
        _normalize_state_movement_stats(raw) for raw in raw_stats if raw is not None
    ]
    if not stats:
        raise RuntimeError(
            "Could not locate RWKV7ModelState.get_state_movement_stats() "
            "through offline LLM worker/model_runner attributes."
        )
    return {
        name: sum(worker_stats[name] for worker_stats in stats)
        for name in STATE_MOVEMENT_COUNTERS
    }


def _iter_runner_state_children(obj: Any) -> list[Any]:
    children: list[Any] = []
    for attr in _RUNNER_STATE_SEARCH_ATTRS:
        try:
            child = getattr(obj, attr)
        except Exception:
            continue
        if child is not None:
            children.append(child)
    if isinstance(obj, dict):
        children.extend(obj.values())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        children.extend(obj)
    return children


def _collect_runner_state_movement_stats_from_object(
    root: Any,
    *,
    reset: bool,
) -> dict[str, int] | None:
    queue = [root]
    seen: set[int] = set()
    matches: list[dict[str, int]] = []
    while queue and len(seen) < 512:
        obj = queue.pop(0)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        getter = getattr(obj, "get_state_movement_stats", None)
        if callable(getter):
            if reset:
                resetter = getattr(obj, "reset_state_movement_stats", None)
                if not callable(resetter):
                    raise RuntimeError(
                        "Located RWKV7 state movement stats without "
                        "reset_state_movement_stats()."
                    )
                resetter()
            matches.append(_normalize_state_movement_stats(getter()))
            continue

        queue.extend(_iter_runner_state_children(obj))

    if not matches:
        return None
    return _merge_state_movement_stat_dicts(matches)


def _collect_runner_state_movement_stats_from_worker(
    worker: Any,
    reset: bool = False,
) -> dict[str, int] | None:
    return _collect_runner_state_movement_stats_from_object(worker, reset=reset)


def _extract_runner_state_movement_stats(llm: Any) -> dict[str, int]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        stats = collective_rpc(_collect_runner_state_movement_stats_from_worker)
        try:
            return _merge_state_movement_stat_dicts(list(stats))
        except RuntimeError:
            local_stats = _collect_runner_state_movement_stats_from_object(
                llm,
                reset=False,
            )
            if local_stats is not None:
                return local_stats
            raise

    local_stats = _collect_runner_state_movement_stats_from_object(
        llm,
        reset=False,
    )
    if local_stats is None:
        return _merge_state_movement_stat_dicts([])
    return local_stats


def _reset_runner_state_movement_stats(llm: Any) -> None:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        stats = collective_rpc(
            _collect_runner_state_movement_stats_from_worker,
            args=(True,),
        )
        _merge_state_movement_stat_dicts(list(stats))
        return

    local_stats = _collect_runner_state_movement_stats_from_object(
        llm,
        reset=True,
    )
    if local_stats is None:
        _merge_state_movement_stat_dicts([])


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


def _worker_prompt_token_ids(batch_size: int, prompt_len: int) -> list[list[int]]:
    return [
        [(idx + position) % 1024 for position in range(prompt_len)]
        for idx in range(batch_size)
    ]


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
    )
    prefix = f"rwkv7-prefill-{id(worker)}-{time.perf_counter_ns()}"
    cuda_events = _worker_cuda_event_pair()
    timing_clock = "cuda_event" if cuda_events is not None else "wall_clock"
    iteration_durations_s: list[float] = []
    unit_durations_s: list[float] = []
    unit_tokens: list[int] = []

    for warmup_iteration in range(warmup):
        req_ids = [
            f"{prefix}-warmup-{warmup_iteration}-{idx}"
            for idx in range(batch_size)
        ]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
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
        req_ids = [
            f"{prefix}-measure-{iteration}-{idx}"
            for idx in range(batch_size)
        ]
        chunk_durations_s, chunk_token_counts = _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
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
                prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
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

    for iteration in range(iters):
        req_ids = [f"{prefix}-{iteration}-{idx}" for idx in range(batch_size)]
        _worker_execute_prefill_chunks(
            worker,
            req_ids=req_ids,
            prompt_token_ids=_worker_prompt_token_ids(batch_size, prompt_len),
            sampling_params=sampling_params,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            measure=False,
        )
        worker.sample_tokens(None)
        _worker_cuda_synchronize()

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

    _reset_runner_state_movement_stats(llm)

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
    _reset_runner_state_movement_stats(llm)
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
    _reset_runner_state_movement_stats(llm)
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
    try:
        llm.llm_engine = None
    except Exception:
        pass
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
        decode_tokens=decode_tokens,
    )
    prefill_chunk_tokens = _runner_prefill_chunk_tokens(runner_config)
    capacity_config = replace(
        runner_config,
        decode_tokens=decode_tokens + warmup,
    )
    llm = _create_vllm_runner_llm(capacity_config)
    try:
        parsed = _time_vllm_runner_steady_decode(
            llm,
            batch_size=batch_size,
            prompt_len=prompt_len,
            prefill_chunk_tokens=prefill_chunk_tokens,
            decode_tokens=decode_tokens,
            warmup=warmup,
            iters=iters,
        )
        state_movement = _extract_runner_state_movement_stats(llm)
    finally:
        _shutdown_vllm_runner_llm(llm)

    runner_metrics: dict[str, Any] = {
        "runner_batch_size": batch_size,
        "runner_prompt_len": prompt_len,
        "runner_prefill_chunk_tokens": prefill_chunk_tokens,
        "runner_decode_tokens": decode_tokens,
        "runner_warmup": warmup,
        "runner_warmup_mode": "same_request_decode_steps",
        "runner_warmup_decode_tokens": warmup,
        "runner_iters": iters,
        "runner_measurement_mode": parsed.get("measurement_mode", VLLM_RUNNER_MODE),
        "runner_internal_timing_target": parsed.get(
            "internal_timing_target",
            VLLM_RUNNER_TIMING_TARGET,
        ),
        "runner_timing_clock": parsed.get(
            "timing_clock",
            VLLM_RUNNER_TIMING_CLOCK,
        ),
        "runner_collective_rpc_serialization": (
            "pickle_enabled"
            if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") == "1"
            else "msgpack_only"
        ),
    }
    if "blockers" in parsed:
        runner_metrics["blockers"] = parsed["blockers"]
    else:
        runner_metrics.update(
            {
                "runner_tokens_per_s": parsed["tokens_per_s"],
                "runner_p10_ms": parsed["p10_ms"],
                "runner_p50_ms": parsed["p50_ms"],
                "runner_p90_ms": parsed["p90_ms"],
                "runner_execute_model_p50_ms": parsed.get(
                    "execute_model_p50_ms"
                ),
                "runner_execute_model_p50_tokens_per_s": parsed.get(
                    "execute_model_p50_tokens_per_s"
                ),
                "runner_sample_tokens_p50_ms": parsed.get(
                    "sample_tokens_p50_ms"
                ),
                "runner_sample_tokens_p50_tokens_per_s": parsed.get(
                    "sample_tokens_p50_tokens_per_s"
                ),
                "runner_decode_step_p50_ms": parsed.get("decode_step_p50_ms"),
                "runner_decode_step_p50_tokens_per_s": parsed.get(
                    "decode_step_p50_tokens_per_s"
                ),
                "runner_postprocess_p50_ms": parsed.get("postprocess_p50_ms"),
                "runner_postprocess_timing_available": parsed.get(
                    "postprocess_timing_available",
                    False,
                ),
                "runner_decode_steps": parsed["decode_steps"],
                "runner_worker_count": parsed["worker_count"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "runner_steady_decode": runner_metrics,
        "state_movement": state_movement,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": f"vllm_runner_{VLLM_RUNNER_MODE}",
            "provenance": _benchmark_provenance(runner_config),
        },
    }


def _runner_phase_metrics(
    *,
    batch_size: int,
    seq_len: int,
    phase: str,
    prefill_chunk_tokens: int,
    iters: int,
    parsed: dict[str, Any],
    warmup: int | None = None,
    include_sampling: bool | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "case": _format_bxt_case(batch_size, seq_len),
        "phase": phase,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "iters": iters,
        "measurement_mode": parsed.get("measurement_mode", VLLM_RUNNER_MODE),
        "internal_timing_target": parsed.get("internal_timing_target"),
        "timing_clock": parsed.get("timing_clock", VLLM_RUNNER_TIMING_CLOCK),
        "worker_count": parsed.get("worker_count"),
    }
    if warmup is not None:
        metrics["warmup_decode_tokens"] = warmup
    if "warmup_iterations" in parsed:
        metrics["warmup_iterations"] = parsed["warmup_iterations"]
    if include_sampling is not None:
        metrics["sampling_included"] = include_sampling
        metrics["sampling_mode"] = "sample_tokens" if include_sampling else "skip"
    if "blockers" in parsed:
        metrics["blockers"] = parsed["blockers"]
        return metrics
    metrics.update(
        {
            "avg_tokens_per_s": parsed["avg_tokens_per_s"],
            "peak_tokens_per_s": parsed["peak_tokens_per_s"],
            "peak_iteration_tokens_per_s": parsed.get(
                "peak_iteration_tokens_per_s"
            ),
            "peak_unit_tokens_per_s": parsed.get("peak_unit_tokens_per_s"),
            "total_tokens": parsed["total_tokens"],
            "total_duration_ms": parsed["total_duration_ms"],
            "p10_ms": parsed["p10_ms"],
            "p50_ms": parsed["p50_ms"],
            "p90_ms": parsed["p90_ms"],
            "unit_p10_ms": parsed["unit_p10_ms"],
            "unit_p50_ms": parsed["unit_p50_ms"],
            "unit_p90_ms": parsed["unit_p90_ms"],
        }
    )
    if "sample_tokens_p50_ms" in parsed:
        metrics["sample_tokens_p50_ms"] = parsed["sample_tokens_p50_ms"]
    return metrics


def generate_vllm_runner_pd_single_measurement(
    config: BenchmarkConfig,
    *,
    phase: str,
    case: tuple[int, int],
    prefill_chunk_tokens: int,
    decode_prompt_len: int,
    warmup: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    batch_size, seq_len = case
    case_name = _format_bxt_case(batch_size, seq_len)
    if phase == "prefill":
        case_prefill_chunk_tokens = min(prefill_chunk_tokens, seq_len)
        capture_size = max(batch_size * case_prefill_chunk_tokens, batch_size)
        runner_config = replace(
            config,
            batch_size=batch_size,
            prompt_len=seq_len,
            decode_tokens=1,
            runner_prefill_chunk_tokens=case_prefill_chunk_tokens,
            runner_cudagraph_capture_sizes=(capture_size,),
        )
        llm = _create_vllm_runner_llm(runner_config)
        try:
            parsed = _time_vllm_runner_prefill_phase(
                llm,
                batch_size=batch_size,
                prompt_len=seq_len,
                prefill_chunk_tokens=case_prefill_chunk_tokens,
                warmup=warmup,
                iters=iters,
            )
            state_movement = _extract_runner_state_movement_stats(llm)
        finally:
            _shutdown_vllm_runner_llm(llm)
        metrics = _runner_phase_metrics(
            batch_size=batch_size,
            seq_len=seq_len,
            phase="prefill",
            prefill_chunk_tokens=case_prefill_chunk_tokens,
            iters=iters,
            parsed=parsed,
        )
    elif phase == "decode":
        case_prefill_chunk_tokens = min(prefill_chunk_tokens, decode_prompt_len)
        capture_size = max(batch_size * case_prefill_chunk_tokens, batch_size)
        runner_config = replace(
            config,
            batch_size=batch_size,
            prompt_len=decode_prompt_len,
            decode_tokens=seq_len + warmup,
            runner_prefill_chunk_tokens=case_prefill_chunk_tokens,
            runner_cudagraph_capture_sizes=(capture_size,),
        )
        llm = _create_vllm_runner_llm(runner_config)
        try:
            parsed = _time_vllm_runner_decode_phase(
                llm,
                batch_size=batch_size,
                prompt_len=decode_prompt_len,
                prefill_chunk_tokens=case_prefill_chunk_tokens,
                decode_tokens=seq_len,
                warmup=warmup,
                iters=iters,
                include_sampling=include_sampling,
            )
            state_movement = _extract_runner_state_movement_stats(llm)
        finally:
            _shutdown_vllm_runner_llm(llm)
        metrics = _runner_phase_metrics(
            batch_size=batch_size,
            seq_len=seq_len,
            phase="decode",
            prefill_chunk_tokens=case_prefill_chunk_tokens,
            warmup=warmup,
            iters=iters,
            include_sampling=include_sampling,
            parsed=parsed,
        )
    else:
        raise ValueError(f"unsupported runner pd phase: {phase}")
    metrics["state_movement"] = state_movement
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "phase": phase,
        "case": case_name,
        "metrics": metrics,
        "state_movement": state_movement,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": "vllm_runner_prefill_decode_single",
            "provenance": _benchmark_provenance(runner_config),
        },
    }


def _run_vllm_runner_pd_case_subprocess(
    config: BenchmarkConfig,
    *,
    phase: str,
    case: tuple[int, int],
    prefill_chunk_tokens: int,
    decode_prompt_len: int,
    warmup: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rwkv7-runner-pd-") as tmpdir:
        output_path = Path(tmpdir) / "measurement.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--repo-root",
            str(config.repo_root),
            "--measure-vllm-runner-pd-single",
            "--runner-pd-single-phase",
            phase,
            "--runner-pd-single-case",
            _format_bxt_case(*case),
            "--runner-pd-prefill-chunk-tokens",
            str(prefill_chunk_tokens),
            "--runner-pd-decode-prompt-len",
            str(decode_prompt_len),
            "--runner-warmup",
            str(warmup),
            "--runner-iters",
            str(iters),
            "--measurement-output",
            str(output_path),
        ]
        if config.runner_enforce_eager:
            command.append("--runner-enforce-eager")
        if config.runner_disable_rapid_sampler:
            command.append("--runner-disable-rapid-sampler")
        if config.model is not None:
            command.extend(["--model", config.model])
        if include_sampling:
            command.append("--runner-pd-include-sampling")
        env = os.environ.copy()
        if config.runner_disable_rapid_sampler:
            env["VLLM_USE_RAPID_SAMPLER"] = "0"
        env["VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP"] = "1"
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "vLLM runner PD subprocess failed for "
                f"{phase}:{_format_bxt_case(*case)} with exit code "
                f"{result.returncode}"
            )
        with output_path.open(encoding="utf-8") as measurement_file:
            measurement = json.load(measurement_file)
    if not isinstance(measurement, dict):
        raise ValueError("vLLM runner PD subprocess did not emit a JSON object")
    return measurement


def generate_vllm_runner_pd_measurement(
    config: BenchmarkConfig,
    *,
    prefill_cases: list[tuple[int, int]],
    decode_cases: list[tuple[int, int]],
    prefill_chunk_tokens: int,
    decode_prompt_len: int,
    warmup: int,
    iters: int,
    include_sampling: bool,
) -> dict[str, Any]:
    if prefill_chunk_tokens <= 0:
        raise ValueError("runner prefill chunk tokens must be positive")
    if decode_prompt_len <= 0:
        raise ValueError("runner decode prompt len must be positive")
    if warmup < 0:
        raise ValueError("runner warmup must be non-negative")
    if iters <= 0:
        raise ValueError("runner iters must be positive")

    prefill_metrics: dict[str, Any] = {}
    decode_metrics: dict[str, Any] = {}
    state_movement_by_case: dict[str, Any] = {}
    case_provenance_by_case: dict[str, Any] = {}

    for batch_size, seq_len in prefill_cases:
        case = _format_bxt_case(batch_size, seq_len)
        case_prefill_chunk_tokens = min(prefill_chunk_tokens, seq_len)
        measurement = _run_vllm_runner_pd_case_subprocess(
            config,
            phase="prefill",
            case=(batch_size, seq_len),
            prefill_chunk_tokens=case_prefill_chunk_tokens,
            decode_prompt_len=decode_prompt_len,
            warmup=warmup,
            iters=iters,
            include_sampling=include_sampling,
        )
        prefill_metrics[case] = measurement["metrics"]
        state_movement_by_case[f"prefill:{case}"] = measurement["state_movement"]
        case_provenance_by_case[f"prefill:{case}"] = measurement.get(
            "config", {}
        ).get("provenance")

    for batch_size, decode_tokens in decode_cases:
        case = _format_bxt_case(batch_size, decode_tokens)
        measurement = _run_vllm_runner_pd_case_subprocess(
            config,
            phase="decode",
            case=(batch_size, decode_tokens),
            prefill_chunk_tokens=min(prefill_chunk_tokens, decode_prompt_len),
            warmup=warmup,
            iters=iters,
            include_sampling=include_sampling,
            decode_prompt_len=decode_prompt_len,
        )
        decode_metrics[case] = measurement["metrics"]
        state_movement_by_case[f"decode:{case}"] = measurement["state_movement"]
        case_provenance_by_case[f"decode:{case}"] = measurement.get(
            "config", {}
        ).get("provenance")

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "runner_prefill": prefill_metrics,
        "runner_decode": decode_metrics,
        "state_movement_by_case": state_movement_by_case,
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "measurement_source": "vllm_runner_prefill_decode",
            "runner_pd_prefill_cases": [
                _format_bxt_case(batch_size, seq_len)
                for batch_size, seq_len in prefill_cases
            ],
            "runner_pd_decode_cases": [
                _format_bxt_case(batch_size, seq_len)
                for batch_size, seq_len in decode_cases
            ],
            "runner_pd_decode_prompt_len": decode_prompt_len,
            "runner_pd_include_sampling": include_sampling,
            "case_provenance_by_case": case_provenance_by_case,
            "provenance": _benchmark_provenance(config),
        },
    }


def _merge_vllm_runner_measurement(
    measurements: dict[str, Any],
    runner_measurement: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(measurements)
    merged["schema_version"] = merged.get("schema_version", SCHEMA_VERSION)
    merged["benchmark"] = merged.get("benchmark", BENCHMARK_NAME)
    merged["runner_steady_decode"] = copy.deepcopy(
        runner_measurement.get("runner_steady_decode", {})
    )
    if "state_movement" in runner_measurement:
        merged["state_movement"] = copy.deepcopy(runner_measurement["state_movement"])
    merged_config = dict(merged.get("config", {}))
    merged_config.update(
        {
            "model": runner_measurement.get("config", {}).get("model"),
            "measurement_source": f"merged_vllm_runner_{VLLM_RUNNER_MODE}",
            "provenance": runner_measurement.get("config", {}).get("provenance"),
        }
    )
    merged["config"] = merged_config
    return merged


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
            "runner_decode_step_p50_ms": raw_metrics.get(
                "runner_decode_step_p50_ms"
            ),
            "runner_decode_step_p50_tokens_per_s": raw_metrics.get(
                "runner_decode_step_p50_tokens_per_s"
            ),
            "runner_postprocess_p50_ms": raw_metrics.get(
                "runner_postprocess_p50_ms"
            ),
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


def _evaluate_state_movement(
    measurements: dict[str, Any] | None,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = _empty_state_movement_metrics()
    check = {
        "status": "blocked",
        "thresholds": ACCEPTANCE_THRESHOLDS["state_movement"],
        "metrics": metrics,
        "blockers": blockers,
        "errors": [],
    }
    if measurements is None:
        check["blockers"] = _measurement_blockers(blockers)
        return check

    raw_metrics = measurements.get("state_movement", {})
    missing = [name for name in STATE_MOVEMENT_COUNTERS if name not in raw_metrics]
    if missing:
        check["blockers"] = [
            _blocker(
                "missing_state_movement_counters",
                "Measurement JSON must include all RWKV7 state movement counters.",
                missing=missing,
            )
        ]
        return check

    for name in STATE_MOVEMENT_COUNTERS:
        metrics[name] = int(raw_metrics[name])
    errors = []
    movement_thresholds = ACCEPTANCE_THRESHOLDS["state_movement"]
    zero_copy_counters = (
        (
            "resident_to_decode_copies",
            "max_resident_to_decode_copies",
            "steady decode resident-to-decode copies must remain zero",
        ),
        (
            "decode_compactions",
            "max_decode_compactions",
            "steady decode recurrent-state compactions must remain zero",
        ),
        (
            "decode_full_row_copies",
            "max_decode_full_row_copies",
            "steady decode full recurrent-state row copies must remain zero",
        ),
    )
    for metric_name, threshold_name, message in zero_copy_counters:
        if metrics[metric_name] > movement_thresholds[threshold_name]:
            errors.append(message)
    check["status"] = "passed" if not errors else "failed"
    check["errors"] = errors
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
        require_albatross_checkpoint=True,
    )
    model_only_check = _evaluate_model_only(measurements, runtime_blockers)
    runner_check = _evaluate_runner(measurements, runtime_blockers)
    state_check = _evaluate_state_movement(measurements, runtime_blockers)
    checks = {
        "model_only_steady_decode": model_only_check,
        "runner_steady_decode": runner_check,
        "state_movement": state_check,
    }
    statuses = [check["status"] for check in checks.values()]
    if "failed" in statuses:
        overall_status = "failed"
    elif "blocked" in statuses:
        overall_status = "blocked"
    else:
        overall_status = "passed"
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "overall_status": overall_status,
        "source": _source_metadata(config),
        "config": {
            "repo_root": str(config.repo_root),
            "model": config.model,
            "albatross_root": (
                str(config.albatross_root)
                if config.albatross_root is not None
                else None
            ),
            "albatross_impl": config.albatross_impl,
            "albatross_checkpoint": (
                str(config.albatross_checkpoint)
                if config.albatross_checkpoint is not None
                else None
            ),
            "batch_size": config.batch_size,
            "prompt_len": config.prompt_len,
            "warmup_tokens": config.warmup_tokens,
            "decode_tokens": config.decode_tokens,
            "cuda_available": cuda,
            "measurement_source": "json" if measurements is not None else None,
            "provenance": _benchmark_provenance(config),
        },
        "acceptance": ACCEPTANCE_THRESHOLDS,
        "checks": checks,
    }


def _default_repo_root() -> Path:
    return REPO_ROOT


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    model = args.model or os.environ.get("VLLM_RWKV7_MODEL") or None
    albatross_root = _optional_path(
        args.albatross_root
        or os.environ.get(
            "ALBATROSS_ROOT",
            str(Path.home() / "Projects/MachineLearning/albatross"),
        )
    )
    checkpoint_env = os.environ.get("ALBATROSS_PTH") or os.environ.get(
        "VLLM_RWKV7_MODEL"
    )
    return BenchmarkConfig(
        repo_root=args.repo_root.resolve(),
        model=model,
        albatross_root=albatross_root,
        albatross_impl=args.albatross_impl,
        albatross_checkpoint=_optional_path(
            args.albatross_checkpoint or checkpoint_env
        ),
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        warmup_tokens=args.warmup_tokens,
        decode_tokens=args.decode_tokens,
        runner_prefill_chunk_tokens=args.runner_prefill_chunk_tokens,
        runner_enforce_eager=args.runner_enforce_eager,
        runner_disable_rapid_sampler=args.runner_disable_rapid_sampler,
        albatross_wkv=args.albatross_wkv
        or os.environ.get("ALBATROSS_WKV")
        or ALBATROSS_DEFAULTS["wkv"],
        albatross_emb=args.albatross_emb
        or os.environ.get("ALBATROSS_EMB")
        or ALBATROSS_DEFAULTS["emb"],
        albatross_batched_rkv=args.albatross_batched_rkv
        or os.environ.get("ALBATROSS_BATCHED_RKV")
        or ALBATROSS_DEFAULTS["batched_rkv"],
        albatross_cmix_sparse=args.albatross_cmix_sparse
        or os.environ.get("ALBATROSS_CMIX_SPARSE")
        or ALBATROSS_DEFAULTS["cmix_sparse"],
        albatross_lowrank_weight=args.albatross_lowrank_weight
        or os.environ.get("ALBATROSS_LOWRANK_WEIGHT")
        or ALBATROSS_DEFAULTS["lowrank_weight"],
        albatross_orig_linear_groups=(
            args.albatross_orig_linear_groups
            or os.environ.get("ALBATROSS_ORIG_LINEAR_GROUPS")
            or ALBATROSS_DEFAULTS["orig_linear_groups"]
        ),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RWKV7 faster3a performance acceptance harness."
    )
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--model", help="vLLM-loadable RWKV7 model path or URL.")
    parser.add_argument("--albatross-root")
    parser.add_argument(
        "--albatross-impl",
        default=ALBATROSS_IMPL,
    )
    parser.add_argument("--albatross-checkpoint")
    parser.add_argument("--albatross-wkv", choices=("fp16", "fp32io16"))
    parser.add_argument("--albatross-emb", choices=("gpu", "cpu"))
    parser.add_argument("--albatross-batched-rkv", choices=("auto", "on", "off"))
    parser.add_argument("--albatross-cmix-sparse", choices=("auto", "no-fc", "off"))
    parser.add_argument("--albatross-lowrank-weight", choices=("orig", "transpose", "both"))
    parser.add_argument("--albatross-orig-linear-groups")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument(
        "--measurement-json",
        type=Path,
        help="Optional JSON metrics file to evaluate against thresholds.",
    )
    parser.add_argument(
        "--measure-albatross-model-only",
        action="store_true",
        help="Run the canonical Albatross model-only benchmark for one BxT case.",
    )
    parser.add_argument(
        "--measure-vllm-model-only",
        action="store_true",
        help="Run the vLLM RWKV7 model-only steady decode benchmark.",
    )
    parser.add_argument(
        "--measure-vllm-runner",
        action="store_true",
        help="Run the vLLM offline LLM.generate runner steady decode benchmark.",
    )
    parser.add_argument(
        "--measure-vllm-runner-pd",
        action="store_true",
        help=(
            "Run split vLLM runner prefill/decode execute_model throughput "
            "benchmarks."
        ),
    )
    parser.add_argument(
        "--measure-vllm-runner-pd-single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runner-pd-single-phase",
        choices=("prefill", "decode"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runner-pd-single-case",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--albatross-case",
        help="Single Albatross BxT case for --measure-albatross-model-only.",
    )
    parser.add_argument(
        "--vllm-case",
        help=(
            "Single vLLM BxT case for --measure-vllm-model-only. Defaults to "
            "the Albatross model-only case from --measurement-json, then "
            "--batch-size x --prompt-len."
        ),
    )
    parser.add_argument(
        "--albatross-warmup",
        type=int,
        default=1,
        help="Warmup iterations passed to rwkv7_fast_v3a.py.",
    )
    parser.add_argument(
        "--albatross-iters",
        type=int,
        default=3,
        help="Timed iterations passed to rwkv7_fast_v3a.py.",
    )
    parser.add_argument(
        "--vllm-warmup",
        type=int,
        default=1,
        help="Warmup iterations for vLLM model-only steady decode.",
    )
    parser.add_argument(
        "--vllm-iters",
        type=int,
        default=3,
        help="Timed iterations for vLLM model-only steady decode.",
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
        "--runner-disable-rapid-sampler",
        action="store_true",
        help="Disable rapid sampler allocation for runner measurements.",
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
        "--runner-pd-prefill-cases",
        default=DEFAULT_RUNNER_PD_PREFILL_CASES,
        help=(
            "Comma-separated BxT prefill cases for --measure-vllm-runner-pd."
        ),
    )
    parser.add_argument(
        "--runner-pd-prefill-chunk-tokens",
        type=int,
        help=(
            "Maximum prompt tokens scheduled per request for split prefill "
            "measurements. Defaults to the largest prefill case T so Albatross "
            "B1T1024/B32T32 comparisons measure full-case prefill."
        ),
    )
    parser.add_argument(
        "--runner-pd-decode-cases",
        default=DEFAULT_RUNNER_PD_DECODE_CASES,
        help="Comma-separated BxT decode cases for --measure-vllm-runner-pd.",
    )
    parser.add_argument(
        "--runner-pd-decode-prompt-len",
        type=int,
        default=1,
        help="Prompt length used to initialize state before decode-only timing.",
    )
    parser.add_argument(
        "--runner-pd-include-sampling",
        action="store_true",
        help="Include sample_tokens in decode phase timing.",
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
    measurement_modes = [
        args.measure_albatross_model_only,
        args.measure_vllm_model_only,
        args.measure_vllm_runner,
        args.measure_vllm_runner_pd,
        args.measure_vllm_runner_pd_single,
    ]
    if sum(bool(mode) for mode in measurement_modes) > 1:
        parser.error("choose only one measurement mode")
    if any(measurement_modes) and args.measurement_output is None:
        parser.error("--measurement-output is required with a measurement mode")
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    config = _config_from_args(args)
    if args.measure_albatross_model_only:
        measurement = generate_albatross_model_only_measurement(
            config,
            case=args.albatross_case or f"{config.batch_size}x{config.prompt_len}",
            warmup=args.albatross_warmup,
            iters=args.albatross_iters,
        )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_model_only:
        existing_measurements = _load_measurements(args.measurement_json)
        measurement = generate_vllm_model_only_measurement(
            config,
            case=args.vllm_case
            or _model_only_case_from_measurements(existing_measurements)
            or f"{config.batch_size}x{config.prompt_len}",
            warmup=args.vllm_warmup,
            iters=args.vllm_iters,
        )
        if existing_measurements is not None:
            measurement = _merge_vllm_model_only_measurement(
                existing_measurements,
                measurement,
            )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_runner:
        existing_measurements = _load_measurements(args.measurement_json)
        measurement = generate_vllm_runner_measurement(
            config,
            batch_size=args.runner_batch_size,
            prompt_len=args.runner_prompt_len,
            decode_tokens=args.runner_decode_tokens,
            warmup=args.runner_warmup,
            iters=args.runner_iters,
        )
        if existing_measurements is not None:
            measurement = _merge_vllm_runner_measurement(
                existing_measurements,
                measurement,
            )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_runner_pd_single:
        if args.runner_pd_single_phase is None or args.runner_pd_single_case is None:
            raise ValueError(
                "--runner-pd-single-phase and --runner-pd-single-case are required"
            )
        pd_prefill_chunk_tokens = (
            args.runner_pd_prefill_chunk_tokens
            if args.runner_pd_prefill_chunk_tokens is not None
            else args.runner_prefill_chunk_tokens
        )
        measurement = generate_vllm_runner_pd_single_measurement(
            config,
            phase=args.runner_pd_single_phase,
            case=_parse_bxt_case(
                args.runner_pd_single_case,
                "runner pd single case",
            ),
            prefill_chunk_tokens=pd_prefill_chunk_tokens,
            decode_prompt_len=args.runner_pd_decode_prompt_len,
            warmup=args.runner_warmup,
            iters=args.runner_iters,
            include_sampling=args.runner_pd_include_sampling,
        )
        _write_report(measurement, args.measurement_output)
        return 0
    if args.measure_vllm_runner_pd:
        prefill_cases = _parse_bxt_cases(
            args.runner_pd_prefill_cases,
            "runner prefill cases",
            allow_empty=True,
        )
        decode_cases = _parse_bxt_cases(
            args.runner_pd_decode_cases,
            "runner decode cases",
            allow_empty=True,
        )
        if not prefill_cases and not decode_cases:
            raise ValueError(
                "runner prefill/decode measurement must include at least one "
                "prefill or decode BxT case"
            )
        pd_prefill_chunk_tokens = (
            args.runner_pd_prefill_chunk_tokens
            if args.runner_pd_prefill_chunk_tokens is not None
            else (
                max(seq_len for _batch_size, seq_len in prefill_cases)
                if prefill_cases
                else args.runner_prefill_chunk_tokens
            )
        )
        measurement = generate_vllm_runner_pd_measurement(
            config,
            prefill_cases=prefill_cases,
            decode_cases=decode_cases,
            prefill_chunk_tokens=pd_prefill_chunk_tokens,
            decode_prompt_len=args.runner_pd_decode_prompt_len,
            warmup=args.runner_warmup,
            iters=args.runner_iters,
            include_sampling=args.runner_pd_include_sampling,
        )
        _write_report(measurement, args.measurement_output)
        return 0

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
