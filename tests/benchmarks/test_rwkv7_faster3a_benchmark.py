# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import pytest

from benchmarks.rwkv7 import benchmark_faster3a as bench


def _config(
    repo_root: Path,
    *,
    model: str | None = None,
    albatross_root: Path | None = None,
    albatross_checkpoint: Path | None = None,
) -> bench.BenchmarkConfig:
    return bench.BenchmarkConfig(
        repo_root=repo_root,
        model=model,
        albatross_root=albatross_root,
        albatross_impl=bench.ALBATROSS_IMPL,
        albatross_checkpoint=albatross_checkpoint,
        batch_size=16,
        prompt_len=128,
        warmup_tokens=16,
        decode_tokens=128,
    )


def _command_options(command: list[str]) -> dict[str, str | bool]:
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(command):
        item = command[index]
        if not item.startswith("--"):
            index += 1
            continue
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[item] = command[index + 1]
            index += 2
        else:
            options[item] = True
            index += 1
    return options


def _phase_result(tokens: int = 1024) -> dict[str, Any]:
    return {
        "avg_tokens_per_s": 1.0,
        "peak_tokens_per_s": 1.0,
        "peak_iteration_tokens_per_s": 1.0,
        "peak_unit_tokens_per_s": 1.0,
        "total_tokens": tokens,
        "total_duration_ms": 1.0,
        "p10_ms": 1.0,
        "p50_ms": 1.0,
        "p90_ms": 1.0,
        "unit_p10_ms": 1.0,
        "unit_p50_ms": 1.0,
        "unit_p90_ms": 1.0,
        "worker_count": 1,
    }


def _matching_model_only_contracts(
    *,
    checkpoint: str = "/tmp/model.pth",
    checkpoint_sha256: str = "a" * 64,
    gemm_accumulation_policy: str = "fp16_where_configurable",
) -> dict[str, dict[str, Any]]:
    common = {
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_sha256,
        "batch_size": 2,
        "seq_len": 4,
        "output_boundary": bench.MODEL_ONLY_OUTPUT_BOUNDARY,
        "wkv_mode": "fp16",
        "gemm_accumulation_policy": gemm_accumulation_policy,
        "embedding_device": "cpu",
        "rkv_mode": "off",
        "cmix_sparse": "no-fc",
        "low_rank_weight": "both",
        "orig_linear_groups": ["att_c2c", "ffn_key", "head"],
    }
    device = {
        "available": True,
        "device_uuid": "GPU-test-uuid",
        "device_name": "test-gpu",
        "capability": [9, 0],
        "total_memory": 24 * 1024**3,
    }
    return {
        "albatross_contract": {
            **common,
            "implementation_revision": bench.ALBATROSS_COMMIT,
            "device": dict(device),
        },
        "vllm_contract": {
            **common,
            "implementation_revision": "vllm-test-revision",
            "device": dict(device),
        },
    }


def _fp16_runner_provenance() -> dict[str, Any]:
    return {"env": {"VLLM_RWKV7_WKV_MODE": "fp16"}}


def test_git_revision_reads_remote_source_marker(tmp_path: Path) -> None:
    marker = tmp_path / ".helicopter-source-revision"
    marker.write_text("abc123-dirty\n", encoding="utf-8")

    assert bench._git_revision(tmp_path) == "abc123-dirty"


def test_git_revision_marks_untracked_files_dirty(tmp_path: Path) -> None:
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", tracked.name],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "test fixture"],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (tmp_path / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    assert bench._git_revision(tmp_path) == f"{revision}-dirty"


def test_rwkv_environment_metadata_records_resolved_defaults(monkeypatch) -> None:
    for name in bench.PROVENANCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    env = bench._rwkv_environment_metadata()
    raw_env = bench._rwkv_environment_raw_metadata()

    assert set(env) == set(bench.PROVENANCE_ENV_VARS)
    assert set(raw_env) == set(bench.PROVENANCE_ENV_VARS)
    assert all(value is None for value in raw_env.values())
    assert env["VLLM_RWKV7_WKV_MODE"] == "fp16"


def test_rwkv_environment_metadata_preserves_explicit_values(monkeypatch) -> None:
    for name in bench.PROVENANCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")

    env = bench._rwkv_environment_metadata()
    raw_env = bench._rwkv_environment_raw_metadata()

    assert env["VLLM_RWKV7_WKV_MODE"] == "fp32io16"
    assert raw_env["VLLM_RWKV7_WKV_MODE"] == "fp32io16"


def test_report_blocks_without_runtime_paths_and_records_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for name in bench.PROVENANCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        bench,
        "_cuda_device_metadata",
        lambda: {"available": False},
    )
    report = bench.build_report(
        _config(
            repo_root,
            model=str(tmp_path / "missing-model"),
            albatross_root=tmp_path / "missing-albatross",
            albatross_checkpoint=tmp_path / "missing.pth",
        ),
        cuda_available=False,
    )

    assert report["overall_status"] == "blocked"
    assert report["source"]["albatross_repo"] == bench.ALBATROSS_REPO
    assert report["source"]["albatross_commit"] == bench.ALBATROSS_COMMIT
    assert report["source"]["albatross_impl"] == bench.ALBATROSS_IMPL
    assert Path(report["source"]["albatross_path"]) == (
        tmp_path / "missing-albatross" / bench.ALBATROSS_IMPL
    )
    assert report["source"]["contracts"] == [
        {
            "source_path": entry.source_path,
            "target_path": entry.target_path,
            "correspondence": entry.correspondence,
        }
        for entry in bench.SOURCE_PROVENANCE
    ]
    blocker_codes = {
        blocker["code"]
        for blocker in report["checks"]["model_only_steady_decode"]["blockers"]
    }
    assert blocker_codes >= {
        "cuda_unavailable",
        "missing_vllm_model_path",
        "missing_albatross_impl_path",
        "missing_albatross_checkpoint_path",
    }
    provenance = report["config"]["provenance"]
    assert provenance["git_revision"]
    assert provenance["workload"] == {
        "batch_size": 16,
        "prompt_len": 128,
        "warmup_tokens": 16,
        "decode_tokens": 128,
        "runner_prefill_chunk_tokens": bench.DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS,
        "runner_enforce_eager": False,
        "runner_cudagraph_capture_sizes": None,
    }
    assert provenance["sampling"] == bench.VLLM_RUNNER_SAMPLING
    assert provenance["cuda"]["available"] is False
    assert set(provenance["env"]) == set(bench.PROVENANCE_ENV_VARS)
    assert set(provenance["raw_env"]) == set(bench.PROVENANCE_ENV_VARS)
    assert provenance["env"]["VLLM_RWKV7_WKV_MODE"] == "fp16"
    assert provenance["raw_env"]["VLLM_RWKV7_WKV_MODE"] is None


def test_report_evaluates_passing_measurements() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
            **_matching_model_only_contracts(),
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 91.0,
        },
        "config": {"provenance": _fp16_runner_provenance()},
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=False,
    )

    assert report["overall_status"] == "passed"
    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"


def test_report_blocks_model_only_comparison_without_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        }
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"] == [
        {
            "code": "missing_model_only_comparison_contract",
            "message": "Model-only comparison requires retained Albatross and "
            "vLLM measurement contracts.",
            "missing": ["albatross_contract", "vllm_contract"],
        }
    ]


def test_report_blocks_non_like_for_like_model_only_precision() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contracts = _matching_model_only_contracts()
    contracts["albatross_contract"]["gemm_accumulation_policy"] = "fp32"
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
            **contracts,
        }
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["metrics"]["vllm_to_albatross_ratio"] is None
    assert check["blockers"] == [
        {
            "code": "non_like_for_like_model_only_comparison",
            "message": "Albatross and vLLM model-only measurements use "
            "different comparison contracts.",
            "mismatches": {
                "gemm_accumulation_policy": {
                    "albatross": "fp32",
                    "vllm": "fp16_where_configurable",
                }
            },
        }
    ]


def test_report_blocks_matching_non_fp16_throughput_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
            **_matching_model_only_contracts(gemm_accumulation_policy="fp32"),
        }
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["metrics"]["vllm_to_albatross_ratio"] is None
    assert check["blockers"] == [
        {
            "code": "invalid_model_only_throughput_contract",
            "message": "Model-only performance acceptance requires the FP16 "
            "throughput contract.",
            "violations": {
                "albatross_contract": {
                    "gemm_accumulation_policy": {
                        "required": "fp16_where_configurable",
                        "actual": "fp32",
                    }
                },
                "vllm_contract": {
                    "gemm_accumulation_policy": {
                        "required": "fp16_where_configurable",
                        "actual": "fp32",
                    }
                },
            },
        }
    ]


def test_report_blocks_model_only_checkpoint_digest_mismatch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contracts = _matching_model_only_contracts()
    contracts["vllm_contract"]["checkpoint_sha256"] = "b" * 64
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
                **contracts,
            }
        },
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"][0]["code"] == ("non_like_for_like_model_only_comparison")
    assert check["blockers"][0]["mismatches"] == {
        "checkpoint_sha256": {
            "albatross": "a" * 64,
            "vllm": "b" * 64,
        }
    }


def test_report_blocks_model_only_device_mismatch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contracts = _matching_model_only_contracts()
    contracts["vllm_contract"]["device"]["device_uuid"] = "GPU-different-uuid"
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
                **contracts,
            }
        },
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"][0]["code"] == ("non_like_for_like_model_only_comparison")
    assert (
        check["blockers"][0]["mismatches"]["device"]["vllm"]["device_uuid"]
        == "GPU-different-uuid"
    )


def test_report_blocks_wrong_albatross_implementation_revision() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contracts = _matching_model_only_contracts()
    contracts["albatross_contract"]["implementation_revision"] = "wrong-revision"
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
                **contracts,
            }
        },
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"] == [
        {
            "code": "invalid_model_only_measurement_provenance",
            "message": "Model-only performance acceptance requires identified "
            "artifacts, devices, and clean implementations.",
            "violations": {
                "albatross_contract": {
                    "implementation_revision": {
                        "required": bench.ALBATROSS_COMMIT,
                        "actual": "wrong-revision",
                    }
                }
            },
        }
    ]


def test_report_blocks_dirty_model_only_implementation_revision() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    contracts = _matching_model_only_contracts()
    contracts["vllm_contract"]["implementation_revision"] = "abc123-dirty"
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
                **contracts,
            }
        },
        cuda_available=True,
    )

    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"] == [
        {
            "code": "invalid_model_only_measurement_provenance",
            "message": "Model-only performance acceptance requires identified "
            "artifacts, devices, and clean implementations.",
            "violations": {
                "vllm_contract": {
                    "implementation_revision": {
                        "required": "clean git revision",
                        "actual": "abc123-dirty",
                    }
                }
            },
        }
    ]


def test_report_blocks_only_missing_vllm_when_albatross_measurement_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 3200.0,
            "albatross_batch_size": 2,
            "albatross_seq_len": 4,
            "albatross_p50_ms": 2.5,
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    model_only_check = report["checks"]["model_only_steady_decode"]
    assert model_only_check["status"] == "blocked"
    assert model_only_check["metrics"]["albatross_tokens_per_s"] == 3200.0
    assert model_only_check["metrics"]["vllm_tokens_per_s"] is None
    assert model_only_check["blockers"] == [
        {
            "code": "missing_vllm_model_only_measurement",
            "message": "Measurement JSON must include vllm_tokens_per_s for "
            "model_only_steady_decode. Albatross model-only measurement is "
            "present; generate vLLM model-only metrics with "
            "--measure-vllm-model-only.",
        }
    ]


def test_cli_writes_albatross_model_only_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    albatross_root = tmp_path / "albatross"
    impl_dir = albatross_root / bench.ALBATROSS_IMPL
    impl_dir.mkdir(parents=True)
    (impl_dir / "rwkv7_fast_v3a.py").write_text("", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_path.write_bytes(b"")
    output_path = tmp_path / "measurement.json"
    calls: list[Any] = []
    device = _matching_model_only_contracts()["albatross_contract"]["device"]

    monkeypatch.setattr(bench, "_model_only_device_identity", lambda: dict(device))
    monkeypatch.setattr(bench, "_git_revision", lambda path: bench.ALBATROSS_COMMIT)

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        cmd = args[0]
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="\n".join(
                [
                    "[rwkv7_fast_v3a] start model=/tmp/checkpoint.pth",
                    "csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50",
                    "csv,rwkv7_fast_v3a,2,4,7,1.250000,2.500000,4.000000,3200.000000",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--albatross-root",
            str(albatross_root),
            "--albatross-checkpoint",
            str(checkpoint_path),
            "--measure-albatross-model-only",
            "--albatross-case",
            "2x4",
            "--albatross-warmup",
            "3",
            "--albatross-iters",
            "7",
            "--albatross-wkv",
            "fp16",
            "--albatross-emb",
            "cpu",
            "--albatross-batched-rkv",
            "off",
            "--albatross-cmix-sparse",
            "no-fc",
            "--albatross-lowrank-weight",
            "both",
            "--albatross-orig-linear-groups",
            "att_c2c,head",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["albatross_tokens_per_s"] == 3200.0
    assert model_only["albatross_batch_size"] == 2
    assert model_only["albatross_seq_len"] == 4
    assert model_only["albatross_warmup"] == 3
    assert model_only["albatross_iters"] == 7
    assert model_only["albatross_p50_ms"] == 2.5
    assert model_only["albatross_label"] == "rwkv7_fast_v3a"
    assert model_only["albatross_contract"] == {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": bench._checkpoint_sha256(checkpoint_path),
        "implementation_revision": bench.ALBATROSS_COMMIT,
        "device": device,
        "batch_size": 2,
        "seq_len": 4,
        "output_boundary": bench.MODEL_ONLY_OUTPUT_BOUNDARY,
        "wkv_mode": "fp16",
        "gemm_accumulation_policy": "fp32",
        "embedding_device": "cpu",
        "rkv_mode": "off",
        "cmix_sparse": "no-fc",
        "low_rank_weight": "both",
        "orig_linear_groups": ["att_c2c", "head"],
    }
    assert measurement["config"]["measurement_source"] == "albatross_subprocess"
    assert measurement["config"]["albatross_wkv"] == "fp16"
    assert measurement["config"]["albatross_emb"] == "cpu"
    assert measurement["config"]["albatross_batched_rkv"] == "off"
    assert measurement["config"]["albatross_cmix_sparse"] == "no-fc"
    assert measurement["config"]["albatross_lowrank_weight"] == "both"
    assert measurement["config"]["albatross_orig_linear_groups"] == "att_c2c,head"
    cmd = calls[0][0][0]
    assert cmd[:2] == [
        sys.executable,
        str(impl_dir / "rwkv7_fast_v3a.py"),
    ]
    assert _command_options(cmd) == {
        "--model": str(checkpoint_path),
        "--wkv": "fp16",
        "--emb": "cpu",
        "--batched-rkv": "off",
        "--cmix-sparse": "no-fc",
        "--lowrank-weight": "both",
        "--warmup": "3",
        "--iters": "7",
        "--cases": "2x4",
        "--orig-linear-groups": "att_c2c,head",
    }
    assert calls[0][1]["cwd"] == impl_dir


def test_cli_writes_vllm_model_only_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "measurement.json"
    calls: list[Any] = []
    fake_model = object()
    device = _matching_model_only_contracts()["vllm_contract"]["device"]

    for name in bench.PROVENANCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        bench,
        "_benchmark_provenance",
        lambda config: {"source": "test"},
    )
    monkeypatch.setattr(bench, "_model_only_device_identity", lambda: dict(device))
    monkeypatch.setattr(bench, "_git_revision", lambda path: "vllm-test-revision")

    def fake_load(config):
        calls.append(("load", config.model))
        return fake_model

    def fake_time(model, *, batch_size, seq_len, warmup, iters, capture_cuda_profiler):
        calls.append(
            (
                "time",
                model,
                batch_size,
                seq_len,
                warmup,
                iters,
                capture_cuda_profiler,
            )
        )
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
            "cuda_profiler_capture": capture_cuda_profiler,
            "distributed_backend": "nccl",
        }

    monkeypatch.setattr(bench, "_load_vllm_rwkv7_model", fake_load)
    monkeypatch.setattr(
        bench,
        "_time_vllm_model_only_steady_decode",
        fake_time,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--model",
            str(model_path),
            "--measure-vllm-model-only",
            "--vllm-case",
            "2x4",
            "--vllm-warmup",
            "3",
            "--vllm-iters",
            "7",
            "--cuda-profiler-capture",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["vllm_tokens_per_s"] == 3040.0
    assert model_only["vllm_batch_size"] == 2
    assert model_only["vllm_seq_len"] == 4
    assert model_only["vllm_warmup"] == 3
    assert model_only["vllm_iters"] == 7
    assert model_only["vllm_p50_ms"] == 2.0
    assert model_only["vllm_label"] == "RWKV7ForCausalLM.forward_logits"
    assert model_only["vllm_output"] == "logits"
    assert model_only["vllm_logits_included"] is True
    assert model_only["vllm_graph"] is True
    assert model_only["vllm_measurement_mode"] == "cuda_graph_replay"
    assert model_only["vllm_distributed_backend"] == "nccl"
    assert model_only["vllm_contract"] == {
        "checkpoint": str(model_path.resolve()),
        "checkpoint_sha256": bench._checkpoint_sha256(model_path),
        "implementation_revision": "vllm-test-revision",
        "device": device,
        "batch_size": 2,
        "seq_len": 4,
        "output_boundary": bench.MODEL_ONLY_OUTPUT_BOUNDARY,
        "wkv_mode": "fp16",
        "gemm_accumulation_policy": "fp16_where_configurable",
        "embedding_device": "gpu",
        "rkv_mode": "off",
        "cmix_sparse": "no-fc",
        "low_rank_weight": "both",
        "orig_linear_groups": [],
    }
    assert measurement["config"]["measurement_source"] == "vllm_model_direct"
    assert measurement["config"]["vllm_provenance"] == {"source": "test"}
    assert calls == [
        ("load", str(model_path)),
        ("time", fake_model, 2, 4, 3, 7, True),
    ]


def test_vllm_model_only_timer_includes_logits_for_albatross_parity(
    monkeypatch,
) -> None:
    import torch

    import vllm.model_executor.models.rwkv7 as rwkv7

    real_arange = torch.arange

    class FakeCudaGraph:
        def replay(self) -> None:
            pass

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            self.enable_timing = enable_timing

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return 1.0

    profiler_calls = []

    class FakeCudart:
        def cudaProfilerStart(self) -> None:
            profiler_calls.append("start")

        def cudaProfilerStop(self) -> None:
            profiler_calls.append("stop")

    class FakeModel:
        vocab_size = 16
        emb_cpu = True
        _benchmark_distributed_backend = "fake"

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def zero_state(self, batch_size):
            self.calls.append(("zero_state", batch_size))
            return object()

        def embed(self, tokens):
            self.calls.append(("embed", tuple(tokens.shape)))
            return tokens.float()

        def forward_from_x(self, x, state, path):
            self.calls.append(("forward_from_x", tuple(x.shape), path))
            return x + 1.0

        def compute_logits(self, hidden_states):
            self.calls.append(("compute_logits", tuple(hidden_states.shape)))
            return hidden_states

    def fake_arange(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return real_arange(*args, **kwargs)

    monkeypatch.setattr(bench, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeCudaGraph)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: FakeCudart())
    monkeypatch.setattr(torch, "arange", fake_arange)
    monkeypatch.setattr(rwkv7, "select_path", lambda batch, seq: "fake-path")

    model = FakeModel()
    parsed = bench._time_vllm_model_only_steady_decode(
        model,
        batch_size=2,
        seq_len=1,
        warmup=0,
        iters=1,
        capture_cuda_profiler=True,
    )

    assert model.calls == [
        ("zero_state", 2),
        ("embed", (2, 1)),
        ("forward_from_x", (2, 1), "fake-path"),
        ("compute_logits", (2, 1)),
    ]
    assert parsed["output"] == "logits"
    assert parsed["logits_included"] is True
    assert parsed["tokens_per_s"] == 2000.0
    assert profiler_calls == ["start", "stop"]


def test_vllm_model_loader_initializes_distributed_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import torch

    import vllm.model_executor.models.rwkv7 as rwkv7
    import vllm.transformers_utils.configs.rwkv7 as rwkv7_config

    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    calls: list[Any] = []
    fake_weight = object()

    def fake_init():
        calls.append("init")
        return "gloo"

    class FakeRWKV7ForCausalLM:
        def __init__(self, *, vllm_config):
            assert calls == ["init"]
            calls.append(("construct", vllm_config.model_config.hf_config))

        def load_weights(self, weights):
            calls.append(("load_weights", list(weights)))

        def eval(self):
            calls.append("eval")
            return self

    monkeypatch.setattr(bench, "_initialize_vllm_single_process_distributed", fake_init)
    monkeypatch.setattr(
        rwkv7_config,
        "build_rwkv7_config_from_pth",
        lambda path: SimpleNamespace(
            hidden_size=64,
            vocab_size=128,
            head_size=64,
            num_hidden_layers=1,
            num_attention_heads=1,
        ),
    )
    monkeypatch.setattr(rwkv7, "RWKV7ForCausalLM", FakeRWKV7ForCausalLM)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {"emb.weight": fake_weight},
    )

    model = bench._load_vllm_rwkv7_model(
        _config(Path(__file__).resolve().parents[2], model=str(model_path))
    )

    assert isinstance(model, FakeRWKV7ForCausalLM)
    assert calls[0] == "init"
    assert calls[1][0] == "construct"
    assert calls[2] == ("load_weights", [("emb.weight", fake_weight)])
    assert calls[3] == "eval"


def test_vllm_single_process_distributed_init_uses_canonical_entrypoints(
    monkeypatch,
) -> None:
    import torch

    import vllm.distributed.parallel_state as parallel_state

    calls: list[Any] = []

    monkeypatch.setattr(bench, "_cuda_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        parallel_state,
        "model_parallel_is_initialized",
        lambda: False,
    )

    def fake_init_distributed_environment(**kwargs):
        calls.append(("init", kwargs))

    def fake_ensure_model_parallel_initialized(*args, **kwargs):
        calls.append(("ensure", args, kwargs))

    monkeypatch.setattr(
        parallel_state,
        "init_distributed_environment",
        fake_init_distributed_environment,
    )
    monkeypatch.setattr(
        parallel_state,
        "ensure_model_parallel_initialized",
        fake_ensure_model_parallel_initialized,
    )

    backend = bench._initialize_vllm_single_process_distributed()

    assert backend == "nccl"
    assert calls[0][0] == "init"
    assert calls[0][1]["world_size"] == 1
    assert calls[0][1]["rank"] == 0
    assert calls[0][1]["local_rank"] == 0
    assert calls[0][1]["backend"] == "nccl"
    distributed_init = urlparse(calls[0][1]["distributed_init_method"])
    assert distributed_init.scheme == "tcp"
    assert distributed_init.hostname == "127.0.0.1"
    assert isinstance(distributed_init.port, int)
    assert calls[1] == ("ensure", (1, 1), {"backend": "nccl"})


def test_cli_merges_vllm_model_only_measurement_with_albatross_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    input_path = tmp_path / "albatross.json"
    output_path = tmp_path / "combined.json"
    contracts = _matching_model_only_contracts(
        checkpoint=str(model_path.resolve()),
        checkpoint_sha256=bench._checkpoint_sha256(model_path),
    )
    albatross_contract = contracts["albatross_contract"]
    device = contracts["vllm_contract"]["device"]
    albatross_contract["gemm_accumulation_policy"] = "fp32"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": bench.SCHEMA_VERSION,
                "benchmark": bench.BENCHMARK_NAME,
                "model_only_steady_decode": {
                    "albatross_tokens_per_s": 3200.0,
                    "albatross_batch_size": 2,
                    "albatross_seq_len": 4,
                    "albatross_p50_ms": 2.5,
                    "albatross_contract": albatross_contract,
                },
                "config": {"measurement_source": "albatross_subprocess"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Any] = []

    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp16")
    monkeypatch.setattr(
        bench,
        "_benchmark_provenance",
        lambda config: {"source": "test"},
    )
    monkeypatch.setattr(bench, "_model_only_device_identity", lambda: dict(device))
    monkeypatch.setattr(bench, "_git_revision", lambda path: "vllm-test-revision")

    monkeypatch.setattr(bench, "_load_vllm_rwkv7_model", lambda config: object())

    def fake_time(model, *, batch_size, seq_len, warmup, iters, capture_cuda_profiler):
        calls.append((batch_size, seq_len, warmup, iters, capture_cuda_profiler))
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
            "cuda_profiler_capture": capture_cuda_profiler,
            "distributed_backend": "gloo",
        }

    monkeypatch.setattr(
        bench,
        "_time_vllm_model_only_steady_decode",
        fake_time,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measurement-json",
            str(input_path),
            "--measure-vllm-model-only",
            "--vllm-warmup",
            "3",
            "--vllm-iters",
            "7",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    model_only = measurement["model_only_steady_decode"]
    assert rc == 0
    assert model_only["albatross_tokens_per_s"] == 3200.0
    assert model_only["vllm_tokens_per_s"] == 3040.0
    assert model_only["vllm_label"] == "RWKV7ForCausalLM.forward_logits"
    assert model_only["vllm_output"] == "logits"
    assert model_only["vllm_logits_included"] is True
    assert model_only["vllm_graph"] is True
    assert model_only["vllm_measurement_mode"] == "cuda_graph_replay"
    assert model_only["vllm_distributed_backend"] == "gloo"
    assert model_only["albatross_contract"] != model_only["vllm_contract"]
    assert measurement["config"]["measurement_source"] == "merged_vllm_model_direct"
    assert measurement["config"]["vllm_distributed_backend"] == "gloo"
    assert measurement["config"]["vllm_provenance"] == {"source": "test"}
    assert calls == [(2, 4, 3, 7, False)]

    report = bench.build_report(
        _config(repo_root),
        measurements=measurement,
        cuda_available=True,
    )
    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "blocked"
    assert check["metrics"]["vllm_to_albatross_ratio"] is None
    assert check["blockers"][0]["code"] == ("non_like_for_like_model_only_comparison")


def test_cli_writes_vllm_runner_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner.json"
    calls: list[Any] = []
    fake_llm = object()

    def fake_create(config):
        calls.append(("create", config.model, config.decode_tokens))
        return fake_llm

    def fake_time(
        llm,
        *,
        batch_size,
        prompt_len,
        prefill_chunk_tokens,
        decode_tokens,
        warmup,
        iters,
    ):
        calls.append(
            (
                "time",
                llm,
                batch_size,
                prompt_len,
                prefill_chunk_tokens,
                decode_tokens,
                warmup,
                iters,
            )
        )
        return {
            "tokens_per_s": 91.0,
            "p10_ms": 1.9,
            "p50_ms": 2.0,
            "p90_ms": 2.4,
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "execute_model_p50_ms": 0.8,
            "execute_model_p50_tokens_per_s": 3750.0,
            "sample_tokens_p50_ms": 0.2,
            "sample_tokens_p50_tokens_per_s": 15000.0,
            "decode_step_p50_ms": 1.0,
            "decode_step_p50_tokens_per_s": 3000.0,
            "postprocess_p50_ms": None,
            "postprocess_timing_available": False,
            "decode_steps": 7,
            "worker_count": 1,
        }

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", fake_create)
    monkeypatch.setattr(bench, "_time_vllm_runner_steady_decode", fake_time)

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "3",
            "--runner-prompt-len",
            "5",
            "--runner-prefill-chunk-tokens",
            "2",
            "--runner-decode-tokens",
            "7",
            "--runner-warmup",
            "2",
            "--runner-iters",
            "11",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    runner = measurement["runner_steady_decode"]
    assert rc == 0
    assert runner["runner_tokens_per_s"] == 91.0
    assert runner["runner_batch_size"] == 3
    assert runner["runner_prompt_len"] == 5
    assert runner["runner_prefill_chunk_tokens"] == 2
    assert runner["runner_decode_tokens"] == 7
    assert runner["runner_warmup"] == 2
    assert runner["runner_warmup_mode"] == "same_request_decode_steps"
    assert runner["runner_warmup_decode_tokens"] == 2
    assert runner["runner_iters"] == 11
    assert runner["runner_p50_ms"] == 2.0
    assert runner["runner_measurement_mode"] == "worker_execute_model"
    assert runner["runner_internal_timing_target"] == "worker.execute_model"
    assert runner["runner_timing_clock"] == "cuda_event"
    assert runner["runner_execute_model_p50_ms"] == 0.8
    assert runner["runner_execute_model_p50_tokens_per_s"] == 3750.0
    assert runner["runner_sample_tokens_p50_ms"] == 0.2
    assert runner["runner_sample_tokens_p50_tokens_per_s"] == 15000.0
    assert runner["runner_decode_step_p50_ms"] == 1.0
    assert runner["runner_decode_step_p50_tokens_per_s"] == 3000.0
    assert runner["runner_postprocess_p50_ms"] is None
    assert runner["runner_postprocess_timing_available"] is False
    assert runner["runner_decode_steps"] == 7
    assert runner["runner_worker_count"] == 1
    assert measurement["config"]["provenance"]["workload"] == {
        "batch_size": 3,
        "prompt_len": 5,
        "warmup_tokens": 16,
        "decode_tokens": 7,
        "runner_prefill_chunk_tokens": 2,
        "runner_enforce_eager": False,
        "runner_cudagraph_capture_sizes": None,
    }
    assert (
        measurement["config"]["measurement_source"]
        == "vllm_runner_worker_execute_model"
    )
    assert calls == [
        ("create", str(model_path), 9),
        ("time", fake_llm, 3, 5, 2, 7, 2, 11),
    ]


def test_cli_writes_split_runner_prefill_decode_measurement_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd.json"
    calls: list[Any] = []

    def fake_generate(
        config,
        *,
        prefill_cases,
        decode_cases,
        prefill_chunk_tokens,
        decode_prompt_len,
        warmup,
        iters,
        include_sampling,
    ):
        calls.append(
            (
                config.model,
                prefill_cases,
                decode_cases,
                prefill_chunk_tokens,
                decode_prompt_len,
                warmup,
                iters,
                include_sampling,
            )
        )
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "runner_prefill": {
                "1x8": {"avg_tokens_per_s": 100.0, "peak_tokens_per_s": 120.0}
            },
            "runner_decode": {
                "4x1": {"avg_tokens_per_s": 90.0, "peak_tokens_per_s": 95.0}
            },
        }

    monkeypatch.setattr(bench, "generate_vllm_runner_pd_measurement", fake_generate)

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner-pd",
            "--runner-pd-prefill-cases",
            "1x8,2x4",
            "--runner-pd-decode-cases",
            "4x1",
            "--runner-prefill-chunk-tokens",
            "8",
            "--runner-pd-decode-prompt-len",
            "3",
            "--runner-warmup",
            "2",
            "--runner-iters",
            "5",
            "--runner-pd-include-sampling",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert measurement["runner_prefill"]["1x8"]["peak_tokens_per_s"] == 120.0
    assert measurement["runner_decode"]["4x1"]["avg_tokens_per_s"] == 90.0
    assert calls == [
        (
            str(model_path),
            [(1, 8), (2, 4)],
            [(4, 1)],
            8,
            3,
            2,
            5,
            True,
        )
    ]


def test_cli_split_runner_defaults_to_albatross_pd_cases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd-defaults.json"
    calls: list[Any] = []

    def fake_generate(
        config,
        *,
        prefill_cases,
        decode_cases,
        prefill_chunk_tokens,
        decode_prompt_len,
        warmup,
        iters,
        include_sampling,
    ):
        calls.append(
            (
                prefill_cases,
                decode_cases,
                prefill_chunk_tokens,
                decode_prompt_len,
                include_sampling,
            )
        )
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "runner_prefill": {},
            "runner_decode": {},
        }

    monkeypatch.setattr(bench, "generate_vllm_runner_pd_measurement", fake_generate)

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner-pd",
            "--measurement-output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert calls == [
        (
            [(1, 1024), (32, 32)],
            [(1024, 1)],
            1024,
            1,
            False,
        )
    ]


def test_cli_split_runner_allows_prefill_only_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd-prefill-only.json"
    calls: list[Any] = []

    def fake_generate(
        config,
        *,
        prefill_cases,
        decode_cases,
        prefill_chunk_tokens,
        decode_prompt_len,
        warmup,
        iters,
        include_sampling,
    ):
        calls.append(
            (
                prefill_cases,
                decode_cases,
                prefill_chunk_tokens,
                decode_prompt_len,
                warmup,
                iters,
                include_sampling,
            )
        )
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "runner_prefill": {},
            "runner_decode": {},
        }

    monkeypatch.setattr(bench, "generate_vllm_runner_pd_measurement", fake_generate)

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner-pd",
            "--runner-pd-prefill-cases",
            "1x1024,32x32",
            "--runner-pd-decode-cases",
            "",
            "--runner-warmup",
            "5",
            "--runner-iters",
            "10",
            "--measurement-output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert calls == [
        (
            [(1, 1024), (32, 32)],
            [],
            1024,
            1,
            5,
            10,
            False,
        )
    ]


def test_cli_split_runner_allows_decode_only_measurement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd-decode-only.json"
    calls: list[Any] = []

    def fake_generate(
        config,
        *,
        prefill_cases,
        decode_cases,
        prefill_chunk_tokens,
        decode_prompt_len,
        warmup,
        iters,
        include_sampling,
    ):
        calls.append(
            (
                prefill_cases,
                decode_cases,
                prefill_chunk_tokens,
                decode_prompt_len,
                warmup,
                iters,
                include_sampling,
            )
        )
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "runner_prefill": {},
            "runner_decode": {},
        }

    monkeypatch.setattr(bench, "generate_vllm_runner_pd_measurement", fake_generate)

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner-pd",
            "--runner-pd-prefill-cases",
            "",
            "--runner-pd-decode-cases",
            "1024x1",
            "--runner-prefill-chunk-tokens",
            "1",
            "--runner-pd-decode-prompt-len",
            "1",
            "--measurement-output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert calls == [([], [(1024, 1)], 1, 1, 1, 3, False)]


def test_cli_split_runner_rejects_empty_prefill_and_decode_cases(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd-empty.json"

    with pytest.raises(ValueError, match="at least one prefill or decode"):
        bench.main(
            [
                "--repo-root",
                str(repo_root),
                "--model",
                str(model_path),
                "--measure-vllm-runner-pd",
                "--runner-pd-prefill-cases",
                "",
                "--runner-pd-decode-cases",
                "",
                "--measurement-output",
                str(output_path),
            ]
        )


def test_cli_split_runner_single_uses_pd_prefill_chunk_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-pd-single.json"
    calls: list[Any] = []

    def fake_generate(
        config,
        *,
        phase,
        case,
        prefill_chunk_tokens,
        decode_prompt_len,
        warmup,
        iters,
        include_sampling,
    ):
        calls.append(
            (
                phase,
                case,
                prefill_chunk_tokens,
                decode_prompt_len,
                warmup,
                iters,
                include_sampling,
            )
        )
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "phase": phase,
            "case": bench._format_bxt_case(*case),
            "metrics": {},
        }

    monkeypatch.setattr(
        bench,
        "generate_vllm_runner_pd_single_measurement",
        fake_generate,
    )

    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner-pd-single",
            "--runner-pd-single-phase",
            "prefill",
            "--runner-pd-single-case",
            "1x1024",
            "--runner-pd-prefill-chunk-tokens",
            "1024",
            "--runner-pd-decode-prompt-len",
            "1",
            "--runner-warmup",
            "3",
            "--runner-iters",
            "10",
            "--measurement-output",
            str(output_path),
        ]
    )

    assert rc == 0
    assert calls == [("prefill", (1, 1024), 1024, 1, 3, 10, False)]


def test_create_vllm_runner_llm_passes_configured_cudagraph_sizes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            captured["model_env_during_init"] = os.environ.get("VLLM_RWKV7_MODEL")

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.rwkv7_ops", ModuleType("vllm.rwkv7_ops"))
    monkeypatch.setenv("VLLM_RWKV7_MODEL", str(model_path))

    config = replace(
        _config(tmp_path, model=str(model_path)),
        batch_size=1024,
        prompt_len=1,
        decode_tokens=4,
        runner_prefill_chunk_tokens=1,
        runner_cudagraph_capture_sizes=(1024,),
    )

    bench._create_vllm_runner_llm(config)

    assert captured["max_num_seqs"] == 1024
    assert captured["max_num_batched_tokens"] == 1024
    assert captured["compilation_config"] == {
        "cudagraph_capture_sizes": [1024],
    }
    assert captured["model_env_during_init"] is None
    assert os.environ["VLLM_RWKV7_MODEL"] == str(model_path)


def test_create_vllm_runner_llm_does_not_graph_capture_eager_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    captured: dict[str, Any] = {}

    class FakeLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.rwkv7_ops", ModuleType("vllm.rwkv7_ops"))

    config = replace(
        _config(tmp_path, model=str(model_path)),
        runner_enforce_eager=True,
        runner_cudagraph_capture_sizes=(1024,),
    )

    bench._create_vllm_runner_llm(config)

    assert "compilation_config" not in captured


def test_runner_pd_single_uses_exact_cudagraph_capture_size(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_configs: list[bench.BenchmarkConfig] = []

    def create_llm(config: bench.BenchmarkConfig) -> object:
        created_configs.append(config)
        return object()

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", create_llm)
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_decode_phase",
        lambda llm, **kwargs: _phase_result(tokens=1024),
    )
    monkeypatch.setattr(bench, "_shutdown_vllm_runner_llm", lambda llm: None)

    measurement = bench.generate_vllm_runner_pd_single_measurement(
        _config(tmp_path, model="/tmp/model.pth"),
        phase="decode",
        case=(1024, 1),
        prefill_chunk_tokens=1024,
        decode_prompt_len=1,
        warmup=3,
        iters=5,
        include_sampling=False,
    )

    assert created_configs[0].runner_cudagraph_capture_sizes == (1024,)
    workload = measurement["config"]["provenance"]["workload"]
    assert workload["runner_cudagraph_capture_sizes"] == [1024]


def test_runner_pd_aggregate_records_case_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_case(config, *, phase, case, prefill_chunk_tokens, **kwargs):
        batch_size, _seq_len = case
        capture_size = max(batch_size * prefill_chunk_tokens, batch_size)
        return {
            "schema_version": bench.SCHEMA_VERSION,
            "benchmark": bench.BENCHMARK_NAME,
            "metrics": _phase_result(tokens=capture_size),
            "config": {
                "provenance": {
                    "workload": {
                        "runner_cudagraph_capture_sizes": [capture_size],
                    }
                }
            },
        }

    monkeypatch.setattr(bench, "_run_vllm_runner_pd_case_subprocess", fake_case)

    measurement = bench.generate_vllm_runner_pd_measurement(
        _config(tmp_path, model="/tmp/model.pth"),
        prefill_cases=[(1, 8)],
        decode_cases=[(4, 1)],
        prefill_chunk_tokens=8,
        decode_prompt_len=1,
        warmup=2,
        iters=3,
        include_sampling=False,
    )

    provenance = measurement["config"]["case_provenance_by_case"]
    assert provenance["prefill:1x8"]["workload"]["runner_cudagraph_capture_sizes"] == [
        8
    ]
    assert provenance["decode:4x1"]["workload"]["runner_cudagraph_capture_sizes"] == [4]


def test_cli_merges_vllm_runner_measurement_with_model_only_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    input_path = tmp_path / "model-only.json"
    output_path = tmp_path / "combined.json"
    input_path.write_text(
        json.dumps(
            {
                "schema_version": bench.SCHEMA_VERSION,
                "benchmark": bench.BENCHMARK_NAME,
                "model_only_steady_decode": {
                    "albatross_tokens_per_s": 100.0,
                    "vllm_tokens_per_s": 96.0,
                    "albatross_batch_size": 2,
                    "albatross_seq_len": 4,
                    **_matching_model_only_contracts(),
                },
                "config": {"measurement_source": "merged_vllm_model_direct"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", lambda config: object())
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_steady_decode",
        lambda llm, **kwargs: {
            "tokens_per_s": 91.0,
            "p10_ms": 1.9,
            "p50_ms": 2.0,
            "p90_ms": 2.4,
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "decode_steps": 4,
            "worker_count": 1,
        },
    )
    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measurement-json",
            str(input_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "2",
            "--runner-prompt-len",
            "4",
            "--runner-decode-tokens",
            "4",
            "--runner-warmup",
            "1",
            "--runner-iters",
            "3",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert measurement["model_only_steady_decode"]["albatross_tokens_per_s"] == 100.0
    assert measurement["model_only_steady_decode"]["vllm_tokens_per_s"] == 96.0
    assert measurement["runner_steady_decode"]["runner_tokens_per_s"] == 91.0
    assert measurement["runner_steady_decode"]["runner_batch_size"] == 2
    assert measurement["runner_steady_decode"]["runner_timing_clock"] == "cuda_event"
    assert (
        measurement["config"]["measurement_source"]
        == "merged_vllm_runner_worker_execute_model"
    )

    report = bench.build_report(
        _config(repo_root),
        measurements=measurement,
        cuda_available=True,
    )
    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    runner_metrics = report["checks"]["runner_steady_decode"]["metrics"]
    assert runner_metrics == {
        "runner_tokens_per_s": 91.0,
        "runner_measurement_mode": "worker_execute_model",
        "runner_internal_timing_target": "worker.execute_model",
        "runner_timing_clock": "cuda_event",
        "runner_execute_model_p50_ms": None,
        "runner_execute_model_p50_tokens_per_s": None,
        "runner_sample_tokens_p50_ms": None,
        "runner_sample_tokens_p50_tokens_per_s": None,
        "runner_decode_step_p50_ms": None,
        "runner_decode_step_p50_tokens_per_s": None,
        "runner_postprocess_p50_ms": None,
        "runner_postprocess_timing_available": False,
    }


def test_runner_check_does_not_compare_worker_timing_to_logits_baseline() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 1000.0,
            "vllm_tokens_per_s": 950.0,
            **_matching_model_only_contracts(),
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 1.0,
            "runner_measurement_mode": "worker_execute_model",
            "runner_internal_timing_target": "worker.execute_model",
        },
        "config": {"provenance": _fp16_runner_provenance()},
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    runner_check = report["checks"]["runner_steady_decode"]
    assert runner_check["status"] == "passed"
    assert runner_check["thresholds"] == {"min_runner_tokens_per_s": 1.0}
    assert runner_check["metrics"] == {
        "runner_tokens_per_s": 1.0,
        "runner_measurement_mode": "worker_execute_model",
        "runner_internal_timing_target": "worker.execute_model",
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


def test_runner_check_blocks_without_precision_provenance() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "runner_steady_decode": {"runner_tokens_per_s": 1.0},
        },
        cuda_available=True,
    )

    check = report["checks"]["runner_steady_decode"]
    assert check["status"] == "blocked"
    assert check["blockers"] == [
        {
            "code": "missing_runner_throughput_provenance",
            "message": "Runner performance acceptance requires retained "
            "precision provenance.",
        }
    ]


def test_runner_check_blocks_non_fp16_throughput_contract() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = bench.build_report(
        _config(repo_root),
        measurements={
            "runner_steady_decode": {"runner_tokens_per_s": 1.0},
            "config": {
                "provenance": {
                    "env": {"VLLM_RWKV7_WKV_MODE": "fp32io16"}
                }
            },
        },
        cuda_available=True,
    )

    check = report["checks"]["runner_steady_decode"]
    assert check["status"] == "blocked"
    assert check["metrics"]["runner_tokens_per_s"] == 1.0
    assert check["blockers"] == [
        {
            "code": "invalid_runner_throughput_contract",
            "message": "Runner performance acceptance requires the FP16 "
            "throughput contract.",
            "violations": {
                "VLLM_RWKV7_WKV_MODE": {
                    "required": "fp16",
                    "actual": "fp32io16",
                },
            },
        }
    ]


def test_cli_writes_blocked_vllm_runner_json_without_fake_tokens(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    output_path = tmp_path / "runner-blocked.json"

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", lambda config: object())
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_steady_decode",
        lambda llm, **kwargs: {
            "measurement_mode": "worker_execute_model",
            "internal_timing_target": "worker.execute_model",
            "blockers": [
                {
                    "code": "missing_internal_runner_decode_samples",
                    "message": "No internal worker decode timing samples were "
                    "recorded.",
                }
            ],
        },
    )
    rc = bench.main(
        [
            "--repo-root",
            str(repo_root),
            "--model",
            str(model_path),
            "--measure-vllm-runner",
            "--runner-batch-size",
            "1",
            "--runner-prompt-len",
            "4",
            "--runner-decode-tokens",
            "2",
            "--runner-warmup",
            "0",
            "--runner-iters",
            "1",
            "--measurement-output",
            str(output_path),
        ]
    )

    measurement = json.loads(output_path.read_text(encoding="utf-8"))
    runner = measurement["runner_steady_decode"]
    assert rc == 0
    assert set(runner) == {
        "runner_batch_size",
        "runner_prompt_len",
        "runner_prefill_chunk_tokens",
        "runner_decode_tokens",
        "runner_warmup",
        "runner_warmup_mode",
        "runner_warmup_decode_tokens",
        "runner_iters",
        "runner_measurement_mode",
        "runner_internal_timing_target",
        "runner_timing_clock",
        "runner_collective_rpc_serialization",
        "blockers",
    }
    assert runner["runner_measurement_mode"] == "worker_execute_model"
    assert runner["runner_batch_size"] == 1
    assert runner["runner_prompt_len"] == 4
    assert runner["runner_prefill_chunk_tokens"] == 4
    assert runner["runner_decode_tokens"] == 2
    assert runner["runner_warmup_mode"] == "same_request_decode_steps"
    assert runner["runner_warmup_decode_tokens"] == 0
    assert runner["blockers"] == [
        {
            "code": "missing_internal_runner_decode_samples",
            "message": "No internal worker decode timing samples were recorded.",
        }
    ]

    report = bench.build_report(
        _config(repo_root),
        measurements={
            **measurement,
            "model_only_steady_decode": {
                "albatross_tokens_per_s": 100.0,
                "vllm_tokens_per_s": 96.0,
                **_matching_model_only_contracts(),
            },
        },
        cuda_available=True,
    )
    assert report["checks"]["runner_steady_decode"]["status"] == "blocked"
    assert report["checks"]["runner_steady_decode"]["blockers"] == [
        {
            "code": "missing_internal_runner_decode_samples",
            "message": "No internal worker decode timing samples were recorded.",
        }
    ]


def test_internal_runner_timing_syncs_once_around_decode_loop(monkeypatch) -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute_model(self, scheduler_output):
            self.calls.append(
                (
                    "execute",
                    scheduler_output.total_num_scheduled_tokens,
                    bool(scheduler_output.finished_req_ids),
                )
            )

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    sync_calls: list[Any] = []
    monkeypatch.setattr(bench, "_worker_cuda_event_pair", lambda: None)
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: sync_calls.append(1))
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_steady_decode(
        worker,
        batch_size=2,
        prompt_len=3,
        prefill_chunk_tokens=2,
        decode_tokens=4,
        iters=1,
        measure=True,
    )

    assert result["decode_steps"] == 4
    assert len(result["iteration_durations_s"]) == 1
    assert len(result["execute_durations_s"]) == 4
    assert len(result["sample_durations_s"]) == 4
    assert len(result["decode_step_durations_s"]) == 4
    assert result["postprocess_durations_s"] == []
    assert result["postprocess_timing_available"] is False
    assert result["timing_clock"] == "wall_clock"
    assert sync_calls == [1] * 17
    assert worker.calls[:2] == [
        ("execute", 4, False),
        ("execute", 2, False),
    ]
    assert [call[0] for call in worker.calls] == [
        "execute",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
    ]


def test_internal_runner_warmup_uses_same_request_before_timing(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, elapsed_ms: float = 1.0) -> None:
            self.elapsed_ms = elapsed_ms

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return self.elapsed_ms

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute_model(self, scheduler_output):
            cached = scheduler_output.scheduled_cached_reqs
            if cached.req_ids:
                self.calls.append(
                    (
                        "execute_cached",
                        cached.num_computed_tokens[0],
                        cached.num_output_tokens[0],
                    )
                )
            else:
                self.calls.append(
                    (
                        "execute_new",
                        scheduler_output.total_num_scheduled_tokens,
                    )
                )

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    monkeypatch.setattr(
        bench,
        "_worker_cuda_event_pair",
        lambda: (FakeEvent(), FakeEvent()),
    )
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: None)
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_steady_decode(
        worker,
        batch_size=1,
        prompt_len=3,
        prefill_chunk_tokens=3,
        decode_tokens=3,
        iters=1,
        measure=True,
        warmup_decode_tokens=2,
    )

    assert result["decode_steps"] == 3
    assert result["warmup_decode_steps"] == 2
    assert result["iteration_durations_s"] == [0.006]
    assert result["execute_durations_s"] == [0.001] * 3
    assert result["sample_durations_s"] == [0.001] * 3
    assert worker.calls == [
        ("execute_new", 3),
        ("sample", None),
        ("execute_cached", 3, 1),
        ("sample", None),
        ("execute_cached", 4, 2),
        ("sample", None),
        ("execute_cached", 5, 3),
        ("sample", None),
        ("execute_cached", 6, 4),
        ("sample", None),
        ("execute_cached", 7, 5),
        ("sample", None),
        ("execute_new", 0),
    ]


def test_phase_throughput_summary_reports_average_and_peak() -> None:
    summary = bench._phase_throughput_summary(
        total_tokens=40,
        iteration_durations_s=[0.010, 0.030],
        unit_durations_s=[0.010, 0.020, 0.010],
        unit_tokens=[10, 10, 20],
    )

    assert summary["avg_tokens_per_s"] == pytest.approx(1000.0)
    assert summary["peak_tokens_per_s"] == pytest.approx(2000.0)
    assert summary["peak_iteration_tokens_per_s"] == pytest.approx(2000.0)
    assert summary["peak_unit_tokens_per_s"] == pytest.approx(2000.0)
    assert summary["total_duration_ms"] == pytest.approx(40.0)
    assert summary["p50_ms"] == pytest.approx(10.0)
    assert summary["unit_p50_ms"] == pytest.approx(10.0)


def test_internal_runner_decode_only_excludes_sampling_by_default(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, elapsed_ms: float = 1.0) -> None:
            self.elapsed_ms = elapsed_ms

        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return self.elapsed_ms

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def execute_model(self, scheduler_output):
            self.calls.append(
                (
                    "execute",
                    scheduler_output.total_num_scheduled_tokens,
                    len(scheduler_output.scheduled_new_reqs),
                    list(scheduler_output.scheduled_cached_reqs.req_ids),
                    bool(scheduler_output.finished_req_ids),
                )
            )

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    monkeypatch.setattr(
        bench,
        "_worker_cuda_event_pair",
        lambda: (FakeEvent(), FakeEvent()),
    )
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: None)
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_decode_only(
        worker,
        batch_size=2,
        prompt_len=1,
        prefill_chunk_tokens=1,
        decode_tokens=2,
        warmup_decode_tokens=1,
        iters=1,
        include_sampling=False,
    )

    assert result["tokens"] == 4
    assert result["iteration_durations_s"] == [0.002]
    assert result["unit_durations_s"] == [0.001, 0.001]
    assert result["unit_tokens"] == [2, 2]
    assert result["sample_durations_s"] == []
    assert result["internal_timing_target"] == "worker.execute_model.decode"
    assert worker.calls == [
        ("execute", 0, 2, [], False),
        ("execute", 2, 0, worker.calls[1][3], False),
        ("execute", 2, 0, worker.calls[2][3], False),
        ("execute", 2, 0, worker.calls[3][3], False),
        ("execute", 0, 0, [], True),
    ]
    assert all(len(call[3]) == 2 for call in worker.calls[1:4])


def test_internal_runner_prefill_warmup_is_unmeasured(monkeypatch) -> None:
    class FakeEvent:
        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, other) -> float:
            return 1.0

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, ...]] = []

        def execute_model(self, scheduler_output):
            self.calls.append(
                (
                    "execute",
                    scheduler_output.total_num_scheduled_tokens,
                    len(scheduler_output.scheduled_new_reqs),
                    list(scheduler_output.scheduled_cached_reqs.req_ids),
                    bool(scheduler_output.finished_req_ids),
                )
            )

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    sync_calls: list[Any] = []
    monkeypatch.setattr(
        bench,
        "_worker_cuda_event_pair",
        lambda: (FakeEvent(), FakeEvent()),
    )
    monkeypatch.setattr(
        bench,
        "_worker_cuda_synchronize",
        lambda: sync_calls.append(1),
    )
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_prefill(
        worker,
        batch_size=2,
        prompt_len=4,
        prefill_chunk_tokens=4,
        warmup=1,
        iters=2,
    )

    assert result["tokens"] == 16
    assert result["warmup_iterations"] == 1
    assert result["iteration_durations_s"] == [0.001, 0.001]
    assert result["unit_durations_s"] == [0.001, 0.001]
    assert result["unit_tokens"] == [8, 8]
    assert sync_calls == [1]
    assert worker.calls == [
        ("execute", 8, 2, [], False),
        ("execute", 0, 0, [], True),
        ("execute", 8, 2, [], False),
        ("execute", 0, 0, [], True),
        ("execute", 8, 2, [], False),
        ("execute", 0, 0, [], True),
    ]


def test_runner_pd_subprocess_skips_generic_warmup_without_slow_path_flags(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "rwkv7-g1d-0.1b-20260129-ctx8192.pth"
    model_path.write_bytes(b"")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, *, cwd, env, check):
        calls.append((command, env))
        options = _command_options(command)
        output_value = options["--measurement-output"]
        assert isinstance(output_value, str)
        output_path = Path(output_value)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": bench.SCHEMA_VERSION,
                    "benchmark": bench.BENCHMARK_NAME,
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv("VLLM_USE_RAPID_SAMPLER", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    measurement = bench._run_vllm_runner_pd_case_subprocess(
        _config(repo_root, model=str(model_path)),
        phase="decode",
        case=(4, 1),
        prefill_chunk_tokens=1,
        decode_prompt_len=1,
        warmup=0,
        iters=1,
        include_sampling=False,
    )

    assert measurement["benchmark"] == bench.BENCHMARK_NAME
    command, env = calls[0]
    options = _command_options(command)
    assert "--runner-enforce-eager" not in options
    assert "--runner-disable-rapid-sampler" not in options
    assert "VLLM_USE_RAPID_SAMPLER" not in env
    assert "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP" not in env


def test_v2_kernel_warmup_skip_is_fixed_for_rwkv7() -> None:
    from vllm.v1.worker.gpu.warmup import should_skip_v2_kernel_warmup

    class RWKV7ModelState:
        pass

    class DefaultModelState:
        pass

    assert should_skip_v2_kernel_warmup(SimpleNamespace(model_state=RWKV7ModelState()))
    assert not should_skip_v2_kernel_warmup(
        SimpleNamespace(model_state=DefaultModelState())
    )


def test_finish_execute_without_sampling_postprocesses_rwkv_state() -> None:
    calls: list[tuple[Any, int, Any]] = []

    class FakeInputBatch:
        idx_mapping = "idx"

    class FakeExecuteModelState:
        input_batch = FakeInputBatch()

    class FakeModelState:
        def postprocess_state(
            self, idx_mapping: Any, num_sampled: int, num_computed_tokens: Any
        ) -> None:
            calls.append((idx_mapping, num_sampled, num_computed_tokens))

    class FakeNumComputedTokens:
        gpu = "num_computed_gpu"

    class FakeReqStates:
        num_computed_tokens = FakeNumComputedTokens()

    class FakeModelRunner:
        def __init__(self) -> None:
            self.execute_model_state = FakeExecuteModelState()
            self.model_state = FakeModelState()
            self.req_states = FakeReqStates()

    class FakeWorker:
        def __init__(self) -> None:
            self.model_runner = FakeModelRunner()

    worker = FakeWorker()

    bench._worker_finish_execute_without_sampling(worker)

    assert calls == [("idx", 0, "num_computed_gpu")]
    assert worker.model_runner.execute_model_state is None


def test_shutdown_vllm_runner_llm_uses_engine_core_shutdown(monkeypatch) -> None:
    calls: list[float | None] = []

    class FakeEngineCore:
        def shutdown(self, timeout: float | None = None) -> None:
            calls.append(timeout)

    class FakeEngine:
        def __init__(self) -> None:
            self.engine_core = FakeEngineCore()

    class FakeLLM:
        def __init__(self) -> None:
            self.llm_engine = FakeEngine()

    monkeypatch.setattr(bench, "_cuda_available", lambda: False)
    llm = FakeLLM()

    bench._shutdown_vllm_runner_llm(llm)

    assert calls == [30]
    assert llm.llm_engine is None


def test_internal_runner_uses_cuda_event_timing_when_available(monkeypatch) -> None:
    class FakeEvent:
        def __init__(self, elapsed_ms: float = 2.0) -> None:
            self.elapsed_ms = elapsed_ms
            self.records = 0
            self.synchronizes = 0

        def record(self) -> None:
            self.records += 1

        def synchronize(self) -> None:
            self.synchronizes += 1

        def elapsed_time(self, other) -> float:
            return self.elapsed_ms

    class FakeWorker:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute_model(self, scheduler_output):
            self.calls.append(("execute", scheduler_output.total_num_scheduled_tokens))

        def sample_tokens(self, grammar_output):
            self.calls.append(("sample", grammar_output))

    start_event = FakeEvent()
    end_event = FakeEvent()
    sync_calls: list[Any] = []
    monkeypatch.setattr(
        bench,
        "_worker_cuda_event_pair",
        lambda: (start_event, end_event),
    )
    monkeypatch.setattr(bench, "_worker_cuda_synchronize", lambda: sync_calls.append(1))
    worker = FakeWorker()

    result = bench._run_vllm_worker_internal_steady_decode(
        worker,
        batch_size=2,
        prompt_len=3,
        prefill_chunk_tokens=2,
        decode_tokens=4,
        iters=1,
        measure=True,
    )

    assert result["timing_clock"] == "cuda_event"
    assert result["iteration_durations_s"] == [0.016]
    assert result["execute_durations_s"] == [0.002] * 4
    assert result["sample_durations_s"] == [0.002] * 4
    assert result["decode_step_durations_s"] == [0.004] * 4
    assert result["postprocess_durations_s"] == []
    assert result["postprocess_timing_available"] is False
    assert start_event.records == 8
    assert end_event.records == 8
    assert end_event.synchronizes == 8
    assert sync_calls == [1]
    assert worker.calls[:2] == [
        ("execute", 4),
        ("execute", 2),
    ]
    assert [call[0] for call in worker.calls] == [
        "execute",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
        "sample",
        "execute",
    ]


def test_internal_runner_merge_reports_component_timings() -> None:
    result = bench._merge_worker_internal_runner_results(
        [
            {
                "iteration_durations_s": [0.012],
                "execute_durations_s": [0.003, 0.004],
                "sample_durations_s": [0.001, 0.002],
                "decode_step_durations_s": [0.004, 0.006],
                "decode_steps": 2,
                "timing_clock": "cuda_event",
            }
        ],
        batch_size=2,
        decode_tokens=2,
        iters=1,
    )

    assert result["tokens_per_s"] == pytest.approx(333.3333333333333)
    assert result["execute_model_p50_ms"] == pytest.approx(3.0)
    assert result["execute_model_p50_tokens_per_s"] == pytest.approx(666.6666666666666)
    assert result["sample_tokens_p50_ms"] == pytest.approx(1.0)
    assert result["sample_tokens_p50_tokens_per_s"] == pytest.approx(2000.0)
    assert result["decode_step_p50_ms"] == pytest.approx(4.0)
    assert result["decode_step_p50_tokens_per_s"] == pytest.approx(500.0)
    assert result["postprocess_p50_ms"] is None
    assert result["postprocess_timing_available"] is False


def test_report_blocks_when_runner_measurement_is_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
            **_matching_model_only_contracts(),
        },
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "blocked"
    assert report["checks"]["runner_steady_decode"]["blockers"] == [
        {
            "code": "missing_runner_measurement",
            "message": "Measurement JSON must include runner_tokens_per_s for "
            "runner_steady_decode.",
        }
    ]


def test_report_blocks_without_measurement_json_when_runtime_paths_exist(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    albatross_root = tmp_path / "albatross"
    impl_dir = albatross_root / bench.ALBATROSS_IMPL
    impl_dir.mkdir(parents=True)
    model_path = tmp_path / "model.pth"
    checkpoint_path = tmp_path / "checkpoint.pth"
    model_path.write_bytes(b"")
    checkpoint_path.write_bytes(b"")

    report = bench.build_report(
        _config(
            repo_root,
            model=str(model_path),
            albatross_root=albatross_root,
            albatross_checkpoint=checkpoint_path,
        ),
        cuda_available=True,
    )

    assert report["overall_status"] == "blocked"
    for check_name in (
        "model_only_steady_decode",
        "runner_steady_decode",
    ):
        assert report["checks"][check_name]["blockers"] == [
            {
                "code": "missing_measurement_json",
                "message": "Provide --measurement-json with RWKV7 faster3a "
                "benchmark metrics, or run the measurement lane first.",
            }
        ]


def test_report_fails_low_model_only() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 80.0,
            **_matching_model_only_contracts(),
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 70.0,
        },
        "config": {"provenance": _fp16_runner_provenance()},
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["overall_status"] == "failed"
    assert report["checks"]["model_only_steady_decode"]["status"] == "failed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"


def test_cli_writes_structured_blocked_json(tmp_path: Path) -> None:
    output_path = tmp_path / "report.json"
    rc = bench.main(
        [
            "--repo-root",
            str(Path(__file__).resolve().parents[2]),
            "--model",
            str(tmp_path / "missing-model"),
            "--albatross-root",
            str(tmp_path / "missing-albatross"),
            "--albatross-checkpoint",
            str(tmp_path / "missing.pth"),
            "--output",
            str(output_path),
        ]
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert report["benchmark"] == "rwkv7_faster3a"
    assert report["overall_status"] == "blocked"
    assert set(report["checks"]) == {
        "model_only_steady_decode",
        "runner_steady_decode",
    }


def test_script_entrypoint_writes_structured_blocked_json(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    output_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/rwkv7/benchmark_faster3a.py",
            "--repo-root",
            str(repo_root),
            "--model",
            str(tmp_path / "missing-model"),
            "--albatross-root",
            str(tmp_path / "missing-albatross"),
            "--albatross-checkpoint",
            str(tmp_path / "missing.pth"),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["benchmark"] == "rwkv7_faster3a"
    assert report["overall_status"] == "blocked"
