# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
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


def _state_movement(**overrides: int) -> dict[str, int]:
    stats = {name: 0 for name in bench.STATE_MOVEMENT_COUNTERS}
    stats.update(overrides)
    return stats


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


def _empty_state_movement() -> dict[str, int | None]:
    return {name: None for name in bench.STATE_MOVEMENT_COUNTERS}


def test_git_revision_reads_remote_source_marker(tmp_path: Path) -> None:
    marker = tmp_path / ".helicopter-source-revision"
    marker.write_text("abc123-dirty\n", encoding="utf-8")

    assert bench._git_revision(tmp_path) == "abc123-dirty"


def test_report_blocks_without_runtime_paths_and_records_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
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
    assert report["checks"]["state_movement"]["metrics"] == _empty_state_movement()
    provenance = report["config"]["provenance"]
    assert provenance["git_revision"]
    assert provenance["workload"] == {
        "batch_size": 16,
        "prompt_len": 128,
        "warmup_tokens": 16,
        "decode_tokens": 128,
        "runner_prefill_chunk_tokens": bench.DEFAULT_RUNNER_PREFILL_CHUNK_TOKENS,
        "runner_enforce_eager": False,
        "runner_disable_rapid_sampler": False,
        "runner_cudagraph_capture_sizes": None,
    }
    assert provenance["sampling"] == bench.VLLM_RUNNER_SAMPLING
    assert provenance["cuda"]["available"] is False
    assert set(provenance["env"]) == set(bench.PROVENANCE_ENV_VARS)


def test_report_evaluates_passing_measurements() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 91.0,
        },
        "state_movement": _state_movement(),
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=False,
    )

    assert report["overall_status"] == "passed"
    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    assert report["checks"]["state_movement"]["status"] == "passed"
    assert report["checks"]["state_movement"]["metrics"] == _state_movement()


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
    assert measurement["config"]["measurement_source"] == "albatross_subprocess"
    cmd = calls[0][0][0]
    assert cmd[:2] == [
        sys.executable,
        str(impl_dir / "rwkv7_fast_v3a.py"),
    ]
    assert _command_options(cmd) == {
        "--model": str(checkpoint_path),
        "--warmup": "3",
        "--iters": "7",
        "--cases": "2x4",
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

    def fake_load(config):
        calls.append(("load", config.model))
        return fake_model

    def fake_time(model, *, batch_size, seq_len, warmup, iters):
        calls.append(("time", model, batch_size, seq_len, warmup, iters))
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
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
    assert measurement["config"]["measurement_source"] == "vllm_model_direct"
    assert calls == [
        ("load", str(model_path)),
        ("time", fake_model, 2, 4, 3, 7),
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
    monkeypatch.setattr(torch, "arange", fake_arange)
    monkeypatch.setattr(rwkv7, "select_path", lambda batch, seq: "fake-path")

    model = FakeModel()
    parsed = bench._time_vllm_model_only_steady_decode(
        model,
        batch_size=2,
        seq_len=1,
        warmup=0,
        iters=1,
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
                },
                "config": {"measurement_source": "albatross_subprocess"},
            }
        ),
        encoding="utf-8",
    )
    calls: list[Any] = []

    monkeypatch.setattr(bench, "_load_vllm_rwkv7_model", lambda config: object())

    def fake_time(model, *, batch_size, seq_len, warmup, iters):
        calls.append((batch_size, seq_len, warmup, iters))
        return {
            "tokens_per_s": 3040.0,
            "p10_ms": 1.75,
            "p50_ms": 2.0,
            "p90_ms": 2.5,
            "graph": True,
            "measurement_mode": "cuda_graph_replay",
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
    assert measurement["config"]["measurement_source"] == "merged_vllm_model_direct"
    assert calls == [(2, 4, 3, 7)]

    report = bench.build_report(
        _config(repo_root),
        measurements=measurement,
        cuda_available=True,
    )
    check = report["checks"]["model_only_steady_decode"]
    assert check["status"] == "passed"
    assert check["metrics"]["vllm_to_albatross_ratio"] == 0.95


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
            "sample_tokens_p50_ms": 0.2,
            "decode_step_p50_ms": 1.0,
            "postprocess_p50_ms": None,
            "postprocess_timing_available": False,
            "decode_steps": 7,
            "worker_count": 1,
        }

    def fake_state_stats(llm):
        calls.append(("state", llm))
        return _state_movement(
            decode_compactions=1,
            decode_compaction_rows=2,
        )

    monkeypatch.setattr(bench, "_create_vllm_runner_llm", fake_create)
    monkeypatch.setattr(bench, "_time_vllm_runner_steady_decode", fake_time)
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        fake_state_stats,
    )

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
    assert runner["runner_sample_tokens_p50_ms"] == 0.2
    assert runner["runner_decode_step_p50_ms"] == 1.0
    assert runner["runner_postprocess_p50_ms"] is None
    assert runner["runner_postprocess_timing_available"] is False
    assert runner["runner_decode_steps"] == 7
    assert runner["runner_worker_count"] == 1
    assert measurement["state_movement"] == _state_movement(
        decode_compactions=1,
        decode_compaction_rows=2,
    )
    assert measurement["config"]["provenance"]["workload"] == {
        "batch_size": 3,
        "prompt_len": 5,
        "warmup_tokens": 16,
        "decode_tokens": 7,
        "runner_prefill_chunk_tokens": 2,
        "runner_enforce_eager": False,
        "runner_disable_rapid_sampler": False,
        "runner_cudagraph_capture_sizes": None,
    }
    assert (
        measurement["config"]["measurement_source"]
        == "vllm_runner_worker_execute_model"
    )
    assert calls == [
        ("create", str(model_path), 9),
        ("time", fake_llm, 3, 5, 2, 7, 2, 11),
        ("state", fake_llm),
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

    fake_vllm = ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.rwkv7_ops", ModuleType("vllm.rwkv7_ops"))

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
    fake_vllm.LLM = FakeLLM
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

    monkeypatch.setattr(
        bench,
        "_create_vllm_runner_llm",
        lambda config: created_configs.append(config) or object(),
    )
    monkeypatch.setattr(
        bench,
        "_time_vllm_runner_decode_phase",
        lambda llm, **kwargs: _phase_result(tokens=1024),
    )
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        lambda llm: _state_movement(),
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
            "state_movement": _state_movement(),
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
    assert provenance["prefill:1x8"]["workload"][
        "runner_cudagraph_capture_sizes"
    ] == [8]
    assert provenance["decode:4x1"]["workload"][
        "runner_cudagraph_capture_sizes"
    ] == [4]


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
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        lambda llm: _state_movement(),
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
    assert measurement["state_movement"]["resident_to_decode_copies"] == 0
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
    assert report["checks"]["state_movement"]["status"] == "passed"
    runner_metrics = report["checks"]["runner_steady_decode"]["metrics"]
    assert runner_metrics == {
        "runner_tokens_per_s": 91.0,
        "runner_measurement_mode": "worker_execute_model",
        "runner_internal_timing_target": "worker.execute_model",
        "runner_timing_clock": "cuda_event",
        "runner_execute_model_p50_ms": None,
        "runner_sample_tokens_p50_ms": None,
        "runner_decode_step_p50_ms": None,
        "runner_postprocess_p50_ms": None,
        "runner_postprocess_timing_available": False,
    }


def test_runner_check_does_not_compare_worker_timing_to_logits_baseline() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 1000.0,
            "vllm_tokens_per_s": 950.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 1.0,
            "runner_measurement_mode": "worker_execute_model",
            "runner_internal_timing_target": "worker.execute_model",
        },
        "state_movement": _state_movement(),
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
        "runner_sample_tokens_p50_ms": None,
        "runner_decode_step_p50_ms": None,
        "runner_postprocess_p50_ms": None,
        "runner_postprocess_timing_available": None,
    }


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
    monkeypatch.setattr(
        bench,
        "_extract_runner_state_movement_stats",
        lambda llm: _state_movement(),
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
    assert report["checks"]["state_movement"]["status"] == "passed"


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
        output_path = Path(options["--measurement-output"])
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": bench.SCHEMA_VERSION,
                    "benchmark": bench.BENCHMARK_NAME,
                    "metrics": {},
                    "state_movement": {},
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
    assert env["VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP"] == "1"


def test_runner_pd_subprocess_honors_explicit_slow_path_flags(
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
        output_path = Path(options["--measurement-output"])
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": bench.SCHEMA_VERSION,
                    "benchmark": bench.BENCHMARK_NAME,
                    "metrics": {},
                    "state_movement": {},
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = replace(
        _config(repo_root, model=str(model_path)),
        runner_enforce_eager=True,
        runner_disable_rapid_sampler=True,
    )

    bench._run_vllm_runner_pd_case_subprocess(
        config,
        phase="decode",
        case=(4, 1),
        prefill_chunk_tokens=1,
        decode_prompt_len=1,
        warmup=0,
        iters=1,
        include_sampling=False,
    )

    command, env = calls[0]
    options = _command_options(command)
    assert options["--runner-enforce-eager"] is True
    assert options["--runner-disable-rapid-sampler"] is True
    assert env["VLLM_USE_RAPID_SAMPLER"] == "0"
    assert env["VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP"] == "1"


def test_v2_kernel_warmup_skip_only_applies_to_rwkv7(monkeypatch) -> None:
    import vllm.envs as envs
    from vllm.v1.worker.gpu.warmup import should_skip_v2_kernel_warmup

    class RWKV7ModelState:
        pass

    class DefaultModelState:
        pass

    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP",
        lambda: True,
    )
    assert should_skip_v2_kernel_warmup(
        SimpleNamespace(model_state=RWKV7ModelState())
    )
    assert not should_skip_v2_kernel_warmup(
        SimpleNamespace(model_state=DefaultModelState())
    )

    monkeypatch.setitem(
        envs.environment_variables,
        "VLLM_RWKV7_SKIP_V2_KERNEL_WARMUP",
        lambda: False,
    )
    assert not should_skip_v2_kernel_warmup(
        SimpleNamespace(model_state=RWKV7ModelState())
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
    assert result["sample_tokens_p50_ms"] == pytest.approx(1.0)
    assert result["decode_step_p50_ms"] == pytest.approx(4.0)
    assert result["postprocess_p50_ms"] is None
    assert result["postprocess_timing_available"] is False


def test_report_blocks_when_runner_measurement_is_missing() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        },
        "state_movement": _state_movement(),
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
    assert report["checks"]["state_movement"]["status"] == "passed"


def test_extract_runner_state_movement_stats_uses_collective_rpc() -> None:
    class FakeLLM:
        def __init__(self) -> None:
            self.calls: list[Any] = []

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            self.calls.append((method, timeout, args, kwargs))
            return [
                _state_movement(
                    resident_to_decode_copies=1,
                    decode_compactions=2,
                    decode_compaction_rows=3,
                )
            ]

    llm = FakeLLM()

    stats = bench._extract_runner_state_movement_stats(llm)

    assert stats == _state_movement(
        resident_to_decode_copies=1,
        decode_compactions=2,
        decode_compaction_rows=3,
    )
    assert callable(llm.calls[0][0])


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
        "state_movement",
    ):
        assert report["checks"][check_name]["blockers"] == [
            {
                "code": "missing_measurement_json",
                "message": "Provide --measurement-json with RWKV7 faster3a "
                "benchmark metrics, or run the measurement lane first.",
            }
        ]


def test_report_fails_low_model_only_and_resident_decode_copies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 80.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 70.0,
        },
        "state_movement": _state_movement(resident_to_decode_copies=1),
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["overall_status"] == "failed"
    assert report["checks"]["model_only_steady_decode"]["status"] == "failed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    assert report["checks"]["state_movement"]["status"] == "failed"


def test_report_fails_decode_compactions_and_full_row_copies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    measurements = {
        "model_only_steady_decode": {
            "albatross_tokens_per_s": 100.0,
            "vllm_tokens_per_s": 96.0,
        },
        "runner_steady_decode": {
            "runner_tokens_per_s": 70.0,
        },
        "state_movement": _state_movement(
            decode_compactions=1,
            decode_full_row_copies=2,
        ),
    }

    report = bench.build_report(
        _config(repo_root),
        measurements=measurements,
        cuda_available=True,
    )

    assert report["checks"]["model_only_steady_decode"]["status"] == "passed"
    assert report["checks"]["runner_steady_decode"]["status"] == "passed"
    state_check = report["checks"]["state_movement"]
    assert state_check["status"] == "failed"
    assert state_check["errors"] == [
        "steady decode recurrent-state compactions must remain zero",
        "steady decode full recurrent-state row copies must remain zero",
    ]


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
        "state_movement",
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
