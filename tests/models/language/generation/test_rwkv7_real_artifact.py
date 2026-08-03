# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the vLLM serving boundary against the real tokenizer-free RWKV7 artifact."""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
import torch

from vllm import LLM, SamplingParams


CHECKPOINT_SHA256 = "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
FIXED_PROMPT_TOKEN_IDS = [1, 7, 11, 3]
EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"

requires_real_artifact = pytest.mark.skipif(
    os.getenv("RWKV7_RUN_REAL_ARTIFACT") != "1",
    reason="set RWKV7_RUN_REAL_ARTIFACT=1 in the real-checkpoint vLLM job",
)


def _quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _artifact_metadata(artifact: Path) -> dict[str, Any]:
    conversion = json.loads((artifact / "rwkv7_conversion.json").read_text(encoding="utf-8"))
    assert conversion["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert conversion["tokenizer_files"] == {}
    assert conversion["wkv_provider"] == "flash_rwkv"
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "rwkv7"
    assert config["bos_token_id"] == config["eos_token_id"] == config["pad_token_id"] == 0
    assert "auto_map" not in config
    return {"config": config, "conversion": conversion}


@requires_real_artifact
def test_real_rwkv7_artifact_uses_packed_recurrent_serving_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = Path(os.environ["RWKV7_REAL_ARTIFACT"]).expanduser().resolve()
    evidence_path = Path(os.environ["RWKV7_REAL_EVIDENCE"]).expanduser().resolve()
    assert artifact.is_dir(), artifact
    metadata = _artifact_metadata(artifact)
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME

    from fla.ops.rwkv7 import get_last_rwkv7_provider
    from vllm.model_executor.models import rwkv7 as vllm_rwkv7
    from vllm.transformers_utils.rwkv7_provenance import (
        validate_transformers_rwkv7_runtime_provenance,
    )

    provenance = validate_transformers_rwkv7_runtime_provenance()
    forbidden_calls: list[str] = []

    def forbid(name: str):
        def forbidden(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"real vLLM RWKV7 path called forbidden {name}")

        return forbidden

    import flash_rwkv
    import flash_rwkv.ops as flash_rwkv_ops
    import flash_rwkv.reference as flash_rwkv_reference

    for module in (flash_rwkv, flash_rwkv_ops, flash_rwkv_reference):
        monkeypatch.setattr(module, "rwkv7_reference", forbid("reference"), raising=False)
        monkeypatch.setattr(module, "infer_chunk_bf16_forward", forbid("chunk"), raising=False)
        monkeypatch.setattr(module, "infer_chunk_bf16_forward_varlen", forbid("chunk-varlen"), raising=False)

    real_recurrent = vllm_rwkv7.run_fla_rwkv7_recurrent_from_decay_logits
    recurrent_calls: list[dict[str, Any]] = []

    def traced_recurrent(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        state_pool = kwargs["state_pool"]
        state_indices = kwargs["state_indices"].to(dtype=torch.long)
        selected_before = state_pool.index_select(0, state_indices).float().square().sum().item()
        output = real_recurrent(*args, **kwargs)
        selected_after = state_pool.index_select(0, state_indices).float().square().sum().item()
        recurrent_calls.append(
            {
                "after_state_norm_squared": selected_after,
                "before_state_norm_squared": selected_before,
                "cu_seqlens": kwargs["cu_seqlens"].tolist(),
                "mode": kwargs["mode"],
                "state_pointer": state_pool.data_ptr(),
                "state_indices": state_indices.tolist(),
            }
        )
        return output

    monkeypatch.setattr(vllm_rwkv7, "run_fla_rwkv7_recurrent_from_decay_logits", traced_recurrent)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_RAPID_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")

    llm = None
    timings_ms = []
    generated_token_ids = None
    try:
        llm = LLM(
            model=str(artifact),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="float16",
            max_model_len=16,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            gpu_memory_utilization=0.15,
            disable_log_stats=True,
        )
        sampling = SamplingParams(temperature=1.0, top_k=1, max_tokens=4, ignore_eos=True)
        request = [{"prompt_token_ids": FIXED_PROMPT_TOKEN_IDS}]
        for _ in range(2):
            llm.generate(request, sampling, use_tqdm=False)
        for _ in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(request, sampling, use_tqdm=False)
            torch.cuda.synchronize()
            timings_ms.append((time.perf_counter() - started) * 1000.0)
        assert len(outputs) == 1
        assert outputs[0].prompt_token_ids == [0, *FIXED_PROMPT_TOKEN_IDS]
        assert len(outputs[0].outputs) == 1
        generated_token_ids = list(outputs[0].outputs[0].token_ids)
        assert len(generated_token_ids) == 4
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()

    assert forbidden_calls == []
    assert recurrent_calls
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert all(call["mode"] == "fp32io16" for call in recurrent_calls)
    assert all(len(call["state_indices"]) == len(set(call["state_indices"])) for call in recurrent_calls)
    assert recurrent_calls[0]["before_state_norm_squared"] == 0.0
    assert recurrent_calls[0]["after_state_norm_squared"] > 0.0
    assert all(call["state_pointer"] == recurrent_calls[0]["state_pointer"] for call in recurrent_calls)

    evidence = {
        "artifact": str(artifact),
        "artifact_metadata": metadata,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dependencies": provenance,
        "fixed_prompt_token_ids": FIXED_PROMPT_TOKEN_IDS,
        "generated_token_ids": generated_token_ids,
        "gpu": torch.cuda.get_device_name(0),
        "measurement": {
            "clock": "host perf_counter with CUDA synchronize before and after each vLLM request",
            "p10_ms": _quantile(timings_ms, 0.1),
            "p50_ms": _quantile(timings_ms, 0.5),
            "p90_ms": _quantile(timings_ms, 0.9),
            "samples_ms": timings_ms,
            "scope": "vLLM offline engine request, including scheduler and sampler; not a server HTTP boundary",
            "warmup": 2,
            "iterations": len(timings_ms),
        },
        "provider": get_last_rwkv7_provider(),
        "recurrent_calls": recurrent_calls,
        "status": "passed",
        "wkv_mode": "fp32io16",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("RWKV7_REAL_ARTIFACT_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
