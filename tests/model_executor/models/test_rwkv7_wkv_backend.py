# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from packaging.requirements import Requirement

from vllm.model_executor.models import rwkv7_wkv_backend


def test_vllm_fused_ops_use_a_provider_distinct_namespace() -> None:
    source_path = (
        Path(__file__).parents[3]
        / "csrc"
        / "libtorch_stable"
        / "rwkv7"
        / "rwkv7_fast_ops_fp16.cpp"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "TORCH_LIBRARY(vllm_rwkv7_fast_ops_fp16, m)" in source
    assert "TORCH_LIBRARY_IMPL(vllm_rwkv7_fast_ops_fp16, CUDA, m)" in source
    assert "TORCH_LIBRARY(rwkv7_fast_ops_fp16, m)" not in source
    assert "TORCH_LIBRARY_IMPL(rwkv7_fast_ops_fp16, CUDA, m)" not in source


def test_rwkv_requirements_pin_fla_and_flashrwkv_source_revisions() -> None:
    requirements_path = Path(__file__).parents[3] / "requirements" / "rwkv.txt"
    requirement_lines = [
        line
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#") and not line.startswith("-r ")
    ]
    requirements = {
        requirement.name: requirement
        for line in requirement_lines
        for requirement in (Requirement(line),)
    }

    assert requirements["flash-linear-attention"].url == (
        "git+https://github.com/rwkv-rs/fla-rwkv.git@"
        f"{rwkv7_wkv_backend.FLA_RWKV_REVISION}"
    )
    assert requirements["flash-rwkv"].url == (
        "git+https://github.com/rwkv-rs/FlashRWKV.git@"
        f"{rwkv7_wkv_backend.FLASH_RWKV_REVISION}"
    )


def test_real_local_fla_rwkv_source_exposes_recurrent_contract() -> None:
    checkout_value = os.environ.get("VLLM_TEST_FLA_RWKV_CHECKOUT")
    if checkout_value is None:
        pytest.skip("VLLM_TEST_FLA_RWKV_CHECKOUT is not set")
    checkout = Path(checkout_value).resolve()
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert revision == rwkv7_wkv_backend.FLA_RWKV_REVISION

    script = textwrap.dedent(
        """
        import inspect
        import sys

        sys.path.insert(0, sys.argv[1])
        from vllm.model_executor.models.rwkv7_wkv_backend import (
            _load_fla_rwkv7_recurrent_contract,
        )

        recurrent_rwkv7 = _load_fla_rwkv7_recurrent_contract()[0]
        parameters = inspect.signature(recurrent_rwkv7).parameters
        assert {"state_indices", "mode"} <= parameters.keys()
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(checkout)],
        check=True,
        cwd=Path(__file__).parents[3],
    )


def test_fla_rwkv_import_error_reports_required_source_revisions(monkeypatch) -> None:
    def reject_import(name):
        raise ImportError(name)

    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        reject_import,
    )

    with pytest.raises(RuntimeError) as exc_info:
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()

    message = str(exc_info.value)
    assert f"fla-rwkv@{rwkv7_wkv_backend.FLA_RWKV_REVISION}" in message
    assert f"FlashRWKV@{rwkv7_wkv_backend.FLASH_RWKV_REVISION}" in message


@pytest.fixture(autouse=True)
def clear_fla_contract_cache():
    rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract.cache_clear()
    yield
    rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract.cache_clear()


def _inputs(*, packed: bool = True):
    tensors = tuple(torch.empty((1, 3, 2, 64)) for _ in range(6))
    state_pool = torch.empty((4, 2, 64, 64))
    if packed:
        cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
        state_indices = torch.tensor([3, 1], dtype=torch.int32)
    else:
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
        state_indices = torch.tensor([2], dtype=torch.int32)
    return tensors, state_pool, cu_seqlens, state_indices


def _module(recurrent_rwkv7, *, provider="flash_rwkv", kernel=None):
    if kernel is None:
        kernel = rwkv7_wkv_backend._EXPECTED_KERNEL
    return SimpleNamespace(
        recurrent_rwkv7=recurrent_rwkv7,
        get_last_rwkv7_provider=lambda: provider,
        get_last_rwkv7_kernel=lambda: kernel,
    )


def _recurrent(output, calls):
    def recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        calls.append(
            (
                (r, decay_logits, k, v, a, b),
                {
                    "initial_state": initial_state,
                    "decay_bias": decay_bias,
                    "elapsed_t": elapsed_t,
                    "output_final_state": output_final_state,
                    "cu_seqlens": cu_seqlens,
                    "state_indices": state_indices,
                    "mode": mode,
                },
            )
        )
        return output, initial_state

    return recurrent_rwkv7


@pytest.mark.parametrize("packed", [False, True], ids=("ordinary", "packed"))
def test_fused_contract_forwards_raw_decay_metadata_and_state_identity(
    monkeypatch,
    packed,
) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs(packed=packed)
    output = torch.full_like(tensors[3], 2)
    calls = []
    module = _module(_recurrent(output, calls))
    decay_bias = torch.empty((128,))
    elapsed_t = torch.empty((4,), dtype=torch.int32)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    actual = rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
        *tensors,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode="fp16",
    )

    assert actual is output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert all(actual is expected for actual, expected in zip(args, tensors))
    assert args[1] is tensors[1]
    assert kwargs["decay_bias"] is decay_bias
    assert kwargs["elapsed_t"] is elapsed_t
    assert kwargs["initial_state"] is state_pool
    assert kwargs["output_final_state"] is True
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["mode"] == "fp16"


def test_fused_contract_is_resolved_and_inspected_once(monkeypatch) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()
    output = torch.empty_like(tensors[3])
    module = _module(_recurrent(output, []))
    decay_bias = torch.empty((128,))
    elapsed_t = None
    imports = []
    inspections = []

    def import_module(name):
        imports.append(name)
        return module

    original_signature = rwkv7_wkv_backend.inspect.signature

    def signature(value):
        inspections.append(value)
        return original_signature(value)

    monkeypatch.setattr(rwkv7_wkv_backend.importlib, "import_module", import_module)
    monkeypatch.setattr(rwkv7_wkv_backend.inspect, "signature", signature)

    for _ in range(2):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )

    assert imports == ["fla.ops.rwkv7"]
    assert inspections == [module.recurrent_rwkv7]


@pytest.mark.parametrize(
    ("provider", "kernel", "expected"),
    [
        ("triton", rwkv7_wkv_backend._EXPECTED_KERNEL, "fallback is disabled"),
        ("flash_rwkv", "rwkv7_recurrent", "fallback is disabled"),
    ],
)
def test_fused_contract_rejects_other_provider_or_kernel(
    monkeypatch,
    provider,
    kernel,
    expected,
) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()
    output = torch.empty_like(tensors[3])
    module = _module(
        _recurrent(output, []),
        provider=provider,
        kernel=kernel,
    )
    decay_bias = torch.empty((128,))
    elapsed_t = torch.empty((4,), dtype=torch.int32)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match=expected):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp16",
        )


def test_fused_contract_rejects_copied_state(monkeypatch) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()

    def recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        return torch.empty_like(v), initial_state.clone()

    module = _module(recurrent_rwkv7)
    decay_bias = torch.empty((128,))
    elapsed_t = None
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="state pool in place"):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )


def test_noncontiguous_scheduler_metadata_is_not_silently_copied(monkeypatch) -> None:
    tensors, state_pool, _cu_seqlens, _state_indices = _inputs()
    cu_seqlens = torch.arange(6, dtype=torch.int32)[::2]
    state_indices = torch.arange(4, dtype=torch.int32)[::2]

    def recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens: torch.Tensor,
        state_indices: torch.Tensor,
        mode,
    ):
        assert not cu_seqlens.is_contiguous()
        assert not state_indices.is_contiguous()
        raise ValueError("packed metadata must be contiguous")

    module = _module(recurrent_rwkv7)
    decay_bias = torch.empty((128,))
    elapsed_t = torch.empty((4,), dtype=torch.int32)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="packed metadata must be contiguous"):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp16",
        )


def test_missing_fused_api_fails_before_execution(monkeypatch) -> None:
    module = SimpleNamespace(
        recurrent_rwkv7=lambda *args, **kwargs: None,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="get_last_rwkv7_kernel"):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()


def test_fused_api_requires_bias_and_elapsed_parameters(monkeypatch) -> None:
    def recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        raise AssertionError("incomplete API must not execute")

    module = _module(recurrent_rwkv7)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(
        RuntimeError,
        match=r"missing parameters \['decay_bias', 'elapsed_t'\]",
    ):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()


def test_standard_fused_api_rejects_legacy_log_decay_parameter(monkeypatch) -> None:
    def recurrent_rwkv7(
        r,
        log_decay,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        raise AssertionError("legacy log_decay API must not execute")

    module = _module(recurrent_rwkv7)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(
        RuntimeError,
        match="standard recurrent_rwkv7 API still exposes log_decay",
    ):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()
