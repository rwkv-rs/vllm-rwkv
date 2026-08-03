# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    assert checkout_value is not None
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


def _default_prepare_metadata(
    cu_seqlens,
    state_indices,
    *,
    total_tokens,
    state_pool_size,
):
    return object()


def _module(
    recurrent_rwkv7,
    *,
    prepare_metadata=None,
    provider="flash_rwkv",
    kernel=None,
):
    if kernel is None:
        kernel = rwkv7_wkv_backend._EXPECTED_KERNEL
    if prepare_metadata is None:
        prepare_metadata = _default_prepare_metadata

    return SimpleNamespace(
        prepare_rwkv7_recurrent_metadata=prepare_metadata,
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
        validated_metadata,
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
                    "validated_metadata": validated_metadata,
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
    calls: list[Any] = []
    prepare_calls = []
    ticket = object()

    def prepare_metadata(
        cu_seqlens,
        state_indices,
        *,
        total_tokens,
        state_pool_size,
    ):
        prepare_calls.append(
            (
                cu_seqlens,
                state_indices,
                total_tokens,
                state_pool_size,
            )
        )
        return ticket

    module = _module(
        _recurrent(output, calls),
        prepare_metadata=prepare_metadata,
    )
    decay_bias = torch.empty((128,))
    elapsed_t = torch.empty((4,), dtype=torch.int32)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    validated_metadata = rwkv7_wkv_backend.prepare_fla_rwkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=4,
    )
    actual = rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
        *tensors,
        decay_bias=decay_bias,
        elapsed_t=elapsed_t,
        validated_metadata=validated_metadata,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode="fp16",
    )

    assert actual is output
    assert validated_metadata is ticket
    assert prepare_calls == [(cu_seqlens, state_indices, 3, 4)]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert all(actual is expected for actual, expected in zip(args, tensors))
    assert args[1] is tensors[1]
    assert kwargs["decay_bias"] is decay_bias
    assert kwargs["elapsed_t"] is elapsed_t
    assert kwargs["validated_metadata"] is ticket
    assert kwargs["initial_state"] is state_pool
    assert kwargs["output_final_state"] is True
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["mode"] == "fp16"


def test_fused_contract_is_resolved_and_inspected_once(monkeypatch) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()
    output = torch.empty_like(tensors[3])
    ticket = object()

    def prepare_metadata(
        cu_seqlens,
        state_indices,
        *,
        total_tokens,
        state_pool_size,
    ):
        return ticket

    module = _module(_recurrent(output, []), prepare_metadata=prepare_metadata)
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

    validated_metadata = rwkv7_wkv_backend.prepare_fla_rwkv7_recurrent_metadata(
        cu_seqlens,
        state_indices,
        total_tokens=3,
        state_pool_size=4,
    )
    for _ in range(2):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )

    assert imports == ["fla.ops.rwkv7"]
    assert inspections == [
        module.prepare_rwkv7_recurrent_metadata,
        module.recurrent_rwkv7,
    ]


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
    ticket = object()
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
            validated_metadata=ticket,
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
        validated_metadata,
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
    ticket = object()
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
            validated_metadata=ticket,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )


def test_noncontiguous_scheduler_metadata_is_not_silently_copied(monkeypatch) -> None:
    tensors, state_pool, _cu_seqlens, _state_indices = _inputs()
    cu_seqlens = torch.arange(6, dtype=torch.int32)[::2]
    state_indices = torch.arange(4, dtype=torch.int32)[::2]
    expected_cu_seqlens = cu_seqlens
    expected_state_indices = state_indices

    def prepare_metadata(
        cu_seqlens,
        state_indices,
        *,
        total_tokens,
        state_pool_size,
    ):
        assert cu_seqlens is expected_cu_seqlens
        assert state_indices is expected_state_indices
        assert not cu_seqlens.is_contiguous()
        assert not state_indices.is_contiguous()
        raise ValueError("packed metadata must be contiguous")

    module = _module(
        _recurrent(torch.empty_like(tensors[3]), []),
        prepare_metadata=prepare_metadata,
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="packed metadata must be contiguous"):
        rwkv7_wkv_backend.prepare_fla_rwkv7_recurrent_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=3,
            state_pool_size=state_pool.shape[0],
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

    with pytest.raises(RuntimeError, match="prepare_rwkv7_recurrent_metadata"):
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
        match=(
            r"missing parameters \['decay_bias', 'elapsed_t', "
            r"'validated_metadata'\]"
        ),
    ):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()


def test_prepare_api_requires_token_and_state_pool_sizes(monkeypatch) -> None:
    def prepare_metadata(cu_seqlens, state_indices):
        raise AssertionError("incomplete prepare API must not execute")

    module = _module(
        _recurrent(torch.empty((1, 3, 2, 64)), []),
        prepare_metadata=prepare_metadata,
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(
        RuntimeError,
        match=r"missing parameters \['state_pool_size', 'total_tokens'\]",
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
        validated_metadata,
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


@pytest.mark.parametrize("symbol", sorted(rwkv7_wkv_backend._FORBIDDEN_PUBLIC_SYMBOLS))
def test_fused_api_rejects_public_log_decay_symbols(monkeypatch, symbol) -> None:
    module = _module(_recurrent(torch.empty((1, 3, 2, 64)), []))
    setattr(module, symbol, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="forbidden canonical log-decay symbols"):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()


def test_fused_api_rejects_future_exported_log_decay_symbol(monkeypatch) -> None:
    module = _module(_recurrent(torch.empty((1, 3, 2, 64)), []))
    module.__all__ = ["future_log_decay_conversion"]
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="future_log_decay_conversion"):
        rwkv7_wkv_backend._load_fla_rwkv7_recurrent_contract()


def test_prepare_fused_api_rejects_missing_ticket(monkeypatch) -> None:
    def prepare_metadata(
        cu_seqlens,
        state_indices,
        *,
        total_tokens,
        state_pool_size,
    ):
        return None

    module = _module(
        _recurrent(torch.empty((1, 3, 2, 64)), []),
        prepare_metadata=prepare_metadata,
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match="returned no validation ticket"):
        rwkv7_wkv_backend.prepare_fla_rwkv7_recurrent_metadata(
            torch.tensor([0, 3], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            total_tokens=3,
            state_pool_size=1,
        )


def test_fused_execution_rejects_missing_prevalidated_ticket(monkeypatch) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: pytest.fail(f"unexpected backend import: {name}"),
    )

    with pytest.raises(RuntimeError, match="requires a prevalidated metadata ticket"):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent_from_decay_logits(
            *tensors,
            decay_bias=torch.empty((128,)),
            elapsed_t=None,
            validated_metadata=None,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )
