# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import rwkv7_wkv_backend


def _inputs():
    tensors = tuple(torch.empty((1, 3, 2, 64)) for _ in range(6))
    state_pool = torch.empty((4, 2, 64, 64))
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int32)
    state_indices = torch.tensor([3, 1], dtype=torch.int32)
    return tensors, state_pool, cu_seqlens, state_indices


def test_fla_contract_missing_state_indices_fails_closed(monkeypatch) -> None:
    called = False

    def old_chunk_rwkv7(
        r,
        w,
        k,
        v,
        a,
        b,
        *,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        mode="fp32io16",
    ):
        nonlocal called
        called = True

    module = SimpleNamespace(
        chunk_rwkv7=old_chunk_rwkv7,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )
    tensors, state_pool, cu_seqlens, state_indices = _inputs()

    with pytest.raises(
        RuntimeError,
        match=r"missing parameters \['state_indices'\]",
    ):
        rwkv7_wkv_backend.run_fla_rwkv7_stateful(
            *tensors,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )
    assert not called


@pytest.mark.parametrize(
    ("provider", "return_original_pool", "error"),
    [
        ("fla", True, "fallback is disabled"),
        (
            "flash_rwkv",
            False,
            "update the supplied request-indexed state pool in place",
        ),
    ],
)
def test_fla_contract_rejects_fallback_or_copied_state(
    monkeypatch,
    provider,
    return_original_pool,
    error,
) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()

    def chunk_rwkv7(
        r,
        w,
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
        final_state = initial_state if return_original_pool else initial_state.clone()
        return torch.empty_like(v), final_state

    module = SimpleNamespace(
        chunk_rwkv7=chunk_rwkv7,
        get_last_rwkv7_provider=lambda: provider,
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match=error):
        rwkv7_wkv_backend.run_fla_rwkv7_stateful(
            *tensors,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )


def test_fla_contract_forwards_mixed_wave_metadata_and_pool(monkeypatch) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()
    calls = []

    def chunk_rwkv7(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[3], 2), kwargs["initial_state"]

    chunk_rwkv7.__signature__ = inspect.Signature(
        parameters=[
            *(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for name in ("r", "w", "k", "v", "a", "b")
            ),
            *(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                for name in (
                    "initial_state",
                    "output_final_state",
                    "cu_seqlens",
                    "state_indices",
                    "mode",
                )
            ),
        ]
    )
    module = SimpleNamespace(
        chunk_rwkv7=chunk_rwkv7,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    output = rwkv7_wkv_backend.run_fla_rwkv7_stateful(
        *tensors,
        state_pool=state_pool,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        mode="fp16",
    )

    assert torch.equal(output, torch.full_like(tensors[3], 2))
    assert len(calls) == 1
    assert all(actual is expected for actual, expected in zip(calls[0][0], tensors))
    kwargs = calls[0][1]
    assert kwargs["initial_state"] is state_pool
    assert kwargs["output_final_state"] is True
    assert kwargs["cu_seqlens"] is cu_seqlens
    assert kwargs["state_indices"] is state_indices
    assert kwargs["mode"] == "fp16"
