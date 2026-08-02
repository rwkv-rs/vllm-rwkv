# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from packaging.requirements import Requirement

from vllm.model_executor.models import rwkv7_wkv_backend


def test_rwkv_requirements_pin_fla_and_flashrwkv_source_revisions() -> None:
    requirements_path = Path(__file__).parents[3] / "requirements" / "rwkv.txt"
    requirement_lines = [
        line
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
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

        recurrent_rwkv7, _ = _load_fla_rwkv7_recurrent_contract()
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


def test_missing_recurrent_api_fails_before_execution(monkeypatch) -> None:
    called = False

    def incompatible_public_api(*_args, **_kwargs):
        nonlocal called
        called = True

    module = SimpleNamespace(
        incompatible_public_api=incompatible_public_api,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )
    tensors, state_pool, cu_seqlens, state_indices = _inputs()

    with pytest.raises(RuntimeError, match="does not expose recurrent_rwkv7"):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent(
            *tensors,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp16",
        )
    assert not called


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


def test_recurrent_contract_missing_state_indices_fails_before_execution(
    monkeypatch,
) -> None:
    called = False

    def incomplete_recurrent_rwkv7(
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
        recurrent_rwkv7=incomplete_recurrent_rwkv7,
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
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent(
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
        ("triton", True, "fallback is disabled"),
        (
            "flash_rwkv",
            False,
            "update the supplied request-indexed state pool in place",
        ),
    ],
)
def test_recurrent_contract_rejects_fallback_or_copied_state(
    monkeypatch,
    provider,
    return_original_pool,
    error,
) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs()

    def recurrent_rwkv7(
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
        recurrent_rwkv7=recurrent_rwkv7,
        get_last_rwkv7_provider=lambda: provider,
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    with pytest.raises(RuntimeError, match=error):
        rwkv7_wkv_backend.run_fla_rwkv7_recurrent(
            *tensors,
            state_pool=state_pool,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode="fp32io16",
        )


@pytest.mark.parametrize("packed", [False, True], ids=("ordinary", "packed"))
def test_recurrent_contract_forwards_metadata_and_state_pool(
    monkeypatch,
    packed,
) -> None:
    tensors, state_pool, cu_seqlens, state_indices = _inputs(packed=packed)
    calls = []

    def recurrent_rwkv7(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.full_like(args[3], 2), kwargs["initial_state"]

    recurrent_rwkv7.__signature__ = inspect.Signature(
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
        recurrent_rwkv7=recurrent_rwkv7,
        get_last_rwkv7_provider=lambda: "flash_rwkv",
    )
    monkeypatch.setattr(
        rwkv7_wkv_backend.importlib,
        "import_module",
        lambda name: module,
    )

    output = rwkv7_wkv_backend.run_fla_rwkv7_recurrent(
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
