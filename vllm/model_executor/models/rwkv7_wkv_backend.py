# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fla-rwkv public recurrent RWKV7 dispatch contract."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch

from vllm.transformers_utils.rwkv7_runtime_contract import (
    FLA_RWKV_REPOSITORY,
    FLA_RWKV_REVISION,
    FLASH_RWKV_REPOSITORY,
    FLASH_RWKV_REVISION,
)

_REQUIRED_RECURRENT_PARAMETERS = frozenset(
    {
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
)


def _with_required_revisions(message: str) -> str:
    return (
        f"{message}. Required revisions: fla-rwkv@{FLA_RWKV_REVISION}, "
        f"FlashRWKV@{FLASH_RWKV_REVISION}"
    )


def _load_fla_rwkv7_recurrent_contract() -> tuple[
    Callable[..., Any], Callable[[], str | None]
]:
    try:
        rwkv7 = importlib.import_module("fla.ops.rwkv7")
    except ImportError as exc:
        raise RuntimeError(
            _with_required_revisions(
                "RWKV7 WKV execution requires the fla-rwkv public recurrent "
                "FlashRWKV backend "
                "with request-indexed state-pool support"
            )
        ) from exc

    recurrent_rwkv7 = getattr(rwkv7, "recurrent_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    if not callable(recurrent_rwkv7) or not callable(get_last_provider):
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv RWKV7 API does not expose recurrent_rwkv7 "
                "and get_last_rwkv7_provider"
            )
        )

    try:
        parameters = inspect.signature(recurrent_rwkv7).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv recurrent_rwkv7 API is not inspectable"
            )
        ) from exc
    missing = sorted(_REQUIRED_RECURRENT_PARAMETERS - parameters.keys())
    if missing:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv recurrent_rwkv7 API lacks the required "
                f"recurrent inference contract: missing parameters {missing}"
            )
        )
    return recurrent_rwkv7, get_last_provider


def run_fla_rwkv7_recurrent(
    r: torch.Tensor,
    log_decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Run fla-rwkv's public recurrent path with vLLM state ownership."""
    recurrent_rwkv7, get_last_provider = _load_fla_rwkv7_recurrent_contract()
    try:
        result = recurrent_rwkv7(
            r,
            log_decay,
            k,
            v,
            a,
            b,
            initial_state=state_pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            _with_required_revisions(
                f"fla-rwkv recurrent FlashRWKV execution failed: {exc}"
            )
        ) from exc
    if get_last_provider() != "flash_rwkv":
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv did not execute the required recurrent FlashRWKV "
                "backend; "
                "fallback is disabled"
            )
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv recurrent FlashRWKV execution must return "
                "(output, state_pool)"
            )
        )
    output, final_state = result
    if final_state is not state_pool:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv recurrent FlashRWKV must update the supplied "
                "request-indexed "
                "state pool in place"
            )
        )
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv recurrent FlashRWKV returned a non-tensor output"
            )
        )
    return output


__all__ = [
    "FLASH_RWKV_REPOSITORY",
    "FLASH_RWKV_REVISION",
    "FLA_RWKV_REPOSITORY",
    "FLA_RWKV_REVISION",
    "run_fla_rwkv7_recurrent",
]
