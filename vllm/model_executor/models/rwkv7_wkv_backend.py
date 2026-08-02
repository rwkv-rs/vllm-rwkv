# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FLA-owned RWKV7 WKV dispatch contract."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any

import torch

_REQUIRED_PARAMETERS = frozenset(
    {
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
)


def _load_fla_rwkv7_contract() -> tuple[Callable[..., Any], Callable[[], str | None]]:
    try:
        rwkv7 = importlib.import_module("fla.ops.rwkv7")
    except ImportError as exc:
        raise RuntimeError(
            "RWKV7 WKV execution requires the FLA FlashRWKV backend with "
            "request-indexed state-pool support"
        ) from exc

    chunk_rwkv7 = getattr(rwkv7, "chunk_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    if not callable(chunk_rwkv7) or not callable(get_last_provider):
        raise RuntimeError(
            "the installed FLA RWKV7 API does not expose chunk_rwkv7 and "
            "get_last_rwkv7_provider"
        )

    try:
        parameters = inspect.signature(chunk_rwkv7).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "the installed FLA chunk_rwkv7 API is not inspectable"
        ) from exc
    missing = sorted(_REQUIRED_PARAMETERS - parameters.keys())
    if missing:
        raise RuntimeError(
            "the installed FLA chunk_rwkv7 API lacks the required stateful "
            f"inference contract: missing parameters {missing}"
        )
    return chunk_rwkv7, get_last_provider


def run_fla_rwkv7_stateful(
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
    """Run FLA's public FlashRWKV path and retain vLLM state ownership."""
    chunk_rwkv7, get_last_provider = _load_fla_rwkv7_contract()
    result = chunk_rwkv7(
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
    if get_last_provider() != "flash_rwkv":
        raise RuntimeError(
            "FLA did not execute the required FlashRWKV backend; fallback is disabled"
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "FLA FlashRWKV stateful execution must return (output, state_pool)"
        )
    output, final_state = result
    if final_state is not state_pool:
        raise RuntimeError(
            "FLA FlashRWKV must update the supplied request-indexed state pool in place"
        )
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("FLA FlashRWKV returned a non-tensor output")
    return output


__all__ = ["run_fla_rwkv7_stateful"]
