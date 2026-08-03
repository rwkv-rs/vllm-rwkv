# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Public fused RWKV7 recurrent backend contract."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from functools import cache
from typing import Any

import torch

from vllm.transformers_utils.rwkv7_runtime_contract import (
    FLA_RWKV_REPOSITORY,
    FLA_RWKV_REVISION,
    FLASH_RWKV_REPOSITORY,
    FLASH_RWKV_REVISION,
)

_EXPECTED_PROVIDER = "flash_rwkv"
_EXPECTED_KERNEL = "rwkv7_recurrent_stateful"
_REQUIRED_PREPARE_PARAMETERS = frozenset(
    {
        "cu_seqlens",
        "state_indices",
        "total_tokens",
        "state_pool_size",
    }
)
_REQUIRED_RECURRENT_PARAMETERS = frozenset(
    {
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "decay_logits",
        "decay_bias",
        "elapsed_t",
        "validated_metadata",
        "state_indices",
        "mode",
    }
)
_FORBIDDEN_PUBLIC_SYMBOLS = frozenset(
    {
        "chunk_rwkv7_from_log_decay",
        "decay_logits_to_log_decay",
        "fused_mul_recurrent_rwkv7_from_log_decay",
        "fused_recurrent_rwkv7_from_log_decay",
        "recurrent_rwkv7_from_log_decay",
        "rwkv7_from_log_decay",
        "rwkv7_recurrent_from_log_decay",
        "rwkv7_recurrent_stateful_from_log_decay",
    }
)


def _with_required_revisions(message: str) -> str:
    return (
        f"{message}. Required revisions: fla-rwkv@{FLA_RWKV_REVISION}, "
        f"FlashRWKV@{FLASH_RWKV_REVISION}"
    )


@cache
def _load_fla_rwkv7_recurrent_contract() -> tuple[
    Callable[..., object],
    Callable[..., Any],
    Callable[[], str | None],
    Callable[[], str | None],
]:
    """Resolve and validate the pinned FLA contract once per process."""
    try:
        rwkv7 = importlib.import_module("fla.ops.rwkv7")
    except ImportError as exc:
        raise RuntimeError(
            _with_required_revisions(
                "RWKV7 WKV execution requires the pinned fla-rwkv/FlashRWKV "
                "fused recurrent backend"
            )
        ) from exc

    prepare_metadata = getattr(rwkv7, "prepare_rwkv7_recurrent_metadata", None)
    recurrent_rwkv7 = getattr(rwkv7, "recurrent_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    get_last_kernel = getattr(rwkv7, "get_last_rwkv7_kernel", None)
    if not all(
        callable(value)
        for value in (
            prepare_metadata,
            recurrent_rwkv7,
            get_last_provider,
            get_last_kernel,
        )
    ):
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv API must expose "
                "prepare_rwkv7_recurrent_metadata, recurrent_rwkv7, "
                "get_last_rwkv7_provider, and get_last_rwkv7_kernel"
            )
        )

    try:
        prepare_parameters = inspect.signature(prepare_metadata).parameters
        recurrent_parameters = inspect.signature(recurrent_rwkv7).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv recurrent metadata APIs are not inspectable"
            )
        ) from exc
    missing = sorted(_REQUIRED_PREPARE_PARAMETERS - prepare_parameters.keys())
    if missing:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv prepare_rwkv7_recurrent_metadata API "
                f"lacks the required contract: missing parameters {missing}"
            )
        )
    missing = sorted(_REQUIRED_RECURRENT_PARAMETERS - recurrent_parameters.keys())
    if missing:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv recurrent_rwkv7 API lacks the required "
                f"packed stateful contract: missing parameters {missing}"
            )
        )
    if "log_decay" in recurrent_parameters:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv standard recurrent_rwkv7 API still exposes "
                "log_decay"
            )
        )
    public_names = set(vars(rwkv7))
    exported_names = getattr(rwkv7, "__all__", ())
    if isinstance(exported_names, (list, tuple, set, frozenset)):
        public_names.update(name for name in exported_names if isinstance(name, str))
    forbidden = sorted(
        name
        for name in public_names
        if name in _FORBIDDEN_PUBLIC_SYMBOLS
        or (not name.startswith("_") and "log_decay" in name)
    )
    if forbidden:
        raise RuntimeError(
            _with_required_revisions(
                "the installed fla-rwkv public API still exposes forbidden canonical "
                f"log-decay symbols: {forbidden}"
            )
        )
    return prepare_metadata, recurrent_rwkv7, get_last_provider, get_last_kernel


def prepare_fla_rwkv7_recurrent_metadata(
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    total_tokens: int,
    state_pool_size: int,
) -> object:
    """Validate packed scheduler metadata once for a full layer range."""
    prepare_metadata, _, _, _ = _load_fla_rwkv7_recurrent_contract()
    try:
        validated_metadata = prepare_metadata(
            cu_seqlens,
            state_indices,
            total_tokens=total_tokens,
            state_pool_size=state_pool_size,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            _with_required_revisions(
                f"fla-rwkv recurrent metadata preparation failed: {exc}"
            )
        ) from exc
    if validated_metadata is None:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv recurrent metadata preparation returned no validation ticket"
            )
        )
    return validated_metadata


def run_fla_rwkv7_recurrent_from_decay_logits(
    r: torch.Tensor,
    decay_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    decay_bias: torch.Tensor,
    elapsed_t: torch.Tensor | None,
    validated_metadata: object,
    state_pool: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Fuse decay delta, bias, and recurrence into vLLM-owned state slots."""
    if validated_metadata is None:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv fused recurrent execution requires a prevalidated "
                "metadata ticket"
            )
        )
    _, recurrent_rwkv7, get_last_provider, get_last_kernel = (
        _load_fla_rwkv7_recurrent_contract()
    )
    try:
        result = recurrent_rwkv7(
            r,
            decay_logits,
            k,
            v,
            a,
            b,
            decay_bias=decay_bias,
            elapsed_t=elapsed_t,
            validated_metadata=validated_metadata,
            initial_state=state_pool,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            mode=mode,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            _with_required_revisions(f"fla-rwkv fused recurrent execution failed: {exc}")
        ) from exc

    provider = get_last_provider()
    kernel = get_last_kernel()
    if provider != _EXPECTED_PROVIDER or kernel != _EXPECTED_KERNEL:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv did not execute the required fused raw-decay stateful "
                f"kernel: provider={provider!r}, kernel={kernel!r}; fallback is "
                "disabled"
            )
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv fused recurrent execution must return (output, state_pool)"
            )
        )
    output, final_state = result
    if final_state is not state_pool:
        raise RuntimeError(
            _with_required_revisions(
                "fla-rwkv must update the supplied request-indexed state pool in place"
            )
        )
    if not isinstance(output, torch.Tensor):
        raise RuntimeError(
            _with_required_revisions("fla-rwkv returned a non-tensor recurrent output")
        )
    if not output.is_contiguous():
        raise RuntimeError(
            _with_required_revisions("fla-rwkv returned a non-contiguous recurrent output")
        )
    return output


__all__ = [
    "FLASH_RWKV_REPOSITORY",
    "FLASH_RWKV_REVISION",
    "FLA_RWKV_REPOSITORY",
    "FLA_RWKV_REVISION",
    "prepare_fla_rwkv7_recurrent_metadata",
    "run_fla_rwkv7_recurrent_from_decay_logits",
]
