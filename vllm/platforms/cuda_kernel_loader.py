# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the CUDA extensions provided by the installed artifact."""

import contextlib
import importlib
from functools import cache


def _has_rwkv_only_marker() -> bool:
    try:
        rwkv7_ops = importlib.import_module("vllm.rwkv7_ops")
    except ImportError:
        return False
    return bool(getattr(rwkv7_ops, "_rwkv_only_build", False))


@cache
def is_rwkv_only_artifact() -> bool:
    """Return whether this install contains only the RWKV CUDA extension."""
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        return _has_rwkv_only_marker()
    return False


def import_kernels() -> None:
    """Load generic kernels, or accept a verified RWKV-only artifact."""
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError as generic_extension_error:
        if not _has_rwkv_only_marker():
            raise generic_extension_error
        return

    for optional_extension in (
        "vllm._moe_C_stable_libtorch",
        "vllm._qutlass_C",
    ):
        with contextlib.suppress(ImportError):
            importlib.import_module(optional_extension)
