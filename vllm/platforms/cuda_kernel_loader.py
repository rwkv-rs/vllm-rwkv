# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the CUDA extensions provided by the installed artifact."""

import contextlib
import importlib


def import_kernels() -> None:
    """Load generic kernels, or accept a verified RWKV-only artifact."""
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError as generic_extension_error:
        try:
            rwkv7_ops = importlib.import_module("vllm.rwkv7_ops")
        except ImportError:
            raise generic_extension_error from None
        if not getattr(rwkv7_ops, "_rwkv_only_build", False):
            raise generic_extension_error
        return

    for optional_extension in (
        "vllm._moe_C_stable_libtorch",
        "vllm._qutlass_C",
    ):
        with contextlib.suppress(ImportError):
            importlib.import_module(optional_extension)
