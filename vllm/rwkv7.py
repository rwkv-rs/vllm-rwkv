# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load RWKV7 native kernels and register fake implementations."""

import importlib
from typing import TYPE_CHECKING

try:
    importlib.import_module("vllm.rwkv7_ops")
except ImportError as error:
    raise ImportError(
        "RWKV7 requires the vllm.rwkv7_ops CUDA extension. "
        "Build vLLM from this fork with the RWKV-only extension target."
    ) from error

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake


@register_fake("rwkv7_wkv_fp16_v2::wkv")
def _rwkv7_wkv_fp16_fake(*args) -> None:
    return None


@register_fake("rwkv7_wkv_fp32_v2::wkv")
def _rwkv7_wkv_fp32_fake(*args) -> None:
    return None
