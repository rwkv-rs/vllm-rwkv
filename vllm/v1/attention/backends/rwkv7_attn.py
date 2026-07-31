# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.linear_attn import (
    LinearAttentionBackend,
    LinearAttentionMetadataBuilder,
)


class RWKV7AttentionMetadataBuilder(LinearAttentionMetadataBuilder):
    """Build recurrent-state metadata without claiming CUDA graph support."""

    _cudagraph_support = AttentionCGSupport.NEVER


class RWKV7AttentionBackend(LinearAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RWKV7"

    @staticmethod
    def get_builder_cls() -> type[RWKV7AttentionMetadataBuilder]:
        return RWKV7AttentionMetadataBuilder
