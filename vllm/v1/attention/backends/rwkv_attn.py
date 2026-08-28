# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention metadata for FlashRWKV2 recurrent state."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor
from vllm.v1.kv_cache_interface import KVCacheLayout, MambaSpec


@dataclass(frozen=True)
class RwkvStateSpec(MambaSpec):
    """Visible RWKV state plus provider-owned workspace accounting."""

    provider_private_bytes_per_page: int = 0
    provider_fixed_workspace_bytes: int = 0

    @property
    def external_bytes_per_page(self) -> int:
        return self.provider_private_bytes_per_page

    @property
    def external_fixed_memory_bytes(self) -> int:
        return self.provider_fixed_workspace_bytes

    @property
    def requires_block_zeroing(self) -> bool:
        return True


class RwkvAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RWKV_ATTN"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError("RWKV executes through its fused model layer")

    @staticmethod
    def get_builder_cls() -> type["RwkvAttentionMetadataBuilder"]:
        return RwkvAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBNHC,)


@dataclass
class RwkvAttentionMetadata:
    cu_seqlens: torch.Tensor
    state_indices: torch.Tensor
    batch_size: int
    max_seqlen: int
    token_capacity: int
    sequence_capacity: int
    max_seqlen_capacity: int
    num_active_tokens: torch.Tensor | None = None
    num_active_sequences: torch.Tensor | None = None
    validated_metadata: object | None = None

    @property
    def is_live(self) -> bool:
        return self.num_active_tokens is not None


class RwkvAttentionMetadataBuilder(AttentionMetadataBuilder[RwkvAttentionMetadata]):
    kv_cache_spec: RwkvStateSpec

    # The live ticket path is implemented for full graphs, but this must remain
    # disabled until prefill, mixed, and decode captures pass the RWKV GPU gate.
    _cudagraph_support = AttentionCGSupport.NEVER
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: RwkvStateSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.kv_cache_spec = kv_cache_spec
        self._state_layer = vllm_config.compilation_config.static_forward_context[
            layer_names[0]
        ]
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._live_cu_seqlens = torch.empty(
            max_num_seqs + 1, dtype=torch.int32, device=device
        )
        self._live_state_indices = torch.empty(
            max_num_seqs, dtype=torch.int32, device=device
        )
        self._num_active_tokens = torch.empty(1, dtype=torch.int32, device=device)
        self._num_active_sequences = torch.empty(1, dtype=torch.int32, device=device)
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        self._use_live_metadata = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )

    def _state_indices(self, metadata: CommonAttentionMetadata) -> torch.Tensor:
        return mamba_get_block_table_tensor(
            metadata.block_table_tensor,
            metadata.seq_lens,
            self.kv_cache_spec,
            self.vllm_config.cache_config.mamba_cache_mode,
        )[:, 0].contiguous()

    @staticmethod
    def _active_shape(metadata: CommonAttentionMetadata) -> tuple[int, int, int]:
        query_lens = torch.diff(metadata.query_start_loc_cpu)
        num_active_tokens = int(metadata.query_start_loc_cpu[-1])
        num_active_sequences = int(torch.count_nonzero(query_lens))
        max_seqlen = max(int(query_lens.max()), 1)
        return num_active_tokens, num_active_sequences, max_seqlen

    def _build_live(
        self,
        metadata: CommonAttentionMetadata,
        state_indices: torch.Tensor,
        *,
        for_capture: bool,
    ) -> RwkvAttentionMetadata:
        sequence_capacity = metadata.num_reqs
        token_capacity = metadata.num_actual_tokens
        max_seqlen_capacity = metadata.max_query_len
        cu_seqlens = self._live_cu_seqlens[: sequence_capacity + 1]
        live_state_indices = self._live_state_indices[:sequence_capacity]
        cu_seqlens.copy_(metadata.query_start_loc[: sequence_capacity + 1])
        live_state_indices.copy_(state_indices[:sequence_capacity])

        if for_capture:
            self._num_active_tokens.zero_()
            self._num_active_sequences.zero_()
            batch_size = sequence_capacity
            max_seqlen = max_seqlen_capacity
        else:
            num_tokens, batch_size, max_seqlen = self._active_shape(metadata)
            self._num_active_tokens.fill_(num_tokens)
            self._num_active_sequences.fill_(batch_size)

        rwkv_metadata = RwkvAttentionMetadata(
            cu_seqlens=cu_seqlens,
            state_indices=live_state_indices,
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            token_capacity=token_capacity,
            sequence_capacity=sequence_capacity,
            max_seqlen_capacity=max_seqlen_capacity,
            num_active_tokens=self._num_active_tokens,
            num_active_sequences=self._num_active_sequences,
        )
        if for_capture:
            self._state_layer.warmup_live_metadata(rwkv_metadata)
        else:
            rwkv_metadata.validated_metadata = (
                self._state_layer.prepare_metadata_ticket(rwkv_metadata)
            )
        return rwkv_metadata

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> RwkvAttentionMetadata:
        return self._build_live(
            common_attn_metadata,
            self._state_indices(common_attn_metadata),
            for_capture=True,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RwkvAttentionMetadata:
        del common_prefix_len, fast_build
        state_indices = self._state_indices(common_attn_metadata)
        if self._use_live_metadata:
            return self._build_live(
                common_attn_metadata, state_indices, for_capture=False
            )

        num_tokens, batch_size, max_seqlen = self._active_shape(common_attn_metadata)
        metadata = RwkvAttentionMetadata(
            cu_seqlens=common_attn_metadata.query_start_loc,
            state_indices=state_indices,
            batch_size=batch_size,
            max_seqlen=max_seqlen,
            token_capacity=num_tokens,
            sequence_capacity=batch_size,
            max_seqlen_capacity=max_seqlen,
        )
        metadata.validated_metadata = self._state_layer.prepare_metadata_ticket(
            metadata
        )
        return metadata


def get_rwkv_metadata() -> RwkvAttentionMetadata | None:
    """Return the current RWKV metadata without importing the model module."""
    from vllm.forward_context import get_forward_context

    raw_metadata: Any = get_forward_context().attn_metadata
    if isinstance(raw_metadata, RwkvAttentionMetadata):
        return raw_metadata
    if isinstance(raw_metadata, list):
        raw_metadata = raw_metadata[0]
    if not isinstance(raw_metadata, dict):
        return None
    for metadata in raw_metadata.values():
        if isinstance(metadata, RwkvAttentionMetadata):
            return metadata
    return None
