# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request state for RWKV recurrent-state lifecycle management."""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class RwkvModelState(DefaultModelState):
    """Route scheduler block lifecycle events through FlashRWKV2 state handles."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self._state_layer = model.get_rwkv_state_layer()
        self._align_mode = vllm_config.cache_config.mamba_cache_mode == "align"
        block_size = vllm_config.cache_config.mamba_block_size
        assert block_size is not None
        self._block_size = block_size
        self._state_block_columns = np.full(self.max_num_reqs, -1, dtype=np.int32)
        self._state_num_computed_tokens = np.zeros(self.max_num_reqs, dtype=np.int64)
        self._rwkv_group_id: int | None = None

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        self._state_block_columns[req_index] = (
            new_req_data.num_computed_tokens - 1
        ) // self._block_size
        self._state_num_computed_tokens[req_index] = new_req_data.num_computed_tokens

    def reset_kv_cache_blocks(self, block_ids: Sequence[int]) -> bool:
        self._state_layer.reset_state_slots(block_ids)
        return True

    def copy_kv_cache_blocks(self, block_copies: Sequence[tuple[int, int]]) -> bool:
        self._state_layer.copy_state_block_pairs(block_copies)
        return True

    def _get_rwkv_group_id(self, kv_cache_config: KVCacheConfig) -> int:
        if self._rwkv_group_id is None:
            layer_name = self._state_layer.prefix
            group_ids = [
                group_id
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if layer_name in group.layer_names
            ]
            assert len(group_ids) == 1
            self._rwkv_group_id = group_ids[0]
        return self._rwkv_group_id

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        if not self._align_mode or input_batch.num_reqs == 0:
            return

        num_reqs = input_batch.num_reqs
        req_indices = input_batch.idx_mapping_np[:num_reqs]
        computed = self._state_num_computed_tokens[req_indices]
        query_lens = input_batch.num_scheduled_tokens[:num_reqs]
        old_columns = self._state_block_columns[req_indices]
        new_columns = (computed + query_lens - 1) // self._block_size
        block_table = block_tables[self._get_rwkv_group_id(kv_cache_config)]
        reset_rows_np = np.flatnonzero(old_columns < 0)
        if reset_rows_np.size:
            reset_rows = torch.as_tensor(
                reset_rows_np, dtype=torch.int64, device=self.device
            )
            reset_columns = torch.as_tensor(
                new_columns[reset_rows_np], dtype=torch.int64, device=self.device
            )
            reset_indices = block_table[reset_rows, reset_columns].contiguous()
            self._state_layer.reset_state_slots(reset_indices)
        row_indices_np = np.flatnonzero(
            (old_columns >= 0) & (old_columns != new_columns)
        )
        if row_indices_np.size:
            row_indices = torch.as_tensor(
                row_indices_np, dtype=torch.int64, device=self.device
            )
            source_columns = torch.as_tensor(
                old_columns[row_indices_np], dtype=torch.int64, device=self.device
            )
            destination_columns = torch.as_tensor(
                new_columns[row_indices_np], dtype=torch.int64, device=self.device
            )
            source_indices = block_table[row_indices, source_columns].contiguous()
            destination_indices = block_table[
                row_indices, destination_columns
            ].contiguous()
            self._state_layer.copy_state_slots(source_indices, destination_indices)
        self._state_block_columns[req_indices] = new_columns
        self._state_num_computed_tokens[req_indices] = computed + query_lens
        del num_computed_tokens
