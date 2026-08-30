# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request state for RWKV recurrent-state lifecycle management."""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.rwkv import _load_flashrwkv2
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler


class _RwkvSamplingGraph:
    def __init__(
        self,
        model: nn.Module,
        sample: Any,
        penalties: torch.Tensor,
        states: torch.Tensor,
        capacity: int,
    ) -> None:
        device = penalties.device
        hidden_size = model.lm_head.weight.shape[1]
        self.hidden_states = torch.empty(
            capacity, hidden_size, dtype=torch.float16, device=device
        )
        self.slot_indices = torch.arange(capacity, dtype=torch.int32, device=device)
        self.presence_penalty = torch.zeros(capacity, device=device)
        self.frequency_penalty = torch.zeros(capacity, device=device)
        self.penalty_decay = torch.ones(capacity, device=device)
        self.temperature = torch.zeros(capacity, device=device)
        self.top_k = torch.full((capacity,), -1, dtype=torch.int32, device=device)
        self.top_p = torch.ones(capacity, device=device)
        self.num_active = torch.full((1,), capacity, dtype=torch.int32, device=device)

        def forward() -> tuple[torch.Tensor, torch.Tensor]:
            logits = model.compute_logits(self.hidden_states)
            sampled = sample(
                logits,
                penalties,
                states,
                self.slot_indices,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                penalty_decay=self.penalty_decay,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                sample_capacity=capacity,
                num_active_samples=self.num_active,
            )
            return logits, sampled

        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self.logits, self.sampled = forward()
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.logits, self.sampled = forward()


class RwkvSampler(Sampler):
    """FlashRWKV2 Rapid-Sampling with state isolated by request slot."""

    def __init__(
        self,
        sampler: Sampler,
        model: nn.Module,
        capture_graphs: bool,
    ) -> None:
        self.__dict__.update(sampler.__dict__)
        flashrwkv2 = _load_flashrwkv2()
        self._sample = flashrwkv2.infer_sampling_six_parameter_forward_varlen
        self._setup_sampling_states = flashrwkv2.setup_sampling_states
        max_num_reqs = self.sampling_states.max_num_reqs
        vocab_size = self.sampling_states.vocab_size
        device = self.req_states.device

        cast(Any, self.penalties_state).output_bin_counts = None
        self._rapid_penalties = torch.zeros(
            max_num_reqs, vocab_size, dtype=torch.float32, device=device
        )
        self._rapid_sampling_states = self._setup_sampling_states(0, max_num_reqs)
        self._presence_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self._frequency_penalty = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self._penalty_decay = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self._penalty_decay.np.fill(1.0)
        self._penalty_decay.copy_to_uva()
        self._active_rows = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._active_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._preserved_requests: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._new_requests: list[
            tuple[int, tuple[torch.Tensor, torch.Tensor] | None]
        ] = []
        self._input_batch: InputBatch | None = None
        self._sampling_graphs: dict[int, _RwkvSamplingGraph] = {}
        if capture_graphs:
            capacities = (1, 2, 4, 8, 16, 32, 64, 128, 256, 320, 512)
            for capacity in capacities:
                if capacity <= max_num_reqs:
                    self._sampling_graphs[capacity] = _RwkvSamplingGraph(
                        model,
                        self._sample,
                        self._rapid_penalties,
                        self._rapid_sampling_states,
                        capacity,
                    )
            if max_num_reqs not in self._sampling_graphs:
                self._sampling_graphs[max_num_reqs] = _RwkvSamplingGraph(
                    model,
                    self._sample,
                    self._rapid_penalties,
                    self._rapid_sampling_states,
                    max_num_reqs,
                )

    def sample_hidden_states(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any | None,
    ) -> SamplerOutput | None:
        idx_mapping_np = input_batch.idx_mapping_np
        if (
            grammar_output is not None
            or self.compute_nans
            or self.get_logprobs_dims(idx_mapping_np) is not None
            or np.any(self.needs_logits_processing[idx_mapping_np])
            or not self._sampling_graphs
        ):
            return None

        num_reqs = input_batch.num_reqs
        finishes_prefill = (
            input_batch.num_computed_prefill_tokens_np[:num_reqs]
            + input_batch.num_scheduled_tokens[:num_reqs]
            >= input_batch.prefill_len_np[:num_reqs]
        )
        active_rows_np = np.flatnonzero(finishes_prefill)
        num_active = len(active_rows_np)
        if num_active == 0:
            return None
        capacity = next(size for size in self._sampling_graphs if size >= num_active)
        sampling_graph = self._sampling_graphs[capacity]
        active_slots_np = idx_mapping_np[active_rows_np]
        self._active_rows.np[:num_active] = active_rows_np
        self._active_slots.np[:num_active] = active_slots_np
        active_rows = self._active_rows.copy_to_uva(num_active)
        active_slots = self._active_slots.copy_to_uva(num_active)
        active_slots_i64 = input_batch.idx_mapping[active_rows]

        sampling_graph.hidden_states[:num_active].copy_(
            hidden_states.index_select(0, active_rows)
        )
        sampling_graph.slot_indices[:num_active].copy_(active_slots)
        sampling_graph.presence_penalty[:num_active].copy_(
            self._presence_penalty.gpu[active_slots_i64]
        )
        sampling_graph.frequency_penalty[:num_active].copy_(
            self._frequency_penalty.gpu[active_slots_i64]
        )
        sampling_graph.penalty_decay[:num_active].copy_(
            self._penalty_decay.gpu[active_slots_i64]
        )
        sampling_graph.temperature[:num_active].copy_(
            self.sampling_states.temperature.gpu[active_slots_i64]
        )
        sampling_graph.top_k[:num_active].copy_(
            self.sampling_states.top_k.gpu[active_slots_i64]
        )
        sampling_graph.top_p[:num_active].copy_(
            self.sampling_states.top_p.gpu[active_slots_i64]
        )
        sampling_graph.num_active.fill_(num_active)
        sampling_graph.graph.replay()

        sampled = torch.zeros(num_reqs, dtype=torch.int64, device=hidden_states.device)
        sampled[active_rows] = sampling_graph.sampled[:num_active].to(torch.int64)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def stage_request(self, req_idx: int, req_id: str) -> None:
        self._new_requests.append((req_idx, self._preserved_requests.pop(req_id, None)))

    def preempt_request(self, req_id: str) -> None:
        req_idx = self.req_states.req_id_to_index[req_id]
        self._preserved_requests[req_id] = (
            self._rapid_sampling_states[req_idx].to(device="cpu", copy=True),
            self._rapid_penalties[req_idx].to(device="cpu", copy=True),
        )

    def remove_request(self, req_id: str) -> None:
        if req_id not in self.req_states.req_id_to_index:
            self._preserved_requests.pop(req_id, None)

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(req_idx, sampling_params)

        self._presence_penalty.np[req_idx] = sampling_params.presence_penalty
        self._frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self._penalty_decay.np[req_idx] = sampling_params.penalty_decay
        self.needs_logits_processing[req_idx] = (
            self.logit_bias_state.use_logit_bias[req_idx]
            or self.bad_words_state.num_bad_words.np[req_idx] > 0
            or (
                self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx]
            )
        )

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()
        self.thinking_budget_state.apply_staged_writes()
        self._presence_penalty.copy_to_uva()
        self._frequency_penalty.copy_to_uva()
        self._penalty_decay.copy_to_uva()

        for req_idx, preserved in self._new_requests:
            if preserved is None:
                seed = int(self.sampling_states.seeds.np[req_idx])
                request_state = self._setup_sampling_states(seed, 1)
                self._rapid_sampling_states[req_idx].copy_(request_state[0])
                self._rapid_penalties[req_idx].zero_()
            else:
                request_state, penalties = preserved
                self._rapid_sampling_states[req_idx].copy_(request_state)
                self._rapid_penalties[req_idx].copy_(penalties)
        self._new_requests.clear()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        self._input_batch = input_batch
        return super().__call__(logits, input_batch)

    def _apply_logits_processors(
        self,
        logits: torch.Tensor,
        slots: torch.Tensor,
        slots_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        local_pos: torch.Tensor,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[slots_np]):
            return logits

        self.logit_bias_state.apply_logit_bias(logits, slots, slots_np, pos)
        self.bad_words_state.apply_bad_words(
            logits, slots, slots_np, input_ids, local_pos
        )
        self.thinking_budget_state.apply(
            logits,
            slots,
            slots,
            slots_np,
            input_ids,
            local_pos,
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del return_logprobs
        input_batch = self._input_batch
        assert input_batch is not None
        if expanded_idx_mapping.shape != idx_mapping.shape:
            raise ValueError("RWKV Rapid-Sampling does not support expanded logits")

        num_reqs = input_batch.num_reqs
        finishes_prefill = (
            input_batch.num_computed_prefill_tokens_np[:num_reqs]
            + input_batch.num_scheduled_tokens[:num_reqs]
            >= input_batch.prefill_len_np[:num_reqs]
        )
        active_rows_np = np.flatnonzero(finishes_prefill)
        num_active = len(active_rows_np)
        sampled = torch.zeros(num_reqs, dtype=torch.int64, device=logits.device)
        if num_active == 0:
            return sampled, logits

        active_slots_np = idx_mapping_np[active_rows_np]
        self._active_rows.np[:num_active] = active_rows_np
        self._active_slots.np[:num_active] = active_slots_np
        active_rows = self._active_rows.copy_to_uva(num_active)
        active_slots = self._active_slots.copy_to_uva(num_active)
        active_slots_i64 = idx_mapping[active_rows]
        active_logits = logits if num_active == num_reqs else logits[active_rows]
        active_logits = self._apply_logits_processors(
            active_logits,
            active_slots_i64,
            active_slots_np,
            pos[active_rows],
            input_ids[active_rows],
            expanded_local_pos[active_rows],
        )
        sampled_active = self._sample(
            active_logits,
            self._rapid_penalties,
            self._rapid_sampling_states,
            active_slots,
            presence_penalty=self._presence_penalty.gpu[active_slots_i64],
            frequency_penalty=self._frequency_penalty.gpu[active_slots_i64],
            penalty_decay=self._penalty_decay.gpu[active_slots_i64],
            temperature=self.sampling_states.temperature.gpu[active_slots_i64],
            top_k=self.sampling_states.top_k.gpu[active_slots_i64],
            top_p=self.sampling_states.top_p.gpu[active_slots_i64],
        )
        sampled[active_rows] = sampled_active.to(torch.int64)
        return sampled, logits


class RwkvModelState(DefaultModelState):
    """Route scheduler block lifecycle events through FlashRWKV2 state handles."""

    requires_cudagraph_max_query_len = True

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
        self._sampler: RwkvSampler | None = None
        self._model = model
        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        self._capture_sampling_graphs = bool(
            cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs()
        )

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        self._sampler = RwkvSampler(
            sampler,
            self._model,
            self._capture_sampling_graphs,
        )
        return self._sampler, None

    def custom_sample(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: Any | None,
    ) -> SamplerOutput | None:
        del model
        if self._sampler is None:
            return None
        return self._sampler.sample_hidden_states(
            hidden_states,
            input_batch,
            grammar_output,
        )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        if self._sampler is not None:
            self._sampler.stage_request(req_index, new_req_data.req_id)
        self._state_block_columns[req_index] = (
            new_req_data.num_computed_tokens - 1
        ) // self._block_size
        self._state_num_computed_tokens[req_index] = new_req_data.num_computed_tokens

    def preempt_request(self, req_id: str) -> None:
        if self._sampler is not None:
            self._sampler.preempt_request(req_id)

    def remove_request(self, req_id: str) -> None:
        super().remove_request(req_id)
        if self._sampler is not None:
            self._sampler.remove_request(req_id)

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
