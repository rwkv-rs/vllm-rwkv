# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 request state for RWKV recurrent-state lifecycle management."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import regex as re
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.model_executor.models.rwkv import RwkvStateSnapshot, _load_flashrwkv2
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import FinishedRequestData, NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsContext
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

RWKV_STATE_CACHE_PROTOCOL = "vllm-rwkv.state-cache.v1"
RWKV_STATE_READ_REF_ARG = "rwkv_state_read_ref"
RWKV_STATE_READ_LEASE_ARG = "rwkv_state_read_lease"
RWKV_STATE_WRITE_REF_ARG = "rwkv_state_write_ref"
_STATE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_COMMIT_FINISH_REASONS = frozenset({"stop", "length", "repetition"})


@dataclass(frozen=True)
class _RwkvStateCacheEntry:
    snapshot: RwkvStateSnapshot
    parent_ref: str | None
    processed_token_count: int
    pending_tail_token_ids: tuple[int, ...]
    finish_reason: str

    @property
    def nbytes(self) -> int:
        return self.snapshot.nbytes


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
        if self.logits_processors[3:]:
            raise ValueError(
                "RWKV Rapid-Sampling does not support custom logits processors"
            )
        flashrwkv2 = _load_flashrwkv2()
        self._sample = flashrwkv2.infer_sampling_six_parameter_forward_varlen
        self._setup_sampling_states = flashrwkv2.setup_sampling_states
        max_num_reqs = self.sampling_states.max_num_reqs
        vocab_size = self.sampling_states.vocab_size
        device = self.req_states.device

        self._logit_bias_state = cast(LogitBiasState, self.logits_processors[0])
        self._bad_words_state = cast(BadWordsState, self.logits_processors[2])
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

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)
        self._logit_bias_state.add_request(req_idx, sampling_params)
        self._bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        self.thinking_budget_state.add_request(req_idx, sampling_params)

        self._presence_penalty.np[req_idx] = sampling_params.presence_penalty
        self._frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self._penalty_decay.np[req_idx] = sampling_params.penalty_decay
        self.needs_logits_processing[req_idx] = (
            self._logit_bias_state.use_logit_bias[req_idx]
            or self._bad_words_state.num_bad_words.np[req_idx] > 0
            or (
                self.thinking_budget_state.enabled
                and self.thinking_budget_state.use_thinking_budget[req_idx]
            )
        )

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self._logit_bias_state.apply_staged_writes()
        self._bad_words_state.apply_staged_writes()
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
        seq_lens_upper_bound_np: np.ndarray,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[slots_np]):
            return logits

        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
        ctx = LogitsContext(
            expanded_idx_mapping=slots,
            idx_mapping=slots,
            idx_mapping_np=slots_np,
            expanded_local_pos=local_pos,
            input_ids=input_ids,
            pos=pos,
            seq_lens_upper_bound_np=seq_lens_upper_bound_np,
        )
        logits = self._logit_bias_state.apply(logits, ctx)
        logits = self._bad_words_state.apply(logits, ctx)
        self.thinking_budget_state.apply(logits, ctx)
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
        seq_lens_upper_bound_np: np.ndarray,
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
            seq_lens_upper_bound_np[active_rows_np],
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
        self._state_cache: dict[str, _RwkvStateCacheEntry] = {}
        self._state_snapshot_refcounts: dict[int, int] = {}
        self._active_state_readers: dict[str, int] = {}
        self._prepared_state_reads_by_lease: dict[str, str] = {}
        self._state_cache_bytes = 0
        self._reserved_state_cache_bytes = 0
        self._state_cache_max_bytes = int(envs.VLLM_RWKV_STATE_CACHE_MAX_BYTES)
        if self._state_cache_max_bytes < 0:
            raise ValueError("VLLM_RWKV_STATE_CACHE_MAX_BYTES must not be negative")
        if (
            self._state_cache_max_bytes
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            raise ValueError("RWKV State refs do not yet support data parallelism")
        self._req_index_by_id: dict[str, int] = {}
        self._pending_read_ref_by_index: dict[int, str] = {}
        self._pending_write_ref_by_id: dict[str, str] = {}
        self._prepared_state_bytes_by_ref: dict[str, int] = {}
        self._reserved_state_bytes_by_req_id: dict[str, int] = {}
        self._base_processed_tokens_by_req_id: dict[str, int] = {}
        self._parent_state_ref_by_req_id: dict[str, str] = {}
        self._preempted_req_ids: set[str] = set()
        self._active_state_slots = torch.full(
            (self.max_num_reqs,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._has_active_state_slot = np.zeros(self.max_num_reqs, dtype=np.bool_)
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
        sampling_params = new_req_data.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        extra_args = extra_args or {}
        read_ref = self._validate_state_ref(
            extra_args.get(RWKV_STATE_READ_REF_ARG),
            field=RWKV_STATE_READ_REF_ARG,
            allow_empty=True,
        )
        read_lease = self._validate_state_ref(
            extra_args.get(RWKV_STATE_READ_LEASE_ARG),
            field=RWKV_STATE_READ_LEASE_ARG,
            allow_empty=True,
        )
        write_ref = self._validate_state_ref(
            extra_args.get(RWKV_STATE_WRITE_REF_ARG),
            field=RWKV_STATE_WRITE_REF_ARG,
            allow_empty=True,
        )
        resuming = new_req_data.req_id in self._preempted_req_ids
        if resuming and (
            read_ref != self._parent_state_ref_by_req_id.get(new_req_data.req_id, "")
            or write_ref != self._pending_write_ref_by_id.get(new_req_data.req_id, "")
        ):
            raise ValueError("RWKV State refs changed while the request was preempted")
        if (read_ref or write_ref) and self._state_cache_max_bytes == 0:
            raise RuntimeError(
                "RWKV State refs are disabled; set "
                "VLLM_RWKV_STATE_CACHE_MAX_BYTES to a positive byte limit"
            )
        if read_lease and not read_ref:
            raise ValueError("rwkv_state_read_lease requires rwkv_state_read_ref")
        if read_ref:
            if read_ref not in self._state_cache:
                raise KeyError(f"Unknown RWKV State ref: {read_ref}")
            if read_lease and not resuming:
                prepared_ref = self._prepared_state_reads_by_lease.get(read_lease)
                if prepared_ref != read_ref:
                    raise ValueError("RWKV State read lease is missing or mismatched")
            pending_tail = self._state_cache[read_ref].pending_tail_token_ids
            prefill_token_ids = new_req_data.prefill_token_ids or ()
            if tuple(prefill_token_ids[: len(pending_tail)]) != pending_tail:
                raise ValueError(
                    "RWKV State restore input is missing its pending tail tokens"
                )
        if write_ref and not resuming:
            if write_ref == read_ref:
                raise ValueError("RWKV State read/write refs must be different")
            if write_ref in self._state_cache or write_ref in (
                self._pending_write_ref_by_id.values()
            ):
                raise ValueError(f"RWKV State ref already exists: {write_ref}")
            required = self._prepared_state_bytes_by_ref.pop(write_ref, 0)
            if not required:
                required = self._reserve_state_ref()
            self._reserved_state_bytes_by_req_id[new_req_data.req_id] = required
            self._pending_write_ref_by_id[new_req_data.req_id] = write_ref
        if read_ref:
            if read_lease and not resuming:
                del self._prepared_state_reads_by_lease[read_lease]
            if new_req_data.num_computed_tokens == 0:
                self._pending_read_ref_by_index[req_index] = read_ref
            if not resuming:
                self._active_state_readers[read_ref] = (
                    self._active_state_readers.get(read_ref, 0) + 1
                )
                self._base_processed_tokens_by_req_id[new_req_data.req_id] = (
                    self._state_cache[read_ref].processed_token_count
                )
                self._parent_state_ref_by_req_id[new_req_data.req_id] = read_ref
        self._preempted_req_ids.discard(new_req_data.req_id)
        self._req_index_by_id[new_req_data.req_id] = req_index
        self._has_active_state_slot[req_index] = False
        if self._sampler is not None:
            self._sampler.stage_request(req_index, new_req_data.req_id)
        self._state_block_columns[req_index] = (
            new_req_data.num_computed_tokens - 1
        ) // self._block_size
        self._state_num_computed_tokens[req_index] = new_req_data.num_computed_tokens

    def preempt_request(self, req_id: str) -> None:
        self._preempted_req_ids.add(req_id)
        if self._sampler is not None:
            self._sampler.preempt_request(req_id)

    def remove_request(
        self,
        req_id: str,
        finished_data: FinishedRequestData | None = None,
    ) -> None:
        finish_reason = None if finished_data is None else finished_data.finish_reason
        pending_tail_token_ids = (
            () if finished_data is None else finished_data.pending_tail_token_ids
        )
        req_index = self._req_index_by_id.pop(req_id, None)
        preempted = req_id in self._preempted_req_ids and finished_data is None
        write_ref = ""
        reserved = 0
        parent_ref = ""
        if not preempted:
            self._preempted_req_ids.discard(req_id)
            write_ref = self._pending_write_ref_by_id.pop(req_id, "")
            reserved = self._reserved_state_bytes_by_req_id.pop(req_id, 0)
            parent_ref = self._parent_state_ref_by_req_id.pop(req_id, "")
        if parent_ref:
            active_readers = self._active_state_readers[parent_ref] - 1
            if active_readers:
                self._active_state_readers[parent_ref] = active_readers
            else:
                del self._active_state_readers[parent_ref]
        if req_index is not None:
            self._pending_read_ref_by_index.pop(req_index, None)
            if (
                write_ref
                and not preempted
                and finish_reason in _COMMIT_FINISH_REASONS
                and self._has_active_state_slot[req_index]
            ):
                processed_token_count = self._base_processed_tokens_by_req_id.get(
                    req_id, 0
                ) + int(self._state_num_computed_tokens[req_index])
                try:
                    self.capture_state_ref(
                        write_ref,
                        self._active_state_slots[req_index : req_index + 1],
                        parent_ref=parent_ref or None,
                        processed_token_count=processed_token_count,
                        pending_tail_token_ids=pending_tail_token_ids,
                        finish_reason=finish_reason,
                        reserved_nbytes=reserved,
                    )
                except Exception:
                    self._reserved_state_cache_bytes -= reserved
                    reserved = 0
                    raise
                else:
                    reserved = 0
            self._has_active_state_slot[req_index] = False
        if not preempted:
            self._base_processed_tokens_by_req_id.pop(req_id, None)
        if reserved:
            self._reserved_state_cache_bytes -= reserved
        super().remove_request(req_id, finished_data)
        if self._sampler is not None:
            self._sampler.remove_request(req_id)

    @staticmethod
    def _validate_state_ref(
        value: object,
        *,
        field: str,
        allow_empty: bool = False,
    ) -> str:
        if value is None and allow_empty:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        state_ref = value.strip()
        if not state_ref and allow_empty:
            return ""
        if not _STATE_REF_PATTERN.fullmatch(state_ref):
            raise ValueError(f"{field} is invalid")
        return state_ref

    def _state_ref_record(self, state_ref: str) -> dict[str, Any]:
        entry = self._state_cache[state_ref]
        return {
            "protocol": RWKV_STATE_CACHE_PROTOCOL,
            "state_ref": state_ref,
            "nbytes": entry.nbytes,
            "parent_ref": entry.parent_ref,
            "processed_token_count": entry.processed_token_count,
            "pending_tail_token_count": len(entry.pending_tail_token_ids),
            "finish_reason": entry.finish_reason,
            "state_dtype": str(self._state_layer.state_dtype),
            "process_local": True,
            "durable": False,
            "scheduler_slots_reserved": 0,
        }

    def _state_ref_reader_count(self, state_ref: str) -> int:
        prepared = sum(
            prepared_ref == state_ref
            for prepared_ref in self._prepared_state_reads_by_lease.values()
        )
        return self._active_state_readers.get(state_ref, 0) + prepared

    def _reserve_state_ref(self) -> int:
        if self._state_cache_max_bytes == 0:
            raise RuntimeError(
                "RWKV State refs are disabled; set "
                "VLLM_RWKV_STATE_CACHE_MAX_BYTES to a positive byte limit"
            )
        required = self._state_layer.state_snapshot_nbytes
        if (
            self._state_cache_bytes + self._reserved_state_cache_bytes + required
            > self._state_cache_max_bytes
        ):
            raise MemoryError(
                "RWKV State cache capacity exceeded before generation: "
                f"used={self._state_cache_bytes}, "
                f"reserved={self._reserved_state_cache_bytes}, "
                f"required={required}, limit={self._state_cache_max_bytes}"
            )
        self._reserved_state_cache_bytes += required
        return required

    def _store_state_ref(
        self,
        state_ref: str,
        entry: _RwkvStateCacheEntry,
        *,
        reserved_nbytes: int = 0,
    ) -> dict[str, Any]:
        if state_ref in self._state_cache:
            raise ValueError(f"RWKV State ref already exists: {state_ref}")
        snapshot_key = id(entry.snapshot)
        required = 0 if snapshot_key in self._state_snapshot_refcounts else entry.nbytes
        unreserved = self._reserved_state_cache_bytes - reserved_nbytes
        if (
            self._state_cache_bytes + unreserved + required
            > self._state_cache_max_bytes
        ):
            raise MemoryError(
                "RWKV State cache capacity exceeded: "
                f"used={self._state_cache_bytes}, reserved={unreserved}, "
                f"required={required}, "
                f"limit={self._state_cache_max_bytes}"
            )
        if reserved_nbytes:
            self._reserved_state_cache_bytes -= reserved_nbytes
        self._state_cache[state_ref] = entry
        self._state_cache_bytes += required
        self._state_snapshot_refcounts[snapshot_key] = (
            self._state_snapshot_refcounts.get(snapshot_key, 0) + 1
        )
        return self._state_ref_record(state_ref)

    def capture_state_ref(
        self,
        state_ref: str,
        slot_id: int | Sequence[int] | torch.Tensor,
        *,
        parent_ref: str | None = None,
        processed_token_count: int = 0,
        pending_tail_token_ids: Sequence[int] = (),
        finish_reason: str = "snapshot",
        reserved_nbytes: int = 0,
    ) -> dict[str, Any]:
        state_ref = self._validate_state_ref(state_ref, field="state_ref")
        snapshot = self._state_layer.capture_state_slot(slot_id)
        entry = _RwkvStateCacheEntry(
            snapshot=snapshot,
            parent_ref=parent_ref,
            processed_token_count=processed_token_count,
            pending_tail_token_ids=tuple(pending_tail_token_ids),
            finish_reason=finish_reason,
        )
        return self._store_state_ref(state_ref, entry, reserved_nbytes=reserved_nbytes)

    def state_cache_action(
        self,
        action: str,
        state_ref: str = "",
        target_ref: str = "",
    ) -> dict[str, Any]:
        if action == "capabilities":
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "supported": True,
                "enabled": self._state_cache_max_bytes > 0,
                "process_local": True,
                "durable": False,
                "scheduler_slots_reserved_per_ref": 0,
                "max_bytes": self._state_cache_max_bytes,
                "used_bytes": self._state_cache_bytes,
                "reserved_bytes": self._reserved_state_cache_bytes,
                "snapshot_nbytes": self._state_layer.state_snapshot_nbytes,
                "state_count": len(self._state_cache),
                "prepared_read_count": len(self._prepared_state_reads_by_lease),
            }
        state_ref = self._validate_state_ref(state_ref, field="state_ref")
        if action == "inspect_if_exists":
            if state_ref not in self._state_cache:
                return {
                    "protocol": RWKV_STATE_CACHE_PROTOCOL,
                    "state_ref": state_ref,
                    "exists": False,
                }
            return {**self._state_ref_record(state_ref), "exists": True}
        if action == "prepare_read":
            read_lease = self._validate_state_ref(target_ref, field="read_lease")
            if state_ref not in self._state_cache:
                raise KeyError(f"Unknown RWKV State ref: {state_ref}")
            if read_lease in self._prepared_state_reads_by_lease:
                raise ValueError(f"RWKV State read lease already exists: {read_lease}")
            self._prepared_state_reads_by_lease[read_lease] = state_ref
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "read_lease": read_lease,
                "ready": True,
            }
        if action == "cancel_read":
            read_lease = self._validate_state_ref(target_ref, field="read_lease")
            canceled = self._prepared_state_reads_by_lease.get(read_lease) == state_ref
            if canceled:
                del self._prepared_state_reads_by_lease[read_lease]
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "read_lease": read_lease,
                "canceled": canceled,
            }
        if action == "prepare_write":
            if (
                state_ref in self._state_cache
                or state_ref in self._pending_write_ref_by_id.values()
                or state_ref in self._prepared_state_bytes_by_ref
            ):
                raise ValueError(f"RWKV State ref already exists: {state_ref}")
            reserved = self._reserve_state_ref()
            self._prepared_state_bytes_by_ref[state_ref] = reserved
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "ready": True,
                "reserved_bytes": reserved,
            }
        if action == "cancel_write":
            reserved = self._prepared_state_bytes_by_ref.pop(state_ref, 0)
            self._reserved_state_cache_bytes -= reserved
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "canceled": bool(reserved),
                "reserved_bytes": self._reserved_state_cache_bytes,
            }
        if action == "drop_if_exists" and state_ref not in self._state_cache:
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "dropped": False,
                "used_bytes": self._state_cache_bytes,
            }
        if state_ref not in self._state_cache:
            raise KeyError(f"Unknown RWKV State ref: {state_ref}")
        if action == "prepare_drop":
            if self._state_ref_reader_count(state_ref):
                raise ValueError(f"RWKV State ref is in use: {state_ref}")
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "ready": True,
            }
        if action == "prepare_clone":
            target_ref = self._validate_state_ref(target_ref, field="target_ref")
            if (
                target_ref in self._state_cache
                or target_ref in self._pending_write_ref_by_id.values()
                or target_ref in self._prepared_state_bytes_by_ref
            ):
                raise ValueError(f"RWKV State ref already exists: {target_ref}")
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "target_ref": target_ref,
                "ready": True,
            }
        if action == "inspect":
            return self._state_ref_record(state_ref)
        if action in {"drop", "drop_if_exists"}:
            if self._state_ref_reader_count(state_ref):
                raise ValueError(f"RWKV State ref is in use: {state_ref}")
            entry = self._state_cache.pop(state_ref)
            snapshot_key = id(entry.snapshot)
            refcount = self._state_snapshot_refcounts[snapshot_key] - 1
            if refcount:
                self._state_snapshot_refcounts[snapshot_key] = refcount
            else:
                del self._state_snapshot_refcounts[snapshot_key]
                self._state_cache_bytes -= entry.nbytes
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "dropped": True,
                "used_bytes": self._state_cache_bytes,
            }
        if action == "clone":
            target_ref = self._validate_state_ref(target_ref, field="target_ref")
            source = self._state_cache[state_ref]
            entry = _RwkvStateCacheEntry(
                snapshot=source.snapshot,
                parent_ref=state_ref,
                processed_token_count=source.processed_token_count,
                pending_tail_token_ids=source.pending_tail_token_ids,
                finish_reason=source.finish_reason,
            )
            return self._store_state_ref(target_ref, entry)
        if action == "restore_metadata":
            entry = self._state_cache[state_ref]
            return {
                **self._state_ref_record(state_ref),
                "pending_tail_token_ids": list(entry.pending_tail_token_ids),
            }
        raise ValueError(f"unsupported RWKV State cache action: {action}")

    def clear_state_cache(self) -> None:
        """Invalidate detached state before changing model weights."""
        if not self._state_cache_max_bytes:
            return
        if self._req_index_by_id or self._preempted_req_ids:
            raise RuntimeError("Quiesce RWKV requests before updating model weights")
        self._state_cache.clear()
        self._state_snapshot_refcounts.clear()
        self._active_state_readers.clear()
        self._prepared_state_reads_by_lease.clear()
        self._prepared_state_bytes_by_ref.clear()
        self._state_cache_bytes = 0
        self._reserved_state_cache_bytes = 0

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
        if input_batch.num_reqs == 0:
            return
        if (
            not self._align_mode
            and not self._pending_read_ref_by_index
            and not self._pending_write_ref_by_id
        ):
            return

        num_reqs = input_batch.num_reqs
        req_indices = input_batch.idx_mapping_np[:num_reqs]
        computed = self._state_num_computed_tokens[req_indices]
        query_lens = input_batch.num_scheduled_tokens[:num_reqs]
        old_columns = self._state_block_columns[req_indices]
        new_columns = (computed + query_lens - 1) // self._block_size
        block_table = block_tables[self._get_rwkv_group_id(kv_cache_config)]
        reset_rows_np = np.flatnonzero(old_columns < 0)
        if reset_rows_np.size and self._pending_read_ref_by_index:
            zero_rows: list[int] = []
            for row in reset_rows_np.tolist():
                req_index = int(req_indices[row])
                state_ref = self._pending_read_ref_by_index.pop(req_index, "")
                if state_ref:
                    destination = (
                        block_table[row, int(new_columns[row])].reshape(1).contiguous()
                    )
                    self._state_layer.restore_state_slot(
                        self._state_cache[state_ref].snapshot, destination
                    )
                else:
                    zero_rows.append(row)
            reset_rows_np = np.asarray(zero_rows, dtype=np.int64)
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
        if self._pending_write_ref_by_id:
            rows = torch.arange(num_reqs, dtype=torch.int64, device=self.device)
            columns = torch.as_tensor(
                new_columns, dtype=torch.int64, device=self.device
            )
            active_slots = block_table[rows, columns].contiguous()
            request_indices = torch.as_tensor(
                req_indices, dtype=torch.int64, device=self.device
            )
            self._active_state_slots.index_copy_(0, request_indices, active_slots)
            self._has_active_state_slot[req_indices] = True
        self._state_block_columns[req_indices] = new_columns
        self._state_num_computed_tokens[req_indices] = computed + query_lens
        del num_computed_tokens
