# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.models.config import RwkvForCausalLMConfig
from vllm.model_executor.models.rwkv import RwkvChannelMixValue, RwkvStateLayer
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.rwkv_attn import (
    RwkvAttentionMetadata,
    RwkvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.model_states.rwkv import RwkvModelState


class _FakeStateHandle:
    def __init__(
        self,
        state_pool_size: int,
        channels: int,
        state_dtype: torch.dtype,
        has_elapsed: bool,
    ) -> None:
        self.state = torch.zeros((state_pool_size, channels), dtype=state_dtype)
        self.elapsed = (
            torch.zeros(state_pool_size, dtype=torch.int32) if has_elapsed else None
        )
        self.materialized: list[torch.Tensor] = []

    @property
    def memory_layout(self) -> dict[str, int]:
        return {
            "base_bytes_per_slot": 1,
            "private_bytes_per_slot": 17,
            "bytes_per_slot": 18,
            "fixed_workspace_nbytes": 13,
            "total_nbytes": self.state.shape[0] * 18 + 13,
        }

    def reset_slots_(self, indices: torch.Tensor) -> None:
        self.state.index_fill_(0, indices.long(), 0)
        if self.elapsed is not None:
            self.elapsed.index_fill_(0, indices.long(), 0)

    def copy_slots_(
        self,
        source: "_FakeStateHandle",
        source_indices: torch.Tensor,
        destination_indices: torch.Tensor,
    ) -> None:
        source_indices = source_indices.long()
        destination_indices = destination_indices.long()
        self.state.index_copy_(
            0, destination_indices, source.state.index_select(0, source_indices)
        )
        if self.elapsed is not None:
            assert source.elapsed is not None
            self.elapsed.index_copy_(
                0,
                destination_indices,
                source.elapsed.index_select(0, source_indices),
            )

    def materialize_slots_(self, indices: torch.Tensor) -> None:
        self.materialized.append(indices.clone())


def _fake_flashrwkv2() -> ModuleType:
    module: Any = ModuleType("flashrwkv2")
    module.__version__ = "0.1.0a10"

    def prepare_fp16(
        state_pool_size,
        channels,
        *,
        sequence_capacity,
        head_size=64,
        device=None,
    ):
        del sequence_capacity, head_size, device
        return _FakeStateHandle(state_pool_size, channels, torch.float16, True)

    def prepare_fp32(
        state_pool_size,
        channels,
        *,
        sequence_capacity,
        head_size=64,
        device=None,
    ):
        del sequence_capacity, head_size, device
        return _FakeStateHandle(state_pool_size, channels, torch.float32, False)

    module.prepare_tmix_wkv7_recurrent_fp16_state = prepare_fp16
    module.prepare_tmix_wkv7_recurrent_fp32io16_state = prepare_fp32
    module.prepare_tmix_wkv7_recurrent_metadata = lambda *args, **kwargs: (args, kwargs)
    for name in (
        "infer_cmix_forward_varlen",
        "infer_embedding_ln0_forward_varlen",
        "infer_post_norm_output_forward_varlen",
        "infer_tmix_postnorm_tokenshift_forward_varlen",
        "infer_tmix_readout_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
        "infer_tmix_wkv_prepare_forward_varlen",
    ):
        setattr(module, name, lambda *args, **kwargs: None)
    return module


def _state_vllm_config() -> SimpleNamespace:
    hf_config = SimpleNamespace(
        hidden_size=128,
        head_size=64,
        num_attention_heads=2,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype="float16",
            mamba_block_size=16,
            mamba_cache_mode="align",
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


def test_state_spec_accounts_provider_memory_and_slot_lifecycle(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config()
    state_layer = RwkvStateLayer(vllm_config, 2, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)

    assert (
        RwkvAttentionMetadataBuilder.get_cudagraph_support(vllm_config, spec)
        is AttentionCGSupport.ALWAYS
    )
    assert spec.provider_state_bytes_per_page == 36
    assert spec.provider_fixed_workspace_bytes == 26
    assert spec.num_prefill_checkpoint_blocks == 0

    num_slots = 3
    raw = torch.zeros((num_slots, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    tmix, handle, cmix = state_layer.get_layer_state(0)
    assert handle.state.untyped_storage().data_ptr() != raw.untyped_storage().data_ptr()
    tmix[1].fill_(1)
    handle.state[1].fill_(2)
    assert handle.elapsed is not None
    handle.elapsed[1].fill_(3)
    cmix[1].fill_(4)

    source = torch.tensor([1], dtype=torch.int32)
    destination = torch.tensor([2], dtype=torch.int32)
    state_layer.copy_state_slots(source, destination)
    assert torch.all(tmix[2] == 1)
    assert torch.all(handle.state[2] == 2)
    assert torch.all(handle.elapsed[2] == 3)
    assert torch.all(cmix[2] == 4)

    state_layer.reset_state_slots(destination)
    assert torch.count_nonzero(tmix[2]) == 0
    assert torch.count_nonzero(handle.state[2]) == 0
    assert torch.count_nonzero(handle.elapsed[2]) == 0
    assert torch.count_nonzero(cmix[2]) == 0

    tmix[0].fill_(5)
    tmix[1].fill_(1)
    handle.state[0].fill_(6)
    handle.state[1].fill_(2)
    handle.elapsed[0].fill_(7)
    handle.elapsed[1].fill_(3)
    cmix[0].fill_(8)
    cmix[1].fill_(4)
    state_layer.copy_state_slots(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    )
    assert torch.all(tmix[0] == 1) and torch.all(tmix[1] == 5)
    assert torch.all(handle.state[0] == 2) and torch.all(handle.state[1] == 6)
    assert torch.all(handle.elapsed[0] == 3) and torch.all(handle.elapsed[1] == 7)
    assert torch.all(cmix[0] == 4) and torch.all(cmix[1] == 8)

    state_layer.materialize_state_slots(source)
    handles = []
    for layer_idx in range(2):
        layer_handle = state_layer.get_layer_state(layer_idx)[1]
        assert isinstance(layer_handle, _FakeStateHandle)
        handles.append(layer_handle)
    assert all(
        len(layer_handle.materialized) == 1
        and torch.equal(layer_handle.materialized[0], source)
        for layer_handle in handles
    )


def test_live_metadata_reuses_provider_ticket_tensor_identity() -> None:
    builder = object.__new__(RwkvAttentionMetadataBuilder)
    builder._live_cu_seqlens = torch.empty(5, dtype=torch.int32)
    builder._live_state_indices = torch.empty(4, dtype=torch.int32)
    builder._live_cu_seqlens_views = {}
    builder._live_state_indices_views = {}
    builder._num_active_tokens = torch.empty(1, dtype=torch.int32)
    builder._num_active_sequences = torch.empty(1, dtype=torch.int32)
    builder._state_layer = SimpleNamespace(
        warmup_live_metadata=lambda metadata: None,
        prepare_metadata_ticket=lambda metadata: object(),
    )

    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=2,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )
    state_indices = torch.tensor([1, 2], dtype=torch.int32)

    first = builder._build_live(metadata, state_indices, for_capture=True)
    second = builder._build_live(metadata, state_indices, for_capture=True)
    runtime = builder._build_live(metadata, state_indices, for_capture=False)

    assert first.cu_seqlens is second.cu_seqlens
    assert first.state_indices is second.state_indices
    assert first.max_seqlen_capacity == first.token_capacity
    assert first.max_seqlen == first.token_capacity
    assert runtime.max_seqlen == runtime.token_capacity


def test_live_metadata_prepares_and_retains_each_capture_ticket(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config()
    state_layer = RwkvStateLayer(vllm_config, 1, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)
    raw = torch.zeros((1, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    metadata = RwkvAttentionMetadata(
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        state_indices=torch.tensor([0], dtype=torch.int32),
        batch_size=1,
        max_seqlen=1,
        token_capacity=1,
        sequence_capacity=1,
        max_seqlen_capacity=1,
        num_active_tokens=torch.tensor([0], dtype=torch.int32),
        num_active_sequences=torch.tensor([0], dtype=torch.int32),
        retain_ticket=True,
    )

    state_layer.warmup_live_metadata(metadata)
    warmup_ticket = state_layer._live_graph_tickets[-1]
    capture_ticket = state_layer.prepare_metadata_ticket(metadata)
    next_capture_ticket = state_layer.prepare_metadata_ticket(metadata)

    assert capture_ticket is not warmup_ticket
    assert next_capture_ticket is not capture_ticket
    assert len(state_layer._live_graph_tickets) == 3
    assert state_layer._live_graph_tickets[0] is warmup_ticket
    assert state_layer._live_graph_tickets[1] is capture_ticket
    assert state_layer._live_graph_tickets[2] is next_capture_ticket


def test_channel_mix_value_loads_checkpoint_weight_transposed() -> None:
    value = RwkvChannelMixValue(hidden_size=2, intermediate_size=8)
    checkpoint_weight = torch.arange(16, dtype=torch.float16).view(2, 8)
    assert value.load_weights([("weight", checkpoint_weight)]) == {"weight"}
    assert torch.equal(value.weight, checkpoint_weight.T.contiguous())


def test_align_state_advance_copies_only_crossing_requests() -> None:
    copies: list[tuple[torch.Tensor, torch.Tensor]] = []
    resets: list[torch.Tensor] = []
    state_layer = SimpleNamespace(
        prefix="model.layers.0.rwkv_state",
        copy_state_slots=lambda source, destination: copies.append(
            (source.clone(), destination.clone())
        ),
        reset_state_slots=lambda slots: resets.append(slots.clone()),
    )
    model_state = object.__new__(RwkvModelState)
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([0, 0], dtype=np.int32)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state.device = torch.device("cpu")

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        num_computed_tokens_np=np.array([15, 16], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
    )
    block_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    model_state.preprocess_state(
        input_batch,
        (block_table,),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )

    assert len(copies) == 1
    assert not resets
    assert torch.equal(copies[0][0], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(copies[0][1], torch.tensor([4], dtype=torch.int32))
    assert np.array_equal(
        model_state._state_block_columns, np.array([0, 1], dtype=np.int32)
    )


def test_align_state_advance_resets_new_request_destination() -> None:
    resets: list[torch.Tensor] = []
    state_layer = SimpleNamespace(
        prefix="model.layers.0.rwkv_state",
        copy_state_slots=lambda source, destination: None,
        reset_state_slots=lambda slots: resets.append(slots.clone()),
    )
    model_state = object.__new__(RwkvModelState)
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([-1], dtype=np.int32)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state.device = torch.device("cpu")

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        num_scheduled_tokens=np.array([32], dtype=np.int32),
    )
    block_table = torch.tensor([[7, 9]], dtype=torch.int32)
    model_state.preprocess_state(
        input_batch,
        (block_table,),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )

    assert len(resets) == 1
    assert torch.equal(resets[0], torch.tensor([9], dtype=torch.int32))
    assert np.array_equal(model_state._state_block_columns, np.array([1]))


def _config() -> SimpleNamespace:
    model_config = SimpleNamespace(
        dtype=torch.float16,
        architecture="RwkvForCausalLM",
        supports_mamba_prefix_caching=True,
        max_model_len=1024,
        enforce_eager=False,
        quantization=None,
        quantization_config=None,
    )
    cache_config = SimpleNamespace(
        enable_prefix_caching=True,
        mamba_cache_mode="none",
        mamba_block_size=None,
        block_size=16,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        kv_offloading_size=None,
        use_replayssm=False,
    )
    return SimpleNamespace(
        use_v2_model_runner=True,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            async_scheduling=None,
        ),
        speculative_config=None,
        lora_config=None,
        quant_config=None,
        kv_transfer_config=None,
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        compilation_config=SimpleNamespace(mode=None, cudagraph_mode=None),
    )


def test_config_uses_fp16_state_and_full_graph_defaults() -> None:
    vllm_config = _config()
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_dtype == "float16"
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float16"
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16
    assert not vllm_config.scheduler_config.async_scheduling
    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL


def test_config_accepts_fp32_wkv_state() -> None:
    vllm_config = _config()
    vllm_config.cache_config.mamba_ssm_cache_dtype = "float32"
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float32"


def test_config_accepts_pipeline_parallel_size_two() -> None:
    vllm_config = _config()
    vllm_config.parallel_config.pipeline_parallel_size = 2
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.parallel_config.pipeline_parallel_size == 2


def test_config_uses_eager_cuda_graph_mode_for_diagnostics() -> None:
    vllm_config = _config()
    vllm_config.model_config.enforce_eager = True
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


def test_config_rejects_tensor_parallelism() -> None:
    vllm_config = _config()
    vllm_config.parallel_config.tensor_parallel_size = 2
    try:
        RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    except ValueError as error:
        assert "tensor parallelism" in str(error)
    else:
        raise AssertionError("tensor parallelism must be rejected")


def test_config_rejects_async_scheduling() -> None:
    vllm_config = _config()
    vllm_config.scheduler_config.async_scheduling = True
    try:
        RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    except ValueError as error:
        assert "asynchronous scheduling" in str(error)
    else:
        raise AssertionError("asynchronous scheduling must be rejected")
