# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from functools import partial
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.model import ModelConfig
from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.engine.arg_utils import EngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.model_loader.checkpoint_weight_patch import (
    CheckpointWeightPatch,
    load_checkpoint_weight_patches,
)
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    load_planned,
    plan_loads,
    record_metadata_for_reloading,
    start_reload,
)
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.config import RwkvForCausalLMConfig
from vllm.model_executor.models.rwkv import (
    RwkvFeedForward,
    RwkvForCausalLM,
    RwkvModel,
    RwkvStateLayer,
    _load_flashrwkv2,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.rwkv_attn import (
    RwkvAttentionMetadata,
    RwkvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.model_states.rwkv import RwkvModelState, RwkvSampler


class _FakeStateHandle:
    def __init__(self, state: torch.Tensor, private_bytes_per_slot: int = 0) -> None:
        self.state = state
        self.private_bytes_per_slot = private_bytes_per_slot
        self.materialized: list[torch.Tensor] = []

    @property
    def memory_layout(self) -> dict[str, int]:
        per_slot = self.state[0].nbytes + self.private_bytes_per_slot
        return {
            "private_bytes_per_slot": self.private_bytes_per_slot,
            "bytes_per_slot": per_slot,
            "fixed_workspace_nbytes": 0,
            "total_nbytes": len(self.state) * per_slot,
        }

    def reset_slots_(self, indices: torch.Tensor) -> None:
        self.state.index_fill_(0, indices.long(), 0)

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

    def materialize_slots_(self, indices: torch.Tensor) -> None:
        self.materialized.append(indices.clone())


def _fake_flashrwkv2(fp16_private_bytes_per_slot: int = 0) -> ModuleType:
    module: Any = ModuleType("flashrwkv2")
    module.__version__ = "0.2.0a4"

    module.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor = _FakeStateHandle
    module.prepare_tmix_wkv7_recurrent_fp16_state_from_tensor = partial(
        _FakeStateHandle, private_bytes_per_slot=fp16_private_bytes_per_slot
    )
    module.prepare_tmix_wkv7_recurrent_metadata = lambda *args, **kwargs: (args, kwargs)
    for name in (
        "infer_cmix_forward_varlen",
        "infer_embedding_ln0_forward_varlen",
        "infer_head_linear_all_forward_varlen",
        "infer_post_norm_output_forward_varlen",
        "infer_sampling_six_parameter_forward_varlen",
        "infer_tmix_postnorm_tokenshift_forward_varlen",
        "infer_tmix_readout_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
        "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
        "infer_tmix_wkv_prepare_forward_varlen",
        "setup_sampling_states",
    ):
        setattr(module, name, lambda *args, **kwargs: None)
    return module


@pytest.fixture(autouse=True)
def _clear_flashrwkv2_cache():
    _load_flashrwkv2.cache_clear()
    yield
    _load_flashrwkv2.cache_clear()


def _state_vllm_config(state_dtype: str = "float32", heads: int = 2) -> SimpleNamespace:
    hf_config = SimpleNamespace(
        hidden_size=heads * 64,
        head_size=64,
        num_attention_heads=heads,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype=state_dtype,
            mamba_block_size=16,
            mamba_cache_mode="align",
        ),
        compilation_config=SimpleNamespace(static_forward_context={}),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )


@pytest.mark.parametrize(
    ("state_dtype", "heads", "private_bytes"),
    [
        ("float32", 40, 0),
        ("float16", 2, 0),
        ("float16", 64, 64 * 1540),
        ("float16", 64, 64 * 1284),
    ],
)
def test_state_spec_and_caller_backed_slot_lifecycle(
    monkeypatch, state_dtype, heads, private_bytes
) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2(private_bytes))
    vllm_config = _state_vllm_config(state_dtype, heads)
    state_layer = RwkvStateLayer(vllm_config, 2, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)

    assert (
        RwkvAttentionMetadataBuilder.get_cudagraph_support(vllm_config, spec)
        is AttentionCGSupport.ALWAYS
    )
    assert spec.num_prefill_checkpoint_blocks == 0
    assert spec.external_bytes_per_page == 2 * private_bytes

    num_slots = 3
    raw = torch.zeros((num_slots, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    tmix, handle, cmix = state_layer.get_layer_state(0)
    assert handle.state.shape == (num_slots, heads, 64, 64)
    assert handle.state.dtype == getattr(torch, state_dtype)
    assert spec.page_size_bytes == 2 * (
        2 * heads * 64 * torch.bfloat16.itemsize
        + heads * 64 * 64 * handle.state.element_size()
    )
    assert tmix.dtype == cmix.dtype == torch.bfloat16
    assert handle.state.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    tmix[1].fill_(1)
    handle.state[1].fill_(2)
    cmix[1].fill_(4)

    source = torch.tensor([1], dtype=torch.int32)
    destination = torch.tensor([2], dtype=torch.int32)
    state_layer.copy_state_slots(source, destination)
    assert torch.all(tmix[2] == 1)
    assert torch.all(handle.state[2] == 2)
    assert torch.all(cmix[2] == 4)

    state_layer.reset_state_slots(destination)
    assert torch.count_nonzero(tmix[2]) == 0
    assert torch.count_nonzero(handle.state[2]) == 0
    assert torch.count_nonzero(cmix[2]) == 0

    tmix[0].fill_(5)
    tmix[1].fill_(1)
    handle.state[0].fill_(6)
    handle.state[1].fill_(2)
    cmix[0].fill_(8)
    cmix[1].fill_(4)
    state_layer.copy_state_slots(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
    )
    assert torch.all(tmix[0] == 1) and torch.all(tmix[1] == 5)
    assert torch.all(handle.state[0] == 2) and torch.all(handle.state[1] == 6)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_public_rwkv_state_pool_transfers_on_model_stream() -> None:
    pytest.importorskip("flashrwkv2")
    config = _state_vllm_config("float32", 32)
    config.device_config.device = torch.device("cuda")
    layer = RwkvStateLayer(config, 2, "model.layers.0.rwkv_state")
    spec = layer.get_kv_cache_spec(config)
    backing = torch.zeros((3, spec.page_size_bytes), device="cuda", dtype=torch.uint8)
    layer.bind_kv_cache(backing)
    src = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    dst = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    tmix, _, cmix = layer.get_layer_state(0)
    tmix[0].fill_(1)
    cmix[0].fill_(2)
    shift_bytes = (tmix.nbytes + cmix.nbytes) * 2
    wkv = backing.view(-1)[shift_bytes:].view(torch.float32).view(2, 3, 32, 64, 64)
    wkv[:, 0].fill_(3)
    wkv[:, 1].fill_(4)
    wkv[:, 2].fill_(5)
    reset = torch.tensor([2], dtype=torch.int32, device="cuda")

    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer.copy_state_slots(src, dst)
        layer.reset_state_slots(reset)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert torch.equal(tmix[1], torch.ones_like(tmix[1]))
    assert torch.equal(cmix[1], torch.full_like(cmix[1], 2))
    assert torch.equal(wkv[:, 0], torch.full_like(wkv[:, 0], 4))
    assert torch.equal(wkv[:, 1], torch.full_like(wkv[:, 1], 3))
    assert torch.count_nonzero(wkv[:, 2]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(("heads", "prefix_steps"), [(40, 1), (64, 1), (64, 2)])
def test_fp16_graph_prefix_copy_and_raw_slot_read(
    heads: int, prefix_steps: int
) -> None:
    provider = pytest.importorskip("flashrwkv2")
    batch, dim = 16, 64
    config = _state_vllm_config("float16", heads)
    config.device_config.device = torch.device("cuda")
    layer = RwkvStateLayer(config, 1, "model.layers.0.rwkv_state")
    spec = layer.get_kv_cache_spec(config)
    backing = torch.zeros(
        (batch + 1, spec.page_size_bytes), device="cuda", dtype=torch.uint8
    )
    layer.bind_kv_cache(backing)
    _, state, _ = layer.get_layer_state(0)
    assert spec.external_bytes_per_page == state.memory_layout["private_bytes_per_slot"]
    assert spec.external_bytes_per_page > 0
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    count = torch.zeros(1, device="cuda", dtype=torch.int32)
    metadata = RwkvAttentionMetadata(
        cu_seqlens=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
        state_indices=slots.clone(),
        batch_size=batch,
        max_seqlen=1,
        token_capacity=batch,
        sequence_capacity=batch,
        max_seqlen_capacity=1,
        num_active_tokens=count,
        num_active_sequences=count,
        retain_ticket=True,
    )
    ones = torch.ones(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros_like(ones)

    def decode():
        return provider.infer_tmix_wkv7_recurrent_fp16_forward_varlen(
            ones,
            zeros,
            ones,
            ones,
            zeros,
            zeros,
            state=state,
            cu_seqlens=metadata.cu_seqlens,
            state_indices=metadata.state_indices,
            validated_metadata=layer.prepare_metadata_ticket(metadata),
            max_seqlen=1,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        decode()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = decode()

    count.fill_(batch)
    graph.replay()
    base = (
        backing.view(-1)[2 * (batch + 1) * heads * dim * 2 :]
        .view(torch.float16)
        .view(batch + 1, heads, dim, dim)
    )
    layer.materialize_state_slots([2])
    assert torch.all(base[2] == 1)
    if prefix_steps == 2:
        graph.replay()
        # Slot 1 accumulates both updates. Slot 2 was materialized between them.
        layer.materialize_state_slots([1, 2])
        assert torch.equal(base[1], base[2])
        assert torch.all(base[2] > 1)
    layer.copy_state_block_pairs([(0, batch)])
    layer.reset_state_slots([0])
    metadata.state_indices[0] = batch
    graph.replay()
    assert torch.equal(output[0], output[1])
    layer.materialize_state_slots([batch, 1, 0])
    assert torch.equal(base[batch], base[1])
    assert torch.all(base[batch] > 1)
    assert torch.count_nonzero(base[0]) == 0


def test_live_metadata_reuses_tensors_per_graph_descriptor() -> None:
    builder = object.__new__(RwkvAttentionMetadataBuilder)
    builder._device = torch.device("cpu")
    builder._live_buffers = {}
    builder._live_query_layouts = {}
    builder._captured_descriptors = set()
    prepared_ticket = object()
    builder._state_layer = SimpleNamespace(
        warmup_live_metadata=lambda metadata: None,
        prepare_metadata_ticket=lambda metadata: prepared_ticket,
    )

    query_start_loc = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
    )
    state_indices = torch.tensor([1, 2], dtype=torch.int32)

    first = builder._build_live(metadata, state_indices, for_capture=True)
    second = builder._build_live(metadata, state_indices, for_capture=True)
    runtime_query_start_loc = torch.tensor([0, 3, 3], dtype=torch.int32)
    runtime_metadata = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=4,
        query_start_loc=runtime_query_start_loc,
        query_start_loc_cpu=runtime_query_start_loc,
    )
    runtime = builder._build_live(
        runtime_metadata,
        state_indices,
        for_capture=False,
    )
    versions = (
        runtime.cu_seqlens._version,
        runtime.num_active_tokens._version,
        runtime.num_active_sequences._version,
    )
    repeated_runtime = builder._build_live(
        runtime_metadata,
        state_indices,
        for_capture=False,
    )
    repeated_versions = (
        repeated_runtime.cu_seqlens._version,
        repeated_runtime.num_active_tokens._version,
        repeated_runtime.num_active_sequences._version,
    )
    changed_runtime = builder._build_live(
        metadata,
        state_indices,
        for_capture=False,
    )
    other_query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    other_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc=other_query_start_loc,
        query_start_loc_cpu=other_query_start_loc,
    )
    other = builder._build_live(
        other_metadata,
        torch.tensor([3], dtype=torch.int32),
        for_capture=True,
    )

    assert first.cu_seqlens is second.cu_seqlens
    assert first.state_indices is second.state_indices
    assert first.cu_seqlens is runtime.cu_seqlens
    assert first.state_indices is runtime.state_indices
    assert repeated_runtime.cu_seqlens is runtime.cu_seqlens
    assert versions == repeated_versions
    assert repeated_versions == (
        changed_runtime.cu_seqlens._version - 1,
        changed_runtime.num_active_tokens._version - 1,
        changed_runtime.num_active_sequences._version - 1,
    )
    assert changed_runtime.num_active_sequences.item() == 2
    assert other.cu_seqlens is not first.cu_seqlens
    assert other.state_indices is not first.state_indices
    assert first.max_seqlen_capacity == metadata.max_query_len
    assert first.max_seqlen == metadata.max_query_len
    assert runtime.max_seqlen == runtime_metadata.max_query_len
    assert runtime.validated_metadata is None

    eager_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
    eager_metadata = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=3,
        max_query_len=3,
        query_start_loc=eager_query_start_loc,
        query_start_loc_cpu=eager_query_start_loc,
    )
    eager = builder._build_live(
        eager_metadata,
        torch.tensor([4], dtype=torch.int32),
        for_capture=False,
    )
    assert eager.validated_metadata is prepared_ticket


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


@pytest.mark.usefixtures("_single_rank")
def test_feed_forward_loads_value_weight_in_provider_layout() -> None:
    feed_forward = RwkvFeedForward(
        SimpleNamespace(hidden_size=2, intermediate_size=8), "test.mlp"
    )
    checkpoint_weight = torch.arange(16, dtype=torch.bfloat16).view(2, 8)
    storage = feed_forward.value.weight.data_ptr()
    loaded = AutoWeightsLoader(feed_forward).load_weights(
        [("value.weight", checkpoint_weight)]
    )
    assert loaded == {"value.weight"}
    assert feed_forward.value.weight.data_ptr() == storage
    assert torch.equal(feed_forward.value.weight, checkpoint_weight.T.contiguous())


def test_model_folds_embedding_norm_after_loading(monkeypatch) -> None:
    module: Any = _fake_flashrwkv2()
    module.infer_embedding_ln0_forward_varlen = lambda embedding, weight, bias, eps: (
        torch.nn.functional.layer_norm(
            embedding, (embedding.shape[-1],), weight, bias, eps
        ).to(torch.bfloat16)
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    model = object.__new__(RwkvModel)
    torch.nn.Module.__init__(model)
    embedding = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    model.embed_tokens = SimpleNamespace(weight=torch.nn.Parameter(embedding.clone()))
    model.embedding_norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
    model.config = SimpleNamespace(layer_norm_epsilon=1e-5)
    model._embedding_norm_folded = False
    storage = model.embed_tokens.weight.data_ptr()
    expected = torch.nn.functional.layer_norm(
        embedding,
        (4,),
        model.embedding_norm.weight,
        model.embedding_norm.bias,
        model.config.layer_norm_epsilon,
    ).to(torch.bfloat16)

    model._fold_embedding_norm()

    assert model._embedding_norm_folded
    assert model.embed_tokens.weight.data_ptr() == storage
    assert model.embed_tokens.weight.dtype == torch.bfloat16
    assert torch.equal(model.embed_tokens.weight, expected)


@pytest.fixture
def _rwkv_reload_model(monkeypatch, _single_rank):
    provider: Any = _fake_flashrwkv2()
    provider.infer_embedding_ln0_forward_varlen = lambda embedding, weight, bias, eps: (
        torch.nn.functional.layer_norm(
            embedding, (embedding.shape[-1],), weight, bias, eps
        )
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", provider)
    config = SimpleNamespace(
        word_embeddings_untied_by_checkpoint=False,
        quantization=None,
        dtype=torch.bfloat16,
    )
    model = RwkvForCausalLM(vllm_config=_tiny_model_vllm_config(None))
    return model, config


def test_reused_rwkv_weight_plan_folds_each_new_embedding(
    monkeypatch, _rwkv_reload_model
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")
    model, config = _rwkv_reload_model
    cpu = torch.device("cpu")
    embedding = model.model.embed_tokens.weight
    norm = model.model.embedding_norm
    name = "model.embed_tokens.weight"
    generator = torch.Generator().manual_seed(0)

    def checkpoint() -> torch.Tensor:
        return torch.rand(embedding.shape, generator=generator, dtype=embedding.dtype)

    model.load_weights([(name, checkpoint())])
    process_weights_after_loading(model, config, cpu)
    storage = embedding.data_ptr()
    initial = embedding.clone()
    other_name = "model.norm.weight"
    other = model.model.norm.weight

    start_reload(model)
    plan = plan_loads(
        model,
        [
            ParamMeta(name, embedding.dtype, tuple(embedding.shape)),
            ParamMeta(other_name, other.dtype, tuple(other.shape)),
        ],
    )
    load_planned(model, plan, [(other_name, torch.ones_like(other))])
    finish_reload(model, config, cpu)
    assert torch.equal(embedding, initial)

    for _ in range(2):
        start_reload(model)
        raw = checkpoint()
        load_planned(model, plan, [(name, raw)])
        finish_reload(model, config, cpu)
        expected = torch.nn.functional.layer_norm(
            raw, (embedding.shape[-1],), norm.weight, norm.bias, norm.eps
        )
        assert torch.equal(embedding, expected)
        assert embedding.data_ptr() == storage


def test_rwkv_rejects_incomplete_embedding_reload_until_repaired(
    monkeypatch, _rwkv_reload_model
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    model, config = _rwkv_reload_model
    cpu = torch.device("cpu")
    embedding = model.model.embed_tokens.weight
    norm = model.model.embedding_norm
    raw = torch.rand(
        embedding.shape,
        generator=torch.Generator().manual_seed(1),
        dtype=embedding.dtype,
    )
    embedding_name = "model.embed_tokens.weight"
    record_metadata_for_reloading(model)
    model.load_weights([(embedding_name, raw)])
    process_weights_after_loading(model, config, cpu)
    replacement_norm = torch.full_like(norm.weight, 2)

    start_reload(model)
    model.load_weights([("model.embedding_norm.weight", replacement_norm)])
    for _ in range(2):
        with pytest.raises(ValueError, match="unfolded embedding weight"):
            finish_reload(model, config, cpu)

    model.load_weights([(embedding_name, raw)])
    finish_reload(model, config, cpu)
    expected = torch.nn.functional.layer_norm(
        raw, (embedding.shape[-1],), norm.weight, norm.bias, norm.eps
    )
    assert torch.equal(embedding, expected)


def test_rwkv_sparse_embedding_patch_requires_complete_checkpoint(_rwkv_reload_model):
    model, config = _rwkv_reload_model
    embedding = model.model.embed_tokens.weight
    name = "model.embed_tokens.weight"
    model.load_weights([(name, torch.rand_like(embedding))])
    process_weights_after_loading(model, config, torch.device("cpu"))
    folded = embedding.clone()
    patch = CheckpointWeightPatch(
        name=name,
        shape=tuple(embedding.shape),
        dtype=embedding.dtype,
        values=torch.zeros(1, dtype=embedding.dtype),
        indices=torch.zeros(1, dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="require a full load"):
        load_checkpoint_weight_patches(model, [patch])
    assert torch.equal(embedding, folded)


def test_align_state_advance_uses_execution_owned_token_counts() -> None:
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
    model_state._state_num_computed_tokens = np.array([15, 16], dtype=np.int64)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state.device = torch.device("cpu")

    input_batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        num_computed_tokens_np=np.array([31, 32], dtype=np.int32),
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
    assert np.array_equal(model_state._state_num_computed_tokens, np.array([16, 17]))


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
    model_state._state_num_computed_tokens = np.array([0], dtype=np.int64)
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
    assert np.array_equal(model_state._state_num_computed_tokens, np.array([32]))


def _config() -> SimpleNamespace:
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(),
        dtype=torch.bfloat16,
        architecture="RwkvForCausalLM",
        supports_mamba_prefix_caching=True,
        max_model_len=1024,
        enforce_eager=False,
        logprobs_mode="raw_logprobs",
        return_sampling_mask=False,
        enable_trace_replay=False,
        quantization=None,
        quantization_config=None,
    )
    cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_block_size=None,
        block_size=16,
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype="auto",
        kv_offloading_size=None,
        use_replayssm=False,
    )
    return SimpleNamespace(
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
            max_num_batched_tokens=8192,
            max_num_seqs=512,
        ),
        speculative_config=None,
        lora_config=None,
        quant_config=None,
        kv_transfer_config=None,
        mamba_config=SimpleNamespace(enable_stochastic_rounding=False),
        compilation_config=SimpleNamespace(
            mode=None,
            cudagraph_mode=None,
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=None,
        ),
    )


def test_config_uses_bf16_shift_fp16_wkv_and_full_graph_defaults() -> None:
    vllm_config = _config()
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_dtype == "bfloat16"
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float16"
    assert not vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.cache_config.mamba_cache_mode == "none"
    assert vllm_config.cache_config.mamba_block_size == 1024
    assert vllm_config.scheduler_config.async_scheduling is None
    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert vllm_config.model_config.hf_config.head_dtype == "float32"
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 8192
    assert 1024 in vllm_config.compilation_config.cudagraph_capture_sizes
    assert 5120 in vllm_config.compilation_config.cudagraph_capture_sizes
    assert 8192 in vllm_config.compilation_config.cudagraph_capture_sizes


@pytest.mark.parametrize("architecture", ["RwkvForCausalLM", "LlamaForCausalLM"])
@pytest.mark.parametrize("explicit", [None, True, False])
def test_rwkv_prefix_cache_default_preserves_explicit_choice(architecture, explicit):
    args = EngineArgs(enable_prefix_caching=explicit)
    config = SimpleNamespace(
        architecture=architecture,
        is_chunked_prefill_supported=True,
        is_prefix_caching_supported=True,
        runner_type="generate",
    )
    args._set_default_chunked_prefill_and_prefix_caching_args(config)
    expected = architecture != "RwkvForCausalLM" if explicit is None else explicit
    assert args.enable_prefix_caching is expected


@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
def test_config_accepts_explicit_rwkv_prefix_cache(state_dtype) -> None:
    vllm_config = _config()
    vllm_config.cache_config.enable_prefix_caching = True
    vllm_config.cache_config.mamba_ssm_cache_dtype = state_dtype
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == state_dtype
    assert vllm_config.cache_config.enable_prefix_caching
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", torch.float16, "--dtype bfloat16"),
        ("mamba_cache_dtype", "float16", "shift state requires bfloat16"),
        ("mamba_ssm_cache_dtype", "bfloat16", "WKV state supports float16 or float32"),
    ],
)
def test_config_requires_flashrwkv2_numerical_contract(field, value, message) -> None:
    vllm_config = _config()
    target = vllm_config.model_config if field == "dtype" else vllm_config.cache_config
    setattr(target, field, value)
    with pytest.raises(ValueError, match=message):
        RwkvForCausalLMConfig.verify_and_update_config(vllm_config)


def test_head_preserves_provider_fp32_logits_and_trims_padding(monkeypatch) -> None:
    module: Any = _fake_flashrwkv2()
    logits = torch.tensor([[1.0001, 1.0002, -4.0]], dtype=torch.float32)
    module.infer_head_linear_all_forward_varlen = lambda hidden, weight: logits
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=2),
        lm_head=SimpleNamespace(weight=torch.empty(3, 4, dtype=torch.bfloat16)),
        logits_processor=lambda head, values: values,
    )
    result = RwkvForCausalLM.compute_logits(
        model, torch.zeros(1, 4, dtype=torch.bfloat16)
    )
    assert result.dtype == torch.float32
    assert torch.equal(result, logits[:, :2])
    assert result[0, 1] > result[0, 0]


_LORA = {
    "r": 4,
    "lora_alpha": 8,
    "use_nora": True,
    "target_modules": ["r_proj", "o_proj", "mlp.value", "lm_head"],
}


def _tiny_model_vllm_config(rwkv_lora: dict | None) -> SimpleNamespace:
    vllm_config = _state_vllm_config()
    vllm_config.model_config.hf_config = SimpleNamespace(
        architecture_version="rwkv7",
        head_size=64,
        hidden_size=128,
        num_attention_heads=2,
        intermediate_size=512,
        num_hidden_layers=2,
        decay_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        gate_low_rank_dim=8,
        group_norm_epsilon=64e-5,
        layer_norm_epsilon=1e-5,
        vocab_size=256,
        tie_word_embeddings=False,
        rwkv_lora=rwkv_lora,
    )
    return vllm_config


@pytest.fixture
def _single_rank(monkeypatch, default_vllm_config):
    group = SimpleNamespace(
        rank_in_group=0, world_size=1, is_first_rank=True, is_last_rank=True
    )
    monkeypatch.setattr("vllm.distributed.parallel_state._TP", group)
    monkeypatch.setattr("vllm.distributed.parallel_state._PP", group)


def test_config_accepts_rwkv_lora() -> None:
    vllm_config = _config()
    vllm_config.model_config.hf_config.rwkv_lora = dict(_LORA)
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("rwkv_lora", "message"),
    [
        ({**_LORA, "lora_dropout": 0.0}, "exactly the keys"),
        ({k: v for k, v in _LORA.items() if k != "use_nora"}, "exactly the keys"),
        ({**_LORA, "r": 0}, "positive integer"),
        ({**_LORA, "r": True}, "positive integer"),
        ({**_LORA, "lora_alpha": "8"}, "finite number"),
        ({**_LORA, "use_nora": 1}, "must be a bool"),
        ({**_LORA, "target_modules": []}, "target_modules"),
        ({**_LORA, "target_modules": ["q_proj"]}, "target_modules"),
        ({**_LORA, "target_modules": "r_proj"}, "target_modules"),
    ],
)
def test_config_rejects_invalid_rwkv_lora(rwkv_lora, message) -> None:
    vllm_config = _config()
    vllm_config.model_config.hf_config.rwkv_lora = rwkv_lora
    with pytest.raises(ValueError, match=message):
        RwkvForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.usefixtures("_single_rank")
def test_lora_registers_fp32_peft_adapters_loaded_in_place() -> None:
    base = RwkvForCausalLM(vllm_config=_tiny_model_vllm_config(None))
    model = RwkvForCausalLM(vllm_config=_tiny_model_vllm_config(_LORA))
    adapters = {
        name: param for name, param in model.named_parameters() if ".lora_" in name
    }
    expected = {"lm_head.lora_A.weight": (4, 128), "lm_head.lora_B.weight": (256, 4)}
    for layer in range(2):
        for module, in_features in (
            ("linear_attn.r_proj", 128),
            ("linear_attn.o_proj", 128),
            ("mlp.value", 512),
        ):
            prefix = f"model.layers.{layer}.{module}"
            expected[f"{prefix}.lora_A.weight"] = (4, in_features)
            expected[f"{prefix}.lora_B.weight"] = (128, 4)

    assert {name for name, _ in base.named_parameters()} == {
        name for name, _ in model.named_parameters()
    } - set(adapters)
    assert {name: tuple(param.shape) for name, param in adapters.items()} == expected
    assert all(param.dtype == torch.float32 for param in adapters.values())
    assert all(not param.any() for param in adapters.values())
    assert model.lm_head.lora_scale == 2.0
    assert model.lm_head.use_nora

    weights = {name: torch.randn(shape) for name, shape in expected.items()}
    storage = {name: param.data_ptr() for name, param in adapters.items()}
    assert model.load_weights(weights.items()) == set(expected)
    for name, param in adapters.items():
        assert param.data_ptr() == storage[name]
        assert param.dtype == torch.float32
        assert torch.equal(param, weights[name])


def _adapter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: id(value) if isinstance(value, torch.Tensor) else value
        for key, value in kwargs.items()
        if "lora" in key or "nora" in key
    }


@pytest.mark.usefixtures("_single_rank")
@pytest.mark.parametrize("rwkv_lora", [None, _LORA])
def test_lora_passes_raw_adapters_to_flashrwkv2(monkeypatch, rwkv_lora) -> None:
    calls: dict[str, dict[str, Any]] = {}
    hidden = torch.zeros(1, 128, dtype=torch.bfloat16)

    def record(name: str, result: object):
        def operator(*args, **kwargs):
            calls[name] = kwargs
            return result

        return operator

    module: Any = _fake_flashrwkv2()
    for name, result in (
        ("infer_tmix_postnorm_tokenshift_forward_varlen", (hidden,) * 7),
        ("infer_tmix_wkv_prepare_forward_varlen", (hidden,) * 8),
        ("infer_tmix_wkv7_recurrent_fp32io16_forward_varlen", hidden),
        ("infer_tmix_readout_forward_varlen", hidden),
        ("infer_cmix_forward_varlen", (hidden, hidden)),
        ("infer_head_linear_all_forward_varlen", torch.zeros(1, 256)),
    ):
        setattr(module, name, record(name, result))
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)

    model = RwkvForCausalLM(vllm_config=_tiny_model_vllm_config(rwkv_lora))
    layer = model.model.layers[1]
    metadata = SimpleNamespace(
        validated_metadata=None,
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        state_indices=torch.tensor([0], dtype=torch.int32),
        max_seqlen=1,
        batch_size=1,
    )
    layer.linear_attn.process_weights_after_loading()
    layer.linear_attn(
        hidden,
        hidden,
        layer.input_layernorm,
        hidden,
        hidden,
        object(),
        metadata,
        torch.float32,
    )
    layer.mlp(hidden, hidden, layer.post_attention_layernorm, hidden, metadata)
    model.compute_logits(hidden)

    adapter_calls = {name: _adapter_kwargs(kwargs) for name, kwargs in calls.items()}
    assert len(adapter_calls) == 6
    if rwkv_lora is None:
        assert all(not kwargs for kwargs in adapter_calls.values())
        return
    attention, feed_forward = layer.linear_attn, layer.mlp
    assert adapter_calls["infer_tmix_wkv_prepare_forward_varlen"] == {
        "receptance_lora_a": id(attention.r_proj.lora_A.weight),
        "receptance_lora_b": id(attention.r_proj.lora_B.weight),
        "receptance_lora_scale": 2.0,
        "receptance_use_nora": True,
    }
    assert adapter_calls["infer_tmix_readout_forward_varlen"] == {
        "output_lora_a": id(attention.o_proj.lora_A.weight),
        "output_lora_b": id(attention.o_proj.lora_B.weight),
        "output_lora_scale": 2.0,
        "output_use_nora": True,
    }
    assert adapter_calls["infer_cmix_forward_varlen"] == {
        "value_lora_a": id(feed_forward.value.lora_A.weight),
        "value_lora_b": id(feed_forward.value.lora_B.weight),
        "value_lora_scale": 2.0,
        "value_use_nora": True,
    }
    assert adapter_calls["infer_head_linear_all_forward_varlen"] == {
        "lora_a": id(model.lm_head.lora_A.weight),
        "lora_b": id(model.lm_head.lora_B.weight),
        "lora_scale": 2.0,
        "use_nora": True,
    }


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


def test_config_rejects_non_fp32_head_output() -> None:
    vllm_config = _config()
    vllm_config.model_config.hf_config.head_dtype = "model"
    try:
        RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    except ValueError as error:
        assert "float32 lm_head output" in str(error)
    else:
        raise AssertionError("RWKV must produce float32 logits directly")


def test_config_preserves_explicit_cudagraph_sizes() -> None:
    vllm_config = _config()
    vllm_config.compilation_config.cudagraph_capture_sizes = [1, 32, 96]

    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.max_cudagraph_capture_size is None
    assert vllm_config.compilation_config.cudagraph_capture_sizes == [1, 32, 96]


def test_config_rejects_unsupported_rapid_sampling_modes() -> None:
    for field, value, message in (
        ("logprobs_mode", "processed_logprobs", "raw logprobs"),
        ("return_sampling_mask", True, "sampling masks"),
        ("enable_trace_replay", True, "trace replay"),
    ):
        vllm_config = _config()
        setattr(vllm_config.model_config, field, value)
        try:
            RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{field} must be rejected")


def test_sampling_params_verify_rejects_unsupported_rwkv_controls() -> None:
    for model_type, sampling_params, message in (
        ("rwkv", SamplingParams(repetition_penalty=1.1), "repetition_penalty"),
        ("qwen2", SamplingParams(penalty_decay=0.9), "penalty_decay"),
    ):
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            max_logprobs=20,
            get_vocab_size=lambda: 64,
            logits_processors=None,
            is_diffusion=False,
        )
        try:
            sampling_params.verify(model_config, None, None, None)
        except VLLMValidationError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"{message} must be rejected")


def test_rapid_sampler_preserves_preempted_request_state() -> None:
    sampler = object.__new__(RwkvSampler)
    sampler.req_states = SimpleNamespace(req_id_to_index={"request": 1})
    sampler._rapid_sampling_states = torch.arange(8, dtype=torch.int8).view(2, 4)
    sampler._rapid_penalties = torch.arange(12, dtype=torch.float32).view(2, 6)
    sampler._preserved_requests = {}
    sampler._new_requests = []

    sampler.preempt_request("request")
    state, penalties = sampler._preserved_requests["request"]
    sampler._rapid_sampling_states[1].zero_()
    sampler._rapid_penalties[1].zero_()

    assert torch.equal(state, torch.tensor([4, 5, 6, 7], dtype=torch.int8))
    assert torch.equal(
        penalties,
        torch.tensor([6, 7, 8, 9, 10, 11], dtype=torch.float32),
    )
    sampler.remove_request("request")
    assert "request" in sampler._preserved_requests

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    sampler.req_states.req_id_to_index.clear()
    sampler.req_states.add_request = lambda **kwargs: (
        sampler.req_states.req_id_to_index.update({kwargs["req_id"]: 0})
    )
    sampler.req_states.apply_staged_writes = lambda: None
    runner = MagicMock(spec=GPUModelRunner)
    runner.req_states = sampler.req_states
    runner.model_state = SimpleNamespace(
        add_request=lambda idx, data: sampler.stage_request(idx, data.req_id),
        apply_staged_writes=lambda: None,
    )
    runner._remove_request.side_effect = lambda req_id: sampler.remove_request(req_id)
    runner.sampler = None
    runner.block_tables = runner.lora_state = MagicMock()
    runner.is_last_pp_rank = False
    for name in (
        "adaptive_verification",
        "pooling_runner",
        "encoder_cache",
        "speculator",
    ):
        setattr(runner, name, None)
    request = SimpleNamespace(
        req_id="request",
        prefill_token_ids=[1],
        prompt_len=1,
        sampling_params=None,
        num_computed_tokens=0,
        block_ids=[],
        lora_request=None,
    )
    GPUModelRunner.add_requests(runner, SimpleNamespace(scheduled_new_reqs=[request]))
    runner._remove_request.assert_not_called()
    assert "request" not in sampler._preserved_requests
    assert len(sampler._new_requests) == 1
    req_idx, preserved = sampler._new_requests[0]
    assert req_idx == 0
    assert preserved is not None
    assert preserved[0] is state
    assert preserved[1] is penalties

    writes = SimpleNamespace(apply_staged_writes=lambda: None, copy_to_uva=lambda: None)
    for name in (
        "sampling_states",
        "_logit_bias_state",
        "_bad_words_state",
        "logprob_token_ids_state",
        "thinking_budget_state",
        "_presence_penalty",
        "_frequency_penalty",
        "_penalty_decay",
    ):
        setattr(sampler, name, writes)
    sampler.apply_staged_writes()
    assert torch.equal(sampler._rapid_sampling_states[0], state)
    assert torch.equal(sampler._rapid_penalties[0], penalties)
    assert not sampler._rapid_sampling_states[1].any()
    assert not sampler._rapid_penalties[1].any()

    sampler.req_states.req_id_to_index.clear()
    sampler._preserved_requests["request"] = (state, penalties)
    sampler.remove_request("request")
    assert "request" not in sampler._preserved_requests


def test_rapid_sampler_graph_compacts_completed_prefills(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.rwkv.get_num_sampled_and_rejected",
        lambda *args: (
            torch.tensor([1, 0, 1, 1], dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
        ),
    )
    graph = SimpleNamespace(
        hidden_states=torch.empty(4, 3),
        slot_indices=torch.empty(4, dtype=torch.int32),
        presence_penalty=torch.empty(4),
        frequency_penalty=torch.empty(4),
        penalty_decay=torch.empty(4),
        temperature=torch.empty(4),
        top_k=torch.empty(4, dtype=torch.int32),
        top_p=torch.empty(4),
        num_active=torch.empty(1, dtype=torch.int32),
        sampled=torch.tensor([10, 11, 12, -1], dtype=torch.int32),
        graph=SimpleNamespace(replay=lambda: None),
    )
    sampler = object.__new__(RwkvSampler)
    sampler.compute_nans = False
    sampler.get_logprobs_dims = lambda idx_mapping_np: None
    sampler.needs_logits_processing = np.zeros(4, dtype=bool)
    sampler._sampling_graphs = {4: graph}
    active_rows = np.empty(4, dtype=np.int64)
    active_slots = np.empty(4, dtype=np.int32)
    sampler._active_rows = SimpleNamespace(
        np=active_rows,
        copy_to_uva=lambda size: torch.from_numpy(active_rows[:size]),
    )
    sampler._active_slots = SimpleNamespace(
        np=active_slots,
        copy_to_uva=lambda size: torch.from_numpy(active_slots[:size]),
    )
    sampler._presence_penalty = SimpleNamespace(gpu=torch.arange(4).float())
    sampler._frequency_penalty = SimpleNamespace(gpu=torch.arange(4).float() + 4)
    sampler._penalty_decay = SimpleNamespace(gpu=torch.arange(4).float() + 8)
    sampler.sampling_states = SimpleNamespace(
        min_p=SimpleNamespace(np=np.zeros(4)),
        temperature=SimpleNamespace(gpu=torch.arange(4).float() + 12),
        top_k=SimpleNamespace(gpu=torch.arange(4, dtype=torch.int32) + 16),
        top_p=SimpleNamespace(gpu=torch.arange(4).float() + 20),
    )
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([2, 8, 4, 1]))
    )
    input_batch = SimpleNamespace(
        num_reqs=4,
        idx_mapping_np=np.array([2, 1, 3, 0]),
        idx_mapping=torch.tensor([2, 1, 3, 0]),
        num_computed_prefill_tokens_np=np.array([1, 2, 3, 0]),
        num_scheduled_tokens=np.array([1, 2, 1, 1]),
        prefill_len_np=np.array([2, 8, 4, 1]),
        seq_lens=torch.tensor([2, 4, 4, 1]),
        cu_num_logits=torch.tensor([0, 1, 2, 3, 4]),
    )
    hidden_states = torch.arange(12, dtype=torch.bfloat16).view(4, 3)

    output = sampler.sample_hidden_states(hidden_states, input_batch, None)

    assert output is not None
    assert torch.equal(output.sampled_token_ids[:, 0], torch.tensor([10, 0, 11, 12]))
    assert graph.num_active.item() == 3
    assert torch.equal(graph.hidden_states[:3], hidden_states[[0, 2, 3]])
    assert torch.equal(graph.slot_indices[:3], torch.tensor([2, 3, 0]))
    assert torch.equal(graph.presence_penalty[:3], torch.tensor([2.0, 3.0, 0.0]))
    assert torch.equal(graph.top_k[:3], torch.tensor([18, 19, 16]))


def test_rapid_sampler_graph_defers_grammar_to_existing_path() -> None:
    sampler = object.__new__(RwkvSampler)
    sampler.compute_nans = False
    sampler.get_logprobs_dims = lambda idx_mapping_np: None
    sampler.needs_logits_processing = np.zeros(1, dtype=bool)
    sampler._sampling_graphs = {1: object()}
    input_batch = SimpleNamespace(idx_mapping_np=np.array([0]))

    assert (
        sampler.sample_hidden_states(torch.zeros(1, 1), input_batch, SimpleNamespace())
        is None
    )


@pytest.mark.parametrize("factory", [SamplingParams, SamplingParams.from_optional])
def test_penalty_decay_preserves_positional_sampling_arguments(factory) -> None:
    params = factory(1, 0.0, 0.0, 1.0, 0.5)
    assert params.temperature == 0.5
    assert params.penalty_decay == 1.0
    assert factory(penalty_decay=0.9).penalty_decay == 0.9


@pytest.mark.parametrize("model_type", ["rwkv", "qwen2"])
def test_rwkv_generation_defaults_do_not_leak_to_other_models(model_type) -> None:
    model_config = SimpleNamespace(
        generation_config="vllm",
        override_generation_config={"stop_strings": ["END"], "penalty_decay": 0.9},
        hf_config=SimpleNamespace(model_type=model_type),
    )
    expected = {"stop": ["END"], "penalty_decay": 0.9} if model_type == "rwkv" else {}
    assert ModelConfig.get_diff_sampling_param(model_config) == expected


def test_rapid_sampler_rejects_loaded_custom_processors() -> None:
    sampler = SimpleNamespace(logits_processors=[object() for _ in range(4)])
    with pytest.raises(ValueError, match="custom logits processors"):
        RwkvSampler(sampler, torch.nn.Identity(), capture_graphs=False)


@pytest.mark.parametrize("mixed_prefill", [False, True])
def test_rapid_sampler_min_p_preserves_raw_logits(mixed_prefill) -> None:
    sampler = object.__new__(RwkvSampler)
    slots = np.array([1, 0])
    for name, dtype in (("_active_rows", np.int64), ("_active_slots", np.int32)):
        buffer = np.empty(2, dtype=dtype)
        setattr(
            sampler,
            name,
            SimpleNamespace(
                np=buffer,
                copy_to_uva=lambda size, buf=buffer: torch.from_numpy(buf[:size]),
            ),
        )
    values = SimpleNamespace(gpu=torch.ones(2))
    min_p = torch.tensor([0.0, 0.5])
    sampler._presence_penalty = sampler._frequency_penalty = values
    sampler._penalty_decay = values
    sampler.sampling_states = SimpleNamespace(
        min_p=SimpleNamespace(np=min_p.numpy(), gpu=min_p),
        temperature=SimpleNamespace(gpu=torch.tensor([1.0, 0.5])),
        top_k=values,
        top_p=values,
    )
    sampler.needs_logits_processing = np.ones(2, dtype=bool)
    sampler._logit_bias_state = SimpleNamespace(
        apply=lambda logits, ctx: logits.add_(torch.tensor([0.0, 0.125, 0.0, 0.0]))
    )
    sampler._bad_words_state = SimpleNamespace(apply=lambda logits, ctx: logits)
    sampler.thinking_budget_state = SimpleNamespace(apply=lambda logits, ctx: None)
    sampler._rapid_penalties = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.5, 0.0, 1.0, 0.0]]
    )
    sampler._rapid_sampling_states = torch.zeros(2, 2)
    sampler._input_batch = SimpleNamespace(
        num_reqs=2,
        num_computed_prefill_tokens_np=np.zeros(2),
        num_scheduled_tokens=np.ones(2),
        prefill_len_np=np.array([1, 2 if mixed_prefill else 1]),
    )
    logits = torch.tensor(
        [[1.0, -0.25, torch.nan, -torch.inf], [2.0, 4.0, 1.0, -torch.inf]]
    )
    raw = logits.clone()
    expected = torch.tensor(
        [[-torch.inf, -0.125, -torch.inf, -torch.inf], [2.0, 4.125, 1.0, -torch.inf]]
    )

    def sample(active_logits, *args, **kwargs):
        torch.testing.assert_close(active_logits, expected[: 1 if mixed_prefill else 2])
        return active_logits.argmax(dim=-1)

    sampler._sample = sample
    sampled, returned = sampler.sample(
        logits,
        torch.from_numpy(slots),
        torch.from_numpy(slots),
        slots,
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        torch.zeros(2, dtype=torch.int64),
        np.ones(2),
        return_logprobs=True,
    )
    assert torch.equal(sampled, torch.tensor([1, 0 if mixed_prefill else 1]))
    torch.testing.assert_close(logits, raw, equal_nan=True)
    torch.testing.assert_close(returned, raw, equal_nan=True)


@pytest.mark.skipif(
    not os.environ.get("VLLM_RWKV_TEST_MODEL"),
    reason="VLLM_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize("reload_mode", ["layerwise", "direct"])
def test_real_model_checkpoint_reload_preserves_tokens_and_logprobs(
    vllm_runner, monkeypatch, reload_mode
) -> None:
    monkeypatch.setenv("VLLM_RELOAD_MODE", reload_mode)
    model_path = os.environ["VLLM_RWKV_TEST_MODEL"]
    prompt = {"prompt_token_ids": list(range(1, 66))}
    params = SamplingParams(
        temperature=0.0, max_tokens=4, ignore_eos=True, logprobs=3, detokenize=False
    )
    with vllm_runner(
        model_path,
        dtype="bfloat16",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_gpu_blocks_override=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
    ) as runner:
        before = runner.llm.generate([prompt], params)[0].outputs[0]
        runner.llm.collective_rpc("reload_weights", kwargs={"weights_path": model_path})
        after = runner.llm.generate([prompt], params)[0].outputs[0]
    assert len(after.token_ids) == 4
    assert after.token_ids == before.token_ids
    assert after.logprobs == before.logprobs


def test_config_accepts_async_scheduling() -> None:
    vllm_config = _config()
    vllm_config.scheduler_config.async_scheduling = True
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.scheduler_config.async_scheduling


@pytest.mark.skipif(
    not os.environ.get("VLLM_RWKV_TEST_MODEL"),
    reason="VLLM_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="requires two GPUs")
def test_real_model_uneven_pipeline_preserves_tokens(vllm_runner, monkeypatch) -> None:
    model_path = os.environ["VLLM_RWKV_TEST_MODEL"]
    num_layers = get_config(model_path, trust_remote_code=False).num_hidden_layers
    split = num_layers // 2 - 1
    prompt = {"prompt_token_ids": list(range(1, 66))}
    params = SamplingParams(
        temperature=0.0, max_tokens=4, ignore_eos=True, detokenize=False
    )
    options = dict(
        dtype="bfloat16",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_gpu_blocks_override=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.7,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
    )
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with vllm_runner(model_path, **options) as runner:
        expected = runner.llm.generate([prompt], params)[0].outputs[0].token_ids

    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", f"{split},{num_layers - split}")
    with vllm_runner(
        model_path,
        pipeline_parallel_size=2,
        distributed_executor_backend="mp",
        **options,
    ) as runner:
        actual = runner.llm.generate([prompt], params)[0].outputs[0].token_ids

    assert len(actual) == 4
    assert actual == expected


@pytest.mark.skipif(
    not os.environ.get("VLLM_RWKV_TEST_MODEL"),
    reason="VLLM_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_real_model_batching_preserves_tokens_and_logprobs(
    vllm_runner, prefix_caching, state_dtype, temperature
) -> None:
    """Each state mode agrees across prefill, batching and graph execution."""
    model_path = os.environ["VLLM_RWKV_TEST_MODEL"]
    params = SamplingParams(
        temperature=temperature,
        top_k=1 if temperature == 0 else 20,
        top_p=0.9,
        seed=20261026,
        max_tokens=4,
        ignore_eos=True,
        logprobs=3,
        presence_penalty=0.3,
        frequency_penalty=0.2,
        penalty_decay=0.95,
        detokenize=False,
    )
    tokens_only = params.clone()
    tokens_only.logprobs = None
    prompt = {"prompt_token_ids": list(range(1, 66))}
    mixed_prompts = [{"prompt_token_ids": [7, 8, 9]}, prompt]
    reference = None
    for enforce_eager, token_budget in ((True, 128), (False, 32)):
        # Compare default graph execution with the explicit FP16 eager reference.
        state_options = {}
        if enforce_eager or state_dtype == "float32":
            state_options["mamba_ssm_cache_dtype"] = state_dtype
        with vllm_runner(
            model_path,
            dtype="bfloat16",
            max_model_len=128,
            max_num_batched_tokens=token_budget,
            max_num_seqs=4,
            gpu_memory_utilization=float(
                os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8")
            ),
            num_gpu_blocks_override=64,
            enable_chunked_prefill=True,
            enable_prefix_caching=prefix_caching,
            enforce_eager=enforce_eager,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
            **state_options,
        ) as runner:
            cache_config = runner.llm.llm_engine.vllm_config.cache_config
            assert cache_config.mamba_ssm_cache_dtype == state_dtype
            single = runner.llm.generate([prompt], params)[0].outputs[0]
            mixed = runner.llm.generate(mixed_prompts, params)[1].outputs[0]
            if not enforce_eager:
                # Requests without logprobs use the head/sampling CUDA Graph.
                replay = runner.llm.generate(mixed_prompts, tokens_only)[1].outputs[0]
                assert replay.token_ids == mixed.token_ids
        assert len(single.token_ids) == 4
        assert mixed.token_ids == single.token_ids
        assert mixed.logprobs == single.logprobs
        if reference is not None:
            assert single.token_ids == reference.token_ids
            assert single.logprobs == reference.logprobs
        reference = single


@pytest.mark.skipif(
    not os.environ.get("VLLM_RWKV_TEST_MODEL"),
    reason="VLLM_RWKV_TEST_MODEL must point to a local RWKV checkpoint",
)
def test_real_model_rapid_sampling_applies_min_p(vllm_runner) -> None:
    """At min_p=1, only the highest-scoring token can be sampled."""
    prompt = {"prompt_token_ids": list(range(1, 66))}
    with vllm_runner(
        os.environ["VLLM_RWKV_TEST_MODEL"],
        dtype="bfloat16",
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        num_gpu_blocks_override=64,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
    ) as runner:
        params = SamplingParams(
            temperature=1.0,
            min_p=1.0,
            seed=20261026,
            max_tokens=4,
            ignore_eos=True,
            detokenize=False,
        )
        tokens = runner.llm.generate([prompt], params)[0].outputs[0].token_ids
        params.logprobs = 2
        scored = runner.llm.generate([prompt], params)[0].outputs[0]

    assert len(tokens) == 4
    assert scored.token_ids == tokens
    assert scored.logprobs is not None
    for token, logprobs in zip(tokens, scored.logprobs, strict=True):
        assert logprobs[token].logprob == max(
            entry.logprob for entry in logprobs.values()
        )
        assert sum(np.isfinite(entry.logprob) for entry in logprobs.values()) >= 2
