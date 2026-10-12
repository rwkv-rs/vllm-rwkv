# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.config.model import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.config import RwkvForCausalLMConfig
from vllm.model_executor.models.rwkv import (
    RwkvFeedForward,
    RwkvModel,
    RwkvStateLayer,
    _load_flashrwkv2,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.rwkv_attn import (
    RwkvAttentionMetadata,
    RwkvAttentionMetadataBuilder,
)
from vllm.v1.core.sched.output import FinishedRequestData
from vllm.v1.worker.gpu.model_states.rwkv import RwkvModelState, RwkvSampler


class _FakeStateHandle:
    def __init__(
        self,
        state_pool_size: int,
        channels: int,
        state_dtype: torch.dtype,
        has_elapsed: bool,
        *,
        bytes_per_slot: int = 18,
        fixed_workspace_nbytes: int = 13,
    ) -> None:
        self.state = torch.zeros((state_pool_size, channels), dtype=state_dtype)
        self.elapsed = (
            torch.zeros(state_pool_size, dtype=torch.int32) if has_elapsed else None
        )
        self.materialized: list[torch.Tensor] = []
        self.bytes_per_slot = bytes_per_slot
        self.fixed_workspace_nbytes = fixed_workspace_nbytes

    @property
    def memory_layout(self) -> dict[str, int]:
        return {
            "base_bytes_per_slot": 1,
            "private_bytes_per_slot": self.bytes_per_slot - 1,
            "bytes_per_slot": self.bytes_per_slot,
            "fixed_workspace_nbytes": self.fixed_workspace_nbytes,
            "total_nbytes": (
                self.state.shape[0] * self.bytes_per_slot + self.fixed_workspace_nbytes
            ),
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

    def clone_slots(self, indices: torch.Tensor) -> "_FakeStateHandle":
        indices = indices.long()
        clone = object.__new__(_FakeStateHandle)
        clone.state = self.state.index_select(0, indices).clone()
        clone.elapsed = (
            self.elapsed.index_select(0, indices).clone()
            if self.elapsed is not None
            else None
        )
        clone.materialized = []
        clone.bytes_per_slot = self.bytes_per_slot
        clone.fixed_workspace_nbytes = self.fixed_workspace_nbytes
        return clone


def _fake_flashrwkv2() -> ModuleType:
    module: Any = ModuleType("flashrwkv2")
    module.__version__ = "0.1.0a13"

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

    def prepare_fp32_from_tensor(state):
        handle = _FakeStateHandle(
            state.shape[0],
            state[0].numel(),
            torch.float32,
            False,
            bytes_per_slot=state[0].numel() * state.element_size(),
            fixed_workspace_nbytes=0,
        )
        handle.state = state
        return handle

    module.prepare_tmix_wkv7_recurrent_fp16_state = prepare_fp16
    module.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor = (
        prepare_fp32_from_tensor
    )
    module.prepare_tmix_wkv7_recurrent_metadata = lambda *args, **kwargs: (args, kwargs)
    for name in (
        "infer_cmix_forward_varlen",
        "infer_embedding_ln0_forward_varlen",
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


def _state_vllm_config(state_dtype: str = "float16") -> SimpleNamespace:
    hf_config = SimpleNamespace(
        hidden_size=128,
        head_size=64,
        num_attention_heads=2,
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


def test_fp32_state_uses_caller_backed_cache_tensor(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config("float32")
    state_layer = RwkvStateLayer(vllm_config, 2, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)

    assert spec.provider_state_bytes_per_page == 0
    assert spec.provider_fixed_workspace_bytes == 0

    num_slots = 3
    raw = torch.zeros((num_slots, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    _, handle, _ = state_layer.get_layer_state(0)

    assert handle.state.shape == (num_slots, 2, 64, 64)
    assert handle.state.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


def test_state_snapshot_roundtrip_is_independent_of_scheduler_slots(
    monkeypatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config()
    state_layer = RwkvStateLayer(vllm_config, 2, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)
    raw = torch.zeros((3, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)

    for layer_idx in range(2):
        tmix, handle, cmix = state_layer.get_layer_state(layer_idx)
        tmix[1].fill_(11 + layer_idx)
        handle.state[1].fill_(21 + layer_idx)
        assert handle.elapsed is not None
        handle.elapsed[1].fill_(31 + layer_idx)
        cmix[1].fill_(41 + layer_idx)

    snapshot = state_layer.capture_state_slot(1)
    state_layer.reset_state_slots([1, 2])
    state_layer.restore_state_slot(snapshot, 2)

    assert snapshot.nbytes > 0
    assert snapshot.nbytes == state_layer.state_snapshot_nbytes
    for layer_idx in range(2):
        tmix, handle, cmix = state_layer.get_layer_state(layer_idx)
        assert torch.all(tmix[2] == 11 + layer_idx)
        assert torch.all(handle.state[2] == 21 + layer_idx)
        assert torch.all(handle.elapsed[2] == 31 + layer_idx)
        assert torch.all(cmix[2] == 41 + layer_idx)
        assert torch.count_nonzero(tmix[1]) == 0


@pytest.mark.parametrize("state_dtype", ["float16", "float32"])
def test_state_ref_cache_is_bounded_and_supports_clone_drop(
    monkeypatch, state_dtype
) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config(state_dtype)
    state_layer = RwkvStateLayer(vllm_config, 1, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)
    raw = torch.zeros((2, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    model_state = object.__new__(RwkvModelState)
    model_state._state_layer = state_layer
    model_state._state_cache = {}
    model_state._state_snapshot_refcounts = {}
    model_state._active_state_readers = {}
    model_state._prepared_state_reads_by_lease = {}
    model_state._state_cache_bytes = 0
    model_state._reserved_state_cache_bytes = 0
    model_state._state_cache_max_bytes = 1 << 30

    captured = model_state.capture_state_ref("goal-a", 0)
    model_state.state_cache_action("prepare_read", "goal-a", "lease-a")
    model_state.state_cache_action("prepare_read", "goal-a", "lease-b")
    with pytest.raises(ValueError, match="in use"):
        model_state.state_cache_action("drop", "goal-a")
    canceled = model_state.state_cache_action("cancel_read", "goal-a", "lease-a")
    assert canceled["canceled"] is True
    with pytest.raises(ValueError, match="in use"):
        model_state.state_cache_action("drop", "goal-a")
    model_state.state_cache_action("cancel_read", "goal-a", "lease-b")
    cloned = model_state.state_cache_action("clone", "goal-a", "goal-b")
    inspected = model_state.state_cache_action("inspect", "goal-b")
    dropped = model_state.state_cache_action("drop", "goal-a")

    assert captured["state_ref"] == "goal-a"
    assert captured["state_dtype"] == str(getattr(torch, state_dtype))
    assert cloned["state_ref"] == inspected["state_ref"] == "goal-b"
    assert inspected["state_dtype"] == captured["state_dtype"]
    assert model_state._state_cache_bytes == inspected["nbytes"]
    assert dropped["dropped"] is True
    assert (
        model_state.state_cache_action("inspect_if_exists", "goal-a")["exists"] is False
    )
    with pytest.raises(KeyError, match="goal-a"):
        model_state.state_cache_action("inspect", "goal-a")

    model_state._state_cache_max_bytes = inspected["nbytes"] * 2 - 1
    with pytest.raises(MemoryError, match="capacity"):
        model_state.capture_state_ref("goal-c", 1)


@pytest.mark.parametrize("block_size", [16, 16384], ids=["align", "none"])
def test_state_ref_restore_happens_before_first_delta_forward(block_size) -> None:
    restored: list[tuple[object, torch.Tensor]] = []
    resets: list[torch.Tensor] = []
    snapshot = SimpleNamespace(nbytes=10)
    state_layer = SimpleNamespace(
        prefix="model.layers.0.rwkv_state",
        restore_state_slot=lambda value, slot: restored.append((value, slot.clone())),
        reset_state_slots=lambda slots: resets.append(slots.clone()),
        copy_state_slots=lambda source, destination: None,
    )
    model_state = object.__new__(RwkvModelState)
    model_state._align_mode = block_size == 16
    model_state._pending_write_ref_by_id = {}
    model_state._block_size = block_size
    model_state._state_block_columns = np.array([-1], dtype=np.int32)
    model_state._state_num_computed_tokens = np.array([0], dtype=np.int64)
    model_state._state_layer = state_layer
    model_state._rwkv_group_id = 0
    model_state._pending_read_ref_by_index = {0: "parent"}
    model_state._active_state_slots = torch.full((1,), -1, dtype=torch.int32)
    model_state._has_active_state_slot = np.zeros(1, dtype=np.bool_)
    model_state._state_cache = {"parent": SimpleNamespace(snapshot=snapshot)}
    model_state.device = torch.device("cpu")
    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping_np=np.array([0]),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
    )

    model_state.preprocess_state(
        input_batch,
        (torch.tensor([[7]], dtype=torch.int32),),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )

    assert not resets
    assert restored and restored[0][0] is snapshot
    assert torch.equal(restored[0][1], torch.tensor([7], dtype=torch.int32))


def test_state_cache_invalidation_requires_quiescence_and_releases_refs() -> None:
    model_state = object.__new__(RwkvModelState)
    model_state._state_layer = SimpleNamespace(state_snapshot_nbytes=16)
    model_state._state_cache_max_bytes = 64
    model_state._state_cache = {"saved": object()}
    model_state._state_snapshot_refcounts = {1: 1}
    model_state._active_state_readers = {}
    model_state._prepared_state_reads_by_lease = {"lease": "saved"}
    model_state._prepared_state_bytes_by_ref = {"next": 16}
    model_state._state_cache_bytes = 16
    model_state._reserved_state_cache_bytes = 16
    model_state._req_index_by_id = {"running": 0}
    model_state._preempted_req_ids = set()

    with pytest.raises(RuntimeError, match="Quiesce"):
        model_state.clear_state_cache()
    assert "saved" in model_state._state_cache

    model_state._req_index_by_id.clear()
    model_state._preempted_req_ids.add("waiting")
    with pytest.raises(RuntimeError, match="Quiesce"):
        model_state.clear_state_cache()
    model_state._preempted_req_ids.clear()
    model_state.clear_state_cache()

    assert not model_state.state_cache_action("inspect_if_exists", "saved")["exists"]
    capabilities = model_state.state_cache_action("capabilities")
    for field in ("state_count", "used_bytes", "reserved_bytes", "prepared_read_count"):
        assert capabilities[field] == 0
    assert not model_state.state_cache_action("cancel_write", "next")["canceled"]
    assert not model_state.state_cache_action("cancel_read", "saved", "lease")[
        "canceled"
    ]


def test_request_state_refs_restore_then_capture_a_new_atomic_ref(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config()
    state_layer = RwkvStateLayer(vllm_config, 1, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)
    raw = torch.zeros((2, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    tmix, handle, cmix = state_layer.get_layer_state(0)
    tmix[0].fill_(1)
    handle.state[0].fill_(2)
    cmix[0].fill_(3)

    model_state = object.__new__(RwkvModelState)
    model_state.prompt_embeds_state = None
    model_state.rope_state = None
    model_state._sampler = None
    model_state._state_layer = state_layer
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([-1], dtype=np.int32)
    model_state._state_num_computed_tokens = np.array([0], dtype=np.int64)
    model_state._rwkv_group_id = 0
    model_state._state_cache = {}
    model_state._state_snapshot_refcounts = {}
    model_state._active_state_readers = {}
    model_state._prepared_state_reads_by_lease = {}
    model_state._state_cache_bytes = 0
    model_state._reserved_state_cache_bytes = 0
    model_state._state_cache_max_bytes = 1 << 30
    model_state._req_index_by_id = {}
    model_state._pending_read_ref_by_index = {}
    model_state._pending_write_ref_by_id = {}
    model_state._prepared_state_bytes_by_ref = {}
    model_state._reserved_state_bytes_by_req_id = {}
    model_state._base_processed_tokens_by_req_id = {}
    model_state._parent_state_ref_by_req_id = {}
    model_state._preempted_req_ids = set()
    model_state._active_state_slots = torch.full((1,), -1, dtype=torch.int32)
    model_state._has_active_state_slot = np.zeros(1, dtype=np.bool_)
    model_state.device = torch.device("cpu")
    model_state.capture_state_ref("goal-parent", 0)
    state_layer.reset_state_slots([1])
    model_state.state_cache_action("prepare_read", "goal-parent", "request-1-lease")
    with pytest.raises(ValueError, match="in use"):
        model_state.state_cache_action("drop", "goal-parent")
    request = SimpleNamespace(
        req_id="request-1",
        num_computed_tokens=0,
        prefill_token_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(
            extra_args={
                "rwkv_state_read_ref": "goal-parent",
                "rwkv_state_read_lease": "request-1-lease",
                "rwkv_state_write_ref": "goal-child",
            }
        ),
    )

    model_state.add_request(0, request)
    with pytest.raises(ValueError, match="in use"):
        model_state.state_cache_action("drop", "goal-parent")
    model_state.preprocess_state(
        SimpleNamespace(
            num_reqs=1,
            idx_mapping_np=np.array([0]),
            num_scheduled_tokens=np.array([3], dtype=np.int32),
        ),
        (torch.tensor([[1]], dtype=torch.int32),),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )
    reserved = model_state._reserved_state_cache_bytes
    model_state.preempt_request(request.req_id)
    model_state.remove_request(request.req_id)
    with pytest.raises(ValueError, match="in use"):
        model_state.state_cache_action("drop", "goal-parent")
    assert model_state._reserved_state_cache_bytes == reserved
    model_state.add_request(0, request)
    model_state.preprocess_state(
        SimpleNamespace(
            num_reqs=1,
            idx_mapping_np=np.array([0]),
            num_scheduled_tokens=np.array([3], dtype=np.int32),
        ),
        (torch.tensor([[1]], dtype=torch.int32),),
        SimpleNamespace(kv_cache_groups=[]),
        torch.empty(0),
    )
    assert model_state._reserved_state_cache_bytes == reserved
    tmix[1].add_(10)
    handle.state[1].add_(20)
    cmix[1].add_(30)
    model_state.remove_request(
        "request-1",
        FinishedRequestData(
            finish_reason="stop",
            pending_tail_token_ids=(99,),
        ),
    )

    assert (
        model_state.state_cache_action("inspect", "goal-parent")[
            "scheduler_slots_reserved"
        ]
        == 0
    )
    assert (
        model_state.state_cache_action("inspect", "goal-child")[
            "scheduler_slots_reserved"
        ]
        == 0
    )
    state_layer.reset_state_slots([0])
    child = model_state._state_cache["goal-child"]
    assert child.parent_ref == "goal-parent"
    assert child.processed_token_count == 3
    assert child.pending_tail_token_ids == (99,)
    assert model_state._reserved_state_cache_bytes == 0
    state_layer.restore_state_slot(child.snapshot, 0)
    assert torch.all(tmix[0] == 11)
    assert torch.all(handle.state[0] == 22)
    assert torch.all(cmix[0] == 33)

    missing_tail_request = SimpleNamespace(
        req_id="request-2",
        num_computed_tokens=0,
        prefill_token_ids=[100],
        sampling_params=SimpleNamespace(
            extra_args={"rwkv_state_read_ref": "goal-child"}
        ),
    )
    with pytest.raises(ValueError, match="missing its pending tail"):
        model_state.add_request(0, missing_tail_request)

    resumed_request = SimpleNamespace(
        req_id="request-3",
        num_computed_tokens=1,
        prefill_token_ids=[99, 100],
        sampling_params=SimpleNamespace(
            extra_args={"rwkv_state_read_ref": "goal-child"}
        ),
    )
    model_state.add_request(0, resumed_request)
    assert 0 not in model_state._pending_read_ref_by_index


def test_state_write_reserves_capacity_and_abort_does_not_commit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flashrwkv2", _fake_flashrwkv2())
    vllm_config = _state_vllm_config()
    state_layer = RwkvStateLayer(vllm_config, 1, "model.layers.0.rwkv_state")
    spec = state_layer.get_kv_cache_spec(vllm_config)
    raw = torch.zeros((1, 1, 1, spec.page_size_bytes), dtype=torch.int8)
    state_layer.bind_kv_cache(raw)
    model_state = object.__new__(RwkvModelState)
    model_state.prompt_embeds_state = None
    model_state.rope_state = None
    model_state._sampler = None
    model_state._state_layer = state_layer
    model_state._align_mode = True
    model_state._block_size = 16
    model_state._state_block_columns = np.array([-1], dtype=np.int32)
    model_state._state_num_computed_tokens = np.array([0], dtype=np.int64)
    model_state._state_cache = {}
    model_state._state_snapshot_refcounts = {}
    model_state._active_state_readers = {}
    model_state._prepared_state_reads_by_lease = {}
    model_state._state_cache_bytes = 0
    model_state._reserved_state_cache_bytes = 0
    model_state._state_cache_max_bytes = state_layer.state_snapshot_nbytes
    model_state._req_index_by_id = {}
    model_state._pending_read_ref_by_index = {}
    model_state._pending_write_ref_by_id = {}
    model_state._prepared_state_bytes_by_ref = {}
    model_state._reserved_state_bytes_by_req_id = {}
    model_state._base_processed_tokens_by_req_id = {}
    model_state._parent_state_ref_by_req_id = {}
    model_state._preempted_req_ids = set()
    model_state._active_state_slots = torch.full((1,), -1, dtype=torch.int32)
    model_state._has_active_state_slot = np.zeros(1, dtype=np.bool_)
    model_state.device = torch.device("cpu")
    request = SimpleNamespace(
        req_id="request-abort",
        num_computed_tokens=0,
        sampling_params=SimpleNamespace(
            extra_args={"rwkv_state_write_ref": "must-not-exist"}
        ),
    )

    prepared = model_state.state_cache_action("prepare_write", "must-not-exist")
    assert prepared["ready"] is True
    assert model_state._reserved_state_cache_bytes == state_layer.state_snapshot_nbytes

    model_state.add_request(0, request)

    assert model_state._reserved_state_cache_bytes == state_layer.state_snapshot_nbytes
    model_state.preempt_request("request-abort")
    model_state.remove_request("request-abort")
    assert model_state._reserved_state_cache_bytes == state_layer.state_snapshot_nbytes
    model_state.remove_request(
        "request-abort",
        FinishedRequestData(finish_reason="abort", pending_tail_token_ids=()),
    )
    assert model_state._reserved_state_cache_bytes == 0
    assert "must-not-exist" not in model_state._state_cache

    model_state.state_cache_action("prepare_write", "canceled-write")
    canceled = model_state.state_cache_action("cancel_write", "canceled-write")
    assert canceled["canceled"] is True
    assert model_state._reserved_state_cache_bytes == 0


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


def test_feed_forward_prepares_value_weight_for_provider() -> None:
    feed_forward = object.__new__(RwkvFeedForward)
    torch.nn.Module.__init__(feed_forward)
    checkpoint_weight = torch.arange(16, dtype=torch.float16).view(2, 8)
    feed_forward.value = SimpleNamespace(
        weight=torch.nn.Parameter(checkpoint_weight.clone())
    )

    feed_forward.process_weights_after_loading()

    assert torch.equal(feed_forward.value.weight, checkpoint_weight.T.contiguous())


def test_model_folds_embedding_norm_after_loading(monkeypatch) -> None:
    module: Any = _fake_flashrwkv2()
    module.infer_embedding_ln0_forward_varlen = lambda embedding, weight, bias, eps: (
        torch.nn.functional.layer_norm(
            embedding, (embedding.shape[-1],), weight, bias, eps
        ).to(torch.float16)
    )
    monkeypatch.setitem(sys.modules, "flashrwkv2", module)
    monkeypatch.setattr(
        "vllm.model_executor.models.rwkv.get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )
    model = object.__new__(RwkvModel)
    torch.nn.Module.__init__(model)
    embedding = torch.arange(32, dtype=torch.bfloat16).view(8, 4)
    model.embed_tokens = SimpleNamespace(weight=torch.nn.Parameter(embedding.clone()))
    model.embedding_norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
    model.config = SimpleNamespace(layer_norm_epsilon=1e-5)
    model.layers = torch.nn.ModuleList()
    model.start_layer = 0
    model.end_layer = 0
    model._embedding_norm_folded = False
    expected = torch.nn.functional.layer_norm(
        embedding,
        (4,),
        model.embedding_norm.weight,
        model.embedding_norm.bias,
        model.config.layer_norm_epsilon,
    ).to(torch.float16)

    model.process_weights_after_loading()

    assert model._embedding_norm_folded
    assert model.embed_tokens.weight.dtype == torch.float16
    assert torch.equal(model.embed_tokens.weight, expected)


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
    model_state._pending_read_ref_by_index = {}
    model_state._pending_write_ref_by_id = {}
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
    model_state._pending_read_ref_by_index = {}
    model_state._pending_write_ref_by_id = {}
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
        dtype=torch.float16,
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


def test_config_uses_fp16_state_and_full_graph_defaults() -> None:
    vllm_config = _config()
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.mamba_cache_dtype == "float16"
    assert vllm_config.cache_config.mamba_ssm_cache_dtype == "float16"
    assert vllm_config.cache_config.mamba_cache_mode == "align"
    assert vllm_config.cache_config.mamba_block_size == 16
    assert vllm_config.scheduler_config.async_scheduling is None
    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL
    assert vllm_config.model_config.hf_config.head_dtype == "float32"
    assert vllm_config.compilation_config.max_cudagraph_capture_size == 8192
    assert 1024 in vllm_config.compilation_config.cudagraph_capture_sizes
    assert 5120 in vllm_config.compilation_config.cudagraph_capture_sizes
    assert 8192 in vllm_config.compilation_config.cudagraph_capture_sizes


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
        ("rwkv", SamplingParams(min_p=0.1), "min_p"),
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
    for name in ("adaptive_verification", "pooling_runner", "encoder_cache"):
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
    hidden_states = torch.arange(12, dtype=torch.float16).view(4, 3)

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
def test_rapid_sampler_processing_preserves_raw_logits(mixed_prefill) -> None:
    sampler = object.__new__(RwkvSampler)
    slots = np.arange(2)
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
    sampler._presence_penalty = sampler._frequency_penalty = values
    sampler._penalty_decay = values
    sampler.sampling_states = SimpleNamespace(
        temperature=values, top_k=values, top_p=values
    )
    sampler.needs_logits_processing = np.ones(2, dtype=bool)
    sampler._logit_bias_state = SimpleNamespace(
        apply=lambda logits, ctx: logits.add_(torch.tensor([0.0, 10.0]))
    )
    sampler._bad_words_state = SimpleNamespace(apply=lambda logits, ctx: logits)
    sampler.thinking_budget_state = SimpleNamespace(apply=lambda logits, ctx: None)
    sampler._sample = lambda logits, *args, **kwargs: logits.argmax(dim=-1)
    sampler._rapid_penalties = sampler._rapid_sampling_states = torch.zeros(2, 2)
    sampler._input_batch = SimpleNamespace(
        num_reqs=2,
        num_computed_prefill_tokens_np=np.zeros(2),
        num_scheduled_tokens=np.ones(2),
        prefill_len_np=np.array([1, 2 if mixed_prefill else 1]),
    )
    logits = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    raw = logits.clone()
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
    assert torch.equal(logits, raw)
    assert torch.equal(returned, raw)


def test_config_accepts_async_scheduling() -> None:
    vllm_config = _config()
    vllm_config.scheduler_config.async_scheduling = True
    RwkvForCausalLMConfig.verify_and_update_config(vllm_config)
    assert vllm_config.scheduler_config.async_scheduling


@pytest.mark.skipif(
    not os.environ.get("VLLM_RWKV_TEST_MODEL"),
    reason="Set VLLM_RWKV_TEST_MODEL to a local RWKV artifact for GPU evaluation",
)
@pytest.mark.parametrize("state_dtype", ["float16", "float32"])
def test_real_model_state_restore_preserves_continuation(tmp_path, state_dtype) -> None:
    """Restoring a snapshot must preserve logits for the same continuation calls."""
    from transformers import AutoTokenizer

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.model_loader import get_model

    model_path = os.environ["VLLM_RWKV_TEST_MODEL"]
    config = EngineArgs(
        model=model_path,
        dtype="float16",
        mamba_ssm_cache_dtype=state_dtype,
        enforce_eager=True,
        max_num_seqs=3,
        max_model_len=1024,
        max_num_batched_tokens=256,
        enable_prefix_caching=False,
    ).create_engine_config()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        with set_current_vllm_config(config), torch.inference_mode():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{tmp_path / 'rendezvous'}",
                local_rank=0,
                backend="nccl",
            )
            initialize_model_parallel(1, 1)
            model = get_model(vllm_config=config)
            state = model.get_rwkv_state_layer()
            spec = state.get_kv_cache_spec(config)
            raw = torch.zeros(
                3 * spec.page_size_bytes, dtype=torch.uint8, device="cuda"
            )
            state.bind_kv_cache(raw)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            prompt = tokenizer.encode(
                "User: Remember project 42 and the color blue.\n\nAssistant:"
            )
            delta = tokenizer.encode("User: What is the project number?\n\nAssistant:")

            def forward(tokens, slot):
                ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
                count = ids.numel()
                metadata = RwkvAttentionMetadata(
                    cu_seqlens=torch.tensor(
                        [0, count], dtype=torch.int32, device="cuda"
                    ),
                    state_indices=torch.tensor(
                        [slot], dtype=torch.int32, device="cuda"
                    ),
                    batch_size=1,
                    max_seqlen=count,
                    token_capacity=count,
                    sequence_capacity=1,
                    max_seqlen_capacity=count,
                )
                with set_forward_context(metadata, config, num_tokens=count):
                    hidden = model(ids, torch.arange(count, device="cuda"))
                    return model.compute_logits(hidden[-1:]).clone()

            for length in (1, 20, 64, 129):
                state.reset_state_slots([0, 1, 2])
                forward((prompt * 129)[:length], 0)
                snapshot = state.capture_state_slot(0)
                expected = forward(delta, 0)
                forward(prompt[::-1], 1)
                state.restore_state_slot(snapshot, 1)
                restored = forward(delta, 1)
                torch.testing.assert_close(restored, expected, rtol=0, atol=0)
                for _ in range(4):
                    token = expected.argmax(-1).tolist()
                    expected = forward(token, 0)
                    restored = forward(token, 1)
                    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
                state.restore_state_slot(snapshot, 2)
                expected = forward(delta, 2)
                state.restore_state_slot(snapshot, 0)
                torch.testing.assert_close(forward(delta, 0), expected, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        cleanup_dist_env_and_memory()
