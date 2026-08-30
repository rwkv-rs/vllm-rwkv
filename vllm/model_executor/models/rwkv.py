# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native RWKV-7 inference backed exclusively by FlashRWKV2."""

from collections.abc import Iterable, Sequence
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsMambaPrefixCaching,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backends.rwkv_attn import (
    RwkvAttentionBackend,
    RwkvAttentionMetadata,
    RwkvStateSpec,
    get_rwkv_metadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec

from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

_FLASHRWKV2_VERSION = "0.1.0a13"
_FLASHRWKV2_APIS = (
    "infer_cmix_forward_varlen",
    "infer_embedding_ln0_forward_varlen",
    "infer_post_norm_output_forward_varlen",
    "infer_sampling_six_parameter_forward_varlen",
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "infer_tmix_readout_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_tmix_wkv_prepare_forward_varlen",
    "prepare_tmix_wkv7_recurrent_fp16_state",
    "prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor",
    "prepare_tmix_wkv7_recurrent_metadata",
    "setup_sampling_states",
)


def _load_flashrwkv2() -> Any:
    try:
        import flashrwkv2
    except ImportError as error:
        raise RuntimeError(
            f"RWKV requires FlashRWKV2 {_FLASHRWKV2_VERSION} with the public "
            "recurrent-state provider API"
        ) from error

    version = getattr(flashrwkv2, "__version__", None)
    if version != _FLASHRWKV2_VERSION:
        raise RuntimeError(
            f"RWKV requires FlashRWKV2 {_FLASHRWKV2_VERSION}, got {version!r}"
        )
    missing = [name for name in _FLASHRWKV2_APIS if not hasattr(flashrwkv2, name)]
    if missing:
        raise RuntimeError(
            "FlashRWKV2 is missing required public APIs: " + ", ".join(missing)
        )
    return flashrwkv2


def _validate_config(config: Any) -> None:
    if getattr(config, "architecture_version", None) != "rwkv7":
        raise ValueError("RWKV requires architecture_version='rwkv7'")
    if config.head_size != 64:
        raise ValueError("RWKV requires head_size=64")
    if config.hidden_size % config.head_size:
        raise ValueError("RWKV hidden_size must be divisible by head_size")
    if config.num_attention_heads != config.hidden_size // config.head_size:
        raise ValueError("RWKV num_attention_heads must equal hidden_size // head_size")
    if config.intermediate_size != 4 * config.hidden_size:
        raise ValueError("RWKV intermediate_size must equal 4 * hidden_size")
    if config.tie_word_embeddings:
        raise ValueError("RWKV requires untied embedding and LM-head weights")


class RwkvStateLayer(nn.Module, AttentionLayerBase):
    """One scheduler-visible state layer for all local RWKV decoder layers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        num_local_layers: int,
        prefix: str,
    ) -> None:
        super().__init__()
        if num_local_layers <= 0:
            raise ValueError("each RWKV pipeline rank must own a decoder layer")

        config = vllm_config.model_config.hf_config
        self.prefix = prefix
        self.num_local_layers = num_local_layers
        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        self.num_heads = config.num_attention_heads
        self.sequence_capacity = vllm_config.scheduler_config.max_num_seqs
        self.state_dtype = (
            torch.float32
            if vllm_config.cache_config.mamba_ssm_cache_dtype == "float32"
            else torch.float16
        )
        self._vllm_config = vllm_config
        self._spec: RwkvStateSpec | None = None
        self._num_slots = 0
        self._tmix_shift_pool: torch.Tensor | None = None
        self._cmix_shift_pool: torch.Tensor | None = None
        self._wkv_state_pool: torch.Tensor | None = None
        self._wkv_handles: list[object] = []
        self._live_graph_tickets: list[object] = []
        self.kv_cache = torch.tensor([])

        static_context = vllm_config.compilation_config.static_forward_context
        if prefix in static_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        static_context[prefix] = self

    def _prepare_wkv_state(
        self,
        state_pool_size: int,
        device: torch.device,
    ) -> object:
        flashrwkv2 = _load_flashrwkv2()
        if self.state_dtype == torch.float32:
            state = torch.zeros(
                (
                    state_pool_size,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                ),
                dtype=torch.float32,
                device=device,
            )
            return flashrwkv2.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
                state
            )
        return flashrwkv2.prepare_tmix_wkv7_recurrent_fp16_state(
            state_pool_size,
            self.hidden_size,
            sequence_capacity=self.sequence_capacity,
            head_size=self.head_size,
            device=device,
        )

    def _probe_wkv_state_layout(self, device: torch.device) -> dict[str, int]:
        handle = self._prepare_wkv_state(1, device)
        try:
            return handle.memory_layout
        finally:
            del handle
            if device.type == "cuda":
                torch.accelerator.empty_cache()

    def _build_spec(self, vllm_config: VllmConfig) -> RwkvStateSpec:
        device = torch.device(vllm_config.device_config.device)
        layout = self._probe_wkv_state_layout(device)
        layer_count = self.num_local_layers
        shapes: list[tuple[int, ...]] = [
            (layer_count, self.hidden_size),
            (layer_count, self.hidden_size),
        ]
        dtypes = [torch.float16, torch.float16]
        if self.state_dtype == torch.float32:
            shapes.append(
                (
                    layer_count,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                )
            )
            dtypes.append(torch.float32)

        block_size = vllm_config.cache_config.mamba_block_size
        assert block_size is not None
        cache_mode = vllm_config.cache_config.mamba_cache_mode
        return RwkvStateSpec(
            block_size=block_size,
            shapes=tuple(shapes),
            dtypes=tuple(dtypes),
            mamba_cache_mode=cache_mode,
            num_prefill_checkpoint_blocks=0,
            provider_state_bytes_per_page=(
                layer_count * layout["bytes_per_slot"]
                if self.state_dtype == torch.float16
                else 0
            ),
            provider_fixed_workspace_bytes=(
                layer_count * layout["fixed_workspace_nbytes"]
                if self.state_dtype == torch.float16
                else 0
            ),
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        if self._spec is None:
            self._spec = self._build_spec(vllm_config)
        return self._spec

    def get_attn_backend(self) -> type[RwkvAttentionBackend]:
        return RwkvAttentionBackend

    @staticmethod
    def _typed_view(
        raw: torch.Tensor,
        offset: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        elements = 1
        for size in shape:
            elements *= size
        byte_count = elements * torch.empty((), dtype=dtype).element_size()
        view = raw[offset : offset + byte_count].view(dtype).view(shape)
        return view, offset + byte_count

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        spec = self.get_kv_cache_spec(self._vllm_config)
        if not kv_cache.is_contiguous():
            raise ValueError("RWKV state allocation must be contiguous")
        total_bytes = kv_cache.numel() * kv_cache.element_size()
        if total_bytes % spec.page_size_bytes:
            raise ValueError("RWKV state allocation has a partial page")
        self._num_slots = total_bytes // spec.page_size_bytes
        raw = kv_cache.view(torch.int8).view(-1)
        offset = 0
        layer_count = self.num_local_layers
        slot_count = self._num_slots

        self._tmix_shift_pool, offset = self._typed_view(
            raw,
            offset,
            (layer_count, slot_count, self.hidden_size),
            torch.float16,
        )
        self._cmix_shift_pool, offset = self._typed_view(
            raw,
            offset,
            (layer_count, slot_count, self.hidden_size),
            torch.float16,
        )
        if self.state_dtype == torch.float32:
            self._wkv_state_pool, offset = self._typed_view(
                raw,
                offset,
                (
                    layer_count,
                    slot_count,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                ),
                torch.float32,
            )
        if offset != total_bytes:
            raise ValueError("RWKV state pool layout does not fill its allocation")

        flashrwkv2 = _load_flashrwkv2()
        self._wkv_handles = []
        for layer_idx in range(layer_count):
            if self._wkv_state_pool is None:
                handle = self._prepare_wkv_state(slot_count, kv_cache.device)
                expected_state_bytes = spec.provider_state_bytes_per_page // layer_count
                expected_fixed_bytes = (
                    spec.provider_fixed_workspace_bytes // layer_count
                )
            else:
                handle = (
                    flashrwkv2.prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(
                        self._wkv_state_pool[layer_idx]
                    )
                )
                expected_state_bytes = (
                    self.num_heads
                    * self.head_size
                    * self.head_size
                    * torch.empty((), dtype=torch.float32).element_size()
                )
                expected_fixed_bytes = 0
            handle_layout = handle.memory_layout
            if (
                handle_layout["bytes_per_slot"] != expected_state_bytes
                or handle_layout["fixed_workspace_nbytes"] != expected_fixed_bytes
                or handle_layout["total_nbytes"]
                != slot_count * handle_layout["bytes_per_slot"]
                + handle_layout["fixed_workspace_nbytes"]
            ):
                raise RuntimeError(
                    "FlashRWKV2 allocated a different recurrent state layout "
                    "than it reported during cache planning"
                )
            self._wkv_handles.append(handle)
        self.kv_cache = kv_cache

    def clear_kv_cache(self) -> None:
        self._wkv_handles.clear()
        self._tmix_shift_pool = None
        self._cmix_shift_pool = None
        self._wkv_state_pool = None
        self._num_slots = 0
        self._live_graph_tickets.clear()
        self.kv_cache = torch.tensor([])

    @property
    def state_pool_size(self) -> int:
        if self._num_slots == 0:
            raise RuntimeError("RWKV state pool has not been bound")
        return self._num_slots

    def get_layer_state(
        self, local_layer_idx: int
    ) -> tuple[torch.Tensor, object, torch.Tensor]:
        if self._tmix_shift_pool is None or self._cmix_shift_pool is None:
            raise RuntimeError("RWKV state pool has not been bound")
        return (
            self._tmix_shift_pool[local_layer_idx],
            self._wkv_handles[local_layer_idx],
            self._cmix_shift_pool[local_layer_idx],
        )

    def _slot_tensor(self, slot_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        if self._tmix_shift_pool is None:
            raise RuntimeError("RWKV state pool has not been bound")
        device = self._tmix_shift_pool.device
        if isinstance(slot_ids, torch.Tensor):
            return slot_ids.to(device=device, dtype=torch.int32).contiguous()
        return async_tensor_h2d(slot_ids, dtype=torch.int32, device=device)

    def reset_state_slots(self, slot_ids: Sequence[int] | torch.Tensor) -> None:
        indices = self._slot_tensor(slot_ids)
        if indices.numel() == 0:
            return
        long_indices = indices.long()
        assert self._tmix_shift_pool is not None
        assert self._cmix_shift_pool is not None
        self._tmix_shift_pool.index_fill_(1, long_indices, 0)
        self._cmix_shift_pool.index_fill_(1, long_indices, 0)
        for handle in self._wkv_handles:
            handle.reset_slots_(indices)

    def copy_state_slots(
        self,
        source_ids: Sequence[int] | torch.Tensor,
        destination_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        source = self._slot_tensor(source_ids)
        destination = self._slot_tensor(destination_ids)
        if source.numel() != destination.numel():
            raise ValueError("RWKV state copy source/destination lengths differ")
        if source.numel() == 0:
            return
        source_long = source.long()
        destination_long = destination.long()
        assert self._tmix_shift_pool is not None
        assert self._cmix_shift_pool is not None
        tmix_values = self._tmix_shift_pool.index_select(1, source_long)
        cmix_values = self._cmix_shift_pool.index_select(1, source_long)
        self._tmix_shift_pool.index_copy_(1, destination_long, tmix_values)
        self._cmix_shift_pool.index_copy_(1, destination_long, cmix_values)
        for handle in self._wkv_handles:
            handle.copy_slots_(handle, source, destination)

    def copy_state_block_pairs(self, block_copies: Sequence[tuple[int, int]]) -> None:
        if not block_copies:
            return
        source, destination = zip(*block_copies, strict=True)
        self.copy_state_slots(source, destination)

    def materialize_state_slots(self, slot_ids: Sequence[int] | torch.Tensor) -> None:
        indices = self._slot_tensor(slot_ids)
        for handle in self._wkv_handles:
            handle.materialize_slots_(indices)

    def _prepare_live_ticket(self, metadata: RwkvAttentionMetadata) -> object:
        flashrwkv2 = _load_flashrwkv2()
        assert metadata.num_active_tokens is not None
        assert metadata.num_active_sequences is not None
        return flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
            metadata.cu_seqlens,
            metadata.state_indices,
            state_pool_size=self.state_pool_size,
            token_capacity=metadata.token_capacity,
            sequence_capacity=metadata.sequence_capacity,
            max_seqlen_capacity=metadata.max_seqlen_capacity,
            num_active_tokens=metadata.num_active_tokens,
            num_active_sequences=metadata.num_active_sequences,
        )

    def prepare_metadata_ticket(self, metadata: RwkvAttentionMetadata) -> object:
        if metadata.is_live:
            assert metadata.num_active_tokens is not None
            assert metadata.num_active_sequences is not None
            ticket = self._prepare_live_ticket(metadata)
            if metadata.retain_ticket:
                self._live_graph_tickets.append(ticket)
            return ticket
        flashrwkv2 = _load_flashrwkv2()
        return flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
            metadata.cu_seqlens,
            metadata.state_indices,
            state_pool_size=self.state_pool_size,
            total_tokens=metadata.token_capacity,
            max_seqlen=metadata.max_seqlen,
        )

    def warmup_live_metadata(self, metadata: RwkvAttentionMetadata) -> None:
        self._live_graph_tickets.append(self._prepare_live_ticket(metadata))


class RwkvAttention(nn.Module):
    def __init__(self, config: Any, layer_idx: int, prefix: str) -> None:
        super().__init__()
        channels = config.hidden_size
        dtype = torch.float16
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = channels
        self.head_size = config.head_size
        self.num_heads = config.num_attention_heads

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(channels, dtype=dtype)))
        self.w0 = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.w1 = nn.Parameter(
            torch.empty(channels, config.decay_low_rank_dim, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.decay_low_rank_dim, channels, dtype=dtype)
        )
        self.a0 = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.a1 = nn.Parameter(
            torch.empty(channels, config.a_low_rank_dim, dtype=dtype)
        )
        self.a2 = nn.Parameter(
            torch.empty(config.a_low_rank_dim, channels, dtype=dtype)
        )
        if layer_idx != 0:
            self.v0 = nn.Parameter(torch.empty(channels, dtype=dtype))
            self.v1 = nn.Parameter(
                torch.empty(channels, config.v_low_rank_dim, dtype=dtype)
            )
            self.v2 = nn.Parameter(
                torch.empty(config.v_low_rank_dim, channels, dtype=dtype)
            )
        self.g1 = nn.Parameter(
            torch.empty(channels, config.gate_low_rank_dim, dtype=dtype)
        )
        self.g2 = nn.Parameter(
            torch.empty(config.gate_low_rank_dim, channels, dtype=dtype)
        )
        self.k_k = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.k_a = nn.Parameter(torch.empty(channels, dtype=dtype))
        self.r_k = nn.Parameter(
            torch.empty(self.num_heads, self.head_size, dtype=dtype)
        )

        self.r_proj = ReplicatedLinear(
            channels,
            channels,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.r_proj",
        )
        self.k_proj = ReplicatedLinear(
            channels,
            channels,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ReplicatedLinear(
            channels,
            channels,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.v_proj",
        )
        self.o_proj = ReplicatedLinear(
            channels,
            channels,
            bias=False,
            params_dtype=dtype,
            prefix=f"{prefix}.o_proj",
        )
        self.g_norm = nn.GroupNorm(
            self.num_heads,
            channels,
            eps=config.group_norm_epsilon,
            affine=True,
            dtype=dtype,
        )

        for name in (
            "w1_canonical",
            "a1_canonical",
            "g1_canonical",
            "v1_canonical",
            "w2_canonical",
            "a2_canonical",
            "g2_canonical",
            "v2_canonical",
            "layer_zero_v0",
            "layer_zero_v1_runtime",
            "layer_zero_v2_runtime",
        ):
            self.register_buffer(name, torch.empty(0), persistent=False)

    def process_weights_after_loading(self) -> None:
        self.w1_canonical = self.w1.T.contiguous()
        self.a1_canonical = self.a1.T.contiguous()
        self.g1_canonical = self.g1.T.contiguous()
        self.w2_canonical = self.w2.T.contiguous()
        self.a2_canonical = self.a2.T.contiguous()
        self.g2_canonical = self.g2.T.contiguous()
        if self.layer_idx == 0:
            rank = self.config.v_low_rank_dim
            self.v1_canonical = self.w1.new_zeros((rank, self.hidden_size))
            self.v2_canonical = self.w1.new_zeros((self.hidden_size, rank))
            self.layer_zero_v0 = self.w1.new_zeros(self.hidden_size)
            self.layer_zero_v1_runtime = self.w1.new_zeros((self.hidden_size, rank))
            self.layer_zero_v2_runtime = self.w1.new_zeros((rank, self.hidden_size))
        else:
            self.v1_canonical = self.v1.T.contiguous()
            self.v2_canonical = self.v2.T.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor,
        wkv_state: object,
        metadata: RwkvAttentionMetadata,
        state_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flashrwkv2 = _load_flashrwkv2()
        ticket = metadata.validated_metadata
        residual, xr, xw, xk, xv, xa, xg = (
            flashrwkv2.infer_tmix_postnorm_tokenshift_forward_varlen(
                hidden_states.contiguous(),
                residual.contiguous(),
                layer_norm.weight.contiguous(),
                layer_norm.bias.contiguous(),
                self.x_r,
                self.x_w,
                self.x_k,
                self.x_v,
                self.x_a,
                self.x_g,
                shift_state_pool=attention_shift,
                cu_seqlens=metadata.cu_seqlens,
                state_indices=metadata.state_indices,
                max_seqlen=metadata.max_seqlen,
                eps=self.config.layer_norm_epsilon,
                validated_metadata=ticket,
            )
        )
        if self.layer_idx == 0:
            v0 = self.layer_zero_v0
            v1_runtime = self.layer_zero_v1_runtime
            v2_runtime = self.layer_zero_v2_runtime
        else:
            v0 = self.v0
            v1_runtime = self.v1
            v2_runtime = self.v2

        receptance, decay, key, value, recurrent_a, recurrent_b, gate, v_first = (
            flashrwkv2.infer_tmix_wkv_prepare_forward_varlen(
                xr,
                xw,
                xk,
                xv,
                xa,
                xg,
                self.r_proj.weight,
                self.k_proj.weight,
                self.v_proj.weight,
                self.w1_canonical,
                self.a1_canonical,
                self.g1_canonical,
                self.v1_canonical,
                self.w2_canonical,
                self.a2_canonical,
                self.g2_canonical,
                self.v2_canonical,
                v0,
                self.k_k,
                self.a0,
                self.k_a,
                v_first=v_first,
                w1_runtime=self.w1,
                a1_runtime=self.a1,
                g1_runtime=self.g1,
                v1_runtime=v1_runtime,
                w2_runtime=self.w2,
                a2_runtime=self.a2,
                g2_runtime=self.g2,
                v2_runtime=v2_runtime,
                head_size=self.head_size,
                batch_size=metadata.batch_size,
                max_seqlen=metadata.max_seqlen,
            )
        )
        recurrent = (
            flashrwkv2.infer_tmix_wkv7_recurrent_fp16_forward_varlen
            if state_dtype == torch.float16
            else flashrwkv2.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen
        )
        wkv_output = recurrent(
            receptance.view(-1, self.num_heads, self.head_size),
            decay.view(-1, self.num_heads, self.head_size),
            key.view(-1, self.num_heads, self.head_size),
            value.view(-1, self.num_heads, self.head_size),
            recurrent_a.view(-1, self.num_heads, self.head_size),
            recurrent_b.view(-1, self.num_heads, self.head_size),
            state=wkv_state,
            cu_seqlens=metadata.cu_seqlens,
            state_indices=metadata.state_indices,
            decay_bias=self.w0,
            max_seqlen=metadata.max_seqlen,
            validated_metadata=ticket,
        )
        output = flashrwkv2.infer_tmix_readout_forward_varlen(
            wkv_output.view(-1, self.hidden_size),
            receptance,
            key,
            value,
            self.r_k.flatten().contiguous(),
            self.g_norm.weight,
            self.g_norm.bias,
            gate,
            self.o_proj.weight,
            head_size=self.head_size,
            batch_size=metadata.batch_size,
            max_seqlen=metadata.max_seqlen,
        )
        return output, residual, v_first


class RwkvFeedForward(nn.Module):
    def __init__(self, config: Any, prefix: str) -> None:
        super().__init__()
        self.config = config
        self.x_k = nn.Parameter(torch.empty(config.hidden_size, dtype=torch.float16))
        self.key = ReplicatedLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.key",
        )
        self.value = ReplicatedLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.float16,
            prefix=f"{prefix}.value",
        )

    def process_weights_after_loading(self) -> None:
        self.value.weight.data = self.value.weight.T.contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        feed_forward_shift: torch.Tensor,
        metadata: RwkvAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flashrwkv2 = _load_flashrwkv2()
        return flashrwkv2.infer_cmix_forward_varlen(
            hidden_states.contiguous(),
            residual.contiguous(),
            layer_norm.weight,
            layer_norm.bias,
            self.x_k,
            self.key.weight,
            self.value.weight,
            shift_state_pool=feed_forward_shift,
            cu_seqlens=metadata.cu_seqlens,
            state_indices=metadata.state_indices,
            max_seqlen=metadata.max_seqlen,
            eps=self.config.layer_norm_epsilon,
            validated_metadata=metadata.validated_metadata,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )


class RwkvDecoderLayer(nn.Module):
    def __init__(self, config: Any, prefix: str) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.linear_attn = RwkvAttention(config, layer_idx, f"{prefix}.linear_attn")
        self.mlp = RwkvFeedForward(config, f"{prefix}.mlp")
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=torch.float16,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=torch.float16,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        v_first: torch.Tensor | None,
        state: tuple[torch.Tensor, object, torch.Tensor] | None,
        metadata: RwkvAttentionMetadata | None,
        state_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if metadata is None:
            receptance, _ = self.linear_attn.r_proj(hidden_states)
            key, _ = self.linear_attn.k_proj(hidden_states)
            value, _ = self.linear_attn.v_proj(hidden_states)
            hidden_states, _ = self.linear_attn.o_proj(receptance + key + value)
            channel, _ = self.mlp.key(hidden_states)
            return torch.relu(channel) @ self.mlp.value.weight, residual, v_first

        assert state is not None
        attention_shift, wkv_state, feed_forward_shift = state
        attention_output, residual, v_first = self.linear_attn(
            hidden_states,
            residual,
            self.input_layernorm,
            v_first,
            attention_shift,
            wkv_state,
            metadata,
            state_dtype,
        )
        hidden_states, residual = self.mlp(
            residual,
            attention_output,
            self.post_attention_layernorm,
            feed_forward_shift,
            metadata,
        )
        return hidden_states, residual, v_first


class RwkvModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        _validate_config(config)
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.bfloat16,
                prefix=maybe_prefix(prefix, "embed_tokens"),
                disable_tp=True,
            )
            self.embedding_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
                dtype=torch.bfloat16,
            )
            self._embedding_norm_folded = False
        else:
            self.embed_tokens = PPMissingLayer()
            self.embedding_norm = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RwkvDecoderLayer(config, prefix),
            prefix=maybe_prefix(prefix, "layers"),
        )
        state_prefix = maybe_prefix(prefix, f"layers.{self.start_layer}.rwkv_state")
        self.rwkv_state = RwkvStateLayer(
            vllm_config,
            self.end_layer - self.start_layer,
            state_prefix,
        )

        if get_pp_group().is_last_rank:
            self.norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_epsilon,
                dtype=torch.float16,
            )
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual", "v_first"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def process_weights_after_loading(self) -> None:
        if get_pp_group().is_first_rank:
            flashrwkv2 = _load_flashrwkv2()
            embedding = self.embed_tokens.weight
            folded = torch.empty_like(embedding, dtype=torch.float16)
            for start in range(0, embedding.shape[0], 4096):
                end = min(start + 4096, embedding.shape[0])
                folded[start:end].copy_(
                    flashrwkv2.infer_embedding_ln0_forward_varlen(
                        embedding[start:end],
                        self.embedding_norm.weight,
                        self.embedding_norm.bias,
                        eps=self.config.layer_norm_epsilon,
                    )
                )
            self.embed_tokens.weight.data = folded
            self._embedding_norm_folded = True
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            assert isinstance(layer, RwkvDecoderLayer)
            layer.linear_attn.process_weights_after_loading()
            layer.mlp.process_weights_after_loading()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors:
        flashrwkv2 = _load_flashrwkv2()
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                assert input_ids is not None
                inputs_embeds = self.embed_input_ids(input_ids)
                assert self._embedding_norm_folded
                hidden_states = inputs_embeds
            else:
                inputs_embeds = inputs_embeds.to(torch.bfloat16).contiguous()
                hidden_states = flashrwkv2.infer_embedding_ln0_forward_varlen(
                    inputs_embeds,
                    self.embedding_norm.weight,
                    self.embedding_norm.bias,
                    eps=self.config.layer_norm_epsilon,
                ).to(torch.float16)
            residual = torch.zeros_like(hidden_states)
            v_first = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
            v_first = intermediate_tensors["v_first"]

        metadata = get_rwkv_metadata()
        if metadata is not None and metadata.validated_metadata is None:
            metadata.validated_metadata = self.rwkv_state.prepare_metadata_ticket(
                metadata
            )
        for local_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            assert isinstance(layer, RwkvDecoderLayer)
            state = (
                None if metadata is None else self.rwkv_state.get_layer_state(local_idx)
            )
            hidden_states, residual, v_first = layer(
                hidden_states,
                residual,
                v_first,
                state,
                metadata,
                self.rwkv_state.state_dtype,
            )
        if metadata is None:
            residual = torch.zeros_like(hidden_states)
            v_first = torch.zeros_like(hidden_states)

        if not get_pp_group().is_last_rank:
            assert v_first is not None
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                    "v_first": v_first,
                }
            )
        return flashrwkv2.infer_post_norm_output_forward_varlen(
            hidden_states,
            residual,
            self.norm.weight,
            self.norm.bias,
            eps=self.config.layer_norm_epsilon,
        )


class RwkvForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
    SupportsMambaPrefixCaching,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = RwkvModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                params_dtype=torch.float16,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_rwkv_state_layer(self) -> RwkvStateLayer:
        return self.model.rwkv_state

    @staticmethod
    def get_model_state_cls():
        from vllm.v1.worker.gpu.model_states.rwkv import RwkvModelState

        return RwkvModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del positions, kwargs
        return self.model(input_ids, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)

    def process_weights_after_loading(self) -> None:
        self.model.process_weights_after_loading()
