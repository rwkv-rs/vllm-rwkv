# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import vllm.envs as envs
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.linear_attn import LinearAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec

RWKV7_HEAD_SIZE = 64


def rwkv7_state_shapes(
    *,
    hidden_size: int,
    num_attention_heads: int,
    tensor_parallel_size: int,
) -> tuple[tuple[int, ...], ...]:
    if hidden_size != num_attention_heads * RWKV7_HEAD_SIZE:
        raise ValueError("RWKV7 hidden size must match its 64-wide attention heads")
    if num_attention_heads % tensor_parallel_size:
        raise ValueError(
            "RWKV7 attention heads must be divisible by tensor parallel size"
        )
    return (
        (2, hidden_size),
        (
            num_attention_heads // tensor_parallel_size,
            RWKV7_HEAD_SIZE,
            RWKV7_HEAD_SIZE,
        ),
        (1,),
    )


def rwkv7_state_dtypes(
    model_dtype: torch.dtype,
) -> tuple[torch.dtype, ...]:
    wkv_dtype = (
        torch.float32 if envs.VLLM_RWKV7_WKV_MODE == "fp32io16" else torch.float16
    )
    return model_dtype, wkv_dtype, torch.int32


def _parameter(
    *shape: int,
    dtype: torch.dtype,
    shard_axis: int | None = None,
) -> Parameter:
    parameter = Parameter(torch.empty(shape, dtype=dtype))
    if shard_axis is not None:
        set_weight_attrs(
            parameter,
            {"weight_loader": sharded_weight_loader(shard_axis)},
        )
    return parameter


def clear_rwkv7_state_for_new_sequences(
    states: tuple[torch.Tensor, ...],
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    state_indices: torch.Tensor,
) -> None:
    """Clear recycled cache slots only for requests with no prior context."""
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    fresh = seq_lens[: query_lens.shape[0]] == query_lens
    slots = state_indices[: query_lens.shape[0]]
    fresh_slots = slots[fresh]
    for state in states:
        state[fresh_slots] = 0


def rwkv7_time_shift(
    hidden_states: torch.Tensor,
    shift_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Apply RWKV's one-token shift to packed variable-length requests."""
    previous = torch.roll(hidden_states, shifts=1, dims=0)
    starts = query_start_loc[:-1]
    slots = state_indices[: starts.shape[0]]
    previous[starts] = shift_state[slots]
    shift_state[slots] = hidden_states[query_start_loc[1:] - 1]
    return previous - hidden_states


class RWKV7TimeMix(nn.Module):
    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        state_shapes = rwkv7_state_shapes(
            hidden_size=hidden_size,
            num_attention_heads=config.num_attention_heads,
            tensor_parallel_size=tp_size,
        )

        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.local_num_heads = state_shapes[1][0]
        self.local_hidden_size = self.local_num_heads * RWKV7_HEAD_SIZE
        self.wkv_mode = envs.VLLM_RWKV7_WKV_MODE
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config

        self.x_r = _parameter(1, 1, hidden_size, dtype=dtype)
        self.x_w = _parameter(1, 1, hidden_size, dtype=dtype)
        self.x_k = _parameter(1, 1, hidden_size, dtype=dtype)
        self.x_v = _parameter(1, 1, hidden_size, dtype=dtype)
        self.x_a = _parameter(1, 1, hidden_size, dtype=dtype)
        self.x_g = _parameter(1, 1, hidden_size, dtype=dtype)

        self.w0 = _parameter(1, 1, self.local_hidden_size, dtype=dtype, shard_axis=2)
        self.a0 = _parameter(1, 1, self.local_hidden_size, dtype=dtype, shard_axis=2)
        self.k_k = _parameter(1, 1, self.local_hidden_size, dtype=dtype, shard_axis=2)
        self.k_a = _parameter(1, 1, self.local_hidden_size, dtype=dtype, shard_axis=2)
        self.r_k = _parameter(
            self.local_num_heads,
            RWKV7_HEAD_SIZE,
            dtype=dtype,
            shard_axis=0,
        )

        self.w1 = _parameter(hidden_size, config.decay_rank, dtype=dtype)
        self.w2 = _parameter(
            config.decay_rank,
            self.local_hidden_size,
            dtype=dtype,
            shard_axis=1,
        )
        self.a1 = _parameter(
            hidden_size,
            config.in_context_learning_rank,
            dtype=dtype,
        )
        self.a2 = _parameter(
            config.in_context_learning_rank,
            self.local_hidden_size,
            dtype=dtype,
            shard_axis=1,
        )
        self.g1 = _parameter(hidden_size, config.gate_rank, dtype=dtype)
        self.g2 = _parameter(
            config.gate_rank,
            self.local_hidden_size,
            dtype=dtype,
            shard_axis=1,
        )

        if layer_idx:
            self.v0 = _parameter(
                1,
                1,
                self.local_hidden_size,
                dtype=dtype,
                shard_axis=2,
            )
            self.v1 = _parameter(
                hidden_size,
                config.value_residual_rank,
                dtype=dtype,
            )
            self.v2 = _parameter(
                config.value_residual_rank,
                self.local_hidden_size,
                dtype=dtype,
                shard_axis=1,
            )
        else:
            self.register_parameter("v0", None)
            self.register_parameter("v1", None)
            self.register_parameter("v2", None)

        linear_kwargs = {
            "bias": False,
            "quant_config": quant_config,
            "return_bias": False,
        }
        self.receptance = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            prefix=f"{prefix}.receptance",
            **linear_kwargs,
        )
        self.key = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            prefix=f"{prefix}.key",
            **linear_kwargs,
        )
        self.value = ColumnParallelLinear(
            hidden_size,
            hidden_size,
            prefix=f"{prefix}.value",
            **linear_kwargs,
        )
        self.output = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True,
            prefix=f"{prefix}.output",
            **linear_kwargs,
        )
        self.ln_x = nn.GroupNorm(
            self.local_num_heads,
            self.local_hidden_size,
            eps=64e-5,
            dtype=dtype,
        )
        set_weight_attrs(
            self.ln_x.weight,
            {"weight_loader": sharded_weight_loader(0)},
        )
        set_weight_attrs(
            self.ln_x.bias,
            {"weight_loader": sharded_weight_loader(0)},
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        shifted: torch.Tensor,
        v_first: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed: torch.Tensor,
        query_start_loc: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xr = hidden_states + shifted * self.x_r.view(-1)
        xw = hidden_states + shifted * self.x_w.view(-1)
        xk = hidden_states + shifted * self.x_k.view(-1)
        xv = hidden_states + shifted * self.x_v.view(-1)
        xa = hidden_states + shifted * self.x_a.view(-1)
        xg = hidden_states + shifted * self.x_g.view(-1)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        w = torch.tanh(xw @ self.w1) @ self.w2
        a = torch.sigmoid(self.a0.view(-1) + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        if self.layer_idx == 0:
            v_first = v
        else:
            assert self.v0 is not None and self.v1 is not None and self.v2 is not None
            value_gate = torch.sigmoid(self.v0.view(-1) + (xv @ self.v1) @ self.v2)
            v = v + (v_first - v) * value_gate

        kk = F.normalize(
            (k * self.k_k.view(-1)).view(
                -1,
                self.local_num_heads,
                RWKV7_HEAD_SIZE,
            ),
            dim=-1,
        ).view(-1, self.local_hidden_size)
        k = k * (1 + (a - 1) * self.k_a.view(-1))

        y = torch.empty_like(v)
        import vllm.rwkv7  # noqa: F401

        if self.wkv_mode == "fp32io16":
            torch.ops.rwkv7_wkv_fp32_v2.wkv(
                query_start_loc,
                state_indices,
                wkv_state,
                r.contiguous(),
                (w + self.w0.view(-1)).contiguous(),
                k.contiguous(),
                v.contiguous(),
                (-kk).contiguous(),
                (kk * a).contiguous(),
                y,
            )
        else:
            torch.ops.rwkv7_wkv_fp16_v2.wkv(
                query_start_loc,
                state_indices,
                wkv_state,
                r.contiguous(),
                w.contiguous(),
                self.w0.view(-1),
                k.contiguous(),
                v.contiguous(),
                (-kk).contiguous(),
                (kk * a).contiguous(),
                y,
                elapsed,
            )

        elapsed[state_indices, 0] += query_start_loc[1:] - query_start_loc[:-1]
        y = F.group_norm(
            y,
            self.local_num_heads,
            self.ln_x.weight,
            self.ln_x.bias,
            self.ln_x.eps,
        )
        bonus = (
            (
                r.view(-1, self.local_num_heads, RWKV7_HEAD_SIZE)
                * k.view(-1, self.local_num_heads, RWKV7_HEAD_SIZE)
                * self.r_k
            ).sum(dim=-1, keepdim=True)
            * v.view(-1, self.local_num_heads, RWKV7_HEAD_SIZE)
        ).view(-1, self.local_hidden_size)
        return self.output((y + bonus) * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, *, config, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        dtype = vllm_config.model_config.dtype
        quant_config = vllm_config.quant_config
        self.x_k = _parameter(1, 1, hidden_size, dtype=dtype)
        self.key = ColumnParallelLinear(
            hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key",
            return_bias=False,
        )
        self.value = RowParallelLinear(
            config.intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.value",
            return_bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        shifted: torch.Tensor,
    ) -> torch.Tensor:
        mixed = hidden_states + shifted * self.x_k.view(-1)
        return self.value(F.relu(self.key(mixed)).square())


@PluggableLayer.register("rwkv7_block")
class RWKV7Block(MambaBase, PluggableLayer):
    def __init__(
        self,
        *,
        config,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str,
    ) -> None:
        super().__init__()
        if config.head_size != RWKV7_HEAD_SIZE:
            raise ValueError("RWKV7 currently requires head_size=64")

        self.config = config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.layer_idx = layer_idx
        self.prefix = prefix
        dtype = self.model_config.dtype

        self.ln0 = (
            nn.LayerNorm(config.hidden_size, dtype=dtype) if layer_idx == 0 else None
        )
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=dtype)
        self.att = RWKV7TimeMix(
            config=config,
            vllm_config=vllm_config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.att",
        )
        self.ffn = RWKV7ChannelMix(
            config=config,
            vllm_config=vllm_config,
            prefix=f"{prefix}.ffn",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.RWKV7

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return rwkv7_state_shapes(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            tensor_parallel_size=get_tensor_model_parallel_world_size(),
        )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return rwkv7_state_dtypes(self.model_config.dtype)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MambaSpec)
        return replace(
            spec,
            page_size_padded=round_up(spec.page_size_bytes, 16),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(hidden_states)
        next_v_first = hidden_states.new_empty(
            hidden_states.shape[0],
            self.att.local_hidden_size,
        )
        if v_first is None:
            v_first = hidden_states.new_empty((0, self.att.local_hidden_size))
        torch.ops.vllm.rwkv7_block(
            hidden_states,
            v_first,
            output,
            next_v_first,
            _encode_layer_name(self.prefix),
        )
        return output, next_v_first

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        output: torch.Tensor,
        next_v_first: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        metadata_raw = forward_context.attn_metadata
        if metadata_raw is None:
            output.copy_(hidden_states)
            next_v_first.zero_()
            return

        assert isinstance(metadata_raw, dict)
        metadata = metadata_raw[self.prefix]
        assert isinstance(metadata, LinearAttentionMetadata)
        num_tokens = metadata.num_decode_tokens + metadata.num_prefill_tokens
        num_requests = metadata.num_decodes + metadata.num_prefills
        query_start_loc = metadata.query_start_loc[: num_requests + 1]
        state_indices = metadata.state_indices_tensor[:num_requests]
        seq_lens = metadata.seq_lens[:num_requests]
        shift_state, wkv_state, elapsed = self.kv_cache
        if metadata.num_prefills:
            clear_rwkv7_state_for_new_sequences(
                self.kv_cache,
                query_start_loc,
                seq_lens,
                state_indices,
            )

        x = hidden_states[:num_tokens]
        if self.ln0 is not None:
            x = self.ln0(x)
        time_input = self.ln1(x)
        time_shift = rwkv7_time_shift(
            time_input,
            shift_state[:, 0],
            query_start_loc,
            state_indices,
        )
        time_output, current_v_first = self.att(
            time_input,
            time_shift,
            v_first[:num_tokens],
            wkv_state,
            elapsed,
            query_start_loc,
            state_indices,
        )
        x = x + time_output

        channel_input = self.ln2(x)
        channel_shift = rwkv7_time_shift(
            channel_input,
            shift_state[:, 1],
            query_start_loc,
            state_indices,
        )
        x = x + self.ffn(channel_input, channel_shift)
        output[:num_tokens].copy_(x)
        output[num_tokens:].zero_()
        next_v_first[:num_tokens].copy_(current_v_first)
        next_v_first[num_tokens:].zero_()


def rwkv7_block(
    hidden_states: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    next_v_first: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(hidden_states, v_first, output, next_v_first)


def rwkv7_block_fake(
    hidden_states: torch.Tensor,
    v_first: torch.Tensor,
    output: torch.Tensor,
    next_v_first: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="rwkv7_block",
    op_func=rwkv7_block,
    mutates_args=["output", "next_v_first"],
    fake_impl=rwkv7_block_fake,
)
