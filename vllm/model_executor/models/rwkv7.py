# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV-7 model implemented on vLLM's standard recurrent-state interfaces."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.mamba.rwkv7 import (
    RWKV7_HEAD_SIZE,
    RWKV7Block,
    rwkv7_state_dtypes,
    rwkv7_state_shapes,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
)
from vllm.sequence import IntermediateTensors

from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_layers,
    maybe_prefix,
)


def _make_rwkv7_intermediate_tensors_factory(
    hidden_size: int,
    local_hidden_size: int,
):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "v_first": torch.zeros(
                    (batch_size, local_hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }
        )

    return make_empty_intermediate_tensors


@support_torch_compile
class RWKV7Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RWKV7Block(
                config=config,
                vllm_config=vllm_config,
                layer_idx=int(prefix.rsplit(".", 1)[-1]),
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = nn.LayerNorm(
                config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self.norm = PPMissingLayer()

        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.make_empty_intermediate_tensors = _make_rwkv7_intermediate_tensors_factory(
            config.hidden_size,
            config.hidden_size // tp_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        del positions
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds
                if inputs_embeds is not None
                else self.embed_input_ids(input_ids)
            )
            v_first = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            v_first = intermediate_tensors["v_first"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, v_first = layer(hidden_states, v_first)

        if not get_pp_group().is_last_rank:
            assert v_first is not None
            return IntermediateTensors(
                {"hidden_states": hidden_states, "v_first": v_first}
            )
        return self.norm(hidden_states)


class RWKV7ForCausalLM(
    nn.Module,
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "emb.": "model.embed_tokens.",
            "blocks.": "model.layers.",
            "ln_out.": "model.norm.",
            "head.": "lm_head.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        if vllm_config.model_config.dtype != torch.float16:
            raise ValueError("RWKV7 currently requires model dtype float16")
        if config.head_size != RWKV7_HEAD_SIZE:
            raise ValueError("RWKV7 currently requires head_size=64")

        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.model = RWKV7Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return rwkv7_state_dtypes(vllm_config.model_config.dtype)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], ...]:
        config = vllm_config.model_config.hf_config
        return rwkv7_state_shapes(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, ...]:
        return (
            get_temporal_copy_spec,
            get_temporal_copy_spec,
            get_temporal_copy_spec,
        )

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
