# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec
from vllm.model_executor.layers.mamba.rwkv7 import (
    RWKV7Block,
    clear_rwkv7_state_for_new_sequences,
    rwkv7_state_shapes,
    rwkv7_time_shift,
)
from vllm.model_executor.models.config import RWKV7ModelConfig
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM


def test_rwkv7_weight_mapper_preserves_checkpoint_parameter_layout() -> None:
    assert RWKV7ForCausalLM.hf_to_vllm_mapper.apply_list(
        [
            "emb.weight",
            "blocks.0.att.w1",
            "blocks.3.ffn.key.weight",
            "ln_out.bias",
            "head.weight",
        ]
    ) == [
        "model.embed_tokens.weight",
        "model.layers.0.att.w1",
        "model.layers.3.ffn.key.weight",
        "model.norm.bias",
        "lm_head.weight",
    ]


def test_rwkv7_time_shift_respects_packed_request_boundaries() -> None:
    hidden_states = torch.tensor([[2.0], [3.0], [10.0], [20.0], [30.0]])
    shift_state = torch.tensor([[7.0], [100.0], [1.0]])
    query_start_loc = torch.tensor([0, 2, 5], dtype=torch.int32)
    state_indices = torch.tensor([2, 0], dtype=torch.int32)

    shifted = rwkv7_time_shift(
        hidden_states,
        shift_state,
        query_start_loc,
        state_indices,
    )

    torch.testing.assert_close(
        shifted,
        torch.tensor([[-1.0], [-1.0], [-3.0], [-10.0], [-10.0]]),
    )
    torch.testing.assert_close(
        shift_state,
        torch.tensor([[30.0], [100.0], [3.0]]),
    )


def test_rwkv7_state_reset_only_clears_requests_without_context() -> None:
    shift_state = torch.arange(18, dtype=torch.float32).view(3, 2, 3)
    wkv_state = torch.arange(24, dtype=torch.float32).view(3, 2, 2, 2)
    shift_state[2, 0, 0] = torch.nan
    wkv_state[2, 0, 0, 0] = torch.inf
    elapsed = torch.tensor([[9], [10], [11]], dtype=torch.int32)
    original_slot_zero = tuple(
        state[0].clone() for state in (shift_state, wkv_state, elapsed)
    )

    clear_rwkv7_state_for_new_sequences(
        (shift_state, wkv_state, elapsed),
        query_start_loc=torch.tensor([0, 2, 3], dtype=torch.int32),
        seq_lens=torch.tensor([8, 1], dtype=torch.int32),
        state_indices=torch.tensor([0, 2], dtype=torch.int32),
    )

    for state, expected in zip(
        (shift_state, wkv_state, elapsed),
        original_slot_zero,
    ):
        torch.testing.assert_close(state[0], expected)
        torch.testing.assert_close(state[2], torch.zeros_like(state[2]))


def test_rwkv7_state_contract_has_matching_shape_dtype_and_copy_entries() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.float16,
            hf_config=SimpleNamespace(
                hidden_size=2048,
                num_attention_heads=32,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )

    shapes = RWKV7ForCausalLM.get_mamba_state_shape_from_config(vllm_config)
    dtypes = RWKV7ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)
    copy_funcs = RWKV7ForCausalLM.get_mamba_state_copy_func()

    assert shapes == ((2, 2048), (16, 64, 64), (1,))
    assert dtypes[0] == torch.float16
    assert dtypes[2] == torch.int32
    assert len(shapes) == len(dtypes) == len(copy_funcs)
    assert all(copy_func is get_temporal_copy_spec for copy_func in copy_funcs)


def test_rwkv7_state_contract_rejects_fractional_head_shards() -> None:
    with pytest.raises(ValueError, match="attention heads"):
        rwkv7_state_shapes(
            hidden_size=768,
            num_attention_heads=12,
            tensor_parallel_size=8,
        )


def _rwkv7_vllm_config(
    *,
    enable_prefix_caching: bool,
    speculative_config: object | None,
    model: str = (
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1h-1.5b-20260710-ctx10240.pth"
    ),
    load_format: str = "auto",
) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            mamba_cache_mode="none",
            mamba_block_size=None,
            mamba_page_size_padded=None,
        ),
        speculative_config=speculative_config,
        load_config=SimpleNamespace(load_format=load_format),
        model_config=SimpleNamespace(
            model=model,
            max_model_len=4096,
            dtype=torch.float16,
            hf_config=SimpleNamespace(
                hidden_size=2048,
                num_attention_heads=32,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


def test_rwkv7_config_rejects_speculative_decoding() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=object(),
    )

    with pytest.raises(ValueError, match="speculative decoding"):
        RWKV7ModelConfig.verify_and_update_config(vllm_config)


def test_rwkv7_config_disables_unsupported_prefix_caching() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=True,
        speculative_config=None,
    )

    RWKV7ModelConfig.verify_and_update_config(vllm_config)

    assert not vllm_config.cache_config.enable_prefix_caching


@pytest.mark.parametrize("load_format", ["auto", "hf", "pt", "rwkv_pth"])
def test_rwkv7_config_selects_raw_pth_loader(load_format: str) -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
        load_format=load_format,
    )

    RWKV7ModelConfig.verify_and_update_config(vllm_config)

    assert vllm_config.load_config.load_format == "rwkv_pth"


def test_rwkv7_config_preserves_dummy_loader_for_initialization() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
        load_format="dummy",
    )

    RWKV7ModelConfig.verify_and_update_config(vllm_config)

    assert vllm_config.load_config.load_format == "dummy"


def test_rwkv7_config_rejects_incompatible_raw_pth_loader() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
        load_format="safetensors",
    )

    with pytest.raises(ValueError, match="RWKV7 raw .pth checkpoints require"):
        RWKV7ModelConfig.verify_and_update_config(vllm_config)


def test_rwkv7_config_rejects_rwkv_loader_for_non_raw_source() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
        model="ordinary/model",
        load_format="rwkv_pth",
    )

    with pytest.raises(ValueError, match="requires a supported RWKV-7"):
        RWKV7ModelConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    ("wkv_mode", "expected_page_size"),
    [
        ("fp16", 8_464),
        ("fp32io16", 16_656),
    ],
)
def test_rwkv7_cache_spec_aligns_physical_pages(
    monkeypatch: pytest.MonkeyPatch,
    wkv_mode: str,
    expected_page_size: int,
) -> None:
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", wkv_mode)
    monkeypatch.setattr(
        RWKV7Block,
        "get_state_shape",
        lambda _self: ((2, 64), (1, 64, 64), (1,)),
    )
    block = RWKV7Block.__new__(RWKV7Block)
    block.model_config = SimpleNamespace(dtype=torch.float16)
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
    )
    vllm_config.cache_config.mamba_block_size = 4096

    spec = block.get_kv_cache_spec(vllm_config)

    assert spec is not None
    assert spec.page_size_bytes == expected_page_size
    assert spec.page_size_padded == expected_page_size


def test_rwkv7_config_rejects_non_none_mamba_cache_mode() -> None:
    vllm_config = _rwkv7_vllm_config(
        enable_prefix_caching=False,
        speculative_config=None,
    )
    vllm_config.cache_config.mamba_cache_mode = "align"

    with pytest.raises(ValueError, match="mamba_cache_mode"):
        RWKV7ModelConfig.verify_and_update_config(vllm_config)
