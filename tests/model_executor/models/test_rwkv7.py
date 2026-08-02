# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.model_executor.models.rwkv7 as rwkv7
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.models.config import RWKV7ForCausalLMConfig
from vllm.model_executor.models.interfaces import requires_uniform_decode_wave
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine.core import _supports_no_kv_cache_chunked_prefill
from vllm.v1.worker.gpu.model_states.rwkv import RWKV7ModelState
from vllm.v1.worker.gpu_input_batch import (
    CachedRequestState,
)
from vllm.v1.worker.gpu_input_batch import (
    InputBatch as LegacyInputBatch,
)


def test_rwkv7_declares_uniform_decode_wave_capability():
    assert requires_uniform_decode_wave(RWKV7ForCausalLM)


def test_rwkv7_cuda_ops_match_torch_reference():
    if (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("CUDA is required for RWKV7 custom op numerical test")

    import vllm.rwkv7_ops  # noqa: F401

    eps = 1e-5
    x = torch.tensor(
        [
            [1.0, -2.0, 0.5, 4.0],
            [-3.5, 0.25, 2.0, -0.75],
        ],
        dtype=torch.float16,
        device="cuda",
    ).contiguous()
    weight = torch.tensor(
        [0.5, -1.25, 2.0, 0.75], dtype=torch.float16, device="cuda"
    ).contiguous()
    bias = torch.tensor(
        [0.125, -0.5, 1.0, -1.5], dtype=torch.float16, device="cuda"
    ).contiguous()

    y = torch.ops.rwkv7_v3a_ops.layer_norm_f16(x, weight, bias, eps)
    z = torch.ops.rwkv7_fast_ops_fp16.relu_square(x)
    torch.accelerator.synchronize()

    expected_y = F.layer_norm(
        x.float(), (x.shape[-1],), weight.float(), bias.float(), eps
    ).to(torch.float16)
    expected_z = torch.relu(x).square()
    torch.testing.assert_close(y, expected_y, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(z, expected_z, rtol=0, atol=0)


def _new_request(req_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=(),
        num_computed_tokens=0,
        lora_request=None,
    )


def _new_cached_request_state(req_id: str) -> CachedRequestState:
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=8),
        generator=None,
        block_ids=([],),
        num_computed_tokens=0,
        output_token_ids=[],
    )


def _new_rwkv7_forward_test_model(**attrs: Any) -> RWKV7ForCausalLM:
    model = object.__new__(RWKV7ForCausalLM)
    nn.Module.__init__(model)
    model.hidden_size = attrs.pop("hidden_size", 3)
    model.head_size = attrs.pop("head_size", 1)
    model.num_attention_heads = attrs.pop(
        "num_attention_heads", model.hidden_size // model.head_size
    )
    model.vocab_size = attrs.pop("vocab_size", 128)
    model.total_num_layers = attrs.pop("total_num_layers", 1)
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: True
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def _new_rwkv7_for_weight_tests() -> RWKV7ForCausalLM:
    model = _new_rwkv7_forward_test_model()
    model.z = {"old": torch.tensor([1])}
    model._dummy_param = nn.Parameter(torch.empty(0), requires_grad=False)
    return model


def _new_default_loader_for_weight_tests(
    load_format: str = "auto",
) -> DefaultModelLoader:
    return DefaultModelLoader(
        SimpleNamespace(
            load_format=load_format,
            model_loader_extra_config={},
            download_dir=None,
            ignore_patterns=None,
            use_tqdm_on_load=False,
            safetensors_load_strategy="lazy",
            safetensors_prefetch_num_threads=1,
            safetensors_prefetch_block_size=1024,
            pt_load_map_location="cpu",
        )
    )


def test_rwkv7_forward_tokens_propagates_canonical_wkv_metadata(monkeypatch):
    seen = []

    def embed(tokens):
        return torch.zeros((*tokens.shape, 4), dtype=torch.float32)

    def forward_from_x(
        x,
        state,
        path,
        *,
        slot_indices=None,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        seen.append(
            (
                path.cmix_mode,
                path.rows,
                slot_indices.tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        return x.squeeze(1)

    model = _new_rwkv7_forward_test_model(
        hidden_size=4,
        embed=embed,
        forward_from_x=forward_from_x,
    )
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32)
    slot_indices = torch.tensor([3], dtype=torch.int32)

    out = RWKV7ForCausalLM.forward_tokens(
        model,
        torch.tensor([7], dtype=torch.long),
        [torch.empty(0)],
        slot_indices=slot_indices,
        query_start_loc=query_start_loc,
        wkv_slot_indices=slot_indices,
    )

    assert out.shape == (1, 4)
    assert seen == [(rwkv7.select_path(1, 1).cmix_mode, 1, [3], [0, 1], [3])]
    assert seen[0][0] != rwkv7.CMIX_DENSE


@pytest.mark.parametrize(
    ("batch", "tokens", "expected"),
    [(1, 2, True), (1, 8, False), (1, 16, True), (2, 2, True), (2, 1, False)],
)
def test_rwkv7_mix_3d_tuned_shapes(
    batch: int,
    tokens: int,
    expected: bool,
) -> None:
    assert rwkv7.use_tmix_mix6_3d(batch, tokens, 4096) is expected
    assert rwkv7.use_cmix_mix_3d(batch, tokens, 4096) is expected


def test_rwkv7_auxiliary_tuned_shape_guards() -> None:
    assert rwkv7.use_tmix_lnx_warp(1, 64, 4096, 64)
    assert not rwkv7.use_tmix_lnx_warp(1, 32, 4096, 64)
    assert rwkv7.use_tmix_lnx_warp(64, 1, 4096, 64)
    assert rwkv7.use_ln1_tmix_fusion(1, 1)
    for batch in (2, 4, 8, 16):
        assert not rwkv7.use_ln1_tmix_fusion(batch, 1)
    assert rwkv7.use_ln1_tmix_fusion(32, 1)
    assert rwkv7.use_ln1_tmix_fusion(320, 1)
    assert not rwkv7.use_ln1_tmix_fusion(1, 2)


@pytest.mark.parametrize("wkv_mode", ["fp16", "fp32io16"])
def test_rwkv7_run_wkv_uses_canonical_packed_op(monkeypatch, wkv_mode) -> None:
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def record(name):
        def op(*args):
            calls.append((name, args))

        return op

    monkeypatch.setattr(
        torch.ops.rwkv7_wkv_fp32_v2,
        "wkv",
        record("fp32"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_wkv_fp16_v2,
        "wkv",
        record("fp16"),
        raising=False,
    )

    query_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32)
    slot_indices = torch.tensor([3, 1], dtype=torch.int32)
    state_dtype = torch.float32 if wkv_mode == "fp32io16" else torch.float16
    state = torch.empty((4, 2, 64, 64), dtype=state_dtype)
    tensors = [torch.empty((1, 3, 128), dtype=torch.float16) for _ in range(6)]
    r, w, k, v, neg_kk, kka = tensors
    w0 = torch.empty((128,), dtype=torch.float16)
    y = torch.empty_like(r)
    elapsed = torch.empty((4,), dtype=torch.int32)
    w_raw = torch.empty_like(w.reshape(3, 128))

    def add_vec(channels, rows, bias):
        assert channels == 128
        assert rows.shape == (3, 128)
        assert bias is w0
        calls.append(("add_vec", (channels, rows, bias)))
        return w_raw

    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        "add_vec",
        add_vec,
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(wkv_mode=wkv_mode)

    RWKV7ForCausalLM._run_wkv(
        model,
        query_start_loc,
        slot_indices,
        state,
        r,
        w,
        w0,
        k,
        v,
        neg_kk,
        kka,
        y,
        elapsed,
    )

    expected_call_names = ["fp16"] if wkv_mode == "fp16" else ["add_vec", "fp32"]
    assert [name for name, _args in calls] == expected_call_names
    args = calls[-1][1]
    assert args[:3] == (query_start_loc, slot_indices, state)
    if wkv_mode == "fp16":
        assert args[5] is w0
        assert args[-1] is elapsed
        row_args = (args[3], args[4], args[6], args[7], args[8], args[9], args[10])
        row_inputs = (r, w, k, v, neg_kk, kka, y)
    else:
        assert args[4] is w_raw
        row_args = args[3:]
        row_inputs = (r, w_raw, k, v, neg_kk, kka, y)
    for actual, expected in zip(row_args, row_inputs):
        assert actual.shape == (3, 128)
        assert (
            actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
        )


def test_rwkv7_forward_layer_range_uses_slot_fused_tmix_for_batched_decode(
    monkeypatch,
):
    calls: list[Any] = []

    def fused_cmix(x, residual, shift_state, weight, bias, x_k, slot_indices):
        calls.append(("fused_cmix", slot_indices.tolist()))
        return x + residual, torch.ones_like(x)

    def fused_tmix(
        x,
        residual,
        shift_state,
        weight,
        bias,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
        slot_indices,
    ):
        calls.append(("fused_tmix", x.shape[0], slot_indices.tolist()))
        mixed = tuple(torch.full_like(x, fill_value=float(i)) for i in range(6))
        return (x + residual, *mixed)

    def advance_slots(elapsed, slot_indices, amount):
        calls.append(("advance", slot_indices.tolist(), amount))

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "add_layer_norm_cmix_mix_f16_slots",
        fused_cmix,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "add_layer_norm_tmix_mix6_f16_slots",
        fused_tmix,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "advance_i32_slots",
        advance_slots,
        raising=False,
    )

    model = _new_rwkv7_forward_test_model(hidden_size=4)
    model.start_layer = 0
    model.end_layer = 2
    model.z = {}
    for layer in range(2):
        model.z.update(
            {
                f"blocks.{layer}.ln1.weight": torch.empty(4),
                f"blocks.{layer}.ln1.bias": torch.empty(4),
                f"blocks.{layer}.ln2.weight": torch.empty(4),
                f"blocks.{layer}.ln2.bias": torch.empty(4),
                f"blocks.{layer}.ffn.x_k": torch.empty(4),
                f"blocks.{layer}.att.x_r": torch.empty(4),
                f"blocks.{layer}.att.x_w": torch.empty(4),
                f"blocks.{layer}.att.x_k": torch.empty(4),
                f"blocks.{layer}.att.x_v": torch.empty(4),
                f"blocks.{layer}.att.x_a": torch.empty(4),
                f"blocks.{layer}.att.x_g": torch.empty(4),
            }
        )
    model.ln = lambda x, weight, bias: x
    model.add = lambda x, y: x + y

    def add_ln(x, residual, weight, bias):
        calls.append(("split_add_ln", x.shape[0]))
        return x + residual, x + residual

    def tmix(
        layer,
        x,
        shift_state,
        wkv_state,
        elapsed_t,
        v_first,
        p,
        path,
        pre_mix=None,
        *,
        slot_indices=None,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        calls.append(
            (
                "tmix",
                layer,
                pre_mix is not None,
                slot_indices.tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        return x, v_first

    def cmix_from_mixed(mixed, p, path):
        return mixed

    model.add_ln = add_ln
    model.tmix = tmix
    model.cmix_from_mixed = cmix_from_mixed
    # B=2/4/8/16 deliberately use separate LN1/TMix kernels; B=3 keeps the
    # slot-aware fused owner and exercises that production branch.
    slot_indices = torch.tensor([3, 1, 4], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    state = [
        torch.zeros((2, 2, 6, 4)),
        torch.empty((2,)),
        torch.zeros((6,), dtype=torch.int32),
    ]

    RWKV7ForCausalLM.forward_layer_range(
        model,
        torch.zeros((3, 1, 4)),
        state,
        rwkv7.PathConfig(3, rwkv7.CMIX_ROWS2_NOFC),
        v_first=None,
        final=False,
        all_logits=False,
        slot_indices=slot_indices,
        query_start_loc=query_start_loc,
        wkv_slot_indices=slot_indices,
    )

    assert calls == [
        ("tmix", 0, False, [3, 1, 4], [0, 1, 2, 3], [3, 1, 4]),
        ("fused_cmix", [3, 1, 4]),
        ("fused_tmix", 3, [3, 1, 4]),
        ("tmix", 1, True, [3, 1, 4], [0, 1, 2, 3], [3, 1, 4]),
        ("fused_cmix", [3, 1, 4]),
        ("advance", [3, 1, 4], 1),
    ]


def _rwkv7_vllm_config(
    *,
    enforce_eager: bool,
    compilation_mode: CompilationMode = CompilationMode.NONE,
) -> SimpleNamespace:
    return SimpleNamespace(
        compilation_config=CompilationConfig(mode=compilation_mode),
        model_config=SimpleNamespace(
            enforce_eager=enforce_eager,
            hf_config=SimpleNamespace(
                hidden_size=64,
                vocab_size=128,
                head_size=64,
                num_hidden_layers=1,
            ),
        ),
    )


def _new_rwkv7_model_state(
    *,
    max_num_reqs: int = 4,
    num_hidden_layers: int = 1,
    hidden_size: int = 64,
    head_size: int = 64,
    num_attention_heads: int = 1,
) -> RWKV7ModelState:
    hf_config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        head_size=head_size,
        num_attention_heads=num_attention_heads,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
    )
    return RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )


def _rwkv7_input_batch(
    state: RWKV7ModelState,
    *,
    idx_mapping_np: np.ndarray,
    query_start_loc: torch.Tensor,
    is_prefilling_np: np.ndarray,
    **kwargs: Any,
) -> SimpleNamespace:
    req_id_by_slot = {
        req_slot: req_id for req_id, req_slot in state.req_id_to_index.items()
    }
    req_slots = np.asarray(idx_mapping_np, dtype=np.int32)
    query_start_loc = query_start_loc.to(dtype=torch.int32, device="cpu")
    return SimpleNamespace(
        req_ids=[req_id_by_slot[int(req_slot)] for req_slot in req_slots],
        idx_mapping_np=req_slots,
        num_reqs=len(req_slots),
        query_start_loc=query_start_loc,
        query_start_loc_np=query_start_loc.numpy().copy(),
        is_prefilling_np=np.asarray(is_prefilling_np, dtype=np.bool_),
        **kwargs,
    )


def _assert_same_storage_view(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.stride() == expected.stride()
    assert actual.storage_offset() == expected.storage_offset()
    assert actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()


def _assert_decode_wkv_metadata(
    inputs: dict[str, Any],
    expected_slots: list[int],
) -> None:
    decode_batch_size = len(expected_slots)
    assert inputs["rwkv_decode_query_start_loc"].tolist() == list(
        range(decode_batch_size + 1)
    )
    slots = inputs["slot_indices"]
    if slots is None:
        slots = inputs["idx_mapping"][:decode_batch_size]
    assert slots.tolist() == expected_slots


def _assert_prefill_wkv_metadata(
    inputs: dict[str, Any],
    *,
    query_start_loc: list[int],
    slot_indices: list[int],
    token_positions: list[int],
    req_id: list[int],
) -> None:
    assert inputs["rwkv_prefill_query_start_loc"].tolist() == query_start_loc
    assert inputs["rwkv_prefill_slot_indices"].tolist() == slot_indices
    assert inputs["rwkv_prefill_token_positions"].tolist() == token_positions
    assert inputs["rwkv_prefill_req_id"].tolist() == req_id


def test_rwkv7_rejects_torch_compile():
    with pytest.raises(ValueError, match="RWKV7 does not support torch.compile"):
        RWKV7ForCausalLM(
            vllm_config=_rwkv7_vllm_config(
                enforce_eager=False,
                compilation_mode=CompilationMode.VLLM_COMPILE,
            )
        )


def test_rwkv7_init_preserves_process_wide_torch_state(
    monkeypatch, default_vllm_config
):
    monkeypatch.setattr(rwkv7, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(rwkv7, "get_tensor_model_parallel_rank", lambda: 0)

    old_grad_enabled = torch.is_grad_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    old_cuda_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_grad_enabled(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        RWKV7ForCausalLM(vllm_config=_rwkv7_vllm_config(enforce_eager=False))

        assert torch.is_grad_enabled() is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_grad_enabled(old_grad_enabled)
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = old_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_cuda_matmul_allow_tf32
        torch.set_float32_matmul_precision(old_matmul_precision)


@pytest.mark.parametrize("load_format", ["auto", "hf"])
def test_rwkv7_raw_pth_file_uses_single_weight_file(tmp_path, load_format):
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()
    loader = _new_default_loader_for_weight_tests(load_format=load_format)

    _, weight_files, use_safetensors = loader._prepare_weights(
        str(checkpoint),
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert weight_files == [str(checkpoint)]
    assert use_safetensors is False


def test_rwkv7_raw_pth_url_downloads_single_weight_file(tmp_path, monkeypatch):
    downloaded = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    downloaded.touch()
    calls: list[Any] = []

    def fake_hf_hub_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(
        "vllm.transformers_utils.configs.rwkv7.hf_hub_download",
        fake_hf_hub_download,
    )
    loader = _new_default_loader_for_weight_tests()

    _, weight_files, use_safetensors = loader._prepare_weights(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
        subfolder=None,
        revision=None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )

    assert calls == [
        {
            "repo_id": "BlinkDL/rwkv7-g1",
            "filename": "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
            "revision": "main",
            "cache_dir": None,
        }
    ]
    assert weight_files == [str(downloaded)]
    assert use_safetensors is False


def test_rwkv7_raw_pth_preprocess_validates_config_shape():
    model = _new_rwkv7_for_weight_tests()
    model.tp_size = 1
    model.tp_rank = 0
    model.org_vocab_size = 65536
    model.config = SimpleNamespace(
        hidden_size=2048,
        vocab_size=65536,
        head_size=64,
        num_hidden_layers=24,
    )
    weights = {
        "emb.weight": torch.empty(65536, 2048),
        "blocks.0.att.r_k": torch.empty(24, 64),
    }

    with pytest.raises(ValueError, match="RWKV7 config hidden_size"):
        RWKV7ForCausalLM._validate_raw_weight_shapes(model, weights)


@pytest.mark.parametrize(
    "compilation_mode",
    [
        CompilationMode.STOCK_TORCH_COMPILE,
        CompilationMode.DYNAMO_TRACE_ONCE,
        CompilationMode.VLLM_COMPILE,
    ],
)
def test_rwkv7_config_rejects_torch_compile(compilation_mode):
    vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(mode=compilation_mode)
    )

    with pytest.raises(ValueError, match="RWKV7 does not support torch.compile"):
        RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)


def test_rwkv7_config_defaults_to_no_compile():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.mode == CompilationMode.NONE
    assert (
        vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )


def test_rwkv7_config_disables_prefix_caching():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=None,
        ),
        compilation_config=CompilationConfig(),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.cache_config.enable_prefix_caching is False


def test_rwkv7_config_captures_full_decode_capacity():
    configured_capacity = 768
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        scheduler_config=SimpleNamespace(
            max_num_seqs=configured_capacity,
            max_num_batched_tokens=8192,
            max_num_scheduled_tokens=None,
        ),
        compilation_config=CompilationConfig(),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert (
        vllm_config.compilation_config.max_cudagraph_capture_size == configured_capacity
    )


@pytest.mark.parametrize(
    "compilation_config",
    [
        CompilationConfig(max_cudagraph_capture_size=768),
        CompilationConfig(cudagraph_capture_sizes=[64, 256, 512]),
    ],
)
def test_rwkv7_config_preserves_explicit_cudagraph_sizes(compilation_config):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        scheduler_config=SimpleNamespace(
            max_num_seqs=768,
            max_num_batched_tokens=8192,
            max_num_scheduled_tokens=None,
        ),
        compilation_config=compilation_config,
    )
    expected_max_size = compilation_config.max_cudagraph_capture_size
    expected_capture_sizes = compilation_config.cudagraph_capture_sizes

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert (
        vllm_config.compilation_config.max_cudagraph_capture_size == expected_max_size
    )
    assert (
        vllm_config.compilation_config.cudagraph_capture_sizes == expected_capture_sizes
    )


def test_rwkv7_allows_chunked_prefill_without_kv_cache():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate",
            architectures=["RWKV7ForCausalLM"],
            hf_config=SimpleNamespace(architectures=[]),
        )
    )

    assert _supports_no_kv_cache_chunked_prefill(vllm_config)


def test_rwkv7_config_rejects_decode_budget_below_max_running_reqs():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=8,
            max_num_scheduled_tokens=2,
        ),
        compilation_config=CompilationConfig(),
    )

    with pytest.raises(ValueError, match="max_num_seqs"):
        RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "cudagraph_mode",
    [None, CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE],
)
def test_rwkv7_config_uses_decode_cudagraph(cudagraph_mode):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=cudagraph_mode,
        ),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert (
        vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    )


def test_rwkv7_config_preserves_disabled_cudagraph():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )

    RWKV7ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert vllm_config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE


@pytest.mark.parametrize(
    ("mode", "state_dtype", "allow_fp16_accumulation", "accumulation_policy"),
    [
        ("fp32io16", torch.float32, False, "fp32"),
        ("fp16", torch.float16, True, "fp16"),
    ],
)
def test_rwkv7_execution_profile_is_coherent(
    mode,
    state_dtype,
    allow_fp16_accumulation,
    accumulation_policy,
):
    profile = rwkv7.resolve_execution_profile(mode)

    assert profile.wkv_mode == mode
    assert profile.wkv_state_dtype == state_dtype
    assert profile.allow_fp16_accumulation is allow_fp16_accumulation
    assert profile.gemm_accumulation_policy == accumulation_policy


def test_rwkv7_execution_profile_rejects_unknown_mode():
    with pytest.raises(ValueError, match="VLLM_RWKV7_WKV_MODE"):
        rwkv7.resolve_execution_profile("legacy")


@pytest.mark.parametrize("mode", ["fp32io16", "fp16"])
def test_rwkv7_passes_profile_accumulation_to_custom_op(monkeypatch, mode):
    calls: list[Any] = []

    def linear_f16(x, weight, allow):
        calls.append(("linear_f16", allow))
        return torch.empty((x.size(0), weight.size(1)))

    monkeypatch.setattr(torch.ops.rwkv7_v3a_ops, "linear_f16", linear_f16)
    profile = rwkv7.resolve_execution_profile(mode)
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=profile.allow_fp16_accumulation
    )
    x = torch.empty((2, 4))

    model.linear(x, torch.empty((4, 3)))

    assert calls == [("linear_f16", profile.allow_fp16_accumulation)]


@pytest.mark.parametrize(
    ("helper_name", "weight_shape", "rows", "expected_cfg"),
    [
        ("linear_att_c2c", (4096, 4096), 64, (32, 2)),
        ("linear_ffn_down", (16384, 4096), 64, (32, 3)),
    ],
)
def test_rwkv7_fp16_tuned_linear_helpers_use_exact_lt_configs(
    monkeypatch,
    helper_name,
    weight_shape,
    rows,
    expected_cfg,
):
    calls = []
    expected = torch.empty(
        (rows, weight_shape[1]),
        device="meta",
        dtype=torch.float16,
    )

    def linear_f16_lt_cfg(
        x,
        weight,
        workspace_mb,
        heuristic_index,
        strict_algo,
    ):
        calls.append(
            (
                x,
                weight,
                workspace_mb,
                heuristic_index,
                strict_algo,
            )
        )
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_lt_cfg",
        linear_f16_lt_cfg,
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=True,
        hidden_size=4096,
    )
    model.linear = lambda *_args: pytest.fail("exact tuned shape must use Lt")
    x = torch.empty((rows, weight_shape[0]), device="meta", dtype=torch.float16)
    weight = torch.empty(weight_shape, device="meta", dtype=torch.float16)

    output = getattr(model, helper_name)(x, weight, rows)

    assert output is expected
    assert calls == [(x, weight, *expected_cfg, False)]


def test_rwkv7_project_att_rkv_groups_exact_m1_splitk(monkeypatch):
    model = _new_rwkv7_forward_test_model(hidden_size=4096)
    prefix = "blocks.0.att."
    weights = tuple(
        torch.empty((4096, 4096), device="meta", dtype=torch.float16) for _ in range(3)
    )
    model.z = {
        prefix + "receptance.weight": weights[0],
        prefix + "key.weight": weights[1],
        prefix + "value.weight": weights[2],
    }
    inputs = tuple(
        torch.empty((1, 1, 4096), device="meta", dtype=torch.float16) for _ in range(3)
    )
    expected = tuple(torch.empty_like(inputs[0]) for _ in range(3))
    calls = []

    def grouped(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_rkv_f16_m1_splitk",
        grouped,
        raising=False,
    )
    model.linear_att_c2c = lambda *_args: pytest.fail(
        "exact M1 must use grouped split-K"
    )
    model.linear = lambda *_args: pytest.fail("exact M1 must use grouped split-K")

    output = model.project_att_rkv(*inputs, prefix, 1)

    assert output is expected
    assert calls == [(*inputs, *weights)]


@pytest.mark.parametrize(
    ("rows", "weight_n"),
    [
        (2, 4096),
        (1, 4000),
    ],
)
def test_rwkv7_project_att_rkv_falls_back_outside_exact_contract(
    monkeypatch,
    rows,
    weight_n,
):
    model = _new_rwkv7_forward_test_model(hidden_size=4096)
    prefix = "blocks.0.att."
    weights = tuple(
        torch.empty((4096, weight_n), device="meta", dtype=torch.float16)
        for _ in range(3)
    )
    model.z = {
        prefix + "receptance.weight": weights[0],
        prefix + "key.weight": weights[1],
        prefix + "value.weight": weights[2],
    }
    inputs = tuple(
        torch.empty((rows, 4096), device="meta", dtype=torch.float16) for _ in range(3)
    )
    expected = tuple(
        torch.empty((rows, weight_n), device="meta", dtype=torch.float16)
        for _ in range(3)
    )
    calls: list[tuple[torch.Tensor, torch.Tensor, int | None]] = []

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_rkv_f16_m1_splitk",
        lambda *_args: pytest.fail("non-contract shape must not use grouped op"),
        raising=False,
    )

    def project(value, weight, runtime_rows=None):
        index = len(calls)
        calls.append((value, weight, runtime_rows))
        return expected[index]

    if rows == 1:
        model.linear = project
        model.linear_att_c2c = lambda *_args: pytest.fail(
            "M1 fallback must stay pinned to independent split-K"
        )
    else:
        model.linear = lambda *_args: pytest.fail(
            "multi-row fallback must retain attention Lt dispatch"
        )
        model.linear_att_c2c = project

    output = model.project_att_rkv(*inputs, prefix, rows)

    assert all(actual is reference for actual, reference in zip(output, expected))
    assert calls == [
        (
            inputs[index],
            weights[index],
            None if rows == 1 else rows,
        )
        for index in range(3)
    ]


def test_rwkv7_project_att_rkv_can_disable_grouped_route_for_benchmark(
    monkeypatch,
):
    model = _new_rwkv7_forward_test_model(hidden_size=4096)
    prefix = "blocks.0.att."
    weights = tuple(
        torch.empty((4096, 4096), device="meta", dtype=torch.float16) for _ in range(3)
    )
    model.z = {
        prefix + "receptance.weight": weights[0],
        prefix + "key.weight": weights[1],
        prefix + "value.weight": weights[2],
    }
    inputs = tuple(
        torch.empty((1, 4096), device="meta", dtype=torch.float16) for _ in range(3)
    )
    expected = tuple(torch.empty_like(inputs[0]) for _ in range(3))
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    monkeypatch.setattr(rwkv7, "M1_RKV_GROUPED_ROWS", set())
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_rkv_f16_m1_splitk",
        lambda *_args: pytest.fail("disabled grouped route must not run"),
        raising=False,
    )

    def linear(value, weight):
        index = len(calls)
        calls.append((value, weight))
        return expected[index]

    model.linear = linear
    model.linear_att_c2c = lambda *_args: pytest.fail(
        "benchmark baseline must stay pinned to independent split-K"
    )

    output = model.project_att_rkv(*inputs, prefix, 1)

    assert all(actual is reference for actual, reference in zip(output, expected))
    assert calls == [(inputs[index], weights[index]) for index in range(3)]


def test_rwkv7_cmix_m1_prepares_sparse_accumulator_during_splitk(
    monkeypatch,
):
    model = _new_rwkv7_forward_test_model(hidden_size=8)
    prefix = "blocks.0.ffn."
    key_weight = torch.empty((8, 64), device="meta", dtype=torch.float16)
    value_weight = torch.empty((64, 8), device="meta", dtype=torch.float16)
    mixed = torch.empty((1, 1, 8), device="meta", dtype=torch.float16)
    hid = torch.empty((1, 1, 64), device="meta", dtype=torch.float16)
    sparse_out = torch.empty((1, 1, 8), device="meta", dtype=torch.float16)
    seen: dict[str, tuple[Any, ...]] = {}
    model.z = {
        prefix + "key.weight": key_weight,
        prefix + "value.weight": value_weight,
    }
    model._tp_all_reduce = lambda value: value
    model.linear = lambda *_args: pytest.fail(
        "exact M1 no-fc must use split-K prepare-zero"
    )

    def prepare_zero(value, weight, zero_features):
        seen["prepare_zero"] = (value, weight, zero_features)
        return hid, sparse_out

    def sparse_down_out(C, F, preact, weight, out):
        seen["sparse_down_out"] = (C, F, preact, weight, out)

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk_prepare_zero",
        prepare_zero,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        "cmix_sparse_down_relu_one_out",
        sparse_down_out,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        "cmix_sparse_down_relu_one",
        lambda *_args: pytest.fail("prezero route must skip the zeroing op"),
        raising=False,
    )

    output = model.cmix_from_mixed(
        mixed,
        prefix,
        rwkv7.PathConfig(1, rwkv7.CMIX_B1T1_NOFC),
    )

    assert output is sparse_out
    prepared_x, prepared_weight, zero_features = seen["prepare_zero"]
    assert prepared_x is mixed
    assert prepared_weight is key_weight
    assert zero_features == 8
    C, F, preact, sparse_weight, out = seen["sparse_down_out"]
    assert (C, F) == (8, 64)
    assert tuple(preact.shape) == (64,)
    assert sparse_weight is value_weight
    assert out is sparse_out


@pytest.mark.parametrize("disable_tuning", [False, True])
def test_rwkv7_cmix_m1_falls_back_to_independent_zero_path(
    monkeypatch,
    disable_tuning,
):
    model = _new_rwkv7_forward_test_model(hidden_size=8)
    prefix = "blocks.0.ffn."
    out_features = 64 if disable_tuning else 65
    key_weight = torch.empty((8, out_features), device="meta", dtype=torch.float16)
    value_weight = torch.empty((out_features, 8), device="meta", dtype=torch.float16)
    mixed = torch.empty((1, 1, 8), device="meta", dtype=torch.float16)
    hid = torch.empty((1, 1, out_features), device="meta", dtype=torch.float16)
    sparse_out = torch.empty((1, 1, 8), device="meta", dtype=torch.float16)
    seen: dict[str, tuple[Any, ...]] = {}
    model.z = {
        prefix + "key.weight": key_weight,
        prefix + "value.weight": value_weight,
    }
    model._tp_all_reduce = lambda value: value
    if disable_tuning:
        monkeypatch.setattr(rwkv7, "M1_CMIX_PREZERO_ROWS", set())

    def linear(value, weight):
        seen["linear"] = (value, weight)
        return hid

    def splitk(value, weight):
        seen["splitk"] = (value, weight)
        return hid

    def sparse_down(C, F, preact, weight):
        seen["sparse_down"] = (C, F, preact, weight)
        return sparse_out

    model.linear = linear
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk_prepare_zero",
        lambda *_args: pytest.fail("fallback must not prepare sparse output"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk",
        splitk,
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        "cmix_sparse_down_relu_one_out",
        lambda *_args: pytest.fail("fallback must use independent zero path"),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        "cmix_sparse_down_relu_one",
        sparse_down,
        raising=False,
    )

    output = model.cmix_from_mixed(
        mixed,
        prefix,
        rwkv7.PathConfig(1, rwkv7.CMIX_B1T1_NOFC),
    )

    assert output is sparse_out
    projection = "splitk" if disable_tuning else "linear"
    projected_x, projected_weight = seen[projection]
    assert projected_x is mixed
    assert projected_weight is key_weight
    assert ("linear" in seen) == (not disable_tuning)
    assert ("splitk" in seen) == disable_tuning
    C, F, preact, sparse_weight = seen["sparse_down"]
    assert (8, out_features) == (C, F)
    assert tuple(preact.shape) == (out_features,)
    assert sparse_weight is value_weight


@pytest.mark.parametrize(
    ("helper_name", "rows", "rank", "expected_cfg"),
    [
        ("linear_rank_in", 8, 128, (128, 0)),
        ("linear_rank_in", 8, 480, (128, 0)),
        ("linear_rank_in", 8, 96, (128, 0)),
        ("linear_rank_in", 16, 480, (32, 1)),
        ("linear_rank_in", 16, 96, (32, 0)),
        ("linear_rank_out", 8, 128, (0, 5)),
        ("linear_rank_out", 8, 480, (0, 3)),
        ("linear_rank_out", 8, 96, (0, 4)),
        ("linear_rank_out", 16, 128, (0, 1)),
        ("linear_rank_out", 16, 480, (32, 2)),
    ],
)
def test_rwkv7_fp16_lowrank_helpers_use_exact_runtime_lt_configs(
    monkeypatch,
    helper_name,
    rows,
    rank,
    expected_cfg,
):
    calls = []
    is_rank_in = helper_name == "linear_rank_in"
    weight_shape = (4096, rank) if is_rank_in else (rank, 4096)
    weight_t_shape = (rank, 4096) if is_rank_in else (4096, rank)
    expected = torch.empty(
        (rows, weight_shape[1]),
        device="meta",
        dtype=torch.float16,
    )

    def linear_f16_lt_cfg(
        x,
        weight,
        workspace_mb,
        heuristic_index,
        strict_algo,
    ):
        calls.append(
            (
                x,
                weight,
                workspace_mb,
                heuristic_index,
                strict_algo,
            )
        )
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_lt_cfg",
        linear_f16_lt_cfg,
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=True,
        hidden_size=4096,
    )
    model.linear = lambda *_args: pytest.fail("exact low-rank shape must use Lt")
    x = torch.empty((rows, weight_shape[0]), device="meta", dtype=torch.float16)
    weight = torch.empty(weight_shape, device="meta", dtype=torch.float16)
    weight_t = torch.empty(weight_t_shape, device="meta", dtype=torch.float16)

    output = getattr(model, helper_name)(x, weight, weight_t, rows)

    assert output is expected
    assert calls == [(x, weight, *expected_cfg, False)]


@pytest.mark.parametrize(
    ("act", "op_name"),
    [
        (1, "act_tanh"),
        (2, "act_sigmoid"),
    ],
)
def test_rwkv7_lowrank_out_activation_delegates_to_tuned_helper(
    monkeypatch,
    act,
    op_name,
):
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=True,
        hidden_size=4096,
    )
    x = torch.empty((8, 480), device="meta", dtype=torch.float16)
    activated = torch.empty_like(x)
    weight = torch.empty((480, 4096), device="meta", dtype=torch.float16)
    weight_t = torch.empty((4096, 480), device="meta", dtype=torch.float16)
    expected = torch.empty((8, 4096), device="meta", dtype=torch.float16)
    activation_calls = []
    projection_calls = []

    def activate(value):
        activation_calls.append(value)
        return activated

    def linear_rank_out(value, runtime_weight, transposed_weight, rows):
        projection_calls.append((value, runtime_weight, transposed_weight, rows))
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_fast_ops_fp16,
        op_name,
        activate,
        raising=False,
    )
    model.linear_rank_out = linear_rank_out

    output = model.linear_rank_out_act(x, weight, weight_t, 8, act)

    assert output is expected
    assert activation_calls == [x]
    assert projection_calls == [(activated, weight, weight_t, 8)]


@pytest.mark.parametrize(
    ("helper_name", "weight_shape", "weight_t_shape"),
    [
        ("linear_rank_in", (4096, 128), (128, 4096)),
        ("linear_rank_out", (128, 4096), (4096, 128)),
    ],
)
def test_rwkv7_lowrank_lt_preserves_fp32_accumulation_path(
    monkeypatch,
    helper_name,
    weight_shape,
    weight_t_shape,
):
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_lt_cfg",
        lambda *_args: pytest.fail("fp32 accumulation must not use tuned FP16 Lt"),
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=False,
        hidden_size=4096,
    )
    expected = torch.empty(
        (8, weight_shape[1]),
        device="meta",
        dtype=torch.float16,
    )
    model.linear = lambda *_args: expected
    x = torch.empty((8, weight_shape[0]), device="meta", dtype=torch.float16)
    weight = torch.empty(weight_shape, device="meta", dtype=torch.float16)
    weight_t = torch.empty(weight_t_shape, device="meta", dtype=torch.float16)

    assert getattr(model, helper_name)(x, weight, weight_t, 8) is expected


@pytest.mark.parametrize(
    ("helper_name", "weight_shape"),
    [
        ("linear_att_c2c", (4096, 4096)),
        ("linear_ffn_down", (16384, 4096)),
    ],
)
def test_rwkv7_tuned_linear_helpers_preserve_fp32_accumulation_path(
    monkeypatch,
    helper_name,
    weight_shape,
):
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_lt_cfg",
        lambda *_args: pytest.fail("fp32 accumulation must not use tuned FP16 Lt"),
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=False,
        hidden_size=4096,
    )
    expected = torch.empty(
        (64, weight_shape[1]),
        device="meta",
        dtype=torch.float16,
    )
    fallback_calls = []

    def fallback(x, weight):
        fallback_calls.append((x, weight))
        return expected

    model.linear = fallback
    x = torch.empty((64, weight_shape[0]), device="meta", dtype=torch.float16)
    weight = torch.empty(weight_shape, device="meta", dtype=torch.float16)

    output = getattr(model, helper_name)(x, weight, 64)

    assert output is expected
    assert fallback_calls == [(x, weight)]


@pytest.mark.parametrize(
    ("helper_name", "weight_shape"),
    [
        ("linear_att_c2c", (4096, 4096)),
        ("linear_ffn_down", (16384, 4096)),
    ],
)
def test_rwkv7_tuned_linear_helpers_require_exact_rows(
    monkeypatch,
    helper_name,
    weight_shape,
):
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_lt_cfg",
        lambda *_args: pytest.fail("untuned row count must not use Lt"),
        raising=False,
    )
    model = _new_rwkv7_forward_test_model(
        allow_fp16_accumulation=True,
        hidden_size=4096,
    )
    expected = torch.empty(
        (63, weight_shape[1]),
        device="meta",
        dtype=torch.float16,
    )
    model.linear = lambda *_args: expected
    x = torch.empty((63, weight_shape[0]), device="meta", dtype=torch.float16)
    weight = torch.empty(weight_shape, device="meta", dtype=torch.float16)

    assert getattr(model, helper_name)(x, weight, 63) is expected


def test_rwkv7_dummy_inputs_decode_capture_advertises_contiguous_rows():
    state = object.__new__(RWKV7ModelState)
    state.num_layers = 2
    state.max_num_reqs = 4
    state.hidden_size = 8
    state.num_heads = 2
    state.head_size = 4
    state.device = torch.device("cpu")
    state.shift_state = torch.zeros((2, 2, 4, 8), dtype=torch.float16)
    state.wkv_state = torch.zeros((2, 4, 2, 4, 4), dtype=torch.float32)
    state.elapsed = torch.zeros((4,), dtype=torch.int32)
    state.execution_idx_mapping = torch.arange(4, dtype=torch.int32)
    state.decode_slot_indices = torch.empty((4,), dtype=torch.int32)
    state.decode_token_positions = torch.empty((4,), dtype=torch.long)
    state.decode_query_start_loc = torch.arange(5, dtype=torch.int32)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=3)

    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert inputs["rwkv_decode_batch_size"] == 3
    assert inputs["rwkv_decode_rows"] == [0, 1, 2]
    assert inputs["rwkv_decode_token_positions"] == [0, 1, 2]
    _assert_decode_wkv_metadata(inputs, [0, 1, 2])


def test_rwkv7_dummy_inputs_decode_capture_uses_persistent_state_buffers():
    state = object.__new__(RWKV7ModelState)
    state.num_layers = 2
    state.max_num_reqs = 4
    state.hidden_size = 8
    state.num_heads = 2
    state.head_size = 4
    state.device = torch.device("cpu")
    state.shift_state = torch.ones((2, 2, 4, 8), dtype=torch.float16)
    state.wkv_state = torch.ones((2, 4, 2, 4, 4), dtype=torch.float32)
    state.elapsed = torch.ones((4,), dtype=torch.int32)
    state.execution_idx_mapping = torch.arange(4, dtype=torch.int32)
    state.decode_slot_indices = torch.empty((4,), dtype=torch.int32)
    state.decode_token_positions = torch.empty((4,), dtype=torch.long)
    state.decode_query_start_loc = torch.arange(5, dtype=torch.int32)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=3)

    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["query_start_loc"].tolist() == [0, 1, 2, 3]
    assert inputs["shift_state"] is state.shift_state
    assert inputs["wkv_state"] is state.wkv_state
    assert inputs["elapsed"] is state.elapsed
    assert inputs["rwkv_decode_batch_size"] == 3
    assert inputs["rwkv_decode_rows"] == [0, 1, 2]
    assert inputs["rwkv_decode_token_positions"] == [0, 1, 2]
    assert inputs["slot_indices"] is None
    _assert_decode_wkv_metadata(inputs, [0, 1, 2])


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({}, True),
        ({"rwkv_decode_batch_size": 2}, False),
        ({"rwkv_decode_rows": [1, 2, 3]}, False),
        ({"slot_indices": torch.tensor([1, 2, 3], dtype=torch.int32)}, False),
        ({"rwkv_decode_rows": None}, False),
    ],
)
def test_rwkv7_full_cudagraph_replay_requires_exact_contiguous_decode(
    updates,
    expected,
):
    model_inputs = {
        "positions": torch.arange(3, dtype=torch.int64),
        "rwkv_decode_batch_size": 3,
        "rwkv_decode_rows": [0, 1, 2],
        "slot_indices": None,
    }
    model_inputs.update(updates)

    assert RWKV7ModelState.can_replay_full_cudagraph(model_inputs) is expected


def test_rwkv7_load_weights_preprocesses_full_raw_weights(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    raw_weights = {
        "emb.weight": torch.tensor([1.0]),
        "head.weight": torch.tensor([2.0]),
    }
    calls: list[set[str]] = []

    def fake_preprocess(self, z):
        calls.append(set(z))
        z["processed"] = torch.tensor([3.0])
        self.z = z

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    loaded = model.load_weights(raw_weights.items())

    assert loaded == {"emb.weight", "head.weight"}
    assert model.raw_weight_names == {"emb.weight", "head.weight"}
    assert calls == [{"emb.weight", "head.weight"}]
    assert set(model.z) == {"emb.weight", "head.weight", "processed"}
    torch.testing.assert_close(model.z["emb.weight"], raw_weights["emb.weight"])
    torch.testing.assert_close(model.z["head.weight"], raw_weights["head.weight"])
    torch.testing.assert_close(model.z["processed"], torch.tensor([3.0]))


def _standard_hf_weights_for_test(
    model: RWKV7ForCausalLM,
) -> dict[str, torch.Tensor]:
    weights = {}
    for raw_name, shape in model._expected_standard_raw_shapes().items():
        if raw_name == "emb.weight":
            name = "model.embeddings.weight"
        elif raw_name == "head.weight":
            name = raw_name
        else:
            name = f"model.{raw_name}"
        weights[name] = torch.zeros(shape)
    return weights


def test_rwkv7_load_weights_maps_complete_standard_hf_artifact(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.config = SimpleNamespace(
        vocab_size=17,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        head_size=64,
        embedding_layer_norm_fused=False,
    )
    standard = _standard_hf_weights_for_test(model)
    assert len(standard) == 69
    assert {
        "model.embeddings.weight",
        "model.blocks.0.ln0.weight",
        "model.blocks.0.att.r_k",
        "model.blocks.1.att.v2",
        "model.ln_out.weight",
        "head.weight",
    }.issubset(standard)
    captured = {}

    def capture_preprocess(self, weights):
        captured.update(weights)

    monkeypatch.setattr(
        RWKV7ForCausalLM,
        "_preprocess_weights",
        capture_preprocess,
    )
    monkeypatch.setattr(
        RWKV7ForCausalLM,
        "_commit_preprocessed_weights",
        lambda self, weights: setattr(self, "z", weights),
    )

    loaded = model.load_weights(standard.items())

    assert loaded == set(standard)
    assert model.weight_source_format == "standard_hf"
    assert model.raw_weight_names == set(model._expected_standard_raw_shapes())
    assert set(captured) == model.raw_weight_names
    torch.testing.assert_close(
        captured["emb.weight"], standard["model.embeddings.weight"]
    )
    torch.testing.assert_close(
        captured["blocks.1.att.v2"], standard["model.blocks.1.att.v2"]
    )
    torch.testing.assert_close(captured["head.weight"], standard["head.weight"])


@pytest.mark.parametrize("failure", ["missing", "unknown", "shape", "mixed"])
def test_rwkv7_standard_hf_weights_fail_closed(failure):
    model = _new_rwkv7_for_weight_tests()
    model.config = SimpleNamespace(
        vocab_size=17,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        head_size=64,
        embedding_layer_norm_fused=False,
    )
    weights = _standard_hf_weights_for_test(model)
    if failure == "missing":
        del weights["head.weight"]
        match = "key mismatch.*head.weight"
    elif failure == "unknown":
        weights["model.unexpected.weight"] = torch.zeros(1)
        match = "unsupported key"
    elif failure == "shape":
        weights["model.blocks.0.att.r_k"] = torch.zeros(2, 64)
        match = "shape mismatch.*blocks.0.att.r_k"
    else:
        weights["emb.weight"] = weights["model.embeddings.weight"]
        match = "unsupported key.*emb.weight"

    with pytest.raises(ValueError, match=match):
        model.load_weights(weights.items())


def test_rwkv7_standard_hf_weights_reject_duplicate_source_key():
    model = _new_rwkv7_for_weight_tests()
    duplicate = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="duplicate key.*model.embeddings.weight"):
        model.load_weights(
            [
                ("model.embeddings.weight", duplicate),
                ("model.embeddings.weight", duplicate),
            ]
        )


def test_rwkv7_default_loader_validation_uses_raw_weight_names():
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}

    DefaultModelLoader.track_weights_loading(
        object.__new__(DefaultModelLoader),
        model,
        {"emb.weight", "head.weight"},
    )


def test_rwkv7_online_weight_update_preprocesses_only_on_finish(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    calls: list[Any] = []

    def fake_preprocess(self, z):
        calls.append(dict(z))
        z["processed"] = torch.tensor([5.0])
        self.z = z

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    assert model.start_weight_update() is True
    model.load_weights([("emb.weight", torch.tensor([10.0]))])
    model.load_weights([("head.weight", torch.tensor([20.0]))])

    assert model.z is old_z
    assert calls == []

    model.finish_weight_update()

    assert calls[0]["emb.weight"].item() == 10.0
    assert calls[0]["head.weight"].item() == 20.0
    assert model.z["processed"].item() == 5.0
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_abort_weight_update_allows_retry():
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}

    assert model.start_weight_update() is True
    model.load_weights([("emb.weight", torch.tensor([10.0]))])
    model.abort_weight_update()

    assert not hasattr(model, "_pending_weight_update")
    assert model.start_weight_update() is True


def test_rwkv7_online_weight_update_reuses_existing_tensor_storage(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.z = {
        "emb.weight": torch.tensor([1.0]),
        "head.weight": torch.tensor([2.0]),
        "stale.weight": torch.tensor([99.0]),
    }
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    old_emb = model.z["emb.weight"]
    old_head = model.z["head.weight"]

    def fake_preprocess(self, z):
        z["emb.weight"] = torch.tensor([10.0])
        z["head.weight"] = torch.tensor([20.0])
        z["derived.weight"] = torch.tensor([30.0])

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([10.0]))])
    model.load_weights([("head.weight", torch.tensor([20.0]))])
    model.finish_weight_update()

    assert model.z is old_z
    assert model.z["emb.weight"] is old_emb
    assert model.z["head.weight"] is old_head
    assert model.z["emb.weight"].item() == 10.0
    assert model.z["head.weight"].item() == 20.0
    assert model.z["derived.weight"].item() == 30.0
    assert set(model.z) == {"emb.weight", "head.weight", "derived.weight"}


def test_rwkv7_online_weight_update_rejects_missing_and_unexpected_keys(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight", "head.weight"}
    old_z = model.z
    monkeypatch.setattr(
        RWKV7ForCausalLM,
        "_preprocess_weights",
        lambda self, z: pytest.fail("preprocess should not run"),
    )

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(ValueError, match="missing.*head.weight"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")

    model.start_weight_update()
    model.load_weights(
        [
            ("emb.weight", torch.tensor([1.0])),
            ("head.weight", torch.tensor([2.0])),
            ("extra.weight", torch.tensor([3.0])),
        ]
    )

    with pytest.raises(ValueError, match="unexpected.*extra.weight"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_keeps_old_weights_on_preprocess_error(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    old_z = model.z

    def fail_preprocess(self, z):
        raise RuntimeError("bad preprocess")

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fail_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(RuntimeError, match="bad preprocess"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_restores_old_weights_after_partial_preprocess(
    monkeypatch,
):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    old_z = model.z

    def fail_after_partial_preprocess(self, z):
        self.z = {"partial": torch.tensor([9.0])}
        raise RuntimeError("post preprocess failure")

    monkeypatch.setattr(
        RWKV7ForCausalLM, "_preprocess_weights", fail_after_partial_preprocess
    )

    model.start_weight_update()
    model.load_weights([("emb.weight", torch.tensor([1.0]))])

    with pytest.raises(RuntimeError, match="post preprocess failure"):
        model.finish_weight_update()

    assert model.z is old_z
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_online_weight_update_cuda_snapshots_pending_tensors_on_cpu(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CUDA staging")

    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    source = torch.tensor([1.0], device="cuda")
    captured = {}

    def fake_preprocess(self, z):
        captured.update(z)
        self.z = z

    monkeypatch.setattr(RWKV7ForCausalLM, "_preprocess_weights", fake_preprocess)

    model.start_weight_update()
    model.load_weights([("emb.weight", source)])
    pending = model._pending_weight_update["emb.weight"]
    source.fill_(99.0)

    assert pending.device.type == "cpu"
    assert pending.data_ptr() != source.data_ptr()

    model.finish_weight_update()

    assert captured["emb.weight"].item() == 1.0


def test_rwkv7_streaming_update_copies_regular_and_derived_weights_in_place():
    model = _new_rwkv7_for_weight_tests()
    model.total_num_layers = 1
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 1
    model.tp_rank = 0
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "head.weight",
        "blocks.0.att.w1",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
        "head.weight": (2, 2),
        "blocks.0.att.w1": (2, 3),
    }
    model.z = {
        "emb.weight": torch.zeros((2, 2)),
        "blocks.0.ln0.weight": torch.zeros(2),
        "blocks.0.ln0.bias": torch.zeros(2),
        "head.weight": torch.zeros((2, 2)),
        "blocks.0.att.w1": torch.zeros((2, 3)),
        "blocks.0.att.w1.t": torch.zeros((3, 2)),
    }
    head_destination = model.z["head.weight"]
    lowrank_destination = model.z["blocks.0.att.w1"]
    lowrank_t_destination = model.z["blocks.0.att.w1.t"]
    head = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lowrank = torch.arange(6, dtype=torch.float32).view(2, 3)

    assert model.start_weight_update() is True
    assert model._streaming_weight_update is True
    model.load_weights([("head.weight", head), ("blocks.0.att.w1", lowrank)])

    assert model.z["head.weight"] is head_destination
    assert model.z["blocks.0.att.w1"] is lowrank_destination
    assert model.z["blocks.0.att.w1.t"] is lowrank_t_destination
    torch.testing.assert_close(head_destination, head.t())
    torch.testing.assert_close(lowrank_destination, lowrank)
    torch.testing.assert_close(lowrank_t_destination, lowrank.t())
    assert model._pending_weight_update["head.weight"] is None
    assert model._pending_weight_update["blocks.0.att.w1"] is None
    model.abort_weight_update()


def test_rwkv7_streaming_update_stages_only_embedding_dependencies(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.hidden_size = 2
    model.total_num_layers = 1
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 1
    model.tp_rank = 0
    model.vocab_size = 2
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
    }
    model.z = {
        "emb.weight": torch.zeros((2, 2)),
        "blocks.0.ln0.weight": torch.zeros(2),
        "blocks.0.ln0.bias": torch.zeros(2),
    }
    emb = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    ln0_w = torch.tensor([2.0, 3.0])
    ln0_b = torch.tensor([5.0, 7.0])
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "emb_ln0_bf16_to_f16",
        lambda value, weight, bias: value * weight + bias,
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    model.start_weight_update()
    model.load_weights(
        [
            ("emb.weight", emb),
            ("blocks.0.ln0.weight", ln0_w),
            ("blocks.0.ln0.bias", ln0_b),
        ]
    )

    assert all(value is not None for value in model._pending_weight_update.values())
    assert all(
        value.device.type == "cpu"
        for value in model._pending_weight_update.values()
        if value is not None
    )

    model.finish_weight_update()

    torch.testing.assert_close(model.z["emb.weight"], emb * ln0_w + ln0_b)
    torch.testing.assert_close(model.z["blocks.0.ln0.weight"], ln0_w)
    torch.testing.assert_close(model.z["blocks.0.ln0.bias"], ln0_b)


def test_rwkv7_streaming_validation_error_is_delayed_until_finish(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.total_num_layers = 1
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 1
    model.tp_rank = 0
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "head.weight",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
        "head.weight": (2, 2),
    }
    model.z = {
        "emb.weight": torch.zeros((2, 2)),
        "blocks.0.ln0.weight": torch.zeros(2),
        "blocks.0.ln0.bias": torch.zeros(2),
        "head.weight": torch.zeros((2, 2)),
    }
    old_head = model.z["head.weight"].clone()

    model.start_weight_update()
    loaded = set()
    for bucket in (
        [("head.weight", torch.zeros((1, 4)))],
        [("emb.weight", torch.zeros((2, 2)))],
        [("blocks.0.ln0.weight", torch.zeros(2))],
        [("blocks.0.ln0.bias", torch.zeros(2))],
    ):
        loaded.update(model.load_weights(bucket))

    assert loaded == model.raw_weight_names
    assert set(model._pending_weight_update) == model.raw_weight_names
    torch.testing.assert_close(model.z["head.weight"], old_head)
    with pytest.raises(ValueError, match="raw weight shape mismatch"):
        model.finish_weight_update()
    assert not hasattr(model, "_pending_weight_update")


def test_rwkv7_streaming_update_matches_tp2_runtime_layouts():
    model = _new_rwkv7_for_weight_tests()
    model.hidden_size = 4
    model.total_num_layers = 1
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 2
    model.tp_rank = 1
    model.tp_hidden_size = 2
    model.vocab_size = 3
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "head.weight",
        "blocks.0.att.r_k",
        "blocks.0.att.output.weight",
        "blocks.0.ffn.key.weight",
    }
    model.raw_weight_shapes = {
        "emb.weight": (3, 4),
        "blocks.0.ln0.weight": (4,),
        "blocks.0.ln0.bias": (4,),
        "head.weight": (3, 4),
        "blocks.0.att.r_k": (4, 2),
        "blocks.0.att.output.weight": (4, 4),
        "blocks.0.ffn.key.weight": (4, 6),
    }
    model.z = {
        "emb.weight": torch.zeros((2, 4)),
        "blocks.0.ln0.weight": torch.zeros(4),
        "blocks.0.ln0.bias": torch.zeros(4),
        "head.weight": torch.zeros((4, 2)),
        "blocks.0.att.r_k": torch.zeros(4),
        "blocks.0.att.output.weight": torch.zeros((2, 4)),
        "blocks.0.ffn.key.weight": torch.zeros((6, 2)),
    }
    head = torch.arange(12, dtype=torch.float32).view(3, 4)
    r_k = torch.arange(8, dtype=torch.float32).view(4, 2)
    output = torch.arange(16, dtype=torch.float32).view(4, 4)
    ffn_key = torch.arange(24, dtype=torch.float32).view(4, 6)

    model.start_weight_update()
    assert model._streaming_weight_update is True
    model.load_weights(
        [
            ("head.weight", head),
            ("blocks.0.att.r_k", r_k),
            ("blocks.0.att.output.weight", output),
            ("blocks.0.ffn.key.weight", ffn_key),
        ]
    )

    expected_head_shard = torch.zeros((2, 4))
    expected_head_shard[0].copy_(head[2])
    torch.testing.assert_close(model.z["head.weight"], expected_head_shard.t())
    torch.testing.assert_close(model.z["blocks.0.att.r_k"], r_k[2:4].flatten())
    torch.testing.assert_close(
        model.z["blocks.0.att.output.weight"],
        output[:, 2:4].t(),
    )
    torch.testing.assert_close(
        model.z["blocks.0.ffn.key.weight"],
        ffn_key[2:4].t(),
    )
    model.abort_weight_update()


def test_rwkv7_strict_online_update_forbids_full_model_staging(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.raw_weight_names = {"emb.weight"}
    model.raw_weight_shapes = None
    monkeypatch.setenv("VLLM_RWKV7_STRICT_STREAMING_WEIGHT_UPDATE", "1")

    with pytest.raises(RuntimeError, match="streaming weight update capability"):
        model.start_weight_update()

    assert not hasattr(model, "_streaming_weight_update")
    assert not hasattr(model, "_pending_weight_update")
    assert not hasattr(model, "_weight_update_error")


def test_rwkv7_streaming_update_skips_off_rank_embedding_on_pp_last(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.hidden_size = 2
    model.total_num_layers = 2
    model.start_layer = 1
    model.end_layer = 2
    model.tp_size = 1
    model.tp_rank = 0
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "head.weight",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
        "head.weight": (2, 2),
    }
    model.z = {"head.weight": torch.zeros((2, 2))}
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    model.start_weight_update()
    assert model._streaming_weight_update is True
    model.load_weights(
        [
            ("emb.weight", torch.ones((2, 2))),
            ("blocks.0.ln0.weight", torch.ones(2)),
            ("blocks.0.ln0.bias", torch.ones(2)),
            ("head.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ]
    )

    assert all(value is None for value in model._pending_weight_update.values())
    model.finish_weight_update()
    torch.testing.assert_close(
        model.z["head.weight"],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_rwkv7_streaming_update_drains_after_non_validation_error(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.total_num_layers = 1
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 1
    model.tp_rank = 0
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "head.weight",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
        "head.weight": (2, 2),
    }
    model.z = {
        "emb.weight": torch.zeros((2, 2)),
        "blocks.0.ln0.weight": torch.zeros(2),
        "blocks.0.ln0.bias": torch.zeros(2),
        "head.weight": torch.zeros((2, 2)),
    }
    monkeypatch.setattr(
        model,
        "_copy_raw_weight_to_runtime",
        lambda name, weight: (_ for _ in ()).throw(OSError("copy failed")),
    )

    model.start_weight_update()
    for bucket in (
        [("head.weight", torch.zeros((2, 2)))],
        [("emb.weight", torch.zeros((2, 2)))],
        [("blocks.0.ln0.weight", torch.zeros(2))],
        [("blocks.0.ln0.bias", torch.zeros(2))],
    ):
        model.load_weights(bucket)

    assert set(model._pending_weight_update) == model.raw_weight_names
    with pytest.raises(OSError, match="copy failed"):
        model.finish_weight_update()


def test_rwkv7_streaming_update_matches_fresh_preprocess_for_all_runtime_keys(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "emb_ln0_bf16_to_f16",
        lambda value, weight, bias: value * weight + bias,
    )

    def new_model():
        model = _new_rwkv7_for_weight_tests()
        model.config = SimpleNamespace(
            hidden_size=128,
            vocab_size=2,
            head_size=64,
            num_hidden_layers=1,
        )
        model.hidden_size = 128
        model.head_size = 64
        model.num_attention_heads = 2
        model.vocab_size = 2
        model.total_num_layers = 1
        model.start_layer = 0
        model.end_layer = 1
        model.tp_size = 1
        model.tp_rank = 0
        model.tp_hidden_size = 128
        model.execution_profile = rwkv7.resolve_execution_profile("fp16")
        model._get_layer_range = lambda: (0, 1)
        return model

    old_raw = {
        "emb.weight": torch.arange(256, dtype=torch.float32).view(2, 128),
        "blocks.0.ln0.weight": torch.ones(128),
        "blocks.0.ln0.bias": torch.zeros(128),
        "blocks.0.att.r_k": torch.arange(128, dtype=torch.float32).view(2, 64),
        "blocks.0.att.w1": torch.arange(384, dtype=torch.float32).view(128, 3),
        "blocks.0.att.output.weight": torch.arange(16384, dtype=torch.float32).view(
            128, 128
        ),
        "blocks.0.ffn.key.weight": torch.arange(32768, dtype=torch.float32).view(
            128, 256
        ),
        "head.weight": torch.arange(256, dtype=torch.float32).view(2, 128),
    }
    new_raw = {name: value + 1 for name, value in old_raw.items()}

    model = new_model()
    old_runtime = {name: value.clone() for name, value in old_raw.items()}
    model._preprocess_weights(old_runtime)
    model.z = old_runtime
    model.raw_weight_names = set(old_raw)
    model.raw_weight_shapes = {
        name: tuple(value.shape) for name, value in old_raw.items()
    }
    old_storage = {name: value.data_ptr() for name, value in model.z.items()}

    expected_model = new_model()
    expected_runtime = {name: value.clone() for name, value in new_raw.items()}
    expected_model._preprocess_weights(expected_runtime)

    model.start_weight_update()
    assert model._streaming_weight_update is True
    for item in new_raw.items():
        model.load_weights([item])
    model.finish_weight_update()

    assert set(model.z) == set(expected_runtime)
    for name, expected in expected_runtime.items():
        assert model.z[name].data_ptr() == old_storage[name]
        torch.testing.assert_close(model.z[name], expected)


def test_rwkv7_streaming_update_skips_off_rank_weights_on_pp_middle(monkeypatch):
    model = _new_rwkv7_for_weight_tests()
    model.total_num_layers = 3
    model.start_layer = 1
    model.end_layer = 2
    model.tp_size = 1
    model.tp_rank = 0
    model.raw_weight_names = {
        "emb.weight",
        "blocks.0.ln0.weight",
        "blocks.0.ln0.bias",
        "blocks.1.ffn.key.weight",
        "head.weight",
    }
    model.raw_weight_shapes = {
        "emb.weight": (2, 2),
        "blocks.0.ln0.weight": (2,),
        "blocks.0.ln0.bias": (2,),
        "blocks.1.ffn.key.weight": (2, 2),
        "head.weight": (2, 2),
    }
    model.z = {"blocks.1.ffn.key.weight": torch.zeros((2, 2))}
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)

    model.start_weight_update()
    assert model._streaming_weight_update is True
    for item in (
        ("emb.weight", torch.ones((2, 2))),
        ("blocks.0.ln0.weight", torch.ones(2)),
        ("blocks.0.ln0.bias", torch.ones(2)),
        ("blocks.1.ffn.key.weight", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("head.weight", torch.ones((2, 2))),
    ):
        model.load_weights([item])

    assert all(value is None for value in model._pending_weight_update.values())
    model.finish_weight_update()
    torch.testing.assert_close(
        model.z["blocks.1.ffn.key.weight"],
        torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
    )


def test_rwkv7_direct_parameter_update_is_not_supported():
    model = _new_rwkv7_for_weight_tests()

    with pytest.raises(NotImplementedError, match="checkpoint-format"):
        model.get_parameter("emb.weight")


def test_rwkv7_model_state_reset_after_weight_update_restores_runtime_metadata():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.add_request(1, _new_request("req-1"))
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)
    state.execution_idx_mapping.fill_(-1)
    state.decode_query_start_loc.fill_(-1)

    state.reset_after_weight_update()

    assert torch.count_nonzero(state.shift_state) == 0
    assert torch.count_nonzero(state.wkv_state) == 0
    assert torch.count_nonzero(state.elapsed) == 0
    assert state.execution_idx_mapping.tolist() == [0, 1, 2]
    assert state.decode_query_start_loc.tolist() == [0, 1, 2, 3]
    assert state.req_id_to_index == {"req-1": 1}
    assert state.req_slot_to_row[1] != -1


def test_rwkv7_model_state_allocates_only_local_pp_layers():
    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    model = _new_rwkv7_forward_test_model(
        wkv_mode="fp16",
        start_layer=1,
        end_layer=3,
    )

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert state.layer_offset == 1
    assert state.num_layers == 2
    assert state.shift_state.shape == (2, 2, 3, 64)
    assert state.wkv_state.shape == (2, 3, 1, 64, 64)


def test_rwkv7_model_state_allocates_only_local_tp_heads():
    hf_config = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=256,
        head_size=64,
        num_attention_heads=4,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=3),
    )
    model = _new_rwkv7_forward_test_model(wkv_mode="fp16", tp_num_heads=2)

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    assert state.num_heads == 2
    assert state.shift_state.shape == (2, 2, 3, 256)
    assert state.wkv_state.shape == (2, 3, 2, 64, 64)


def test_rwkv7_pipeline_rank_keeps_only_stage_weights():
    middle = object.__new__(RWKV7ForCausalLM)
    middle.start_layer = 1
    middle.end_layer = 3
    middle.total_num_layers = 4

    assert not middle._is_weight_needed_on_rank("emb.weight")
    assert not middle._is_weight_needed_on_rank("blocks.0.att.r_k")
    assert middle._is_weight_needed_on_rank("blocks.1.att.r_k")
    assert middle._is_weight_needed_on_rank("blocks.2.ffn.key.weight")
    assert not middle._is_weight_needed_on_rank("blocks.3.att.r_k")
    assert not middle._is_weight_needed_on_rank("ln_out.weight")
    assert not middle._is_weight_needed_on_rank("head.weight")

    last = object.__new__(RWKV7ForCausalLM)
    last.start_layer = 3
    last.end_layer = 4
    last.total_num_layers = 4

    assert last._is_weight_needed_on_rank("blocks.3.att.r_k")
    assert last._is_weight_needed_on_rank("ln_out.weight")
    assert last._is_weight_needed_on_rank("head.weight")


def test_rwkv7_tensor_parallel_shards_weights_for_rank():
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.tp_rank = 1
    model.tp_num_heads = 2
    model.tp_hidden_size = 128

    def values(*shape: int) -> torch.Tensor:
        return torch.arange(int(np.prod(shape)), dtype=torch.float32).view(*shape)

    for key in ("emb.weight", "head.weight"):
        weight = values(10, 256)
        torch.testing.assert_close(
            model._shard_weight_for_tp(key, weight),
            weight[5:10],
            rtol=0,
            atol=0,
        )

    weight = values(4, 64)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.r_k", weight),
        weight[2:4],
        rtol=0,
        atol=0,
    )

    weight = values(256, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.receptance.weight", weight),
        weight[128:256],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.output.weight", weight),
        weight[:, 128:256],
        rtol=0,
        atol=0,
    )

    weight = values(512, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.ffn.key.weight", weight),
        weight[256:512],
        rtol=0,
        atol=0,
    )

    weight = values(256, 512)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.ffn.value.weight", weight),
        weight[:, 256:512],
        rtol=0,
        atol=0,
    )

    weight = values(32, 256)
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.w2", weight),
        weight[:, 128:256],
        rtol=0,
        atol=0,
    )

    weight = values(256)
    assert model._shard_weight_for_tp("blocks.1.att.x_r", weight) is weight
    torch.testing.assert_close(
        model._shard_weight_for_tp("blocks.1.att.ln_x.weight", weight),
        weight[128:256],
        rtol=0,
        atol=0,
    )


def test_rwkv7_project_logits_fp32_uses_fp32_lt_op_for_batched_input(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.z = {"head.weight": torch.empty(4, 5, dtype=torch.float16)}
    hidden_states = torch.empty(2, 3, 4, dtype=torch.float16)
    expected = torch.ones(2, 3, 5, dtype=torch.float32)
    calls = []

    def fake_linear(x, weight):
        calls.append((x, weight))
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_fp32_lt",
        fake_linear,
    )

    logits = model.project_logits_fp32(hidden_states)

    assert logits.shape == (2, 3, 5)
    assert logits.dtype == torch.float32
    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is model.z["head.weight"]


def test_rwkv7_project_logits_fp32_uses_m1_splitk(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.z = {"head.weight": torch.empty(4, 64, dtype=torch.float16)}
    hidden_states = torch.empty(1, 4, dtype=torch.float16)
    expected = torch.ones(1, 64, dtype=torch.float32)
    calls = []

    def fake_splitk(x, weight):
        calls.append((x, weight))
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk_fp32",
        fake_splitk,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_fp32_lt",
        lambda *_args, **_kwargs: pytest.fail("M=1 must use split-K"),
    )

    logits = model.project_logits_fp32(hidden_states)

    assert logits is expected
    assert len(calls) == 1
    assert calls[0][0] is hidden_states
    assert calls[0][1] is model.z["head.weight"]


def test_rwkv7_project_logits_fp32_zero_k_multirow_uses_fp32_lt(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.z = {"head.weight": torch.empty(0, 64, dtype=torch.float16)}
    hidden_states = torch.empty(2, 0, dtype=torch.float16)
    expected = torch.zeros(2, 64, dtype=torch.float32)
    calls = []

    def fake_linear(x, weight):
        calls.append((x, weight))
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_fp32_lt",
        fake_linear,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk_fp32",
        lambda *_args: pytest.fail("multiple rows must not use split-K"),
    )

    logits = model.project_logits_fp32(hidden_states)

    assert logits is expected
    assert calls == [(hidden_states, model.z["head.weight"])]


def test_rwkv7_project_logits_fp32_m1_non_aligned_vocab_uses_fp32_lt(
    monkeypatch,
):
    model = object.__new__(RWKV7ForCausalLM)
    model.z = {"head.weight": torch.empty(4, 65, dtype=torch.float16)}
    hidden_states = torch.empty(1, 4, dtype=torch.float16)
    expected = torch.ones(1, 65, dtype=torch.float32)
    calls = []

    def fake_linear(x, weight):
        calls.append((x, weight))
        return expected

    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_fp32_lt",
        fake_linear,
    )
    monkeypatch.setattr(
        torch.ops.rwkv7_v3a_ops,
        "linear_f16_m1_splitk_fp32",
        lambda *_args: pytest.fail("unaligned vocab must not use split-K"),
    )

    logits = model.project_logits_fp32(hidden_states)

    assert logits is expected
    assert calls == [(hidden_states, model.z["head.weight"])]


def test_rwkv7_compute_logits_tp1_preserves_fp32_contract(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 1
    expected = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    model.project_logits_fp32 = lambda hidden_states: expected
    processor_calls = []

    def logits_processor(lm_head, logits):
        processor_calls.append((lm_head, logits))
        return logits

    model.logits_processor = logits_processor
    monkeypatch.setattr(
        rwkv7,
        "tensor_model_parallel_all_gather",
        lambda _logits: pytest.fail("TP=1 must not all-gather logits"),
    )

    logits = model.compute_logits(torch.empty(2, 4))

    assert logits is expected
    assert logits.dtype == torch.float32
    assert processor_calls == [(None, expected)]


def test_rwkv7_compute_logits_all_gathers_tensor_parallel_vocab(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.vocab_size = 5
    model.z = {"head.weight": torch.empty(3, 4)}
    model.project_logits_fp32 = lambda hidden_states: torch.ones(
        (2, 3), dtype=torch.float32
    )
    model.logits_processor = lambda lm_head, logits: logits

    def fake_all_gather(logits):
        assert logits.shape == (2, 3)
        assert logits.dtype == torch.float32
        return torch.cat([logits, logits + 10], dim=-1)

    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_gather", fake_all_gather)

    logits = model.compute_logits(torch.empty(2, 4))

    assert logits.shape == (2, 5)
    assert logits.dtype == torch.float32
    assert logits.tolist() == [
        [1.0, 1.0, 1.0, 11.0, 11.0],
        [1.0, 1.0, 1.0, 11.0, 11.0],
    ]


def test_rwkv7_compute_sampling_logits_uses_packed_decode_view(monkeypatch):
    model = _new_rwkv7_forward_test_model()
    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    logits_indices = torch.arange(3, dtype=torch.int64)
    expected_view = hidden_states[:3]
    expected_logits = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    seen_sample_hidden_states = []

    def compute_logits(sample_hidden_states):
        seen_sample_hidden_states.append(sample_hidden_states)
        _assert_same_storage_view(sample_hidden_states, expected_view)
        return expected_logits

    model.compute_logits = compute_logits

    def fail_arange(*_args, **_kwargs):
        pytest.fail("sampling logits fast path must not allocate expected indices")

    def fail_sync_guard(*_args, **_kwargs):
        pytest.fail("sampling logits fast path must not run tensor equality guards")

    monkeypatch.setattr(torch, "arange", fail_arange)
    monkeypatch.setattr(torch, "equal", fail_sync_guard)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=0,
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
        is_prefilling_np=np.zeros(3, dtype=np.bool_),
        logits_indices=logits_indices,
        rwkv_sampling_logits_contiguous=True,
    )

    logits = model.compute_sampling_logits(hidden_states, logits_indices, input_batch)

    assert logits is expected_logits
    assert len(seen_sample_hidden_states) == 1


@pytest.mark.parametrize(
    ("case_name", "logits_indices", "input_batch"),
    [
        (
            "prefill",
            torch.tensor([0, 2], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.array([1, 2], dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 3], dtype=np.int32),
                is_prefilling_np=np.array([False, True], dtype=np.bool_),
            ),
        ),
        (
            "spec_decode",
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=1,
                num_draft_tokens=1,
                num_draft_tokens_per_req=np.array([1], dtype=np.int32),
                num_scheduled_tokens=np.array([1], dtype=np.int32),
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1], dtype=np.int32),
                is_prefilling_np=np.array([False], dtype=np.bool_),
            ),
        ),
        (
            "non_contiguous_logits",
            torch.tensor([0, 2], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                is_prefilling_np=np.zeros(2, dtype=np.bool_),
            ),
        ),
        (
            "metadata_declines",
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                is_prefilling_np=np.zeros(2, dtype=np.bool_),
                rwkv_sampling_logits_contiguous=False,
            ),
        ),
        (
            "missing_prefill_metadata",
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
            ),
        ),
        (
            "floating_logits_indices",
            torch.tensor([0.0, 1.0], dtype=torch.float32),
            SimpleNamespace(
                num_reqs=2,
                num_draft_tokens=0,
                num_scheduled_tokens=np.ones(2, dtype=np.int32),
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
                is_prefilling_np=np.zeros(2, dtype=np.bool_),
            ),
        ),
    ],
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_rwkv7_compute_sampling_logits_declines_non_decode_or_spec_batch(
    case_name,
    logits_indices,
    input_batch,
):
    del case_name
    model = _new_rwkv7_forward_test_model()
    hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    def compute_logits(_sample_hidden_states):
        pytest.fail("declined sampling logits hook must not compute logits")

    model.compute_logits = compute_logits

    logits = model.compute_sampling_logits(hidden_states, logits_indices, input_batch)

    assert logits is None


def test_rwkv7_tensor_parallel_embedding_masks_remote_tokens(monkeypatch):
    model = object.__new__(RWKV7ForCausalLM)
    model.tp_size = 2
    model.tp_rank = 0
    model.vocab_size = 5
    model.z = {
        "emb.weight": torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ]
        )
    }
    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_reduce", lambda x: x)

    out = model.embed(torch.tensor([[0, 4]], dtype=torch.int64))

    assert out.tolist() == [[[1.0, 1.0], [0.0, 0.0]]]


def test_rwkv7_model_state_lifecycle_resets_reused_rows():
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        hidden_size=128,
        head_size=64,
        num_attention_heads=2,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    model = _new_rwkv7_forward_test_model(wkv_mode="fp16")

    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=model,
        encoder_cache=None,
        device=torch.device("cpu"),
    )

    state.shift_state[:, :, 0].fill_(1)
    state.wkv_state[:, 0].fill_(1)
    state.elapsed[0] = 7
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    assert row == 0
    assert torch.count_nonzero(state.shift_state[:, :, row]) == 0
    assert torch.count_nonzero(state.wkv_state[:, row]) == 0
    assert state.elapsed[row].item() == 0
    assert state.req_id_to_index["req-2"] == 2

    state.elapsed[row] = 11
    state.remove_request("req-2")
    assert "req-2" not in state.req_id_to_index
    assert state.elapsed[row].item() == 0


def test_rwkv7_model_state_allows_permuted_decode_schedule_with_slots():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[:2] = torch.tensor([10, 20], dtype=torch.int32)

    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert inputs["idx_mapping"].tolist() == [1, 0]
    assert inputs["rwkv_decode_rows"] == [1, 0]
    assert inputs["slot_indices"].tolist() == [1, 0]
    _assert_decode_wkv_metadata(inputs, [1, 0])
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert state.elapsed.tolist()[:2] == [10, 20]


def test_rwkv7_model_state_sorts_decode_wave_by_resident_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))

    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1, 2, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    req_states = SimpleNamespace(
        req_id_to_index={f"req-{req_slot}": req_slot for req_slot in range(4)},
        num_computed_prefill_tokens=np.array([3, 3, 3, 3], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([3, 3, 3, 3], dtype=np.int32)),
    )

    req_ids = ["req-1", "req-2", "req-3", "req-0"]
    sorted_req_ids = state.sort_scheduled_req_ids(
        req_ids,
        {req_id: 1 for req_id in req_ids},
        req_states,
    )

    assert sorted_req_ids == ["req-0", "req-1", "req-2", "req-3"]


def test_rwkv7_model_state_sort_keeps_one_token_prefill_after_decode():
    state = _new_rwkv7_model_state(max_num_reqs=3)
    for req_slot in range(3):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    req_states = SimpleNamespace(
        req_id_to_index={f"req-{req_slot}": req_slot for req_slot in range(3)},
        num_computed_prefill_tokens=np.array([3, 3, 0], dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.array([3, 3, 1], dtype=np.int32)),
    )

    req_ids = ["req-2", "req-1", "req-0"]
    sorted_req_ids = state.sort_scheduled_req_ids(
        req_ids,
        {req_id: 1 for req_id in req_ids},
        req_states,
    )

    assert sorted_req_ids == ["req-0", "req-1", "req-2"]


def test_rwkv7_model_state_uses_contiguous_decode_path_for_prefix_rows():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["rwkv_decode_batch_size"] == 2
    assert inputs["rwkv_decode_rows"] == [0, 1]
    assert inputs["rwkv_decode_token_positions"] == [0, 1]
    assert inputs["slot_indices"] is None
    _assert_decode_wkv_metadata(inputs, [0, 1])


def test_rwkv7_model_state_keeps_steady_decode_slot_after_row_removal():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 1].fill_(30)
    state.elapsed[1] = 40
    state.remove_request("req-0")

    assert state.req_slot_to_row[:2] == [-1, 1]
    assert state.row_to_req_slot[:2] == [-1, 1]
    assert torch.count_nonzero(state.shift_state[:, :, 0]) == 0
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 0]) == 0
    assert torch.all(state.wkv_state[:, 1] == 30)
    assert state.elapsed.tolist()[:2] == [0, 40]
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(decode_batch, req_states=None)

    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["idx_mapping"].tolist() == [1]
    assert inputs["rwkv_decode_rows"] == [1]
    assert inputs["rwkv_decode_token_positions"].tolist() == [0]
    assert inputs["slot_indices"].tolist() == [1]
    _assert_decode_wkv_metadata(inputs, [1])
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()


def test_rwkv7_model_state_keeps_slots_after_generic_input_batch_condense():
    state = _new_rwkv7_model_state(
        max_num_reqs=4,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    input_batch = LegacyInputBatch(
        max_num_reqs=4,
        max_model_len=16,
        max_num_batched_tokens=16,
        device=torch.device("cpu"),
        vocab_size=128,
        block_sizes=[1],
        kernel_block_sizes=[1],
        max_num_blocks_per_req=[16],
    )
    for req_slot in range(3):
        req_id = f"req-{req_slot}"
        assert input_batch.add_request(_new_cached_request_state(req_id)) == req_slot
        state.add_request(req_slot, _new_request(req_id))

    first_decode_batch = SimpleNamespace(
        req_ids=["req-0", "req-1", "req-2"],
        idx_mapping_np=np.array([0, 1, 2], dtype=np.int32),
        num_reqs=3,
        query_start_loc=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        query_start_loc_np=np.array([0, 1, 2, 3], dtype=np.int32),
        is_prefilling_np=np.array([False, False, False], dtype=np.bool_),
    )
    state.prepare_inputs(first_decode_batch, req_states=None)
    state.shift_state[:, :, 2].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.elapsed[2] = 40

    input_batch.remove_request("req-1")
    state.remove_request("req-1")
    input_batch.condense()
    assert input_batch.req_ids == ["req-0", "req-2"]
    assert input_batch.req_id_to_index == {"req-0": 0, "req-2": 1}

    input_batch.idx_mapping_np = np.array([0, 1], dtype=np.int32)
    input_batch.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    input_batch.query_start_loc_np = np.array([0, 1, 2], dtype=np.int32)
    input_batch.is_prefilling_np = np.array([False, False], dtype=np.bool_)
    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert state.req_id_to_index == {"req-0": 0, "req-2": 2}
    assert state.req_slot_to_row[:3] == [0, -1, 2]
    assert state.row_to_req_slot[:3] == [0, -1, 2]
    assert inputs["idx_mapping"].tolist() == [0, 2]
    assert inputs["rwkv_decode_rows"] == [0, 2]
    assert inputs["slot_indices"].tolist() == [0, 2]
    _assert_decode_wkv_metadata(inputs, [0, 2])
    assert torch.all(state.shift_state[:, :, 2] == 20)
    assert torch.all(state.wkv_state[:, 2] == 30)
    assert state.elapsed.tolist()[:3] == [0, 0, 40]


def test_rwkv7_model_state_keeps_decode_prefix_when_prefill_reuses_free_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("decode-low"))
    state.add_request(1, _new_request("decode-high"))
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[1] = 20

    state.remove_request("decode-low")
    state.add_request(2, _new_request("prefill-low"))
    decode_row = state.req_slot_to_row[1]
    prefill_row = state.req_slot_to_row[2]
    assert decode_row == 1
    assert prefill_row == 0

    mixed_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([1, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        is_prefilling_np=np.array([False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(mixed_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [1, 0]
    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["rwkv_decode_rows"] == [1]
    assert inputs["rwkv_decode_token_positions"].tolist() == [0]
    assert inputs["slot_indices"].tolist() == [1]
    assert inputs["rwkv_prefill_rows"] == [prefill_row]
    _assert_decode_wkv_metadata(inputs, [1])
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[prefill_row],
        token_positions=[1, 2, 3],
        req_id=[0, 0, 0],
    )
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert inputs["prefill_shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["prefill_wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["prefill_elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert state.decode_req_slots == {1}
    assert state.req_slot_to_row[:3] == [-1, decode_row, prefill_row]
    assert state.row_to_req_slot[:3] == [2, 1, -1]
    assert 2 in state.free_rows
    assert torch.count_nonzero(state.shift_state[:, :, prefill_row]) == 0
    assert torch.all(state.shift_state[:, :, decode_row] == 20)
    assert torch.count_nonzero(state.shift_state[:, :, 2]) == 0
    assert torch.count_nonzero(state.wkv_state[:, prefill_row]) == 0
    assert torch.all(state.wkv_state[:, decode_row] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 2]) == 0
    assert state.elapsed.tolist()[:3] == [0, 20, 0]


def test_rwkv7_prepare_permuted_decode_returns_slot_indices_before_forward():
    state = _new_rwkv7_model_state(
        max_num_reqs=2,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(30)
    state.wkv_state[:, 1].fill_(40)
    state.elapsed[:2] = torch.tensor([50, 60], dtype=torch.int32)
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([1, 0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert inputs["idx_mapping"].tolist() == [1, 0]
    assert inputs["rwkv_decode_rows"] == [1, 0]
    assert inputs["slot_indices"].tolist() == [1, 0]
    _assert_decode_wkv_metadata(inputs, [1, 0])
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 30)
    assert torch.all(state.wkv_state[:, 1] == 40)
    assert state.elapsed.tolist() == [50, 60]


def test_rwkv7_model_state_keeps_decode_and_resident_prefill_slots_separate():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.elapsed[:2] = torch.tensor([10, 20], dtype=torch.int32)
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3, 4], dtype=torch.int32),
        is_prefilling_np=np.array([True, False], dtype=np.bool_),
        num_scheduled_tokens=np.array([3, 1], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0], dtype=np.int32),
        prefill_len_np=np.array([3, 1], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert state.req_slot_to_row[:2] == [0, 1]
    assert state.row_to_req_slot[:2] == [0, 1]
    assert inputs["idx_mapping"].tolist() == [0, 1]
    assert inputs["rwkv_decode_batch_size"] == 1
    assert inputs["rwkv_decode_rows"] == [1]
    assert inputs["slot_indices"].tolist() == [1]
    assert inputs["rwkv_prefill_rows"] == [0]
    _assert_decode_wkv_metadata(inputs, [1])
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[0],
        token_positions=[0, 1, 2],
        req_id=[0, 0, 0],
    )
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert state.elapsed.tolist()[:2] == [10, 20]


@pytest.mark.parametrize("scheduled_slot", [0, 1])
def test_rwkv7_model_state_rejects_partial_live_decode_wave(scheduled_slot: int):
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)

    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([scheduled_slot], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )

    with pytest.raises(RuntimeError, match="all live decode rows"):
        state.prepare_inputs(input_batch, req_states=None)
    assert len(state.decode_req_slots) == 2
    assert state.req_slot_to_row[:2] == [0, 1]


def test_rwkv7_model_state_allows_permuted_decode_with_resident_prefill():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.elapsed[:3] = torch.tensor([10, 20, 30], dtype=torch.int32)
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([1, 0, 2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 1, 3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [1, 0, 2]
    assert inputs["rwkv_decode_rows"] == [1, 0]
    assert inputs["slot_indices"].tolist() == [1, 0]
    assert inputs["rwkv_prefill_rows"] == [2]
    _assert_decode_wkv_metadata(inputs, [1, 0])
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[2],
        token_positions=[2, 3, 4],
        req_id=[0, 0, 0],
    )
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert state.elapsed.tolist()[:3] == [10, 20, 30]


def test_rwkv7_model_state_keeps_prefill_to_decode_slot_stable():
    state = _new_rwkv7_model_state(
        max_num_reqs=4,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    for req_slot in range(3):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    state.remove_request("req-1")
    assert state.req_slot_to_row[:3] == [0, -1, 2]

    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)
    state.shift_state[:, :, 2].fill_(20)
    state.wkv_state[:, 2].fill_(21)
    state.elapsed[2] = 22

    prefill_to_decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
    )
    prefill_inputs = state.prepare_inputs(prefill_to_decode_batch, req_states=None)
    _assert_prefill_wkv_metadata(
        prefill_inputs,
        query_start_loc=[0, 2],
        slot_indices=[2],
        token_positions=[0, 1],
        req_id=[0, 0],
    )

    state.postprocess_state(
        prefill_inputs["idx_mapping"],
        torch.tensor([1], dtype=torch.int32),
    )

    assert state.decode_req_slots == {0, 2}
    assert state.req_slot_to_row[:3] == [0, -1, 2]
    assert state.row_to_req_slot[:3] == [0, -1, 2]
    assert 1 in state.free_rows
    assert torch.all(state.shift_state[:, :, 2] == 20)
    assert torch.all(state.wkv_state[:, 2] == 21)
    assert state.elapsed.tolist()[:3] == [0, 0, 22]


def _fragmented_mixed_rwkv7_inputs() -> tuple[RWKV7ModelState, dict[str, Any]]:
    state = _new_rwkv7_model_state(
        max_num_reqs=4,
        hidden_size=3,
        head_size=1,
        num_attention_heads=1,
    )
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    state.remove_request("req-1")

    decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        is_prefilling_np=np.array([False], dtype=np.bool_),
    )
    state.prepare_inputs(decode_batch, req_states=None)

    prefill_to_decode_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([2], dtype=np.int32),
    )
    prefill_inputs = state.prepare_inputs(prefill_to_decode_batch, req_states=None)
    _assert_prefill_wkv_metadata(
        prefill_inputs,
        query_start_loc=[0, 2],
        slot_indices=[2],
        token_positions=[0, 1],
        req_id=[0, 0],
    )
    state.postprocess_state(
        prefill_inputs["idx_mapping"],
        torch.tensor([1], dtype=torch.int32),
    )

    assert state.req_slot_to_row[:4] == [0, -1, 2, 3]
    assert state.row_to_req_slot[:4] == [0, -1, 2, 3]
    assert state.decode_req_slots == {0, 2}
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 2].fill_(20)
    state.shift_state[:, :, 3].fill_(30)
    state.wkv_state[:, 0].fill_(11)
    state.wkv_state[:, 2].fill_(21)
    state.wkv_state[:, 3].fill_(31)
    state.elapsed[:4] = torch.tensor([12, 0, 22, 32], dtype=torch.int32)
    mixed_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 2, 3], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2, 5], dtype=torch.int32),
        is_prefilling_np=np.array([False, False, True], dtype=np.bool_),
        num_scheduled_tokens=np.array([1, 1, 3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 0, 0], dtype=np.int32),
        prefill_len_np=np.array([1, 1, 3], dtype=np.int32),
    )

    return state, state.prepare_inputs(mixed_batch, req_states=None)


def test_rwkv7_model_state_keeps_prefill_transition_slots_before_forward():
    state, inputs = _fragmented_mixed_rwkv7_inputs()

    assert inputs["rwkv_decode_batch_size"] == 2
    assert inputs["rwkv_decode_rows"] == [0, 2]
    assert inputs["rwkv_decode_token_positions"].tolist() == [0, 1]
    assert inputs["idx_mapping"].tolist() == [0, 2, 3]
    assert inputs["slot_indices"].tolist() == [0, 2]
    assert inputs["rwkv_prefill_rows"] == [3]
    _assert_decode_wkv_metadata(inputs, [0, 2])
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[3],
        token_positions=[2, 3, 4],
        req_id=[0, 0, 0],
    )
    assert state.req_slot_to_row[:4] == [0, -1, 2, 3]
    assert state.row_to_req_slot[:4] == [0, -1, 2, 3]
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 2] == 20)
    assert torch.all(state.shift_state[:, :, 3] == 30)
    assert state.elapsed.tolist()[:4] == [12, 0, 22, 32]


def test_rwkv7_model_state_prefill_uses_resident_state():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(2, _new_request("req-2"))
    state.shift_state.fill_(7)
    state.wkv_state.fill_(8)
    state.elapsed.fill_(9)
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [0]
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    assert inputs["rwkv_prefill_rows"] == [0]
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[0],
        token_positions=[0, 1, 2],
        req_id=[0, 0, 0],
    )
    assert torch.all(state.shift_state == 7)
    assert torch.all(state.wkv_state == 8)
    assert torch.all(state.elapsed == 9)


def test_rwkv7_model_state_keeps_resident_prefill_row_when_decode_row_starts():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    assert row == 2
    state.shift_state[:, :, 0].fill_(3)
    state.shift_state[:, :, 1].fill_(5)
    state.wkv_state[:, 0].fill_(7)
    state.wkv_state[:, 1].fill_(9)
    state.elapsed[:2] = torch.tensor([11, 13], dtype=torch.int32)
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[row],
        token_positions=[0, 1, 2],
        req_id=[0, 0, 0],
    )
    inputs["shift_state"][:, :, row].fill_(11)
    inputs["wkv_state"][:, row].fill_(13)
    inputs["elapsed"][row].fill_(17)

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    decode_row = state.req_slot_to_row[2]
    assert 2 in state.decode_req_slots
    assert decode_row == row
    assert torch.all(state.shift_state[:, :, decode_row] == 11)
    assert torch.all(state.wkv_state[:, decode_row] == 13)
    assert state.elapsed[decode_row].item() == 17
    assert state.req_slot_to_row[:3] == [0, 1, 2]
    assert state.row_to_req_slot[:3] == [0, 1, 2]
    assert torch.all(state.shift_state[:, :, 0] == 3)
    assert torch.all(state.shift_state[:, :, 1] == 5)
    assert torch.all(state.wkv_state[:, 0] == 7)
    assert torch.all(state.wkv_state[:, 1] == 9)
    assert state.elapsed.tolist()[:3] == [11, 13, 17]


def test_rwkv7_model_state_prefill_becomes_decode_without_resident_copy():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    row = state.req_slot_to_row[2]
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([2], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[row],
        token_positions=[0, 1, 2],
        req_id=[0, 0, 0],
    )
    inputs["shift_state"][:, :, row].fill_(11)
    inputs["wkv_state"][:, row].fill_(13)
    inputs["elapsed"][row].fill_(17)

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    assert 2 in state.decode_req_slots
    assert len(state.decode_req_slots) == 1
    assert state.req_slot_to_row[2] == row
    assert state.row_to_req_slot[row] == 2
    assert torch.all(state.shift_state[:, :, row] == 11)
    assert torch.all(state.wkv_state[:, row] == 13)
    assert state.elapsed[row].item() == 17
    assert not state.has_pending_postprocess_state()


def test_rwkv7_model_state_reports_pending_prefill_state_postprocess():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([3], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([3], dtype=np.int32),
    )

    assert not state.has_pending_postprocess_state()
    inputs = state.prepare_inputs(input_batch, req_states=None)
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3],
        slot_indices=[0],
        token_positions=[0, 1, 2],
        req_id=[0, 0, 0],
    )
    assert state.has_pending_postprocess_state()

    state.postprocess_state(inputs["idx_mapping"], torch.tensor([1], dtype=torch.int32))

    assert not state.has_pending_postprocess_state()


def test_rwkv7_non_last_pp_postprocesses_pending_prefill_when_all_decode_next():
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([0, 1], dtype=torch.int32))
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.is_last_pp_rank = False
    runner.pp_handler = SimpleNamespace(receive=lambda _input_batch: True)
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda _finished_req_ids: SimpleNamespace(is_empty=lambda: True)
    )
    runner.eplb = SimpleNamespace(step=lambda **_kwargs: None)

    calls: list[Any] = []
    runner.postprocess_num_computed_tokens = lambda _input_batch: calls.append(
        "num_computed"
    )
    runner.model_state = SimpleNamespace(
        has_pending_postprocess_state=lambda: True,
        postprocess_state=lambda idx_mapping, num_sampled: calls.append(
            ("state", idx_mapping.tolist(), num_sampled)
        ),
    )

    GPUModelRunner.sample_tokens(runner, grammar_output=None)

    assert calls == ["num_computed", ("state", [0, 1], 0)]


def test_non_rwkv_non_last_pp_rank_does_not_require_pending_state_hook():
    from vllm.v1.worker.gpu.model_runner import ExecuteModelState, GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    input_batch = SimpleNamespace(idx_mapping=torch.tensor([0, 1], dtype=torch.int32))
    runner.execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner.is_last_pp_rank = False
    runner.pp_handler = SimpleNamespace(receive=lambda _input_batch: True)
    runner.kv_connector = SimpleNamespace(
        post_forward=lambda _finished_req_ids: SimpleNamespace(is_empty=lambda: True)
    )
    runner.eplb = SimpleNamespace(step=lambda **_kwargs: None)

    calls: list[Any] = []
    runner.postprocess_num_computed_tokens = lambda _input_batch: calls.append(
        "num_computed"
    )
    runner.model_state = SimpleNamespace(
        postprocess_state=lambda idx_mapping, num_sampled: calls.append(
            ("state", idx_mapping.tolist(), num_sampled)
        ),
    )

    GPUModelRunner.sample_tokens(runner, grammar_output=None)

    assert calls == ["num_computed"]


def test_rwkv7_model_state_remove_request_recycles_stable_row():
    state = _new_rwkv7_model_state(max_num_reqs=4)
    state.add_request(0, _new_request("req-0"))
    state.add_request(1, _new_request("req-1"))
    state.add_request(2, _new_request("req-2"))
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.elapsed[:3] = torch.tensor([10, 20, 30], dtype=torch.int32)

    state.remove_request("req-1")

    assert state.req_slot_to_row[0] == 0
    assert state.req_slot_to_row[1] == -1
    assert state.req_slot_to_row[2] == 2
    assert state.row_to_req_slot[:3] == [0, -1, 2]
    assert 1 in state.free_rows
    assert torch.count_nonzero(state.shift_state[:, :, 1]) == 0
    assert torch.count_nonzero(state.wkv_state[:, 1]) == 0
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert torch.all(state.wkv_state[:, 2] == 30)
    assert state.elapsed.tolist()[:3] == [10, 0, 30]

    state.add_request(3, _new_request("req-3"))

    assert state.req_slot_to_row[3] == 1


def test_rwkv7_model_state_remove_decode_row_keeps_other_resident_rows_stable():
    state = _new_rwkv7_model_state(max_num_reqs=5)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(input_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.shift_state[:, :, 3].fill_(40)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.wkv_state[:, 3].fill_(40)
    state.elapsed[:4] = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    state.remove_request("req-0")

    assert len(state.decode_req_slots) == 1
    assert state.req_slot_to_row[:4] == [-1, 1, 2, 3]
    assert state.row_to_req_slot[:4] == [-1, 1, 2, 3]
    assert 0 in state.free_rows
    assert torch.count_nonzero(state.shift_state[:, :, 0]) == 0
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.all(state.shift_state[:, :, 2] == 30)
    assert torch.all(state.shift_state[:, :, 3] == 40)
    assert torch.count_nonzero(state.wkv_state[:, 0]) == 0
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert torch.all(state.wkv_state[:, 2] == 30)
    assert torch.all(state.wkv_state[:, 3] == 40)
    assert state.elapsed.tolist()[:4] == [0, 20, 30, 40]


def test_rwkv7_model_state_remove_prefill_row_preserves_decode_prefix():
    state = _new_rwkv7_model_state(max_num_reqs=5)
    for req_slot in range(4):
        state.add_request(req_slot, _new_request(f"req-{req_slot}"))
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        is_prefilling_np=np.array([False, False], dtype=np.bool_),
    )
    state.prepare_inputs(input_batch, req_states=None)
    state.shift_state[:, :, 0].fill_(10)
    state.shift_state[:, :, 1].fill_(20)
    state.shift_state[:, :, 2].fill_(30)
    state.shift_state[:, :, 3].fill_(40)
    state.wkv_state[:, 0].fill_(10)
    state.wkv_state[:, 1].fill_(20)
    state.wkv_state[:, 2].fill_(30)
    state.wkv_state[:, 3].fill_(40)
    state.elapsed[:4] = torch.tensor([10, 20, 30, 40], dtype=torch.int32)

    state.remove_request("req-2")

    assert len(state.decode_req_slots) == 2
    assert state.req_slot_to_row[:4] == [0, 1, -1, 3]
    assert state.row_to_req_slot[:4] == [0, 1, -1, 3]
    assert 2 in state.free_rows
    assert torch.all(state.shift_state[:, :, 0] == 10)
    assert torch.all(state.shift_state[:, :, 1] == 20)
    assert torch.count_nonzero(state.shift_state[:, :, 2]) == 0
    assert torch.all(state.shift_state[:, :, 3] == 40)
    assert torch.all(state.wkv_state[:, 0] == 10)
    assert torch.all(state.wkv_state[:, 1] == 20)
    assert torch.count_nonzero(state.wkv_state[:, 2]) == 0
    assert torch.all(state.wkv_state[:, 3] == 40)
    assert state.elapsed.tolist()[:4] == [10, 20, 0, 40]


def test_rwkv7_model_state_dummy_batch_uses_scratch_state():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.req_slot_to_row = [3, 2, 1, 0]
    state.row_to_req_slot = [3, 2, 1, 0]
    state.free_rows.clear()
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)
    input_batch = SimpleNamespace(
        req_ids=["_warmup_0_", "_warmup_1_"],
        idx_mapping_np=np.array([0, 1], dtype=np.int32),
        num_reqs=2,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    inputs = state.prepare_inputs(input_batch, req_states=None)

    assert inputs["idx_mapping"].tolist() == [0, 1]
    assert inputs["shift_state"].data_ptr() != state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() != state.wkv_state.data_ptr()
    assert torch.count_nonzero(inputs["shift_state"]) == 0
    assert torch.count_nonzero(inputs["wkv_state"]) == 0
    assert torch.count_nonzero(inputs["elapsed"]) == 0
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 1, 2],
        slot_indices=[0, 1],
        token_positions=[0, 1],
        req_id=[0, 1],
    )
    assert state.req_slot_to_row == [-1, -1, -1, -1]
    assert state.row_to_req_slot == [-1, -1, -1, -1]
    assert state.free_rows == {0, 1, 2, 3}
    assert torch.all(state.shift_state == 1)
    assert torch.all(state.wkv_state == 2)
    assert torch.all(state.elapsed == 3)


def test_rwkv7_model_state_requires_rapid_sampler():
    state = object.__new__(RWKV7ModelState)
    sampler = SimpleNamespace(use_rapid=True, require_rapid=False)

    assert "uniform recurrent decode waves" in state.get_v2_kernel_warmup_skip_reason()
    assert state.custom_sampler(sampler) == (sampler, None)
    assert sampler.require_rapid is True


def test_rwkv7_model_state_rejects_unavailable_rapid_sampler():
    state = object.__new__(RWKV7ModelState)
    sampler = SimpleNamespace(use_rapid=False, require_rapid=False)

    with pytest.raises(RuntimeError, match="requires rapid-sampling"):
        state.custom_sampler(sampler)


def test_rwkv7_dummy_inputs_cover_all_tokens():
    hf_config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        head_size=64,
        num_attention_heads=1,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    state = RWKV7ModelState(
        vllm_config=vllm_config,
        model=_new_rwkv7_forward_test_model(wkv_mode="fp32io16"),
        encoder_cache=None,
        device=torch.device("cpu"),
    )
    state.shift_state.fill_(1)
    state.wkv_state.fill_(2)
    state.elapsed.fill_(3)

    inputs = state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)

    assert inputs["query_start_loc"].tolist() == [0, 3, 6, 8]
    assert inputs["idx_mapping"].tolist() == [0, 1, 2]
    assert inputs["wkv_state"].dtype == torch.float32
    assert inputs["shift_state"].data_ptr() == state.shift_state.data_ptr()
    assert inputs["wkv_state"].data_ptr() == state.wkv_state.data_ptr()
    assert inputs["elapsed"].data_ptr() == state.elapsed.data_ptr()
    _assert_prefill_wkv_metadata(
        inputs,
        query_start_loc=[0, 3, 6, 8],
        slot_indices=[0, 1, 2],
        token_positions=list(range(8)),
        req_id=[0, 0, 0, 1, 1, 1, 2, 2],
    )
    assert torch.all(state.shift_state == 1)
    assert torch.all(state.wkv_state == 2)
    assert torch.all(state.elapsed == 3)


def test_rwkv7_model_state_rejects_nonpositive_prefill_length():
    state = _new_rwkv7_model_state(max_num_reqs=2)
    state.add_request(0, _new_request("req-0"))
    input_batch = _rwkv7_input_batch(
        state,
        idx_mapping_np=np.array([0], dtype=np.int32),
        query_start_loc=torch.tensor([0, 0], dtype=torch.int32),
        is_prefilling_np=np.array([True], dtype=np.bool_),
        num_scheduled_tokens=np.array([0], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([1], dtype=np.int32),
    )

    with pytest.raises(RuntimeError, match="fast prefill requires"):
        state.prepare_inputs(input_batch, req_states=None)


def test_rwkv7_vllm_forward_uses_dense_decode_input_view_for_contiguous_rows(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    seen_tokens = []
    input_ids = torch.tensor([10, 20], dtype=torch.int64)
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    def forward_tokens(
        tokens,
        state,
        *,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        seen_tokens.append(tokens.tolist())
        assert query_start_loc.tolist() == [0, 1, 2]
        assert wkv_slot_indices.tolist() == [0, 1]
        _assert_same_storage_view(tokens.view(-1), input_ids[:2])
        _assert_same_storage_view(state[0], shift_state[:, :, :2, :])
        _assert_same_storage_view(state[1], wkv_state[:, :2, :, :, :])
        _assert_same_storage_view(state[2], elapsed[:2])
        state[0].add_(1)
        state[1].add_(2)
        state[2].add_(3)
        return torch.tensor(
            [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]],
            dtype=torch.float32,
        )

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "pure contiguous decode must not run prefill path"
        ),
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    assert seen_tokens == [[[10], [20]]]
    assert out.tolist() == [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]]
    assert torch.all(shift_state == 1)
    assert torch.all(wkv_state == 2)
    assert elapsed.tolist() == [3, 3]


@pytest.mark.parametrize(
    "decode_rows", [[1, 0], [2, 0]], ids=["permuted-prefix", "non-prefix"]
)
def test_rwkv7_vllm_forward_rejects_noncontiguous_decode_rows(
    monkeypatch, decode_rows: list[int]
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    seen_tokens = []

    def forward_tokens(tokens, state):
        seen_tokens.append(tokens.tolist())
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "pure decode must not run prefill path"
        ),
    )
    num_rows = max(decode_rows) + 1
    shift_state = torch.zeros((1, 2, num_rows, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, num_rows, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((num_rows,), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="contiguous prefix rows"):
        RWKV7ForCausalLM.forward(
            model,
            torch.tensor([10, 20], dtype=torch.int64),
            positions=None,
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            idx_mapping=torch.tensor(decode_rows, dtype=torch.int32),
            shift_state=shift_state,
            wkv_state=wkv_state,
            elapsed=elapsed,
            rwkv_decode_batch_size=2,
            rwkv_decode_rows=decode_rows,
            rwkv_decode_token_positions=[0, 1],
            rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

    assert seen_tokens == []
    assert torch.count_nonzero(shift_state) == 0
    assert torch.count_nonzero(wkv_state) == 0
    assert torch.count_nonzero(elapsed) == 0


def test_rwkv7_vllm_forward_uses_slot_indices_for_permuted_decode_rows(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))
    monkeypatch.setattr(
        RWKV7ForCausalLM,
        "_contiguous_decode_token_range",
        staticmethod(
            lambda *args: pytest.fail("slot decode must not require contiguous rows")
        ),
    )

    seen = []
    input_ids = torch.tensor([10, 20], dtype=torch.int64)
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)
    slot_indices = torch.tensor([2, 0], dtype=torch.int32)

    def forward_tokens(
        tokens,
        state,
        *,
        slot_indices=None,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        assert slot_indices is not None
        seen.append(
            (
                tokens.tolist(),
                slot_indices.tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        assert tuple(tensor.data_ptr() for tensor in state) == (
            shift_state.data_ptr(),
            wkv_state.data_ptr(),
            elapsed.data_ptr(),
        )
        for row in slot_indices.tolist():
            shift_state[:, :, row].add_(1)
            wkv_state[:, row].add_(2)
            elapsed[row].add_(3)
        return torch.tensor(
            [[201.0, 202.0, 203.0], [101.0, 102.0, 103.0]],
            dtype=torch.float32,
        )

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail(
            "pure slot decode must not run prefill path"
        ),
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        idx_mapping=torch.tensor([2, 0], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[2, 0],
        rwkv_decode_token_positions=[1, 0],
        rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        slot_indices=slot_indices,
    )

    assert seen == [([[20], [10]], [2, 0], [0, 1, 2], [2, 0])]
    assert out.tolist() == [[101.0, 102.0, 103.0], [201.0, 202.0, 203.0]]
    assert torch.all(shift_state[:, :, 0] == 1)
    assert torch.count_nonzero(shift_state[:, :, 1]) == 0
    assert torch.all(shift_state[:, :, 2] == 1)
    assert torch.all(wkv_state[:, 0] == 2)
    assert torch.count_nonzero(wkv_state[:, 1]) == 0
    assert torch.all(wkv_state[:, 2] == 2)
    assert elapsed.tolist() == [3, 0, 3]


def test_rwkv7_vllm_forward_uses_varlen_prefill_metadata(monkeypatch):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    calls: list[tuple[Any, ...]] = []
    shift_state = torch.zeros((1, 2, 4, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 4, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((4,), dtype=torch.int32)
    input_ids = torch.tensor([10, 20, 21, 30, 31, 32], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 3, 6], dtype=torch.int32)

    def forward_tokens(
        tokens,
        state,
        *,
        slot_indices=None,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        calls.append(
            (
                "decode",
                tokens.tolist(),
                slot_indices.tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    def forward_varlen_hidden(
        tokens,
        state,
        *,
        query_start_loc,
        slot_indices,
        req_id,
    ):
        calls.append(
            (
                "varlen",
                tokens.tolist(),
                query_start_loc.tolist(),
                slot_indices.tolist(),
                req_id.tolist(),
            )
        )
        assert tuple(tensor.data_ptr() for tensor in state) == (
            shift_state.data_ptr(),
            wkv_state.data_ptr(),
            elapsed.data_ptr(),
        )
        for row, amount in zip(slot_indices.tolist(), [2, 3]):
            shift_state[:, :, row].fill_(amount)
            wkv_state[:, row].fill_(amount + 10)
            elapsed[row] = amount
        return tokens.to(torch.float32).unsqueeze(-1).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_varlen_hidden=forward_varlen_hidden,
        forward_all_hidden=lambda *args: pytest.fail(
            "varlen metadata must bypass grouped prefill fallback"
        ),
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=query_start_loc,
        idx_mapping=torch.tensor([1, 2, 0], dtype=torch.int32),
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=1,
        rwkv_decode_rows=[1],
        rwkv_decode_token_positions=[0],
        rwkv_decode_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        rwkv_prefill_token_ranges=[(1, 1, 3), (2, 3, 6)],
        rwkv_prefill_rows=[2, 0],
        rwkv_prefill_query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        rwkv_prefill_slot_indices=torch.tensor([2, 0], dtype=torch.int32),
        rwkv_prefill_token_positions=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        rwkv_prefill_req_id=torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
        slot_indices=torch.tensor([1], dtype=torch.int32),
    )

    assert calls == [
        ("decode", [[10]], [1], [0, 1], [1]),
        ("varlen", [20, 21, 30, 31, 32], [0, 2, 5], [2, 0], [0, 0, 1, 1, 1]),
    ]
    assert out.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [21.0, 21.0, 21.0],
        [30.0, 30.0, 30.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]
    assert torch.all(shift_state[:, :, 0] == 3)
    assert torch.all(shift_state[:, :, 2] == 2)
    assert torch.all(wkv_state[:, 0] == 13)
    assert torch.all(wkv_state[:, 2] == 12)
    assert elapsed.tolist() == [3, 0, 2, 0]


def test_rwkv7_vllm_forward_uses_slot_mapped_mixed_decode_state_without_gather(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))
    state, model_inputs = _fragmented_mixed_rwkv7_inputs()

    def forbid_index_select(*args, **kwargs):
        pytest.fail("model-side decode state gather must not be used")

    monkeypatch.setattr(torch, "index_select", forbid_index_select)
    calls = []

    def forward_tokens(
        tokens,
        model_state,
        *,
        slot_indices=None,
        query_start_loc=None,
        wkv_slot_indices=None,
    ):
        assert slot_indices is not None
        calls.append(
            (
                "tokens",
                tokens.tolist(),
                slot_indices.tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        assert tuple(tensor.data_ptr() for tensor in model_state) == (
            state.shift_state.data_ptr(),
            state.wkv_state.data_ptr(),
            state.elapsed.data_ptr(),
        )
        assert torch.all(model_state[0][:, :, 0] == 10)
        assert torch.count_nonzero(model_state[0][:, :, 1]) == 0
        assert torch.all(model_state[0][:, :, 2] == 20)
        assert model_state[2].tolist() == [12, 0, 22, 32]
        return tokens.to(torch.float32).expand(tokens.shape[0], 3)

    def forward_varlen_hidden(
        tokens,
        model_state,
        *,
        query_start_loc,
        slot_indices,
        req_id,
    ):
        calls.append(
            (
                "varlen",
                tokens.tolist(),
                query_start_loc.tolist(),
                slot_indices.tolist(),
                req_id.tolist(),
            )
        )
        assert tuple(tensor.data_ptr() for tensor in model_state) == (
            state.shift_state.data_ptr(),
            state.wkv_state.data_ptr(),
            state.elapsed.data_ptr(),
        )
        return tokens.to(torch.float32).unsqueeze(-1).expand(tokens.shape[0], 3)

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_varlen_hidden=forward_varlen_hidden,
        forward_all_hidden=lambda *args, **kwargs: pytest.fail(
            "canonical prefill must not use grouped fixed-shape execution"
        ),
    )

    out = RWKV7ForCausalLM.forward(
        model,
        torch.tensor([10, 20, 30, 31, 32], dtype=torch.int64),
        positions=None,
        **model_inputs,
    )

    assert calls == [
        ("tokens", [[10], [20]], [0, 2], [0, 1, 2], [0, 2]),
        ("varlen", [30, 31, 32], [0, 3], [3], [0, 0, 0]),
    ]
    assert out.tolist() == [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]


def test_rwkv7_vllm_forward_rejects_sparse_active_decode_rows(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))
    seen = []

    def forward_tokens(tokens, state):
        seen.append(
            (
                tokens.tolist(),
                tuple(state[0].shape),
                state[0].storage_offset(),
            )
        )
        return torch.arange(tokens.shape[0] * 3, dtype=torch.float32).view(
            tokens.shape[0], 3
        )

    model = _new_rwkv7_forward_test_model(
        forward_tokens=forward_tokens,
        forward_all_hidden=lambda *args: pytest.fail("decode must stay T=1"),
    )
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)
    shift_state[:, :, 1].fill_(5)
    wkv_state[:, 1].fill_(7)
    elapsed[1] = 11

    with pytest.raises(RuntimeError, match="decode rows must match"):
        RWKV7ForCausalLM.forward(
            model,
            torch.tensor([20], dtype=torch.int64),
            positions=None,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping=torch.tensor([1], dtype=torch.int32),
            shift_state=shift_state,
            wkv_state=wkv_state,
            elapsed=elapsed,
            rwkv_decode_batch_size=2,
            rwkv_decode_rows=[1],
            rwkv_decode_token_positions=[0],
            rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        )

    assert seen == []


def test_rwkv7_vllm_pp_non_last_stage_returns_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.hidden_size = 3
    model.start_layer = 0
    model.end_layer = 1
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: False
    model.embed = lambda tokens: (
        tokens.to(torch.float32).unsqueeze(-1).expand(*tokens.shape, 3)
    )
    seen_calls: list[tuple[Any, ...]] = []

    def forward_layer_range(
        x,
        state,
        path,
        *,
        v_first,
        final,
        all_logits,
        last_indices,
        query_start_loc,
        wkv_slot_indices,
    ):
        assert v_first is None
        seen_calls.append(
            (
                "decode",
                x[:, :, 0].tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        state[0].fill_(5)
        state[1].fill_(7)
        state[2].fill_(11)
        return x + 1, x + 2

    def forward_varlen_layer_range(
        x,
        state,
        *,
        query_start_loc,
        slot_indices,
        req_id,
        v_first,
        final,
    ):
        assert v_first is None
        assert not final
        seen_calls.append(
            (
                "prefill",
                x[:, 0].tolist(),
                query_start_loc.tolist(),
                slot_indices.tolist(),
                req_id.tolist(),
            )
        )
        row = slot_indices.item()
        state[0][:, :, row].fill_(5)
        state[1][:, row].fill_(7)
        state[2][row].fill_(11)
        return x + 1, x + 2

    model.forward_layer_range = forward_layer_range
    model.forward_varlen_layer_range = forward_varlen_layer_range
    input_ids = torch.tensor([10, 20, 30, 31], dtype=torch.int64)
    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids,
        positions=None,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        rwkv_prefill_token_ranges=[(2, 2, 4)],
        rwkv_prefill_rows=[2],
        rwkv_prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        rwkv_prefill_slot_indices=torch.tensor([2], dtype=torch.int32),
        rwkv_prefill_token_positions=torch.tensor([2, 3], dtype=torch.long),
        rwkv_prefill_req_id=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert isinstance(out, IntermediateTensors)
    assert seen_calls == [
        ("decode", [[10.0], [20.0]], [0, 1, 2], [0, 1]),
        ("prefill", [30.0, 31.0], [0, 2], [2], [0, 0]),
    ]
    assert out["hidden_states"].tolist() == [
        [11.0, 11.0, 11.0],
        [21.0, 21.0, 21.0],
        [31.0, 31.0, 31.0],
        [32.0, 32.0, 32.0],
    ]
    assert out["v_first"].tolist() == [
        [12.0, 12.0, 12.0],
        [22.0, 22.0, 22.0],
        [32.0, 32.0, 32.0],
        [33.0, 33.0, 33.0],
    ]
    assert torch.all(shift_state == 5)
    assert torch.all(wkv_state == 7)
    assert elapsed.tolist() == [11, 11, 11]


def test_rwkv7_vllm_pp_non_last_stage_all_gathers_tp_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.hidden_size = 4
    model.start_layer = 0
    model.end_layer = 1
    model.tp_size = 2
    model.tp_rank = 0
    model.tp_hidden_size = 2
    model._is_pp_first_rank = lambda: True
    model._is_pp_last_rank = lambda: False
    model.embed = lambda tokens: (
        tokens.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4)
    )

    def fake_all_gather(value):
        assert value.shape == (1, 1, 2)
        return torch.cat([value, value + 100], dim=-1)

    monkeypatch.setattr(rwkv7, "tensor_model_parallel_all_gather", fake_all_gather)

    def forward_layer_range(
        x,
        state,
        path,
        *,
        v_first,
        final,
        all_logits,
        last_indices,
        query_start_loc,
        wkv_slot_indices,
    ):
        assert v_first is None
        assert query_start_loc.tolist() == [0, 1]
        assert wkv_slot_indices.tolist() == [0]
        return x + 1, x[..., :2] + 2

    model.forward_layer_range = forward_layer_range
    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=torch.tensor([10], dtype=torch.int64),
        positions=None,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        shift_state=torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        wkv_state=torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32),
        elapsed=torch.zeros((1,), dtype=torch.int32),
        rwkv_decode_batch_size=1,
        rwkv_decode_rows=[0],
        rwkv_decode_token_positions=[0],
        rwkv_decode_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert isinstance(out, IntermediateTensors)
    assert out["hidden_states"].tolist() == [[11.0, 11.0, 11.0, 11.0]]
    assert out["v_first"].tolist() == [[12.0, 12.0, 112.0, 112.0]]


def test_rwkv7_vllm_pp_last_stage_uses_intermediate_tensors(monkeypatch):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.hidden_size = 3
    model.start_layer = 1
    model.end_layer = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True
    seen_calls: list[tuple[Any, ...]] = []

    def forward_layer_range(
        x,
        state,
        path,
        *,
        v_first,
        final,
        all_logits,
        last_indices,
        query_start_loc,
        wkv_slot_indices,
    ):
        assert final
        assert all_logits
        assert last_indices is None
        seen_calls.append(
            (
                "decode",
                x[:, :, 0].tolist(),
                v_first[:, :, 0].tolist(),
                query_start_loc.tolist(),
                wkv_slot_indices.tolist(),
            )
        )
        state[0].fill_(13)
        state[1].fill_(17)
        state[2].fill_(19)
        return x + v_first, None

    def forward_varlen_layer_range(
        x,
        state,
        *,
        query_start_loc,
        slot_indices,
        req_id,
        v_first,
        final,
    ):
        assert final
        seen_calls.append(
            (
                "prefill",
                x[:, 0].tolist(),
                v_first[:, 0].tolist(),
                query_start_loc.tolist(),
                slot_indices.tolist(),
                req_id.tolist(),
            )
        )
        row = slot_indices.item()
        state[0][:, :, row].fill_(13)
        state[1][:, row].fill_(17)
        state[2][row].fill_(19)
        return x + v_first, None

    model.forward_layer_range = forward_layer_range
    model.forward_varlen_layer_range = forward_varlen_layer_range
    hidden_states = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    v_first = torch.full_like(hidden_states, 100)
    v_first[0].fill_(100)
    v_first[1].fill_(200)
    v_first[2].fill_(300)
    v_first[3].fill_(400)
    intermediate_tensors = IntermediateTensors(
        {"hidden_states": hidden_states, "v_first": v_first}
    )
    query_start_loc = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 3, 3), dtype=torch.float32)
    wkv_state = torch.zeros((1, 3, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((3,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        rwkv_prefill_token_ranges=[(2, 2, 4)],
        rwkv_prefill_rows=[2],
        rwkv_prefill_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        rwkv_prefill_slot_indices=torch.tensor([2], dtype=torch.int32),
        rwkv_prefill_token_positions=torch.tensor([2, 3], dtype=torch.long),
        rwkv_prefill_req_id=torch.tensor([0, 0], dtype=torch.int32),
    )

    assert isinstance(out, torch.Tensor)
    assert seen_calls == [
        ("decode", [[1.0], [4.0]], [[100.0], [200.0]], [0, 1, 2], [0, 1]),
        ("prefill", [7.0, 10.0], [300.0, 400.0], [0, 2], [2], [0, 0]),
    ]
    assert out.tolist() == [
        [101.0, 102.0, 103.0],
        [204.0, 205.0, 206.0],
        [307.0, 308.0, 309.0],
        [410.0, 411.0, 412.0],
    ]
    assert torch.all(shift_state == 13)
    assert torch.all(wkv_state == 17)
    assert elapsed.tolist() == [19, 19, 19]


def test_rwkv7_vllm_pp_last_stage_slices_full_tp_v_first(monkeypatch):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float32)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.hidden_size = 4
    model.start_layer = 1
    model.end_layer = 2
    model.tp_size = 2
    model.tp_rank = 1
    model.tp_hidden_size = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True

    def forward_layer_range(
        x,
        state,
        path,
        *,
        v_first,
        final,
        all_logits,
        last_indices,
        query_start_loc,
        wkv_slot_indices,
    ):
        assert v_first.tolist() == [[[30.0, 40.0]]]
        assert query_start_loc.tolist() == [0, 1]
        assert wkv_slot_indices.tolist() == [0]
        return x, None

    model.forward_layer_range = forward_layer_range
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.ones((1, 4), dtype=torch.float32),
            "v_first": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        }
    )

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        shift_state=torch.zeros((1, 2, 1, 4), dtype=torch.float32),
        wkv_state=torch.zeros((1, 1, 1, 1, 1), dtype=torch.float32),
        elapsed=torch.zeros((1,), dtype=torch.int32),
        rwkv_decode_batch_size=1,
        rwkv_decode_rows=[0],
        rwkv_decode_token_positions=[0],
        rwkv_decode_query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
    )

    assert isinstance(out, torch.Tensor)
    assert out.tolist() == [[1.0, 1.0, 1.0, 1.0]]


def test_rwkv7_vllm_pp_last_stage_casts_intermediate_tensors_to_internal_dtype(
    monkeypatch,
):
    monkeypatch.setattr(rwkv7, "DTYPE", torch.float16)
    monkeypatch.setattr(rwkv7, "CUDA_DEVICE", torch.device("cpu"))

    model = object.__new__(RWKV7ForCausalLM)
    model.hidden_size = 3
    model.start_layer = 1
    model.end_layer = 2
    model._is_pp_first_rank = lambda: False
    model._is_pp_last_rank = lambda: True

    seen_dtypes = []

    def forward_layer_range(
        x,
        state,
        path,
        *,
        v_first,
        final,
        all_logits,
        last_indices,
        query_start_loc,
        wkv_slot_indices,
    ):
        seen_dtypes.append((x.dtype, v_first.dtype))
        assert query_start_loc.tolist() == [0, 1, 2]
        assert wkv_slot_indices.tolist() == [0, 1]
        return x + v_first, None

    model.forward_layer_range = forward_layer_range
    intermediate_tensors = IntermediateTensors(
        {
            "hidden_states": torch.ones((2, 3), dtype=torch.bfloat16),
            "v_first": torch.full((2, 3), 2, dtype=torch.bfloat16),
        }
    )
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    shift_state = torch.zeros((1, 2, 2, 3), dtype=torch.float16)
    wkv_state = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)
    elapsed = torch.zeros((2,), dtype=torch.int32)

    out = RWKV7ForCausalLM.forward(
        model,
        input_ids=None,
        positions=None,
        intermediate_tensors=intermediate_tensors,
        query_start_loc=query_start_loc,
        idx_mapping=idx_mapping,
        shift_state=shift_state,
        wkv_state=wkv_state,
        elapsed=elapsed,
        rwkv_decode_batch_size=2,
        rwkv_decode_rows=[0, 1],
        rwkv_decode_token_positions=[0, 1],
        rwkv_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
    )

    assert seen_dtypes == [(torch.float16, torch.float16)]
    assert out.dtype == torch.float16
    torch.testing.assert_close(out, torch.full((2, 3), 3, dtype=torch.float16))
