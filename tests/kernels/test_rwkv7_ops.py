# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch


class OpCase(NamedTuple):
    name: str
    args: Callable[[str], tuple]
    expected: Callable[[tuple], list[tuple[tuple[int, ...], torch.dtype]]]


def _h(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.float16)


def _bf(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.bfloat16)


def _i32(device: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.empty(shape, device=device, dtype=torch.int32)


def _same(index: int, count: int = 1):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        return [(tuple(x.shape), x.dtype) for _ in range(count)]

    return expected


def _last_dim(index: int, dim: int):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        return [((*x.shape[:-1], dim), x.dtype)]

    return expected


def _last_dim_with_dtype(index: int, dim: int, dtype: torch.dtype):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        return [((*x.shape[:-1], dim), dtype)]

    return expected


def _last_dim_from_weight(index: int, weight_index: int, weight_dim: int):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        x = args[index]
        weight = args[weight_index]
        return [((*x.shape[:-1], weight.shape[weight_dim]), x.dtype)]

    return expected


def _multi_last_dim(specs: tuple[tuple[int, int, int], ...]):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        return [
            (
                (*args[x_index].shape[:-1], args[w_index].shape[weight_dim]),
                args[x_index].dtype,
            )
            for x_index, w_index, weight_dim in specs
        ]

    return expected


def _add_last_expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
    x = args[0]
    return [((x.shape[0], x.shape[2]), x.dtype)]


def _emb_expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
    emb = args[0]
    return [(tuple(emb.shape), torch.float16)]


def _linear_prepare_zero_expected(
    args: tuple,
) -> list[tuple[tuple[int, ...], torch.dtype]]:
    x, weight, zero_features = args
    return [
        ((*x.shape[:-1], weight.shape[1]), x.dtype),
        ((1, 1, zero_features), x.dtype),
    ]


def _shape_expected(
    shape: tuple[int, ...],
    dtype_index: int,
):
    def expected(args: tuple) -> list[tuple[tuple[int, ...], torch.dtype]]:
        return [(shape, args[dtype_index].dtype)]

    return expected


def _ln_args(device: str, c: int = 8) -> tuple:
    return _h(device, (2, 3, c)), _h(device, (c,)), _h(device, (c,))


def _linear_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (8, 5))


def _linear_orig_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (5, 8))


def _rank_tensor_args(device: str) -> tuple:
    return _h(device, (2, 3, 8)), _h(device, (2, 3, 8)), _h(device, (2, 3, 8))


def _rwkv7_import_or_skip() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 custom op registration")
    try:
        import vllm.rwkv7_ops  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"RWKV7 extension is unavailable: {exc!r}")
    import vllm._custom_ops  # noqa: F401


def _op(name: str):
    namespace, op_name = name.split("::", 1)
    return getattr(getattr(torch.ops, namespace), op_name)


V3A_RETURNING_CASES = [
    OpCase("rwkv7_v3a_ops::layer_norm_f16", _ln_args, _same(0)),
    OpCase(
        "rwkv7_v3a_ops::emb_ln0_bf16_to_f16",
        lambda d: (_bf(d, (11, 8)), _bf(d, (8,)), _bf(d, (8,))),
        _emb_expected,
    ),
    OpCase("rwkv7_v3a_ops::linear_f16", _linear_args, _last_dim(0, 5)),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_lt_cfg",
        lambda d: (*_linear_args(d), 0, 0),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_fp32_lt",
        _linear_args,
        _last_dim_with_dtype(0, 5, torch.float32),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 5))),
        _last_dim(0, 5),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk_fp32",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 64))),
        _last_dim_with_dtype(0, 64, torch.float32),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_f16_m1_splitk_prepare_zero",
        lambda d: (_h(d, (1, 8)), _h(d, (8, 64)), 16),
        _linear_prepare_zero_expected,
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_rkv_f16_m1_splitk",
        lambda d: (
            _h(d, (1, 8)),
            _h(d, (1, 1, 8)),
            _h(d, (1, 1, 1, 8)),
            _h(d, (8, 64)),
            _h(d, (8, 64)),
            _h(d, (8, 64)),
        ),
        _multi_last_dim(((0, 3, 1), (1, 4, 1), (2, 5, 1))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_f16",
        _linear_orig_args,
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_act_f16",
        lambda d: (*_linear_orig_args(d), 1),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_t_vres_f16",
        lambda d: (
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _h(d, (2, 3, 5)),
            _h(d, (2, 3, 5)),
            _h(d, (5,)),
        ),
        _last_dim_from_weight(0, 1, 0),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wag_rank_in_f16",
        lambda d: (*_rank_tensor_args(d), _h(d, (4, 8)), _h(d, (3, 8)), _h(d, (2, 8))),
        _multi_last_dim(((0, 3, 0), (1, 4, 0), (2, 5, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wagv_rank_in_f16",
        lambda d: (
            *_rank_tensor_args(d),
            _h(d, (2, 3, 8)),
            _h(d, (4, 8)),
            _h(d, (3, 8)),
            _h(d, (2, 8)),
            _h(d, (6, 8)),
        ),
        _multi_last_dim(((0, 4, 0), (1, 5, 0), (2, 6, 0), (3, 7, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wag_rank_out_f16",
        lambda d: (
            _h(d, (2, 3, 4)),
            _h(d, (2, 3, 3)),
            _h(d, (2, 3, 2)),
            _h(d, (8, 4)),
            _h(d, (8, 3)),
            _h(d, (8, 2)),
        ),
        _multi_last_dim(((0, 3, 0), (1, 4, 0), (2, 5, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::linear_wagv_rank_out_f16",
        lambda d: (
            _h(d, (2, 3, 4)),
            _h(d, (2, 3, 3)),
            _h(d, (2, 3, 2)),
            _h(d, (2, 3, 6)),
            _h(d, (8, 4)),
            _h(d, (8, 3)),
            _h(d, (8, 2)),
            _h(d, (8, 6)),
            _h(d, (2, 3, 8)),
            _h(d, (2, 3, 8)),
            _h(d, (8,)),
        ),
        _multi_last_dim(((0, 4, 0), (1, 5, 0), (2, 6, 0), (3, 7, 0))),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_f16",
        lambda d: (_h(d, (2, 3, 8)), _h(d, (2, 3, 8))),
        _same(0),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_f16",
        lambda d: (*_ln_args(d), _h(d, (8,))),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_last_layer_norm_f16",
        lambda d: (*_ln_args(d), _h(d, (8,))),
        _add_last_expected,
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (2, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            _h(d, (8,)),
        ),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_cmix_mix_f16_slots",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (4, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            _h(d, (8,)),
            _i32(d, (2,)),
        ),
        _same(0, 2),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (2, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(0, 7),
    ),
    OpCase(
        "rwkv7_v3a_ops::add_layer_norm_tmix_mix6_f16_slots",
        lambda d: (
            _h(d, (2, 1, 8)),
            _h(d, (2, 1, 8)),
            _h(d, (4, 8)),
            _h(d, (8,)),
            _h(d, (8,)),
            *[_h(d, (8,)) for _ in range(6)],
            _i32(d, (2,)),
        ),
        _same(0, 7),
    ),
]


FAST_RETURNING_CASES = [
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_mix6",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (2, 8)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(3, 6),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_mix6_3d",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (2, 8)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(3, 6),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_mix6_slot",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _i32(d, (2,)),
            *[_h(d, (8,)) for _ in range(6)],
        ),
        _same(3, 6),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_mix6_varlen",
        lambda d: (
            2,
            5,
            8,
            _h(d, (5, 8)),
            _h(d, (6, 8)),
            _i32(d, (2,)),
            *[_h(d, (8,)) for _ in range(6)],
            _i32(d, (3,)),
            _i32(d, (5,)),
        ),
        _same(3, 6),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_kk_a_gate",
        lambda d: (
            2,
            3,
            128,
            2,
            _h(d, (2, 3, 128)),
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (2, 3, 128)),
            _h(d, (128,)),
        ),
        _same(4, 3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_kk_a_gate_2d",
        lambda d: (
            2,
            1,
            4096,
            64,
            _h(d, (2, 1, 4096)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            _h(d, (2, 1, 4096)),
            _h(d, (4096,)),
        ),
        _same(4, 3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_lnx_rkvres_xg",
        lambda d: (
            2,
            3,
            128,
            2,
            *[_h(d, (2, 3, 128)) for _ in range(4)],
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (128,)),
            _h(d, (2, 3, 128)),
        ),
        _same(4),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_lnx_rkvres_xg_warp",
        lambda d: (
            64,
            1,
            4096,
            64,
            *[_h(d, (64, 1, 4096)) for _ in range(4)],
            _h(d, (4096,)),
            _h(d, (4096,)),
            _h(d, (4096,)),
            _h(d, (64, 1, 4096)),
        ),
        _same(4),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::tmix_vres_gate",
        lambda d: (
            320,
            1,
            4096,
            _h(d, (320, 1, 4096)),
            _h(d, (320, 1, 4096)),
            _h(d, (4096,)),
            _h(d, (320, 1, 4096)),
        ),
        _same(3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_sparse_down_relu_one",
        lambda d: (256, 128, _h(d, (128,)), _h(d, (128, 256))),
        _shape_expected((1, 1, 256), 2),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows",
        lambda d: (2, 3, 128, 128, _h(d, (2, 3, 128)), _h(d, (128, 128))),
        _shape_expected((2, 3, 128), 4),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_sparse_down_relu_rows_t512",
        lambda d: (2, 3, 512, 512, _h(d, (2, 3, 512)), _h(d, (512, 512))),
        _shape_expected((2, 3, 512), 4),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_mix",
        lambda d: (2, 3, 8, _h(d, (2, 3, 8)), _h(d, (2, 8)), _h(d, (8,))),
        _same(3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_mix_3d",
        lambda d: (2, 3, 8, _h(d, (2, 3, 8)), _h(d, (2, 8)), _h(d, (8,))),
        _same(3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_mix_slot",
        lambda d: (
            2,
            3,
            8,
            _h(d, (2, 3, 8)),
            _h(d, (5, 8)),
            _i32(d, (2,)),
            _h(d, (8,)),
        ),
        _same(3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::cmix_mix_varlen",
        lambda d: (
            2,
            5,
            8,
            _h(d, (5, 8)),
            _h(d, (6, 8)),
            _i32(d, (2,)),
            _h(d, (8,)),
            _i32(d, (3,)),
            _i32(d, (5,)),
        ),
        _same(3),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::relu_square", lambda d: (_h(d, (2, 3, 8)),), _same(0)
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::act_tanh", lambda d: (_h(d, (2, 3, 8)),), _same(0)
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::act_sigmoid", lambda d: (_h(d, (2, 3, 8)),), _same(0)
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::add_vec",
        lambda d: (8, _h(d, (2, 3, 8)), _h(d, (8,))),
        _same(1),
    ),
    OpCase(
        "vllm_rwkv7_fast_ops_fp16::add_vec_2d",
        lambda d: (8, _h(d, (2, 3, 8)), _h(d, (8,))),
        _same(1),
    ),
]


RETURNING_CASES = V3A_RETURNING_CASES + FAST_RETURNING_CASES


@pytest.fixture(scope="module", autouse=True)
def rwkv7_ops_registered() -> None:
    _rwkv7_import_or_skip()


def test_rwkv7_linear_op_honors_fp16_accumulation() -> None:
    torch.manual_seed(20260710)
    x = torch.randn((64, 1024), device="cuda", dtype=torch.float16)
    weight_orig = torch.randn((256, 1024), device="cuda", dtype=torch.float16)
    weight = weight_orig.t().contiguous()
    op = torch.ops.rwkv7_v3a_ops.linear_f16
    reference = torch.nn.functional.linear(x.float(), weight_orig.float()).half()

    default_accumulation = op(x, weight)
    fp32_accumulation = op(x, weight, False)
    fp16_accumulation = op(x, weight, True)
    torch.accelerator.synchronize()

    assert fp16_accumulation.shape == reference.shape
    assert fp16_accumulation.dtype == reference.dtype
    assert fp16_accumulation.device == reference.device
    assert torch.isfinite(fp16_accumulation).all()
    assert torch.equal(default_accumulation, fp32_accumulation)
    fp32_relative_l2 = (
        fp32_accumulation.float() - reference.float()
    ).norm() / reference.float().norm()
    fp16_relative_l2 = (
        fp16_accumulation.float() - reference.float()
    ).norm() / reference.float().norm()
    fp32_cosine = torch.nn.functional.cosine_similarity(
        fp32_accumulation.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    fp16_cosine = torch.nn.functional.cosine_similarity(
        fp16_accumulation.float().flatten(),
        reference.float().flatten(),
        dim=0,
    )
    assert fp32_relative_l2 < 1e-3
    assert fp32_cosine > 0.999999
    assert fp16_relative_l2 < 5e-3
    assert fp16_cosine > 0.99999
    assert torch.count_nonzero(fp16_accumulation != fp32_accumulation) > 0


@pytest.mark.parametrize("workspace_mb", [0, 32, 128])
def test_rwkv7_linear_f16_lt_cfg_matches_fp16_accumulation(
    workspace_mb: int,
) -> None:
    torch.manual_seed(20260723 + workspace_mb)
    x = torch.randn((16, 4096), device="cuda", dtype=torch.float16)
    weight = torch.randn((4096, 128), device="cuda", dtype=torch.float16)
    op = torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg

    output = op(x, weight, workspace_mb, 0, True)
    repeated = op(x, weight, workspace_mb, 0, True)
    reference = torch.mm(x, weight, out_dtype=torch.float32)
    torch.accelerator.synchronize()

    assert output.shape == (16, 128)
    assert output.dtype == torch.float16
    assert output.device == x.device
    assert torch.isfinite(output).all()
    assert torch.equal(repeated, output)

    output_error = (output.float() - reference).norm()
    relative_l2 = output_error / reference.norm()
    cosine = torch.nn.functional.cosine_similarity(
        output.float().flatten(),
        reference.flatten(),
        dim=0,
    )
    assert relative_l2 < 5e-3
    assert cosine > 0.99999


def test_rwkv7_linear_f16_lt_cfg_graph_replays_deterministically() -> None:
    torch.manual_seed(20260723)
    x = torch.randn((2, 4, 1024), device="cuda", dtype=torch.float16)
    weight = torch.randn((1024, 512), device="cuda", dtype=torch.float16)
    op = torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg

    # The plan is deliberately warmed outside capture; a cache miss inside
    # capture fails closed instead of creating host-side Lt descriptors.
    eager_output = op(x, weight, 0, 0, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(x, weight, 0, 0, True)
    graph.replay()
    first_replay = graph_output.clone()
    graph.replay()
    second_replay = graph_output.clone()
    reference = torch.mm(
        x.reshape(-1, x.shape[-1]),
        weight,
        out_dtype=torch.float32,
    ).reshape(2, 4, 512)
    torch.accelerator.synchronize()

    assert torch.equal(first_replay, second_replay)
    assert torch.equal(first_replay, eager_output)
    graph_error = (first_replay.float() - reference).norm()
    relative_l2 = graph_error / reference.norm()
    cosine = torch.nn.functional.cosine_similarity(
        first_replay.float().flatten(),
        reference.flatten(),
        dim=0,
    )
    assert relative_l2 < 5e-3
    assert cosine > 0.99999


def test_rwkv7_linear_f16_lt_cfg_empty_and_invalid_inputs() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg
    empty_rows = op(
        torch.empty((0, 3, 8), device="cuda", dtype=torch.float16),
        torch.empty((8, 5), device="cuda", dtype=torch.float16),
        0,
        0,
        True,
    )
    empty_k = op(
        torch.empty((2, 0), device="cuda", dtype=torch.float16),
        torch.empty((0, 5), device="cuda", dtype=torch.float16),
        0,
        0,
        True,
    )

    assert empty_rows.shape == (0, 3, 5)
    assert empty_rows.dtype == torch.float16
    assert empty_k.shape == (2, 5)
    assert torch.count_nonzero(empty_k) == 0

    x = torch.empty((8, 16), device="cuda", dtype=torch.float16)
    weight = torch.empty((16, 32), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(x.t(), torch.empty((8, 32), device="cuda", dtype=torch.float16), 0, 0)
    with pytest.raises(RuntimeError, match="fp16"):
        op(x.float(), weight, 0, 0)
    with pytest.raises(RuntimeError, match="workspace_mb"):
        op(x, weight, -1, 0)
    with pytest.raises(RuntimeError, match="workspace_mb"):
        op(x, weight, 129, 0)
    with pytest.raises(RuntimeError, match="heuristic_index"):
        op(x, weight, 0, -1)
    with pytest.raises(RuntimeError, match="heuristic_index"):
        op(x, weight, 0, 64)
    with pytest.raises(RuntimeError, match="requested heuristic index"):
        op(x, weight, 0, 63, True)


def test_rwkv7_linear_fp32_lt_3d_graph_matches_fp32_reference() -> None:
    torch.manual_seed(20260723)
    x = torch.randn((2, 3, 256), device="cuda", dtype=torch.float16)
    weight = torch.randn((256, 512), device="cuda", dtype=torch.float16)
    op = torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt

    # Warm the shape-specific Lt plan before graph capture.
    output = op(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(x, weight)
    graph.replay()
    first_replay = graph_output.clone()
    graph.replay()
    second_replay = graph_output.clone()

    reference = (x.float().reshape(-1, x.shape[-1]) @ weight.float()).reshape(2, 3, 512)
    torch.accelerator.synchronize()

    assert output.shape == (2, 3, 512)
    assert output.dtype == torch.float32
    assert output.device == x.device
    assert torch.equal(first_replay, second_replay)
    torch.testing.assert_close(output, reference, rtol=1e-3, atol=2e-2)
    torch.testing.assert_close(first_replay, reference, rtol=1e-3, atol=2e-2)


def test_rwkv7_linear_fp32_lt_empty_shapes_keep_fp32_contract() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt
    empty_rows = op(
        torch.empty((0, 3, 8), device="cuda", dtype=torch.float16),
        torch.empty((8, 5), device="cuda", dtype=torch.float16),
    )
    empty_k = op(
        torch.empty((2, 0), device="cuda", dtype=torch.float16),
        torch.empty((0, 5), device="cuda", dtype=torch.float16),
    )

    assert empty_rows.shape == (0, 3, 5)
    assert empty_rows.dtype == torch.float32
    assert empty_rows.device.type == "cuda"
    assert empty_k.shape == (2, 5)
    assert empty_k.dtype == torch.float32
    assert torch.count_nonzero(empty_k) == 0


def test_rwkv7_linear_fp32_lt_rejects_invalid_inputs() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt
    x = torch.empty((2, 8), device="cuda", dtype=torch.float16)
    weight = torch.empty((8, 5), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="x must be fp16"):
        op(x.float(), weight)
    with pytest.raises(RuntimeError, match="weight must be fp16"):
        op(x, weight.float())
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        op(torch.empty((8, 2), device="cuda", dtype=torch.float16).t(), weight)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        op(x, torch.empty((7, 5), device="cuda", dtype=torch.float16))


def test_rwkv7_linear_fp32_lt_rejects_cross_device() -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    x = torch.empty((2, 8), device="cuda:0", dtype=torch.float16)
    weight = torch.empty((8, 5), device="cuda:1", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt(x, weight)


def test_rwkv7_linear_fp32_lt_supports_misaligned_contiguous_views() -> None:
    torch.manual_seed(20260723)
    x = torch.randn(
        (2 * 256 + 1,),
        device="cuda",
        dtype=torch.float16,
    )[1:].view(2, 256)
    weight = torch.randn(
        (256 * 512 + 1,),
        device="cuda",
        dtype=torch.float16,
    )[1:].view(256, 512)
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.data_ptr() % 256 != 0
    assert weight.data_ptr() % 256 != 0
    op = torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt

    # Populate the same-shape cache with the allocator-aligned plan first.
    op(x.clone(), weight.clone())
    output = op(x, weight)
    reference = x.float() @ weight.float()
    torch.accelerator.synchronize()

    assert output.dtype == torch.float32
    assert output.device == x.device
    torch.testing.assert_close(output, reference, rtol=1e-3, atol=2e-2)


def test_rwkv7_tmix_kk_a_gate_2d_matches_flat_grid() -> None:
    torch.manual_seed(20260723)
    batch_size, token_count, hidden_size, num_heads = 64, 1, 4096, 64
    k = torch.randn(
        (batch_size, token_count, hidden_size),
        device="cuda",
        dtype=torch.float16,
    )
    k_k = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    a0 = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    a12 = torch.randn_like(k)
    k_a = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    args = (batch_size, token_count, hidden_size, num_heads, k, k_k, a0, a12, k_a)

    flat = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_kk_a_gate(*args)
    grid_2d = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_kk_a_gate_2d(*args)
    repeated = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_kk_a_gate_2d(*args)
    torch.accelerator.synchronize()

    for expected, actual, again in zip(flat, grid_2d, repeated, strict=True):
        assert torch.equal(actual, expected)
        assert torch.equal(again, actual)


def test_rwkv7_tmix_lnx_warp_matches_two_warp_kernel() -> None:
    torch.manual_seed(20260723)
    batch_size, token_count, hidden_size, num_heads = 64, 1, 4096, 64
    shape = (batch_size, token_count, hidden_size)
    x, r, k, v, g = [
        0.1 * torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(5)
    ]
    r_k, weight, bias = [
        0.1 * torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
        for _ in range(3)
    ]
    args = (
        batch_size,
        token_count,
        hidden_size,
        num_heads,
        x,
        r,
        k,
        v,
        r_k,
        weight,
        bias,
        g,
    )

    two_warp = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_lnx_rkvres_xg(*args)
    one_warp = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_lnx_rkvres_xg_warp(*args)
    repeated = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_lnx_rkvres_xg_warp(*args)
    torch.accelerator.synchronize()

    assert torch.equal(repeated, one_warp)
    torch.testing.assert_close(one_warp, two_warp, rtol=2e-3, atol=2e-3)


def _tmix_vres_gate_reference(
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    gate = torch.sigmoid(v0.float().view(1, 1, -1) + v12.float())
    return (v.float() + (v_first.float() - v.float()) * gate).half()


@pytest.mark.parametrize("batch_size", [64, 255, 256, 320])
def test_rwkv7_tmix_vres_gate_tuned_vec2_matches_scalar_and_reference(
    batch_size: int,
) -> None:
    torch.manual_seed(20260724 + batch_size)
    token_count, hidden_size = 1, 4096
    shape = (batch_size, token_count, hidden_size)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    v_first = torch.randn_like(v)
    v0 = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    v12 = torch.randn_like(v)
    op = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_vres_gate

    output = op(batch_size, token_count, hidden_size, v, v_first, v0, v12)
    scalar_chunks = []
    for start in range(0, batch_size, 63):
        end = min(start + 63, batch_size)
        scalar_chunks.append(
            op(
                end - start,
                token_count,
                hidden_size,
                v[start:end],
                v_first[start:end],
                v0,
                v12[start:end],
            )
        )
    scalar = torch.cat(scalar_chunks, dim=0)
    repeated = op(batch_size, token_count, hidden_size, v, v_first, v0, v12)
    reference = _tmix_vres_gate_reference(v, v_first, v0, v12)
    torch.accelerator.synchronize()

    assert output.shape == shape
    assert output.dtype == torch.float16
    assert output.device.type == "cuda"
    assert torch.equal(output, scalar)
    assert torch.equal(repeated, output)
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(
    ("batch_size", "token_count", "hidden_size"),
    [(63, 1, 4096), (320, 1, 128)],
)
def test_rwkv7_tmix_vres_gate_fallback_matches_reference(
    batch_size: int,
    token_count: int,
    hidden_size: int,
) -> None:
    torch.manual_seed(20260724 + batch_size + hidden_size)
    shape = (batch_size, token_count, hidden_size)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    v_first = torch.randn_like(v)
    v0 = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)
    v12 = torch.randn_like(v)
    op = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_vres_gate
    args = (batch_size, token_count, hidden_size, v, v_first, v0, v12)

    output = op(*args)
    repeated = op(*args)
    reference = _tmix_vres_gate_reference(v, v_first, v0, v12)
    torch.accelerator.synchronize()

    assert torch.equal(repeated, output)
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)


def _rkv_m1_inputs(
    hidden_size: int,
    out_features: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    input_shapes = (
        (1, hidden_size),
        (1, 1, hidden_size),
        (1, 1, 1, hidden_size),
    )
    inputs = tuple(
        0.1 * torch.randn(shape, device="cuda", dtype=torch.float16)
        for shape in input_shapes
    )
    weights = tuple(
        0.1
        * torch.randn(
            (hidden_size, out_features),
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(3)
    )
    return inputs, weights


def _misaligned_contiguous_half(shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    storage = torch.empty((numel + 1,), device="cuda", dtype=torch.float16)
    result = storage[1:].view(shape)
    assert result.storage_offset() == 1
    assert result.is_contiguous()
    assert result.data_ptr() % 4 == 2
    return result


@pytest.mark.parametrize(
    "op_name",
    [
        "linear_f16_m1_splitk",
        "linear_f16_m1_splitk_fp32",
    ],
)
def test_rwkv7_m1_splitk_rejects_misaligned_contiguous_weight(
    op_name: str,
) -> None:
    x = torch.empty((1, 8), device="cuda", dtype=torch.float16)
    weight = _misaligned_contiguous_half((8, 64))

    with pytest.raises(RuntimeError, match="4-byte aligned"):
        getattr(torch.ops.rwkv7_v3a_ops, op_name)(x, weight)


@pytest.mark.parametrize(
    ("weight_index", "weight_name"),
    [(0, "weight_r"), (1, "weight_k"), (2, "weight_v")],
)
def test_rwkv7_rkv_m1_splitk_rejects_each_misaligned_contiguous_weight(
    weight_index: int,
    weight_name: str,
) -> None:
    inputs = tuple(
        torch.empty((1, 8), device="cuda", dtype=torch.float16) for _ in range(3)
    )
    weights = [
        torch.empty((8, 64), device="cuda", dtype=torch.float16) for _ in range(3)
    ]
    weights[weight_index] = _misaligned_contiguous_half((8, 64))

    with pytest.raises(
        RuntimeError,
        match=rf"{weight_name} must be 4-byte aligned",
    ):
        torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk(
            *inputs,
            *weights,
        )


@pytest.mark.parametrize(
    ("hidden_size", "out_features", "zero_features"),
    [
        pytest.param(1024, 4096, 4096, id="ordinary-reducer"),
        pytest.param(4096, 4096, 4096, id="warp-reducer"),
        pytest.param(1024, 4096, 0, id="empty-zero-output"),
    ],
)
def test_rwkv7_m1_splitk_prepare_zero_matches_preact_bitwise_and_writes_plus_zero(
    hidden_size: int,
    out_features: int,
    zero_features: int,
) -> None:
    torch.manual_seed(20260731 + hidden_size + zero_features)
    x = torch.randn(
        (1, 1, hidden_size),
        device="cuda",
        dtype=torch.float16,
    )
    weight = torch.randn(
        (hidden_size, out_features),
        device="cuda",
        dtype=torch.float16,
    )
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero

    preact, zero_output = op(x, weight, zero_features)
    repeated_preact, repeated_zero = op(x, weight, zero_features)
    baseline = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x, weight)
    torch.accelerator.synchronize()

    assert torch.equal(preact, baseline)
    assert torch.equal(repeated_preact, preact)
    assert zero_output.shape == (1, 1, zero_features)
    assert zero_output.dtype == torch.float16
    assert zero_output.device == x.device
    assert zero_output.is_contiguous()
    assert torch.count_nonzero(zero_output.view(torch.int16)) == 0
    assert torch.count_nonzero(repeated_zero.view(torch.int16)) == 0


def test_rwkv7_m1_splitk_prepare_zero_graph_replays_consecutively() -> None:
    torch.manual_seed(20260731)
    x = torch.randn((1, 1, 1024), device="cuda", dtype=torch.float16)
    weight = torch.randn((1024, 4096), device="cuda", dtype=torch.float16)
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero
    baseline = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x, weight)

    eager_preact, eager_zero = op(x, weight, 4096)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_preact, graph_zero = op(x, weight, 4096)

    replays = []
    for _ in range(3):
        graph_zero.fill_(1.0)
        graph.replay()
        replays.append(
            (
                graph_preact.clone(),
                graph_zero.view(torch.int16).clone(),
            )
        )
    torch.accelerator.synchronize()

    assert torch.equal(eager_preact, baseline)
    assert torch.count_nonzero(eager_zero.view(torch.int16)) == 0
    for preact, zero_bits in replays:
        assert torch.equal(preact, baseline)
        assert torch.count_nonzero(zero_bits) == 0
    assert torch.equal(replays[0][0], replays[1][0])
    assert torch.equal(replays[1][0], replays[2][0])


@pytest.mark.parametrize("F", [128, 256])
def test_rwkv7_cmix_sparse_down_relu_one_out_matches_allocating_op(
    F: int,
) -> None:
    torch.manual_seed(20260731 + F)
    C = 256
    preact = torch.randn((F,), device="cuda", dtype=torch.float16) * 0.25
    value_fc = torch.randn((F, C), device="cuda", dtype=torch.float16) * 0.05
    expected = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one(
        C,
        F,
        preact,
        value_fc,
    )
    actual = torch.zeros((1, 1, C), device="cuda", dtype=torch.float16)

    result = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out(
        C,
        F,
        preact,
        value_fc,
        actual,
    )
    torch.accelerator.synchronize()

    assert result is None
    if F == 128:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-3,
            atol=2e-3,
        )
    reference = torch.relu(preact.float()).square() @ value_fc.float()
    torch.testing.assert_close(
        actual.float().view(-1),
        reference,
        rtol=2e-2,
        atol=2e-2,
    )


def test_rwkv7_cmix_m1_prepare_zero_full_graph_replays_consecutively() -> None:
    torch.manual_seed(20260731)
    C, F = 256, 128
    x = torch.randn((1, 1, C), device="cuda", dtype=torch.float16)
    key_weight = torch.randn((C, F), device="cuda", dtype=torch.float16)
    value_weight = torch.randn((F, C), device="cuda", dtype=torch.float16) * 0.05
    prepare = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero
    sparse_out = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out
    baseline_preact = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(
        x,
        key_weight,
    )
    baseline = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one(
        C,
        F,
        baseline_preact.view(-1),
        value_weight,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_preact, graph_out = prepare(x, key_weight, C)
        sparse_out(
            C,
            F,
            graph_preact.view(-1),
            value_weight,
            graph_out,
        )

    replays = []
    for _ in range(3):
        graph_out.fill_(1.0)
        graph.replay()
        replays.append(graph_out.clone())
    torch.accelerator.synchronize()

    for replay in replays:
        assert torch.equal(replay, baseline)


def test_rwkv7_cmix_sparse_down_relu_one_out_fake_meta_and_schema() -> None:
    op = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out
    args = (
        256,
        128,
        _h("meta", (128,)),
        _h("meta", (128, 256)),
        _h("meta", (1, 1, 256)),
    )

    assert op(*args) is None
    torch.library.opcheck(
        op,
        (
            256,
            128,
            _h("cuda", (128,)),
            _h("cuda", (128, 256)),
            torch.zeros((1, 1, 256), device="cuda", dtype=torch.float16),
        ),
        test_utils=("test_schema",),
    )


def test_rwkv7_cmix_sparse_down_relu_one_out_bad_output_fails_closed() -> None:
    op = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out
    preact = _h("cuda", (128,))
    value_fc = _h("cuda", (128, 256))

    with pytest.raises(RuntimeError, match="C / 256 exceeds"):
        op(
            256 * (65535 + 1),
            128,
            preact,
            value_fc,
            _h("cuda", (1, 1, 256)),
        )
    with pytest.raises(RuntimeError, match="F / 128 exceeds"):
        op(
            256,
            128 * (65535 + 1),
            preact,
            value_fc,
            _h("cuda", (1, 1, 256)),
        )
    with pytest.raises(RuntimeError, match="C must be divisible by 256"):
        torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one(
            128,
            128,
            preact,
            _h("cuda", (128, 128)),
        )
    with pytest.raises(RuntimeError, match="C must be divisible by 256"):
        op(
            128,
            128,
            preact,
            _h("cuda", (128, 128)),
            _h("cuda", (1, 1, 128)),
        )
    with pytest.raises(RuntimeError, match=r"shape \[1,1,C\]"):
        op(256, 128, preact, value_fc, _h("cuda", (256,)))
    with pytest.raises(RuntimeError, match="out must be fp16"):
        op(
            256,
            128,
            preact,
            value_fc,
            torch.empty((1, 1, 256), device="cuda", dtype=torch.float32),
        )
    with pytest.raises(RuntimeError, match="out must be contiguous"):
        op(
            256,
            128,
            preact,
            value_fc,
            _h("cuda", (1, 1, 512))[:, :, ::2],
        )
    with pytest.raises(RuntimeError, match="aligned for fp16x2"):
        op(
            256,
            128,
            preact,
            value_fc,
            _misaligned_contiguous_half((1, 1, 256)),
        )
    shared_preact_out = _h("cuda", (256,))
    with pytest.raises(RuntimeError, match="out must not overlap preact"):
        op(
            256,
            128,
            shared_preact_out[:128],
            value_fc,
            shared_preact_out.view(1, 1, 256),
        )
    shared_value_out = _h("cuda", (128 * 256,))
    with pytest.raises(RuntimeError, match="out must not overlap value_fc"):
        op(
            256,
            128,
            preact,
            shared_value_out.view(128, 256),
            shared_value_out[:256].view(1, 1, 256),
        )


def test_rwkv7_m1_splitk_prepare_zero_empty_inputs() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero
    preact, zero_output = op(
        torch.empty((1, 1, 0), device="cuda", dtype=torch.float16),
        torch.empty((0, 64), device="cuda", dtype=torch.float16),
        16,
    )
    assert preact.shape == (1, 1, 64)
    assert torch.count_nonzero(preact.view(torch.int16)) == 0
    assert zero_output.shape == (1, 1, 16)
    assert torch.count_nonzero(zero_output.view(torch.int16)) == 0

    empty_preact, prepared_zero = op(
        torch.empty((1, 1, 8), device="cuda", dtype=torch.float16),
        torch.empty((8, 0), device="cuda", dtype=torch.float16),
        16,
    )
    assert empty_preact.shape == (1, 1, 0)
    assert empty_preact.is_contiguous()
    assert torch.count_nonzero(prepared_zero.view(torch.int16)) == 0


def test_rwkv7_m1_splitk_prepare_zero_bad_inputs_fail_closed() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero
    x = torch.empty((1, 8), device="cuda", dtype=torch.float16)
    weight = torch.empty((8, 64), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="non-negative"):
        op(x, weight, -8)
    with pytest.raises(RuntimeError, match="multiple of 8"):
        op(x, weight, 7)
    with pytest.raises(RuntimeError, match="too large"):
        op(x, weight, 2**31)
    with pytest.raises(RuntimeError, match="x must be fp16"):
        op(x.float(), weight, 8)
    with pytest.raises(RuntimeError, match="weight must be fp16"):
        op(x, weight.float(), 8)
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        op(
            torch.empty((1, 16), device="cuda", dtype=torch.float16)[:, ::2],
            weight,
            8,
        )
    with pytest.raises(RuntimeError, match="weight must be contiguous"):
        op(
            x,
            torch.empty((64, 8), device="cuda", dtype=torch.float16).t(),
            8,
        )
    with pytest.raises(RuntimeError, match="4-byte aligned"):
        op(x, _misaligned_contiguous_half((8, 64)), 8)
    with pytest.raises(RuntimeError, match="M=1"):
        op(x.expand(2, -1).contiguous(), weight, 8)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        op(
            x,
            torch.empty((9, 64), device="cuda", dtype=torch.float16),
            8,
        )
    with pytest.raises(RuntimeError, match="N multiple of 64"):
        op(
            x,
            torch.empty((8, 65), device="cuda", dtype=torch.float16),
            8,
        )


@pytest.mark.parametrize(
    ("hidden_size", "out_features"),
    [
        pytest.param(4096, 4096, id="warp-reduce"),
        pytest.param(256, 65536, id="large-n"),
        pytest.param(4096, 16384, id="k4096-n16384"),
        pytest.param(8192, 4096, id="large-k"),
        pytest.param(1024, 4096, id="generic"),
    ],
)
def test_rwkv7_rkv_m1_splitk_is_deterministic_and_matches_fp32_reference(
    hidden_size: int,
    out_features: int,
) -> None:
    torch.manual_seed(20260730 + hidden_size + out_features)
    inputs, weights = _rkv_m1_inputs(hidden_size, out_features)
    grouped_op = torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk
    independent_op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk

    grouped = grouped_op(*inputs, *weights)
    repeated = grouped_op(*inputs, *weights)
    independent = tuple(
        independent_op(value, weight)
        for value, weight in zip(inputs, weights, strict=True)
    )
    references = tuple(
        value.float() @ weight.float()
        for value, weight in zip(inputs, weights, strict=True)
    )
    torch.accelerator.synchronize()

    assert isinstance(grouped, tuple)
    for output, again, baseline, reference in zip(
        grouped,
        repeated,
        independent,
        references,
        strict=True,
    ):
        assert output.shape == reference.shape
        assert output.dtype == torch.float16
        assert output.device == reference.device
        assert output.is_contiguous()
        assert torch.isfinite(output).all()
        assert torch.equal(output, again)
        if (hidden_size, out_features) == (4096, 4096):
            # The exact grouped-RKV path has a dedicated split-K schedule, so
            # its FP32 additions need not be bitwise identical to three
            # independently scheduled projections.
            torch.testing.assert_close(
                output,
                baseline,
                rtol=2e-3,
                atol=2e-3,
            )
        else:
            assert torch.equal(output, baseline)

        error = output.float() - reference
        relative_l2 = error.norm() / reference.norm()
        cosine = torch.nn.functional.cosine_similarity(
            output.float().flatten(),
            reference.flatten(),
            dim=0,
        )
        assert error.abs().max() < 1e-2
        assert relative_l2 < 1e-3
        assert cosine > 0.999999
        torch.testing.assert_close(
            output.float(),
            reference,
            rtol=2e-3,
            atol=2e-3,
        )


def test_rwkv7_rkv_m1_splitk_graph_is_repeatable_and_matches_eager() -> None:
    torch.manual_seed(20260730)
    inputs, weights = _rkv_m1_inputs(1024, 4096)
    op = torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk
    independent_op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk

    eager = op(*inputs, *weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = op(*inputs, *weights)
    graph.replay()
    first_replay = tuple(output.clone() for output in graph_outputs)
    graph.replay()
    second_replay = tuple(output.clone() for output in graph_outputs)
    independent = tuple(
        independent_op(value, weight)
        for value, weight in zip(inputs, weights, strict=True)
    )
    torch.accelerator.synchronize()

    for eager_output, first, second, baseline in zip(
        eager,
        first_replay,
        second_replay,
        independent,
        strict=True,
    ):
        assert torch.equal(eager_output, baseline)
        assert torch.equal(first, eager_output)
        assert torch.equal(second, first)


def test_rwkv7_rkv_m1_splitk_empty_inputs_preserve_each_leading_shape() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk
    empty_inputs = (
        torch.empty((1, 0), device="cuda", dtype=torch.float16),
        torch.empty((1, 1, 0), device="cuda", dtype=torch.float16),
        torch.empty((1, 1, 1, 0), device="cuda", dtype=torch.float16),
    )
    empty_weights = tuple(
        torch.empty((0, 64), device="cuda", dtype=torch.float16) for _ in range(3)
    )

    outputs = op(*empty_inputs, *empty_weights)

    assert tuple(output.shape for output in outputs) == (
        (1, 64),
        (1, 1, 64),
        (1, 1, 1, 64),
    )
    for output in outputs:
        assert output.dtype == torch.float16
        assert output.device.type == "cuda"
        assert output.is_contiguous()
        assert torch.count_nonzero(output) == 0

    empty_vocab_inputs = (
        torch.empty((1, 8), device="cuda", dtype=torch.float16),
        torch.empty((1, 1, 8), device="cuda", dtype=torch.float16),
        torch.empty((1, 1, 1, 8), device="cuda", dtype=torch.float16),
    )
    empty_vocab_weights = tuple(
        torch.empty((8, 0), device="cuda", dtype=torch.float16) for _ in range(3)
    )
    empty_vocab_outputs = op(*empty_vocab_inputs, *empty_vocab_weights)
    assert tuple(output.shape for output in empty_vocab_outputs) == (
        (1, 0),
        (1, 1, 0),
        (1, 1, 1, 0),
    )
    for output in empty_vocab_outputs:
        assert output.dtype == torch.float16
        assert output.device.type == "cuda"
        assert output.is_contiguous()


def test_rwkv7_rkv_m1_splitk_invalid_inputs_fail_closed() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk
    inputs = tuple(
        torch.empty((1, 8), device="cuda", dtype=torch.float16) for _ in range(3)
    )
    weights = tuple(
        torch.empty((8, 64), device="cuda", dtype=torch.float16) for _ in range(3)
    )

    with pytest.raises(RuntimeError, match="M=1"):
        op(inputs[0].expand(2, -1).contiguous(), *inputs[1:], *weights)
    with pytest.raises(RuntimeError, match="input/weight shape mismatch"):
        op(
            *inputs,
            torch.empty((9, 64), device="cuda", dtype=torch.float16),
            *weights[1:],
        )
    with pytest.raises(RuntimeError, match="common K"):
        op(
            inputs[0],
            torch.empty((1, 16), device="cuda", dtype=torch.float16),
            inputs[2],
            weights[0],
            torch.empty((16, 64), device="cuda", dtype=torch.float16),
            weights[2],
        )
    with pytest.raises(RuntimeError, match="common N"):
        op(
            *inputs,
            weights[0],
            torch.empty((8, 128), device="cuda", dtype=torch.float16),
            weights[2],
        )
    with pytest.raises(RuntimeError, match="N multiple of 64"):
        op(
            *inputs,
            *(
                torch.empty((8, 65), device="cuda", dtype=torch.float16)
                for _ in range(3)
            ),
        )
    with pytest.raises(RuntimeError, match="x_k must be fp16"):
        op(inputs[0], inputs[1].float(), inputs[2], *weights)
    with pytest.raises(RuntimeError, match="weight_v must be fp16"):
        op(*inputs, *weights[:2], weights[2].float())
    with pytest.raises(RuntimeError, match="x_v must be contiguous"):
        noncontiguous_x = torch.empty((1, 16), device="cuda", dtype=torch.float16)[
            :, ::2
        ]
        op(inputs[0], inputs[1], noncontiguous_x, *weights)
    with pytest.raises(RuntimeError, match="weight_k must be contiguous"):
        noncontiguous_weight = torch.empty(
            (64, 8), device="cuda", dtype=torch.float16
        ).t()
        op(*inputs, weights[0], noncontiguous_weight, weights[2])


def test_rwkv7_m1_splitk_rejects_chunk_grid_y_overflow() -> None:
    hidden_size = 512 * 65535 + 1
    x = torch.empty((1, hidden_size), device="cuda", dtype=torch.float16)
    weight = torch.empty(
        (hidden_size, 0),
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(RuntimeError, match="chunks must be <= 65535"):
        torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x, weight)
    with pytest.raises(RuntimeError, match="chunks must be <= 65535"):
        torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32(x, weight)
    with pytest.raises(RuntimeError, match="chunks must be <= 65535"):
        torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero(
            x,
            weight,
            8,
        )
    with pytest.raises(RuntimeError, match="chunks must be <= 65535"):
        torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk(
            x,
            x,
            x,
            weight,
            weight,
            weight,
        )


def test_rwkv7_rkv_m1_splitk_uses_input_device_stream() -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    inputs = tuple(
        torch.randn((1, 1024), device="cuda:1", dtype=torch.float16) for _ in range(3)
    )
    weights = tuple(
        torch.randn((1024, 4096), device="cuda:1", dtype=torch.float16)
        for _ in range(3)
    )
    torch.accelerator.synchronize(1)
    stream = torch.cuda.Stream(device=1)
    assert stream.cuda_stream != torch.cuda.default_stream(1).cuda_stream

    with torch.accelerator.device_index(0), torch.cuda.stream(stream):
        assert torch.cuda.current_stream(1).cuda_stream == stream.cuda_stream
        outputs = torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk(
            *inputs,
            *weights,
        )
    stream.synchronize()
    references = tuple(
        torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(value, weight)
        for value, weight in zip(inputs, weights, strict=True)
    )
    torch.accelerator.synchronize(1)

    for output, reference in zip(outputs, references, strict=True):
        assert output.device == inputs[0].device
        assert torch.equal(output, reference)

    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk(
            *inputs,
            *weights[:2],
            weights[2].to("cuda:0"),
        )


def test_rwkv7_m1_splitk_prepare_zero_uses_cuda1_non_default_stream() -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    x = torch.randn((1, 1024), device="cuda:1", dtype=torch.float16)
    weight = torch.randn((1024, 4096), device="cuda:1", dtype=torch.float16)
    torch.accelerator.synchronize(1)
    stream = torch.cuda.Stream(device=1)
    assert stream.cuda_stream != torch.cuda.default_stream(1).cuda_stream

    with torch.accelerator.device_index(0), torch.cuda.stream(stream):
        assert torch.cuda.current_stream(1).cuda_stream == stream.cuda_stream
        preact, zero_output = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero(
            x,
            weight,
            4096,
        )
    stream.synchronize()
    baseline = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x, weight)
    torch.accelerator.synchronize(1)

    assert preact.device == x.device
    assert zero_output.device == x.device
    assert torch.equal(preact, baseline)
    assert torch.count_nonzero(zero_output.view(torch.int16)) == 0

    with pytest.raises(RuntimeError, match="same device"):
        torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero(
            x,
            weight.to("cuda:0"),
            4096,
        )


def test_rwkv7_cmix_sparse_down_out_uses_cuda1_non_default_stream() -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    C, F = 256, 128
    preact = torch.randn((F,), device="cuda:1", dtype=torch.float16)
    value_fc = torch.randn((F, C), device="cuda:1", dtype=torch.float16)
    out = torch.zeros((1, 1, C), device="cuda:1", dtype=torch.float16)
    stream = torch.cuda.Stream(device=1)
    assert stream.cuda_stream != torch.cuda.default_stream(1).cuda_stream

    with torch.accelerator.device_index(0), torch.cuda.stream(stream):
        assert torch.cuda.current_stream(1).cuda_stream == stream.cuda_stream
        torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out(
            C,
            F,
            preact,
            value_fc,
            out,
        )
    stream.synchronize()
    expected = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one(
        C,
        F,
        preact,
        value_fc,
    )
    torch.accelerator.synchronize(1)

    assert torch.equal(out, expected)
    with pytest.raises(RuntimeError, match="same CUDA device"):
        torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_sparse_down_relu_one_out(
            C,
            F,
            preact,
            value_fc,
            out.to("cuda:0"),
        )


@pytest.mark.parametrize(
    ("hidden_size", "vocab_size"),
    [(1024, 4096), (4096, 4096), (1024, 65536)],
)
def test_rwkv7_m1_splitk_fp32_is_at_least_as_accurate_as_fp16_output(
    hidden_size: int,
    vocab_size: int,
) -> None:
    torch.manual_seed(20260723 + hidden_size + vocab_size)
    x = torch.randn((1, hidden_size), device="cuda", dtype=torch.float16)
    weight = torch.randn(
        (hidden_size, vocab_size),
        device="cuda",
        dtype=torch.float16,
    )
    fp16_output = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x, weight)
    fp32_output = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32(x, weight)
    repeated = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32(x, weight)
    reference = x.float() @ weight.float()
    torch.accelerator.synchronize()

    assert fp32_output.shape == reference.shape
    assert fp32_output.dtype == torch.float32
    assert fp32_output.device == reference.device
    assert torch.isfinite(fp32_output).all()
    assert torch.equal(repeated, fp32_output)

    fp16_error = (fp16_output.float() - reference).norm()
    fp32_error = (fp32_output - reference).norm()
    max_error = (fp32_output - reference).abs().max()
    relative_l2 = fp32_error / reference.norm()
    cosine = torch.nn.functional.cosine_similarity(
        fp32_output.flatten(),
        reference.flatten(),
        dim=0,
    )
    assert fp32_error <= fp16_error
    assert max_error < 5e-2
    assert relative_l2 < 5e-5
    assert cosine > 0.999999
    torch.testing.assert_close(fp32_output, reference, rtol=2e-3, atol=2e-2)


def test_rwkv7_m1_splitk_fp32_graph_matches_fp32_reference() -> None:
    torch.manual_seed(20260723)
    x = torch.randn((1, 1024), device="cuda", dtype=torch.float16)
    weight = torch.randn((1024, 4096), device="cuda", dtype=torch.float16)
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32

    output = op(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = op(x, weight)
    graph.replay()
    first_replay = graph_output.clone()
    graph.replay()
    second_replay = graph_output.clone()

    reference = x.float() @ weight.float()
    torch.accelerator.synchronize()

    assert graph_output.shape == reference.shape
    assert graph_output.dtype == torch.float32
    assert graph_output.device == x.device
    assert torch.equal(first_replay, second_replay)
    torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-2)
    torch.testing.assert_close(first_replay, reference, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize(
    ("op_name", "output_dtype"),
    [
        ("linear_f16_m1_splitk", torch.float16),
        ("linear_f16_m1_splitk_fp32", torch.float32),
    ],
)
def test_rwkv7_m1_splitk_uses_input_device_non_default_stream(
    op_name: str,
    output_dtype: torch.dtype,
) -> None:
    if torch.accelerator.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    x = torch.randn((1, 1024), device="cuda:1", dtype=torch.float16)
    weight = torch.randn((1024, 4096), device="cuda:1", dtype=torch.float16)
    torch.accelerator.synchronize(1)
    stream = torch.cuda.Stream(device=1)
    assert stream.cuda_stream != torch.cuda.default_stream(1).cuda_stream

    with torch.accelerator.device_index(0), torch.cuda.stream(stream):
        assert torch.cuda.current_stream(1).cuda_stream == stream.cuda_stream
        output = getattr(torch.ops.rwkv7_v3a_ops, op_name)(x, weight)
    stream.synchronize()
    reference = x.float() @ weight.float()

    assert output.device == x.device
    assert output.dtype == output_dtype
    torch.testing.assert_close(
        output.float(),
        reference,
        rtol=2e-3,
        atol=2e-2,
    )


def test_rwkv7_m1_splitk_fp32_empty_and_invalid_inputs() -> None:
    op = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32
    empty = op(
        torch.empty((1, 0), device="cuda", dtype=torch.float16),
        torch.empty((0, 64), device="cuda", dtype=torch.float16),
    )
    assert empty.shape == (1, 64)
    assert empty.dtype == torch.float32
    assert torch.count_nonzero(empty) == 0
    empty_vocab = op(
        torch.empty((1, 8), device="cuda", dtype=torch.float16),
        torch.empty((8, 0), device="cuda", dtype=torch.float16),
    )
    assert empty_vocab.shape == (1, 0)
    assert empty_vocab.dtype == torch.float32
    assert empty_vocab.device.type == "cuda"
    with pytest.raises(RuntimeError, match="M=1"):
        op(
            torch.empty((2, 0), device="cuda", dtype=torch.float16),
            torch.empty((0, 64), device="cuda", dtype=torch.float16),
        )

    x = torch.empty((1, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="N multiple of 64"):
        op(x, torch.empty((8, 65), device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="x must be fp16"):
        op(x.float(), torch.empty((8, 64), device="cuda", dtype=torch.float16))
    with pytest.raises(RuntimeError, match="weight must be fp16"):
        op(x, torch.empty((8, 64), device="cuda", dtype=torch.float32))
    with pytest.raises(RuntimeError, match="x must be contiguous"):
        op(
            torch.empty((1, 16), device="cuda", dtype=torch.float16)[:, ::2],
            torch.empty((8, 64), device="cuda", dtype=torch.float16),
        )
    with pytest.raises(RuntimeError, match="weight must be contiguous"):
        op(
            x,
            torch.empty((64, 8), device="cuda", dtype=torch.float16).t(),
        )


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_schema_matches_meta_contract(case: OpCase) -> None:
    torch.library.opcheck(
        _op(case.name),
        case.args("meta"),
        test_utils=("test_schema",),
    )


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_fake_meta_shapes(case: OpCase) -> None:
    args = case.args("meta")
    result = _op(case.name)(*args)
    outputs = result if isinstance(result, list | tuple) else [result]
    expected = case.expected(args)

    assert len(outputs) == len(expected)
    for output, (shape, dtype) in zip(outputs, expected, strict=True):
        assert tuple(output.shape) == shape
        assert output.dtype == dtype
        assert output.device.type == "meta"


@pytest.mark.parametrize("case", RETURNING_CASES, ids=lambda c: c.name)
def test_rwkv7_returning_ops_opcheck_faketensor(case: OpCase) -> None:
    torch.library.opcheck(
        _op(case.name),
        case.args("meta"),
        test_utils=("test_faketensor",),
    )


def test_rwkv7_advance_i32_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32,
        (_i32("cuda", (4,)), 3),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_slots_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32_slots,
        (_i32("cuda", (6,)), torch.tensor([3, 0], device="cuda", dtype=torch.int32), 3),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_varlen_schema_opcheck() -> None:
    torch.library.opcheck(
        torch.ops.rwkv7_v3a_ops.advance_i32_varlen,
        (
            _i32("cuda", (6,)),
            torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
            torch.tensor([3, 0], device="cuda", dtype=torch.int32),
        ),
        test_utils=("test_schema",),
    )


def test_rwkv7_advance_i32_varlen_updates_only_mapped_slots() -> None:
    elapsed = torch.tensor([10, 20, 30, 40], device="cuda", dtype=torch.int32)
    query_start_loc = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
    slot_indices = torch.tensor([3, 0], device="cuda", dtype=torch.int32)

    torch.ops.rwkv7_v3a_ops.advance_i32_varlen(elapsed, query_start_loc, slot_indices)

    assert elapsed.cpu().tolist() == [13, 20, 30, 42]


@pytest.mark.parametrize(
    "op_name,args",
    [
        (
            "rwkv7_wkv_fp16_v2::wkv",
            lambda: (
                torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                _h("cuda", (5, 1, 64, 64)),
                _h("cuda", (5, 64)),
                _h("cuda", (5, 64)),
                _h("cuda", (64,)),
                *[_h("cuda", (5, 64)) for _ in range(4)],
                _h("cuda", (5, 64)),
                _i32("cuda", (5,)),
            ),
        ),
        (
            "rwkv7_wkv_fp32_v2::wkv",
            lambda: (
                torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32),
                torch.tensor([3, 0], device="cuda", dtype=torch.int32),
                torch.empty((5, 1, 64, 64), device="cuda", dtype=torch.float32),
                *[_h("cuda", (5, 64)) for _ in range(6)],
                _h("cuda", (5, 64)),
            ),
        ),
    ],
)
def test_rwkv7_wkv_canonical_schema_opcheck(op_name, args) -> None:
    torch.library.opcheck(_op(op_name), args(), test_utils=("test_schema",))


@pytest.mark.parametrize(
    ("batch", "tokens", "cache_rounded"),
    [(2, 4, False), (16, 16, True)],
)
def test_rwkv7_add_layer_norm_welford_matches_fp32_reference(
    batch: int,
    tokens: int,
    cache_rounded: bool,
) -> None:
    torch.manual_seed(60 + batch)
    device = "cuda"
    hidden = 4096
    eps = 1e-5
    x = torch.randn((batch, tokens, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)

    x_out, normalized = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(
        x, residual, weight, bias, eps
    )

    unrounded_sum = x.float() + residual.float()
    rounded_sum = unrounded_sum.half()
    stats_input = rounded_sum.float() if cache_rounded else unrounded_sum
    expected = torch.nn.functional.layer_norm(
        stats_input,
        (hidden,),
        weight.float(),
        bias.float(),
        eps,
    ).half()

    torch.testing.assert_close(x_out, rounded_sum, atol=0, rtol=0)
    torch.testing.assert_close(normalized, expected, atol=2e-2, rtol=2e-2)

    repeated = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(
        x, residual, weight, bias, eps
    )
    torch.testing.assert_close(repeated[0], x_out, atol=0, rtol=0)
    torch.testing.assert_close(repeated[1], normalized, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_add_layer_norm_cmix_mix_slot_matches_scattered_reference(
    hidden: int,
) -> None:
    torch.manual_seed(6)
    device = "cuda"
    batch, slots = 3, 6
    eps = 1e-5
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    ref_x_out, ref_mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
        x, residual, compact_shift_state, weight, bias, x_k, eps
    )

    x_out, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_slots(
        x, residual, shift_state, weight, bias, x_k, slot_indices, eps
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state

    torch.testing.assert_close(x_out, ref_x_out, atol=0, rtol=0)
    torch.testing.assert_close(mixed, ref_mixed, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


def test_rwkv7_add_layer_norm_cmix_mix_welford_cache_matches_fp32_reference() -> None:
    torch.manual_seed(61)
    device = "cuda"
    batch, hidden = 320, 4096
    eps = 1e-5
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)

    x_out, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
        x, residual, shift_state, weight, bias, x_k, eps
    )

    rounded_sum = (x.float() + residual.float()).half()
    normalized = (
        torch.nn.functional.layer_norm(
            rounded_sum.float(),
            (hidden,),
            weight.float(),
            bias.float(),
            eps,
        )
        .half()
        .squeeze(1)
    )
    expected_mixed = (
        (
            normalized.float()
            + (initial_shift_state.float() - normalized.float()) * x_k.float()
        )
        .half()
        .unsqueeze(1)
    )

    torch.testing.assert_close(x_out, rounded_sum, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, normalized, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(mixed, expected_mixed, atol=3e-2, rtol=2e-2)

    repeated_state = initial_shift_state.clone()
    repeated = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
        x, residual, repeated_state, weight, bias, x_k, eps
    )
    torch.testing.assert_close(repeated[0], x_out, atol=0, rtol=0)
    torch.testing.assert_close(repeated[1], mixed, atol=0, rtol=0)
    torch.testing.assert_close(repeated_state, shift_state, atol=0, rtol=0)


def test_rwkv7_add_layer_norm_cmix_mix_welford_cache_slots_match_compact() -> None:
    torch.manual_seed(62)
    device = "cuda"
    batch, slots, hidden = 320, 640, 4096
    eps = 1e-5
    slot_indices = torch.arange(
        slots - 1, slots - batch - 1, -1, device=device, dtype=torch.int32
    )
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    ref_x_out, ref_mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
        x, residual, compact_shift_state, weight, bias, x_k, eps
    )
    x_out, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_slots(
        x, residual, shift_state, weight, bias, x_k, slot_indices, eps
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state
    torch.testing.assert_close(x_out, ref_x_out, atol=0, rtol=0)
    torch.testing.assert_close(mixed, ref_mixed, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_add_layer_norm_tmix_mix6_slot_matches_scattered_reference(
    hidden: int,
) -> None:
    torch.manual_seed(7)
    device = "cuda"
    batch, slots = 3, 6
    eps = 1e-5
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    mix_weights = [
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    ]
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    ref_outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
        x, residual, compact_shift_state, weight, bias, *mix_weights, eps
    )

    outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16_slots(
        x, residual, shift_state, weight, bias, *mix_weights, slot_indices, eps
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state

    for output, ref_output in zip(outputs, ref_outputs, strict=True):
        torch.testing.assert_close(output, ref_output, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


def test_rwkv7_add_layer_norm_tmix_mix6_welford_cache_matches_fp32_reference() -> None:
    torch.manual_seed(71)
    device = "cuda"
    batch, hidden = 320, 4096
    eps = 1e-5
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    mix_weights = [
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    ]

    outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
        x,
        residual,
        shift_state,
        weight,
        bias,
        *mix_weights,
        eps,
    )

    rounded_sum = (x.float() + residual.float()).half()
    normalized = (
        torch.nn.functional.layer_norm(
            rounded_sum.float(),
            (hidden,),
            weight.float(),
            bias.float(),
            eps,
        )
        .half()
        .squeeze(1)
    )
    expected_mixed = [
        (
            normalized.float()
            + (initial_shift_state.float() - normalized.float()) * mix.float()
        )
        .half()
        .unsqueeze(1)
        for mix in mix_weights
    ]

    torch.testing.assert_close(outputs[0], rounded_sum, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, normalized, atol=2e-2, rtol=2e-2)
    for actual, expected in zip(outputs[1:], expected_mixed, strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=2e-2)

    repeated_state = initial_shift_state.clone()
    repeated = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
        x,
        residual,
        repeated_state,
        weight,
        bias,
        *mix_weights,
        eps,
    )
    for actual, expected in zip(repeated, outputs, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(repeated_state, shift_state, atol=0, rtol=0)


def test_rwkv7_add_layer_norm_tmix_mix6_welford_cache_slots_match_compact() -> None:
    torch.manual_seed(72)
    device = "cuda"
    batch, slots, hidden = 320, 640, 4096
    eps = 1e-5
    slot_indices = torch.arange(
        slots - 1, slots - batch - 1, -1, device=device, dtype=torch.int32
    )
    x = torch.randn((batch, 1, hidden), device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    weight = torch.randn((hidden,), device=device, dtype=torch.float16)
    bias = torch.randn((hidden,), device=device, dtype=torch.float16)
    mix_weights = [
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    ]
    initial_shift_state = shift_state.clone()
    compact_shift_state = initial_shift_state[slot_indices.long()].clone()

    reference = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
        x,
        residual,
        compact_shift_state,
        weight,
        bias,
        *mix_weights,
        eps,
    )
    outputs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16_slots(
        x,
        residual,
        shift_state,
        weight,
        bias,
        *mix_weights,
        slot_indices,
        eps,
    )

    expected_shift_state = initial_shift_state.clone()
    expected_shift_state[slot_indices.long()] = compact_shift_state
    for actual, expected in zip(outputs, reference, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(shift_state, expected_shift_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_tmix_mix6_batched_shift_state_matches_reference(hidden: int) -> None:
    torch.manual_seed(hidden)
    device = "cuda"
    batch, seq_len = 4, 5
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    op = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_mix6
    args = (batch, seq_len, hidden, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g)
    outputs = op(*args)

    prev = torch.cat((initial_shift_state[:, None, :], x[:, :-1, :]), dim=1)
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(shift_state, x[:, -1, :], atol=0, rtol=0)


@pytest.mark.parametrize(("batch", "seq_len"), [(320, 1), (2, 3)])
def test_rwkv7_tmix_mix6_3d_matches_dense_state_contract(
    batch: int, seq_len: int
) -> None:
    torch.manual_seed(20260723 + batch + seq_len)
    device = "cuda"
    hidden = 4096
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    initial_shift_state = torch.randn(
        (batch, hidden), device=device, dtype=torch.float16
    )
    mix_weights = [
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    ]
    reference_state = initial_shift_state.clone()
    grid_state = initial_shift_state.clone()

    reference = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_mix6(
        batch, seq_len, hidden, x, reference_state, *mix_weights
    )
    actual = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_mix6_3d(
        batch, seq_len, hidden, x, grid_state, *mix_weights
    )
    torch.accelerator.synchronize()

    for expected, output in zip(reference, actual, strict=True):
        assert torch.equal(output, expected)
    assert torch.equal(grid_state, reference_state)
    assert torch.equal(grid_state, x[:, -1, :])


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_tmix_mix6_slot_matches_scattered_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 2)
    device = "cuda"
    batch, seq_len, slots = 3, 4, 6
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    op = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_mix6_slot
    args = (
        batch,
        seq_len,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
    )
    outputs = op(*args)

    prev = torch.cat(
        (initial_shift_state[slot_indices.long(), None, :], x[:, :-1, :]), dim=1
    )
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    expected_state[slot_indices.long()] = x[:, -1, :]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


def test_rwkv7_tmix_mix6_varlen_matches_scattered_reference() -> None:
    torch.manual_seed(4)
    device = "cuda"
    lengths = [2, 4, 1]
    batch, total_tokens, hidden, slots = len(lengths), sum(lengths), 64, 7
    query_start_loc = torch.tensor([0, 2, 6, 7], device=device, dtype=torch.int32)
    req_id = torch.tensor([0, 0, 1, 1, 1, 1, 2], device=device, dtype=torch.int32)
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((total_tokens, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_r, x_w, x_k, x_v, x_a, x_g = (
        torch.randn((hidden,), device=device, dtype=torch.float16) for _ in range(6)
    )
    initial_shift_state = shift_state.clone()

    outputs = torch.ops.vllm_rwkv7_fast_ops_fp16.tmix_mix6_varlen(
        batch,
        total_tokens,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_r,
        x_w,
        x_k,
        x_v,
        x_a,
        x_g,
        query_start_loc,
        req_id,
    )

    prev_parts = []
    for local_req, length in enumerate(lengths):
        start = int(query_start_loc[local_req].item())
        end = int(query_start_loc[local_req + 1].item())
        slot = int(slot_indices[local_req].item())
        prev_parts.append(
            torch.cat((initial_shift_state[slot : slot + 1], x[start : end - 1]))
        )
        assert end - start == length
    prev = torch.cat(prev_parts, dim=0)
    delta = prev.float() - x.float()
    for output, mix_weight in zip(
        outputs,
        (x_r, x_w, x_k, x_v, x_a, x_g),
        strict=True,
    ):
        expected = (x.float() + delta * mix_weight.float()).to(torch.float16)
        torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    for local_req, slot in enumerate(slot_indices.long().tolist()):
        end = int(query_start_loc[local_req + 1].item())
        expected_state[slot] = x[end - 1]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_cmix_mix_batched_shift_state_matches_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 1)
    device = "cuda"
    batch, seq_len = 4, 5
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((batch, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    op = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_mix
    args = (batch, seq_len, hidden, x, shift_state, x_k)
    output = op(*args)

    prev = torch.cat((initial_shift_state[:, None, :], x[:, :-1, :]), dim=1)
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(shift_state, x[:, -1, :], atol=0, rtol=0)


@pytest.mark.parametrize(("batch", "seq_len"), [(320, 1), (2, 3)])
def test_rwkv7_cmix_mix_3d_matches_dense_state_contract(
    batch: int, seq_len: int
) -> None:
    torch.manual_seed(20260724 + batch + seq_len)
    device = "cuda"
    hidden = 4096
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    initial_shift_state = torch.randn(
        (batch, hidden), device=device, dtype=torch.float16
    )
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    reference_state = initial_shift_state.clone()
    grid_state = initial_shift_state.clone()

    reference = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_mix(
        batch, seq_len, hidden, x, reference_state, x_k
    )
    actual = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_mix_3d(
        batch, seq_len, hidden, x, grid_state, x_k
    )
    torch.accelerator.synchronize()

    assert torch.equal(actual, reference)
    assert torch.equal(grid_state, reference_state)
    assert torch.equal(grid_state, x[:, -1, :])


@pytest.mark.parametrize("hidden", [64, 4096])
def test_rwkv7_cmix_mix_slot_matches_scattered_reference(hidden: int) -> None:
    torch.manual_seed(hidden + 3)
    device = "cuda"
    batch, seq_len, slots = 3, 4, 6
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((batch, seq_len, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    op = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_mix_slot
    args = (batch, seq_len, hidden, x, shift_state, slot_indices, x_k)
    output = op(*args)

    prev = torch.cat(
        (initial_shift_state[slot_indices.long(), None, :], x[:, :-1, :]), dim=1
    )
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    expected_state[slot_indices.long()] = x[:, -1, :]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


def test_rwkv7_cmix_mix_varlen_matches_scattered_reference() -> None:
    torch.manual_seed(5)
    device = "cuda"
    lengths = [2, 4, 1]
    batch, total_tokens, hidden, slots = len(lengths), sum(lengths), 64, 7
    query_start_loc = torch.tensor([0, 2, 6, 7], device=device, dtype=torch.int32)
    req_id = torch.tensor([0, 0, 1, 1, 1, 1, 2], device=device, dtype=torch.int32)
    slot_indices = torch.tensor([4, 1, 5], device=device, dtype=torch.int32)
    x = torch.randn((total_tokens, hidden), device=device, dtype=torch.float16)
    shift_state = torch.randn((slots, hidden), device=device, dtype=torch.float16)
    x_k = torch.randn((hidden,), device=device, dtype=torch.float16)
    initial_shift_state = shift_state.clone()

    output = torch.ops.vllm_rwkv7_fast_ops_fp16.cmix_mix_varlen(
        batch,
        total_tokens,
        hidden,
        x,
        shift_state,
        slot_indices,
        x_k,
        query_start_loc,
        req_id,
    )

    prev_parts = []
    for local_req, length in enumerate(lengths):
        start = int(query_start_loc[local_req].item())
        end = int(query_start_loc[local_req + 1].item())
        slot = int(slot_indices[local_req].item())
        prev_parts.append(
            torch.cat((initial_shift_state[slot : slot + 1], x[start : end - 1]))
        )
        assert end - start == length
    prev = torch.cat(prev_parts, dim=0)
    expected = (x.float() + (prev.float() - x.float()) * x_k.float()).to(torch.float16)
    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-3)
    expected_state = initial_shift_state.clone()
    for local_req, slot in enumerate(slot_indices.long().tolist()):
        end = int(query_start_loc[local_req + 1].item())
        expected_state[slot] = x[end - 1]
    torch.testing.assert_close(shift_state, expected_state, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [17, 320])
def test_rwkv7_add_vec_2d_matches_flat_kernel(rows: int) -> None:
    torch.manual_seed(20260725 + rows)
    hidden = 4096
    x = torch.randn((rows, hidden), device="cuda", dtype=torch.float16)
    vec = torch.randn((hidden,), device="cuda", dtype=torch.float16)

    expected = torch.ops.vllm_rwkv7_fast_ops_fp16.add_vec(hidden, x, vec)
    actual = torch.ops.vllm_rwkv7_fast_ops_fp16.add_vec_2d(hidden, x, vec)
    repeated = torch.ops.vllm_rwkv7_fast_ops_fp16.add_vec_2d(hidden, x, vec)
    torch.accelerator.synchronize()

    assert torch.equal(actual, expected)
    assert torch.equal(repeated, actual)


def test_rwkv7_add_vec_2d_rejects_grid_y_overflow() -> None:
    hidden = 2
    x = torch.empty((65536, hidden), device="cuda", dtype=torch.float16)
    vec = torch.empty((hidden,), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="rows <= 65535"):
        torch.ops.vllm_rwkv7_fast_ops_fp16.add_vec_2d(hidden, x, vec)


def test_rwkv7_add_vec_2d_rejects_misaligned_half2_vector() -> None:
    hidden = 8
    x = torch.empty((17, hidden), device="cuda", dtype=torch.float16)
    vec_storage = torch.empty((hidden + 1,), device="cuda", dtype=torch.float16)
    vec = vec_storage[1:]
    assert vec.is_contiguous()

    with pytest.raises(RuntimeError, match="aligned for fp16x2"):
        torch.ops.vllm_rwkv7_fast_ops_fp16.add_vec_2d(hidden, x, vec)
