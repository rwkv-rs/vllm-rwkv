# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import combinations

import pytest

from benchmarks.rwkv7 import benchmark_direct_model as bench


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("all", bench.FP16_LT_FAMILIES),
        ("attention-c2c", ("attention-c2c",)),
        ("ffn-down", ("ffn-down",)),
        ("lowrank", ("lowrank-in", "lowrank-out")),
        ("lowrank-in", ("lowrank-in",)),
        ("lowrank-out", ("lowrank-out",)),
    ],
)
def test_disabled_fp16_lt_families_are_explicit(family, expected) -> None:
    assert bench._disabled_fp16_lt_families(family) == expected


def test_disabled_fp16_lt_families_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown FP16 Lt tuning family"):
        bench._disabled_fp16_lt_families("ffn-key")


@pytest.mark.parametrize(
    ("variant", "batch_size", "token_count", "production_enabled", "expected"),
    [
        ("baseline", 16, 1, True, True),
        ("tuned", 1, 1, True, True),
        ("tuned", 2, 1, True, False),
        ("tuned", 1, 2, True, False),
        ("baseline", 1, 1, False, False),
        ("tuned", 1, 1, False, False),
    ],
)
def test_ln1_tmix_fuse_variant_matches_albatross_policy(
    variant,
    batch_size,
    token_count,
    production_enabled,
    expected,
) -> None:
    assert (
        bench._ln1_tmix_fuse_for_variant(
            variant,
            batch_size,
            token_count,
            production_enabled,
        )
        is expected
    )


def test_ln1_tmix_fuse_variant_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown model comparison variant"):
        bench._ln1_tmix_fuse_for_variant("unknown", 1, 1, True)


def test_measurement_result_reports_shape_dtype_and_throughput() -> None:
    torch = pytest.importorskip("torch")
    output = torch.empty((2, 3), dtype=torch.float32)

    result = bench._measurement_result(
        [2.0, 1.0, 3.0],
        output,
        batch_size=2,
        token_count=4,
        provider="paired",
    )

    assert result["p10_ms"] == pytest.approx(1.2)
    assert result["p50_ms"] == 2.0
    assert result["p90_ms"] == pytest.approx(2.8)
    assert result["output_shape"] == [2, 3]
    assert result["output_dtype"] == "torch.float32"
    assert result["tokens_per_s_p50"] == 4000.0


def test_model_comparison_modes_are_mutually_exclusive() -> None:
    parser = bench._build_parser()

    modes = (
        "--compare-fp16-lt-tuning",
        "--compare-m1-rkv",
        "--compare-m1-cmix-prezero",
        "--compare-gemm-accumulation",
        "--compare-ln1-tmix-fusion",
    )
    for first, second in combinations(modes, 2):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "model.pth", first, second])
