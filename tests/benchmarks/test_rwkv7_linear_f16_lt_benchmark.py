# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.rwkv7 import benchmark_linear_f16_lt as bench


def test_lagging_profile_contains_only_real_runtime_layout_shapes() -> None:
    shapes = bench._candidate_shapes("lagging")

    assert len(shapes) == 25
    assert {shape.m for shape in shapes} == {8, 16, 64}
    assert bench.Shape(64, 16384, 4096) in shapes
    assert bench.Shape(8, 16384, 4096) not in shapes
    assert bench.Shape(16, 16384, 4096) not in shapes
    assert all(
        (shape.k, shape.n) in (*bench.LOWRANK_KN, *bench.DENSE_KN)
        for shape in shapes
        if shape.k != 16384
    )


def test_full_profile_covers_b320_and_ffn_down_threshold() -> None:
    shapes = bench._candidate_shapes("full")

    assert len(shapes) == 61
    assert bench.Shape(320, 4096, 128) in shapes
    assert bench.Shape(320, 16384, 4096) in shapes
    assert bench.Shape(16, 16384, 4096) not in shapes


def test_percentile_uses_linear_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert bench._percentile(values, 0.0) == 1.0
    assert bench._percentile(values, 0.5) == 2.5
    assert bench._percentile(values, 1.0) == 4.0


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        bench._candidate_shapes("other")
