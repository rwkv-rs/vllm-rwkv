# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.build_profile import BuildProfileMetadata
from vllm.model_executor.layers.quantization.utils import w8a8_utils


@pytest.mark.parametrize(
    ("profile", "configured_targets"),
    [
        ("rwkv", ("_rapid_sampling", "cumem_allocator", "rwkv7_ops")),
        (
            "rwkv-nvfp4",
            (
                "_C_stable_libtorch",
                "_rapid_sampling",
                "cumem_allocator",
                "rwkv7_ops",
            ),
        ),
    ],
)
def test_reduced_profile_skips_unbuilt_stable_cutlass_probes(
    monkeypatch,
    profile: str,
    configured_targets: tuple[str, ...],
) -> None:
    metadata = BuildProfileMetadata(
        profile=profile,
        configured_targets=configured_targets,
        external_projects=(),
        unrestricted=False,
    )
    monkeypatch.setattr(w8a8_utils, "get_build_profile_metadata", lambda: metadata)
    monkeypatch.setattr(w8a8_utils.current_platform, "is_cuda", lambda: True)

    def unexpected_probe(*args, **kwargs):
        raise AssertionError("reduced artifact must not call full CUTLASS ops")

    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_scaled_mm_supports_fp8", unexpected_probe
    )
    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_scaled_mm_supports_block_fp8", unexpected_probe
    )
    monkeypatch.setattr(
        w8a8_utils.ops, "cutlass_group_gemm_supported", unexpected_probe
    )

    assert not w8a8_utils.cutlass_fp8_supported()
    assert not w8a8_utils.cutlass_block_fp8_supported()
    assert not w8a8_utils.cutlass_group_gemm_supported()
