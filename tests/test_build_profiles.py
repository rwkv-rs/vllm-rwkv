# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tools.build_profiles import (
    RWKV_EXTENSION_NAMES,
    profile_build_temp,
    resolve_build_profile,
    select_extension_names,
)


def test_build_profile_defaults_to_rwkv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BUILD_PROFILE", raising=False)

    assert resolve_build_profile() == "rwkv"


def test_build_profile_accepts_rwkv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "rwkv")

    assert resolve_build_profile() == "rwkv"


def test_build_profile_rejects_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "full")

    with pytest.raises(
        ValueError,
        match=r"VLLM_BUILD_PROFILE=full is disabled; only rwkv is supported",
    ):
        resolve_build_profile()


def test_build_profile_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "attention")

    with pytest.raises(
        ValueError,
        match=r"attention.*accepted values.*rwkv",
    ):
        resolve_build_profile()


def test_rwkv_profile_selects_only_required_extensions() -> None:
    full_names = [
        "vllm.cumem_allocator",
        "vllm.triton_kernels",
        "vllm.spinloop",
        "vllm.fs_io_C",
        "vllm.vllm_flash_attn._vllm_fa2_C",
        "vllm._flashmla_C",
        "vllm._deep_gemm_C",
        "vllm._qutlass_C",
        "vllm.fmha_sm100",
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm._rapid_sampling",
        "vllm.rwkv7_ops",
    ]

    assert select_extension_names(full_names, "rwkv") == list(RWKV_EXTENSION_NAMES)
    with pytest.raises(ValueError, match=r"full.*disabled.*only rwkv"):
        select_extension_names(full_names, "full")


def test_switching_profile_forces_cmake_reconfiguration() -> None:
    build_temp = "build/temp.linux-x86_64-cpython-312"

    assert profile_build_temp(build_temp, "rwkv") == f"{build_temp}-rwkv"
    with pytest.raises(ValueError, match=r"full.*disabled.*only rwkv"):
        profile_build_temp(build_temp, "full")
