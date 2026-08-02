# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
from packaging.requirements import Requirement

from tools.build_profiles import (
    RWKV_EXTENSION_NAMES,
    RWKV_NVFP4_EXTENSION_NAMES,
    profile_build_temp,
    resolve_build_profile,
    select_extension_names,
)
from vllm.transformers_utils.configs.rwkv7 import (
    RWKV7_COMPRESSED_TENSORS_VERSION,
)


def test_build_profile_defaults_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BUILD_PROFILE", raising=False)

    assert resolve_build_profile() == "full"


def test_build_profile_accepts_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "full")

    assert resolve_build_profile() == "full"


def test_build_profile_accepts_rwkv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "rwkv")

    assert resolve_build_profile() == "rwkv"


def test_build_profile_accepts_rwkv_nvfp4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "rwkv-nvfp4")

    assert resolve_build_profile() == "rwkv-nvfp4"


def test_build_profile_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "attention")

    with pytest.raises(
        ValueError,
        match=r"attention.*accepted values.*full, rwkv, rwkv-nvfp4",
    ):
        resolve_build_profile()


def test_profiles_select_their_owned_extensions() -> None:
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
    assert select_extension_names(full_names, "rwkv-nvfp4") == list(
        RWKV_NVFP4_EXTENSION_NAMES
    )
    assert select_extension_names(full_names, "full") == full_names


def test_switching_profile_forces_cmake_reconfiguration() -> None:
    build_temp = "build/temp.linux-x86_64-cpython-312"

    assert profile_build_temp(build_temp, "rwkv") == f"{build_temp}-rwkv"
    assert profile_build_temp(build_temp, "rwkv-nvfp4") == (f"{build_temp}-rwkv-nvfp4")
    assert profile_build_temp(build_temp, "full") == build_temp


def test_rwkv_nvfp4_dependency_delta_is_profile_local() -> None:
    requirements = Path(__file__).parents[1] / "requirements"
    dense = (requirements / "rwkv.txt").read_text(encoding="utf-8")
    nvfp4 = (requirements / "rwkv-nvfp4.txt").read_text(encoding="utf-8")

    assert "compressed-tensors" not in dense
    assert "-r rwkv.txt" in nvfp4
    assert Requirement(
        f"compressed-tensors=={RWKV7_COMPRESSED_TENSORS_VERSION}"
    ) == Requirement(
        next(line for line in nvfp4.splitlines() if line.startswith("compressed"))
    )


def test_rwkv_profile_includes_no_isolation_rust_build_requirements() -> None:
    requirements = Path(__file__).parents[1] / "requirements"
    dense = (requirements / "rwkv.txt").read_text(encoding="utf-8")
    rust_build = (requirements / "build" / "rust.txt").read_text(encoding="utf-8")

    assert "-r build/rust.txt" in dense
    parsed = {
        Requirement(line).name
        for line in rust_build.splitlines()
        if line and not line.startswith("#")
    }
    assert "setuptools-rust" in parsed
