# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from tools.build_profiles import (
    RWKV_EXTENSION_NAMES,
    profile_build_temp,
    resolve_build_profile,
    select_extension_names,
)


def test_build_profile_defaults_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_BUILD_PROFILE", raising=False)

    assert resolve_build_profile() == "full"


def test_build_profile_accepts_rwkv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "rwkv")

    assert resolve_build_profile() == "rwkv"


def test_build_profile_rejects_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_BUILD_PROFILE", "attention")

    with pytest.raises(
        ValueError,
        match=r"attention.*accepted values.*full.*rwkv",
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
        "vllm.rwkv7_ops",
    ]

    assert select_extension_names(full_names, "full") == full_names
    assert select_extension_names(full_names, "rwkv") == list(RWKV_EXTENSION_NAMES)


def test_switching_profile_forces_cmake_reconfiguration() -> None:
    build_temp = "build/temp.linux-x86_64-cpython-312"
    setup_source = (Path(__file__).parents[1] / "setup.py").read_text()

    assert profile_build_temp(build_temp, "full") == build_temp
    assert profile_build_temp(build_temp, "rwkv") == f"{build_temp}-rwkv"
    assert '"-DVLLM_BUILD_PROFILE={}".format(VLLM_BUILD_PROFILE)' in setup_source
    assert '"-DCMAKE_SUPPRESS_REGENERATION=ON"' in setup_source


def test_cmake_declares_profile_manifest() -> None:
    cmake = (Path(__file__).parents[1] / "CMakeLists.txt").read_text()

    assert 'set(VLLM_BUILD_PROFILE "full" CACHE STRING' in cmake
    assert "vllm_build_profile.json" in cmake
    assert "VLLM_EXTENSION_TARGETS" in cmake
    assert "VLLM_EXTERNAL_PROJECTS" in cmake
    early_return = cmake.index('if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND')
    manifest_index = cmake.index("vllm_write_build_profile_manifest()", early_return)
    assert manifest_index < cmake.index("return()", early_return)


def test_rwkv_profile_carries_model_runner_uva_helper() -> None:
    root = Path(__file__).parents[1]
    cmake = (root / "CMakeLists.txt").read_text()
    registration = (
        root / "csrc/libtorch_stable/rwkv7/rwkv7_registration.cpp"
    ).read_text()

    assert '"csrc/libtorch_stable/cuda_view.cu"' in cmake
    assert "VLLM_RWKV_RUNTIME_OPS" in cmake
    assert "#ifdef VLLM_RWKV_RUNTIME_OPS" in registration
    assert "get_cuda_view_from_cpu_tensor" in registration


def test_sdist_declares_profile_inputs() -> None:
    manifest = (Path(__file__).parents[1] / "MANIFEST.in").read_text()

    assert "include tools/build_profiles.py" in manifest
    assert "include requirements/rwkv.txt" in manifest


def test_editable_build_installs_manifest_into_source_package() -> None:
    setup_source = (Path(__file__).parents[1] / "setup.py").read_text()

    assert 'getattr(self, "editable_mode", False)' in setup_source
    assert 'ROOT_DIR / "vllm" / "_build_profile.json"' in setup_source
