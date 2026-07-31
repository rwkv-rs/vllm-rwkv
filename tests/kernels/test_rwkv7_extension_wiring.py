# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_generic_custom_ops_remain_independent_from_rwkv7() -> None:
    source = (ROOT / "vllm/_custom_ops.py").read_text()
    assert "rwkv7" not in source


def test_rwkv_only_artifact_can_import_generic_wrappers() -> None:
    try:
        rwkv7_ops = importlib.import_module("vllm.rwkv7_ops")
    except ImportError:
        pytest.skip("RWKV CUDA extension is not built")
    if not getattr(rwkv7_ops, "_rwkv_only_build", False):
        pytest.skip("test requires an RWKV-only artifact")

    importlib.import_module("vllm._custom_ops")


def test_rwkv7_loader_reports_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import_module = importlib.import_module

    def import_module(name: str):
        if name == "vllm.rwkv7_ops":
            raise ImportError("missing test extension")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    sys.modules.pop("vllm.rwkv7", None)
    with pytest.raises(ImportError, match="CUDA extension"):
        real_import_module("vllm.rwkv7")
    sys.modules.pop("vllm.rwkv7", None)


def test_rwkv7_extension_preserves_full_upstream_build() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    setup = (ROOT / "setup.py").read_text()

    assert "FetchContent_Declare(cutlass" in cmake
    assert "define_extension_target(\n    rwkv7_ops" in cmake
    assert 'CMakeExtension(name="vllm.rwkv7_ops")' in setup

    assert "from setuptools_rust.build import build_rust" in setup
    assert "rust_build.rust_extensions(" in setup
    assert 'CMakeExtension(name="vllm._C_stable_libtorch")' in setup
    assert 'CMakeExtension(name="vllm._moe_C_stable_libtorch")' in setup


def test_rwkv7_extension_contains_only_runtime_wkv_kernels() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    runtime = (ROOT / "vllm/rwkv7.py").read_text()

    assert "rwkv7_wkv_fp16_v2" in cmake
    assert "rwkv7_wkv_fp32_v2" in cmake
    assert "rwkv7_fast_ops_fp16" not in cmake
    assert "rwkv7_v3a_ops" not in cmake
    assert "rwkv7_fast_ops_fp16" not in runtime
    assert "rwkv7_v3a_ops" not in runtime


def test_rwkv7_profile_only_selects_build_targets() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    setup = (ROOT / "setup.py").read_text()

    assert 'VLLM_BUILD_PROFILE STREQUAL "rwkv"' in cmake
    assert 'if VLLM_BUILD_PROFILE == "rwkv":' in setup
    assert 'extension.name == "vllm.rwkv7_ops"' in setup
    assert 'if VLLM_BUILD_PROFILE == "rwkv"\n    else rust_build' in setup
    assert "-DVLLM_BUILD_PROFILE={}" in setup
    assert 'VLLM_BUILD_PROFILE != "rwkv"' in setup
    assert "VLLM_USE_PRECOMPILED must be disabled" in setup
    assert 'requirements = _read_requirements("rwkv.txt")' in setup

    rwkv_return = cmake.index('if(VLLM_BUILD_PROFILE STREQUAL "rwkv")\n  return()')
    assert rwkv_return < cmake.index("FetchContent_Declare(cutlass")
    assert rwkv_return < cmake.index("external_projects/triton_kernels.cmake")
    assert (
        'NOT VLLM_BUILD_PROFILE STREQUAL "rwkv"'
        in cmake[cmake.index("# spinloop extension") : rwkv_return]
    )

    runtime_occurrences = [
        path
        for path in (ROOT / "vllm").rglob("*.py")
        if "VLLM_BUILD_PROFILE" in path.read_text()
    ]
    assert runtime_occurrences == []


def test_rwkv7_profile_rejects_precompiled_rust_frontend() -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_BUILD_PROFILE": "rwkv",
            "VLLM_TARGET_DEVICE": "cuda",
            "VLLM_USE_PRECOMPILED": "0",
            "VLLM_USE_PRECOMPILED_RUST": "1",
        }
    )
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "VLLM_USE_PRECOMPILED_RUST must be disabled" in result.stderr


def test_cuda_platform_loads_generic_kernels_on_demand() -> None:
    cuda_platform = (ROOT / "vllm/platforms/cuda.py").read_text()
    kernel_loader = (ROOT / "vllm/platforms/cuda_kernel_loader.py").read_text()
    import_kernels = cuda_platform.index("def import_kernels")

    assert "import_kernels as import_cuda_kernels" in cuda_platform[:import_kernels]
    assert 'getattr(rwkv7_ops, "_rwkv_only_build", False)' in kernel_loader
    assert "import_cuda_kernels()" in cuda_platform[import_kernels:]
    assert "Failed to import from vllm._C_stable_libtorch" not in cuda_platform


@pytest.mark.parametrize(
    ("rwkv_only_build", "raises"),
    [(True, False), (False, True)],
)
def test_cuda_kernel_loader_distinguishes_rwkv_only_artifact(
    monkeypatch: pytest.MonkeyPatch,
    rwkv_only_build: bool,
    raises: bool,
) -> None:
    loader = ROOT / "vllm/platforms/cuda_kernel_loader.py"
    rwkv7_ops = types.ModuleType("vllm.rwkv7_ops")
    rwkv7_ops._rwkv_only_build = rwkv_only_build
    real_import_module = importlib.import_module

    def import_module(name: str):
        if name == "vllm._C_stable_libtorch":
            raise ImportError("missing generic extension")
        if name == "vllm.rwkv7_ops":
            return rwkv7_ops
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    namespace = runpy.run_path(str(loader))
    if raises:
        with pytest.raises(ImportError):
            namespace["import_kernels"]()
    else:
        namespace["import_kernels"]()


def test_cuda_kernel_loader_preserves_full_build_eager_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ROOT / "vllm/platforms/cuda_kernel_loader.py"
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    namespace = runpy.run_path(str(loader))
    namespace["import_kernels"]()

    assert imported == [
        "vllm._C_stable_libtorch",
        "vllm._moe_C_stable_libtorch",
        "vllm._qutlass_C",
    ]


def test_full_artifact_detection_does_not_probe_rwkv_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = ROOT / "vllm/platforms/cuda_kernel_loader.py"
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_module)
    namespace = runpy.run_path(str(loader))

    assert not namespace["is_rwkv_only_artifact"]()
    assert imported == ["vllm._C_stable_libtorch"]


@pytest.mark.parametrize(
    ("rwkv_only", "model_arch", "raises"),
    [
        (False, "LlamaForCausalLM", False),
        (True, "RWKV7ForCausalLM", False),
        (True, "LlamaForCausalLM", True),
    ],
)
def test_cuda_platform_rejects_unsupported_rwkv_only_models(
    monkeypatch: pytest.MonkeyPatch,
    rwkv_only: bool,
    model_arch: str,
    raises: bool,
) -> None:
    import vllm.platforms.cuda as cuda_platform

    monkeypatch.setattr(
        cuda_platform,
        "is_rwkv_only_artifact",
        lambda: rwkv_only,
    )

    if raises:
        with pytest.raises(ValueError, match="RWKV-only.*full vLLM build"):
            cuda_platform.CudaPlatformBase.verify_model_arch(model_arch)
    else:
        cuda_platform.CudaPlatformBase.verify_model_arch(model_arch)
