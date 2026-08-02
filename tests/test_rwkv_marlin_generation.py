# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import subprocess
import sys
from pathlib import Path


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_alternating_marlin_arch_generation_is_build_directory_owned(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = repository / "csrc/libtorch_stable/quantization/marlin"
    generator = source / "generate_kernels.py"
    source_outputs = sorted(
        [*source.glob("*kernel_*.cu"), source / "kernel_selector.h"]
    )
    before = {
        path.relative_to(source): _source_digest(path)
        for path in source_outputs
        if path.exists()
    }

    sm80_output = tmp_path / "full-sm80"
    sm120_output = tmp_path / "rwkv-nvfp4-sm120"
    subprocess.run(
        [sys.executable, str(generator), "8.0", str(sm80_output)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(generator), "12.0a", str(sm120_output)],
        check=True,
    )

    after = {
        path.relative_to(source): _source_digest(path)
        for path in source_outputs
        if path.exists()
    }
    assert after == before
    assert list(sm80_output.glob("sm80_kernel_*.cu"))
    assert list(sm120_output.glob("sm80_kernel_*.cu"))
    sm80_selector = (sm80_output / "kernel_selector.h").read_text()
    sm120_selector = (sm120_output / "kernel_selector.h").read_text()
    assert "a_type == vllm::kFE4M3fn" in sm80_selector
    assert "marlin kernel with fp8 activation is not built" in sm80_selector
    assert "a_type == vllm::kFE4M3fn" in sm120_selector
    assert "marlin kernel with fp8 activation is not built" not in sm120_selector
