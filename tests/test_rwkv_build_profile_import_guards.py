# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from vllm.build_profile import BuildProfileMetadata
from vllm.model_executor.layers.quantization.utils import w8a8_utils

ROOT = Path(__file__).parents[1]


def test_rwkv_profile_skips_unbuilt_stable_cutlass_probes(monkeypatch) -> None:
    metadata = BuildProfileMetadata(
        profile="rwkv",
        configured_targets=("rwkv7_ops",),
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


def test_rwkv_profile_guards_generic_warmup_and_flash_attention_imports() -> None:
    gpu_worker = (ROOT / "vllm/v1/worker/gpu_worker.py").read_text()
    warmup_import = "from vllm.model_executor.warmup.kernel_warmup import kernel_warmup"
    assert gpu_worker.index('get_build_profile_metadata().profile == "rwkv"') < (
        gpu_worker.index(warmup_import)
    )

    fa_utils = (ROOT / "vllm/v1/attention/backends/fa_utils.py").read_text()
    flash_attention_import = "from vllm.vllm_flash_attn import"
    assert fa_utils.index('get_build_profile_metadata().profile == "rwkv"') < (
        fa_utils.index(flash_attention_import)
    )
    assert "FlashAttention is unavailable in the RWKV build profile" in fa_utils
