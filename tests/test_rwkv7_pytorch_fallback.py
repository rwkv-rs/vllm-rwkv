"""Smoke test for the pure-PyTorch rwkv7_* op fallback.

Verifies the fallback registers all four op namespaces + a couple of representative
ops, so vllm-rwkv's RWKV7 model code can run on CPU / any non-CUDA backend without
the CUDA kernels. (Numerical correctness vs the CUDA kernels is covered separately;
this just checks the ops exist after install().)
"""
import os
import sys

# Make the repo root importable so `import rwkv7_pytorch_fallback` works without
# pulling in the (heavy, CUDA-expecting) vllm package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def test_fallback_registers_all_namespaces():
    import rwkv7_pytorch_fallback as fb
    fb.install()
    for ns in ("rwkv7_v3a_ops", "rwkv7_fast_ops_fp16",
               "rwkv7_wkv_fp16_v2", "rwkv7_wkv_fp32_v2"):
        assert hasattr(torch.ops, ns), f"fallback did not register op namespace '{ns}'"


def test_fallback_registers_representative_ops():
    import rwkv7_pytorch_fallback as fb
    fb.install()
    assert hasattr(torch.ops.rwkv7_v3a_ops, "layer_norm_f16")
    assert hasattr(torch.ops.rwkv7_v3a_ops, "linear_f16")
    assert hasattr(torch.ops.rwkv7_wkv_fp16_v2, "wkv_seq")
    assert hasattr(torch.ops.rwkv7_wkv_fp32_v2, "forward")
