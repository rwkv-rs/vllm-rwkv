"""Install the target_info API expected by the vLLM 0.18 wheel.

The official triton-ascend 3.2.0.dev20260322 wheel does not ship
triton.language.target_info, while the mirrored vLLM 0.18 wheel imports five
GPU predicates from it. They are optional CUDA/ROCm kernel predicates on
Ascend, so this shim returns False for those backends and delegates target
inspection to Triton's active driver when explicitly requested.
"""
from pathlib import Path
import triton.language
target = Path(triton.language.__file__).with_name("target_info.py")
body = '''"""vLLM 0.18 compatibility predicates for Triton Ascend."""
def current_target():
    from triton.runtime import driver
    return driver.active.get_current_target()

def is_cuda():
    return getattr(current_target(), "backend", None) == "cuda"

def is_hip():
    return getattr(current_target(), "backend", None) == "hip"

def cuda_capability_geq(major, minor=0):
    if not is_cuda():
        return False
    arch = getattr(current_target(), "arch", 0)
    if isinstance(arch, str) and arch.startswith("sm"):
        arch = int(arch[2:])
    return int(arch) >= int(major) * 10 + int(minor)

def is_hip_cdna3():
    return is_hip() and getattr(current_target(), "arch", None) == "gfx942"

def is_hip_cdna4():
    return is_hip() and getattr(current_target(), "arch", None) == "gfx950"
'''
if target.exists() and target.read_text() != body:
    raise SystemExit(f"refusing to overwrite existing nonmatching {target}")
target.write_text(body)
print(target)
init = Path(triton.language.__file__).parent.parent / "__init__.py"
marker = "# rwkv7-vllm-ascend: Triton Ascend constexpr compatibility\n"
text = init.read_text()
patch = marker + "if 'constexpr_function' not in globals():\n    constexpr_function = lambda fn: fn\n"
if marker not in text:
    init.write_text(text + "\n" + patch)
print(init)
