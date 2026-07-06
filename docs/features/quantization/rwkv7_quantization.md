# RWKV-7 Quantized Inference (INT8 + NF4)

vLLM-rwkv supports two offline quantization formats for RWKV-7 models:
**INT8** (8-bit integer) and **NF4** (4-bit float E2M1). Both use the same
CUDA-core kernels (no tensor core required) and are compatible with all GPU
architectures from sm_50 onwards.

## Overview

| Feature | INT8 | NF4 (E2M1) |
|---------|------|------------|
| Weight bits | 8 | 4 |
| Scale format | per-channel [N] fp16 | per-block [N, K/16] fp16 |
| Weight dtype | int8 | uint8 (packed) |
| Weight shape | [N, K] | [N, K/2] (2 E2M1 per byte) |
| Kernel type | GEMV/GEMM (CUDA core) | GEMV/GEMM (CUDA core) |
| Architecture | sm_50+ | sm_50+ |
| Weight compression | 2x | 4x |

### Quantized weights

Only "orig-linear" weights are quantized (same set for both modes):
- `att.receptance.weight`, `att.key.weight`, `att.value.weight`, `att.output.weight`
- `ffn.key.weight`
- `head.weight`

The following weights are **not** quantized:
- `ffn.value.weight` (sparse cmix kernel hardcodes fp16)
- Low-rank weights (minimal benefit)
- Embedding / LayerNorm / r_k

## Offline Quantization

Use the combined quantization tool (`rwkv7_quantize.py`):

```bash
# INT8 quantization
python3 rwkv7_quantize.py --mode int8 \
  --model model.pth --out model-int8.pth --verify

# NF4 quantization
python3 rwkv7_quantize.py --mode nf4 \
  --model model.pth --out model-nf4.pth --verify
```

The tool:
1. Loads the FP16/BF16 model (mmap for low memory)
2. Quantizes all orig-linear weights
3. Stores quantized weight + scale under the original key + suffix
4. NF4 mode auto-repacks to fix mmap file bloat
5. `--verify` prints per-weight CosSim (float64)

### Weight key naming

| Mode | Weight key | Scale key |
|------|-----------|-----------|
| INT8 | `key` (int8 tensor) | `key + ".scale"` (fp16 [N]) |
| NF4 | `key` (uint8 tensor) | `key + ".nf4_b_scale"` (fp16 [N, K/16]) |

## Inference

vLLM automatically detects quantized weights by dtype:
- `int8` -> INT8 kernel path
- `uint8` -> NF4 kernel path
- `float16/bfloat16` -> FP16 path (original)

No configuration needed — just load the quantized model file:

```python
from vllm import LLM
llm = LLM(model="path/to/model-int8.pth")
```

### Dispatch strategy

Both INT8 and NF4 use the same M-based dispatch (identical to FP16 v3a):
- M=1: exact GEMV kernel (2-wide or 4-wide K loop)
- M=2: exact GEMV kernel (2-wide or 4-wide K loop)
- M=3-12: GEMM kernel (row-tiling)
- M>12: dequantize to fp16 + cuBLAS (tiling advantage for large M)

## Precision

### INT8
- Per-weight CosSim: ~1.0 (negligible loss)
- End-to-end greedy match: identical to FP16

### NF4
- Per-weight CosSim: ~0.996
- End-to-end greedy match: identical to INT8
- Logits CosSim (NF4 vs INT8): ~0.9999

## Performance

Tested on RWKV-7-G1G-7.2B, RTX 5070 Ti Laptop (12GB, sm_120):

| Mode | Decode tok/s | VRAM (weights) | File size |
|------|-------------|----------------|-----------|
| FP16 | ~55 | ~14 GB | 14 GB |
| INT8 | ~54 | ~9.3 GB | 9.2 GB |
| NF4 | ~12* | ~7.6 GB | 8.1 GB |

*NF4 decode is slower due to LUT lookup + block scale access overhead.
NF4 is recommended for VRAM-constrained scenarios where INT8 doesn't fit.

### Standalone kernel performance (M=1, 4096x4096)

| Kernel | Time | vs FP16 cuBLAS |
|--------|------|-----------------|
| FP16 cuBLAS | 0.076ms | 1.0x |
| NF4 exact | 0.050ms | 0.66x (faster) |
| INT8 exact | 0.042ms | 0.55x (faster) |

## Building from source

The INT8 and NF4 CUDA kernels are compiled automatically as part of the
`rwkv7_v3a_ops` extension. No additional build flags needed.

For non-sm_120 architectures, recompile from source on the target GPU:
```bash
cd vllm-rwkv
pip install -e .  # recompiles for your GPU's sm version
```

## File structure

```
csrc/libtorch_stable/rwkv7/
├── rwkv7_v3a_ops.cu       # FP16 kernels (original)
├── rwkv7_v3a_ops.cpp      # FP16 C++ binding
├── rwkv7_int8_ops.cu      # INT8 kernels
├── rwkv7_int8_ops.cpp     # INT8 C++ binding
├── rwkv7_nf4_ops.cu       # NF4 kernels
└── rwkv7_nf4_ops.cpp      # NF4 C++ binding

vllm/model_executor/models/
└── rwkv7.py               # Model code (FP16 + INT8 + NF4 dispatch)
```
