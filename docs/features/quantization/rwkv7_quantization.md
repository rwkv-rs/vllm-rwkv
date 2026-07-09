# RWKV-7 Quantized Inference (INT8 + NVFP4)

vLLM-rwkv supports two offline quantization formats for RWKV-7 models:
**INT8** (8-bit integer) and **NVFP4** (NVIDIA standard FP4 — E2M1 4-bit weights
with E4M3 FP8 block scales and FP32 tensor scales). Both use CUDA-core kernels
(no tensor core required) and are compatible with all GPU architectures from
sm_50 onwards.

## Overview

| Feature | INT8 | NVFP4 (E2M1 + E4M3 + FP32) |
|---------|------|----------------------------|
| Weight bits | 8 | 4 (E2M1) |
| Block scale | per-channel [N] fp16 | per-block [N, K/16] FP8 (E4M3) |
| Tensor scale | — | per-tensor FP32 scalar |
| Weight dtype | int8 | uint8 (packed) |
| Weight shape | [N, K] | [N, K/2] (2 E2M1 per byte) |
| Kernel type | GEMV/GEMM (CUDA core) | GEMV/GEMM (CUDA core) |
| Architecture | sm_50+ | sm_50+ |
| Weight compression | 2x | 4x |

### Dequantization formula (NVFP4)

```
actual_value = e2m1_lut[code] * block_scale_e4m3 * tensor_scale
```

- `e2m1_lut[code]`: 4-bit E2M1 look-up table (16 entries)
- `block_scale_e4m3`: per-16-element E4M3 FP8 scale (`float8_e4m3fn`)
- `tensor_scale`: per-tensor FP32 global scale

### Quantized weights

Only "orig-linear" weights are quantized (same set for both modes):
- `att.receptance.weight`, `att.key.weight`, `att.value.weight`, `att.output.weight`
- `ffn.key.weight`, `ffn.value.weight`

**Note**: `head.weight` is intentionally **not** quantized — the LM head must
remain FP16/FP32 to preserve precision for reinforcement learning training.

The following weights are **not** quantized:
- Low-rank weights (minimal benefit)
- Embedding / LayerNorm / r_k

### Weight coverage (7.2B model)

| Weight group | Params | Quantized | Coverage |
|-------------|--------|-----------|----------|
| att.r/k/v/o.weight | 2,178M | Yes | 30.2% |
| ffn.key.weight | 938M | Yes | 13.0% |
| ffn.value.weight | 2,148M | Yes | 29.8% |
| head.weight | 268M | **No** | 3.7% |
| Low-rank (w1/w2/a1/a2/g1/g2/v1/v2) | 294M | No | 4.1% |
| Embedding | 268M | No | 3.7% |
| LN / r_k | ~1M | No | <0.1% |
| **Total quantized** | **6,443M** | | **89.5%** |

## Offline Quantization

Use the NVFP4 quantization tool (`tools/quantize_nf4.py`):

```bash
python3 tools/quantize_nf4.py --model model.pth --out model-nvfp4.pth --verify
```

The tool:
1. Loads the FP16/BF16 model (mmap for low memory)
2. Quantizes all orig-linear weights to the standard NVFP4 format
   (E2M1 4-bit packed + E4M3 FP8 block scale + FP32 tensor scale)
3. Stores the packed uint8 weight in-place under the original key, plus the
   block scale and tensor scale under their suffix keys
4. Auto-repacks the output to fix mmap file bloat
5. `--verify` prints per-weight CosSim (float64)

### Weight key naming

| Mode | Weight key | Scale keys |
|------|-----------|------------|
| INT8 | `key` (int8 tensor) | `key + ".scale"` (fp16 [N]) |
| NVFP4 | `key` (uint8 tensor, in-place) | `key + ".nf4_b_scale"` (float8_e4m3fn [N, K/16]) + `key + ".nvfp4_t_scale"` (fp32 scalar) |

## Inference

vLLM automatically detects quantized weights by dtype + scale keys:
- `int8` weight + `.scale` → INT8 kernel path
- `uint8` weight + `.nvfp4_t_scale` → NVFP4 kernel path
- `float16/bfloat16` → FP16 path (original)

No configuration needed — just load the quantized model file:

```python
from vllm import LLM
llm = LLM(model="path/to/model-nvfp4.pth")
```

### Dispatch strategy (NVFP4)

NVFP4 uses an M-based dispatch (identical M-buckets to FP16/INT8 v3a):
- M=1: optimized NVFP4 blk16 GEMV kernel (`linear_nvfp4_orig_row1_blk16_f16`)
- M=2: optimized NVFP4 blk16 GEMV kernel (`linear_nvfp4_orig_row2_blk16_f16`)
- M=3-12: NVFP4 GEMM kernel (`linear_nf4_orig_rows_f16`, row-tiling)
- M>12: dequantize to fp16 + cuBLAS (tiling advantage for large M)

`ffn.value.weight` is served by dedicated `cmix_sparse` NVFP4 kernels that
fuse the down-projection with the RWKV-7 cmix non-linearity.

## Precision

### INT8
- Per-weight CosSim: ~1.0 (negligible loss)
- End-to-end greedy match: identical to FP16

### NVFP4
- Per-weight CosSim: ~0.996
- End-to-end greedy match: identical to INT8
- Logits CosSim (NVFP4 vs INT8): ~0.9999

## Performance

Tested on RWKV-7-G1G-7.2B, RTX 5070 Ti Laptop (12GB, sm_120):

| Mode | Decode tok/s | VRAM (weights) | File size |
|------|-------------|----------------|-----------|
| FP16 | ~55 | ~14 GB | 14 GB |
| INT8 | ~54 | ~9.3 GB | 9.2 GB |
| NVFP4 | ~30 | ~6.1 GB | 5.2 GB |

NVFP4 quantizes ffn.value.weight via dedicated cmix_sparse NVFP4 kernels,
achieving 89.5% weight coverage. VRAM savings: ~56% vs FP16. The standard
NVFP4 format uses E4M3 FP8 block scales (half the size of the previous fp16
block scales) plus a per-tensor FP32 scale, for a marginally smaller footprint.

### Standalone kernel performance (M=1, 4096x4096)

| Kernel | Time | vs FP16 cuBLAS |
|--------|------|-----------------|
| FP16 cuBLAS | 0.076ms | 1.0x |
| NVFP4 blk16 GEMV | 0.050ms | 0.66x (faster) |
| INT8 exact | 0.042ms | 0.55x (faster) |

## Building from source

The INT8 and NVFP4 CUDA kernels are compiled automatically as part of the
`rwkv7_v3a_ops` extension. The NVFP4 kernels use `cuda_fp8.h` for the E4M3 FP8
block scales; no additional build flags are needed.

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
├── rwkv7_nf4_ops.cu       # NVFP4 kernels (E2M1 + E4M3 FP8 + FP32)
└── rwkv7_nf4_ops.cpp      # NVFP4 C++ binding

tools/
└── quantize_nf4.py        # NVFP4 offline quantization tool

vllm/model_executor/models/
└── rwkv7.py               # Model code (FP16 + INT8 + NVFP4 dispatch)
```
