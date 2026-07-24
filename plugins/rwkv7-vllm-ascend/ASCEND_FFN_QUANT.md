# Ascend 910B3 FFN W8/W4 integration seam

This repository contains a **real but production-disabled** weight-only loading
and dispatch seam for the two RWKV-7 7.2B FFN projections:
`layers.N.ffn.key` (`K=4096,N=16384`) and `layers.N.ffn.value`
(`K=16384,N=4096`).  Dense serving remains the default and its accepted path is
unchanged.

## Admission boundary

The seam fails closed unless every item below is exact:

- device name `Ascend910B3`;
- PyTorch `2.9.0`, `torch_npu` `2.9.0`, CANN `8.5.0`;
- `npu_weight_quant_batchmatmul` dispatcher schema SHA256
  `f99a37dcc4e7d07f803bb83ce3d4c93ccbd15c41af1267d5a07dcc0d62e7dff0`;
- FP16 activations, TP=1/PP=1, and one of the two measured FFN shapes;
- W4 group size 128 with exact rows `M=1,8`, or W8 per-channel with exact
  rows `M=17,28`.

Those row ranges are **raw-operator candidates**, not engine acceptance.  An
unlisted row raises; there is no dense fallback because a selected module does
not retain an FP16 weight.  Consequently this mode is not suitable for general
serving yet.  It exists to run the next real engine E2E quality/HBM/latency gate
without maintaining a second implementation.

## Default-off activation

All three variables are required; partial configuration raises:

```bash
export RWKV7_ASCEND_QUANT=1
export RWKV7_ASCEND_QUANT_MANIFEST=/absolute/path/to/manifest.json
export RWKV7_ASCEND_ALLOW_RAW_CANDIDATE=1
```

The acknowledgement is deliberately named `RAW_CANDIDATE`.  It cannot turn
`production_accepted` on.  A manifest claiming production acceptance is
rejected by code.

The example manifests are:

- `examples/ascend_w4_raw_candidate_manifest.json`
- `examples/ascend_w8_raw_candidate_manifest.json`

Edit `ffn.key_layers` and `ffn.value_layers` to select projections independently.
The examples use `source: fp-checkpoint`: the ordinary floating checkpoint
weight is quantized once while streaming through the engine loader, then is not
retained by the module.

For `source: packed-checkpoint`, replace each selected `.weight` tensor with
exactly `.qweight`, `.scales`, and `.offsets` and list every tensor in the
manifest with shape, dtype, and SHA256.  W8 uses an empty FP16 `offsets` tensor.
The loader authenticates every packed tensor before installing it.

## Storage and hot path

A selected module has no parameters and no `weight` attribute.  Its persistent
buffers are exactly:

- `qweight`: W8 `[K,N] int8`, or W4 `[K,N/8] int32` (eight signed nibbles);
- `scales`: W8 `[N] fp16`, or W4 `[K/128,N] fp16`;
- `offsets`: empty for W8, or `[K/128,N] fp16` zeros for W4.

After checkpoint loading, the adapter moves only these buffers to NPU and binds
one closure per admitted `M`.  Forward performs tensor ABI checks and a direct
pre-bound closure lookup.  It does not repeat device/version/CANN/schema or
acceptance-policy discovery in the layer hot path.  W8 calls
`npu_weight_quant_batchmatmul(x, qweight, scales)`; W4 calls the groupwise ABI
with `offsets`, group size 128, and inner precision mode 1.

## Status

- Dense engine evidence elsewhere in this repository remains valid.
- CPU mock/unit tests cover default-dense regression, FP checkpoint mapping,
  packed-manifest hashes, no dense copy, W8/W4 operator signatures, row gating,
  and exact-runtime rejection.
- No NPU E2E quant artifact is claimed by this change.
- `production_accepted` remains `false` until the backend itself passes output
  quality, peak-HBM reduction, and speed `>= FP16` on every declared scheduler
  shape.

## vLLM loader wiring

`RWKV7ForCausalLM.load_weights` intercepts only the selected canonical
`model.layers.N.ffn.{key,value}` names. All other names continue through the
existing vLLM `default_weight_loader`. The recurrent cache, scheduler metadata,
and dense projections are untouched. vLLM must be started with `--dtype half`.
