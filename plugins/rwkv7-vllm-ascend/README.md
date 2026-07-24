# vllm-rwkv-ascend

RWKV-7 inference on Huawei Ascend through a real out-of-tree **vLLM V1 model
plugin**. The primary implementation is `rwkv7_vllm_ascend/`; the older
`serving/` framework and root op shim remain as historical/reference paths.

## Validated scope

The release gate was executed on one Ascend 910B3 64GB device with CANN 8.5.0,
ATB 8.5.0.B160, PyTorch/torch_npu 2.9.0, vLLM 0.18.0 and vllm-ascend 0.18.0.
Other Ascend cards are unvalidated and fail closed by default. TP, PP, prefix
caching, speculative decoding and Mamba cache modes other than `none` also fail
closed. Set `RWKV7_ALLOW_UNVALIDATED_ASCEND=1` only for explicit bring-up work.

## What works

- vLLM V1 admission, dynamic/continuous batching, cancellation and state-block
  lifecycle.
- Real `MambaSpec` state cache per RWKV layer: fp32 WKV plus attention/FFN
  previous-x states.
- Mixed decode and prefill metadata, irregular chunked prefill and continuation.
- Standard HF safetensors names for the FLA RWKV-7 7.2B checkpoint.
- Native vLLM embedding, LM head, logits processor and OpenAI serving stack.
- Fresh cache slots are zeroed; completed-slot reuse is regression tested.

On-device evidence is under `evidence/`. `real_engine_smoke_7.log` contains a
full 7.2B load and generation ending in `SMOKE_OK`.
`real_engine_acceptance.json` and `real_engine_scheduler_trace.jsonl` gate
different-length dynamic batches, an actual 180-token prefill split by a
32-token scheduler budget, mixed decode+prefill steps, reverse-order output
stability, physical cache-slot reuse and zeroization of stale recurrent state.

## Installation

PyPI metadata for vLLM 0.18 and vllm-ascend 0.18 requests incompatible PyTorch
versions. The plugin therefore intentionally has no resolver-level runtime
dependencies. Install the vendor-matched stack explicitly:

```bash
python -m venv --system-site-packages /data/venvs/vllm
/data/venvs/vllm/bin/pip install --no-deps vllm==0.18.0
/data/venvs/vllm/bin/pip install vllm-ascend==0.18.0
/data/venvs/vllm/bin/pip install --no-deps -e /data/work/vllm-rwkv-ascend
```

See `VLLM_ASCEND_PRODUCTION.md` and `compat/install_triton_target_info_shim.py`
for the exact host procedure and compatibility notes.

## Serve

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /data/venvs/vllm/bin/activate
unset VLLM_PLUGINS

vllm serve /path/to/rwkv7-hf \
  --trust-remote-code --dtype bfloat16 --enforce-eager \
  --enable-chunked-prefill --max-num-batched-tokens 2048 \
  --max-num-seqs 64 --disable-prefix-caching
```

## Repository layout

```text
rwkv7_vllm_ascend/   production vLLM V1 plugin (model, plugin, state oracle)
tests_vllm/           contract, math, scheduler and real-NPU acceptance scripts
evidence/             schema, resolver, smoke and acceptance artifacts
compat/               reproducible host compatibility shim
serving/              historical self-contained server
perf/                  historical C++ op-coalesced forward
harness/               vendored standalone reference
```

## License

Apache 2.0. Derives from [rwkv-rs/vllm-rwkv](https://github.com/rwkv-rs/vllm-rwkv).

## Ascend W8/W4 FFN seam (production disabled)

The default-off, fail-closed loader and `npu_weight_quant_batchmatmul` dispatch
contract is documented in [ASCEND_FFN_QUANT.md](ASCEND_FFN_QUANT.md). It is a
raw-kernel-candidate integration seam only; no quantized vLLM E2E acceptance is
claimed yet.
