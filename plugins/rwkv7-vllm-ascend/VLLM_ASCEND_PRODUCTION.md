# RWKV-7 vLLM-Ascend production plugin

This directory now contains a real vLLM general/model plugin. The historical
serving/serve_*.py programs remain benchmark references and are not the vLLM
deliverable.

## Runtime contract

* vLLM V1 owns admission, continuous batching, cancellation and block-table
  reorder/release.
* Every RWKV block is registered as MambaBase and advertises a MambaSpec with
  three tensors: fp32 [H,N,N] WKV state, activation-dtype attention previous-x
  and FFN previous-x.
* The vLLM MambaManager owns allocation/reuse on request join/exit.
* Mixed decode+prefill batches are split from Mamba1AttentionMetadata.
  query_start_loc_p preserves per-request boundaries, including irregular
  chunked prefill and continuation from existing state.
* Prefix caching is fail-closed. Enabling it raises rather than silently
  reusing a state at a wrong token boundary.
* TP, PP and speculative decoding are fail-closed until parity evidence exists.

## Pinned stack for this 910B3 host

| component | pin |
|---|---|
| CANN toolkit / ATB | 8.5.0 / 8.5.0.B160 |
| driver | 25.5.0 |
| Python | 3.11 |
| PyTorch / torch_npu | 2.9.0 / 2.9.0 |
| vLLM | 0.18.0 compatibility wheel |
| vllm-ascend | 0.18.0 |
| plugin | 0.3.0 |

### Upstream packaging defect

As observed on 2026-07-24, PyPI vllm==0.18.0 declares torch==2.10.0, while
PyPI vllm-ascend==0.18.0 declares torch==2.9.0 and its metadata says CANN 8.5
requires PyTorch/torch_npu 2.9. Pip reports ResolutionImpossible. The
reproducible compatibility installation is:

    python -m venv --system-site-packages /data/venvs/vllm
    /data/venvs/vllm/bin/pip install --no-deps vllm==0.18.0
    /data/venvs/vllm/bin/pip install vllm-ascend==0.18.0
    /data/venvs/vllm/bin/pip install --no-deps -e /data/work/vllm-rwkv-ascend

This preserves the vendor-matched torch 2.9 stack. Import and NPU smoke results
must pass before serving; do not suppress failures.

## HF checkpoint requirements

config.json requires architecture NativeRWKV7ForCausalLM and explicit
hidden_size, num_hidden_layers, intermediate_size, head_size plus
decay_lora_dim, aaa_lora_dim, gate_lora_dim and value_lora_dim.

Weights use the standard HF safetensors names: model.layers.*,
model.embeddings.weight, model.norm.*, and lm_head.weight. The exact real 7.2B
checkpoint schema is validated in evidence/real_7b_schema.txt.

## Commands

    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /data/venvs/vllm/bin/activate
    unset VLLM_PLUGINS  # load both ascend platform and RWKV model plugins
    pytest -q tests_vllm

    vllm serve /path/to/rwkv7-hf \
      --trust-remote-code --dtype bfloat16 --enforce-eager \
      --enable-chunked-prefill --max-num-batched-tokens 2048 \
      --max-num-seqs 64 --disable-prefix-caching

The plugin uses vLLM's OpenAI server. It does not start or wrap FastAPI itself.

## Real 910B3 acceptance

`evidence/real_engine_smoke_7.log` records a full 13.41GB checkpoint load and
real prefill/decode ending in `SMOKE_OK`. `evidence/real_engine_acceptance.json`
and `evidence/real_engine_scheduler_trace.jsonl` record two reverse-order
three-request batches (1/47/180 input tokens), actual scheduler metadata under
a 32-token prefill budget, mixed decode+prefill steps and
fresh-to-continuation transitions. The same atomic run proves identical greedy
outputs under reverse-order scheduling, physical cache-slot reuse, stale state
present before reuse, and explicit zeroization before the fresh request runs.

Only the normalized device name `Ascend 910B3` is enabled by default. Prefix cache, speculative decode, TP,
PP and Mamba cache modes other than `none` fail before serving.

## Required hardware gates before publishing performance

1. Whole-prompt vs 128/256/512 and irregular 123/511/777/637 chunks: logits,
   token and all state tensors must match.
2. Dynamic join, early finish, cancel and request reorder: each stream must
   equal standalone B=1.
3. Cache slot reuse must be zeroized; 30-minute churn must not grow HBM.
4. Same real checkpoint/dtype/prompt/decode for HF oracle and vLLM.
5. W8/W4 remain production-disabled. Selected 4096↔16384 row counts are
   raw-operator candidates only; prior module/model probes did not pass every
   engine latency and quality gate. `production_accepted` stays false until the
   exact scheduler shapes pass output, peak-HBM, and speed `>= FP16` together.

## Host-versus-official matrix risk

The official vllm-ascend 0.18 matrix pins vLLM 0.18.0, CANN 8.5.1,
PyTorch 2.9.0 and torch_npu 2.9.0.post1+git4c901a4. The main-branch matrix also
lists CANN 8.5.0 with torch/torch_npu 2.9.0. This host currently has CANN 8.5.0
and torch_npu 2.9.0 (without post1). The release-matrix mismatch remains a published risk, although the real 7.2B
engine smoke and dynamic/chunked acceptance pass on this exact 910B3 host. Other
Ascend devices remain fail-closed by default and are not claimed supported. Reference:
https://docs.vllm.ai/projects/ascend/en/v0.18.0/community/versioning_policy.html
