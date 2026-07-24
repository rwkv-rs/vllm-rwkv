# Rebuilt vLLM-Ascend 0.18 acceptance

This is a clean rerun on one Ascend 910B3 after the original validation VM was
destroyed. The engine used vLLM 0.18.0, vllm-ascend 0.18.0, plugin 0.3.0,
CANN 8.5.0, torch 2.9.0+cpu and torch_npu 2.9.0.

The real `fla-hub/rwkv7-7.2B-g0a` checkpoint loaded 13.4100 GB of weights.
With `max_num_batched_tokens=32`, two reverse-order three-request batches used
1-, 47-, and 180-token prompts and greedily decoded three tokens each.

## Gates

| Gate | Rebuilt result |
|---|---|
| real vLLM V1 engine load on NPU | pass |
| dynamic multi-request batching | pass |
| mixed decode + prefill steps | 6 |
| multi-prefill-request steps | 3 |
| actual prefill tokens / prefill steps | 455 / 16 |
| largest prefill step | 32 tokens |
| fresh / continuation recurrent segments | 5 / 14 |
| reverse-order outputs | identical |
| physical Mamba state slots reused | `[2, 3]` |
| reused slots nonzero before clear | `[2, 3]` |
| reused slots zero after clear | pass |
| `Hello` greedy IDs | exact `[45, 308, 459]` |

The scheduler trace SHA256,
`2f5a8c408295cb034bec37b135f25e76f5a07c20fd5a31e47cd1b273d139c2d6`,
exactly matches the original run. `SHA256SUMS` pins this rebuild's JSON, trace,
and full log.

The startup log contains a non-fatal generic-vLLM Triton warning about
`triton.tools.ragged_tma`; the model runs in enforced eager mode through the
Ascend platform and the complete acceptance exits successfully.

The `pr_real_engine_*` files are a third run made from the exact ruff-formatted
plugin tree proposed in this repository. It also reports `ACCEPTANCE_OK`; its
scheduler trace has the same SHA256 as both earlier runs. `pr_SHA256SUMS`
authenticates that JSON, trace, and full log.
