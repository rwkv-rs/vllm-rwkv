# Out-of-tree plugins maintained by this fork

## RWKV-7 for Ascend

[`rwkv7-vllm-ascend`](rwkv7-vllm-ascend/) is a vLLM V1 model plugin validated
with vLLM/vllm-ascend 0.18.0 on one Ascend 910B3. It provides recurrent
`MambaSpec` state, continuous batching, mixed decode and chunked prefill,
standard Hugging Face checkpoint loading, fail-closed runtime admission, and
real-engine acceptance artifacts.

It remains out of tree so the Huawei-specific runtime can be installed and
versioned independently of vLLM core. The W8/W4 FFN seam is experimental and
production-disabled; the dense serving path is the accepted default.
