# RWKV development

This project is the RWKV community's authoritative vLLM adaptation repository. It needs to complete RWKV adaptation for upstream in accordance with mainstream community practices (refer to the **functional design** and **code style** of models with Linear RNN Layer, such as Qwen3.5 and Kimi-K3 in vLLM).

Code principle: For every file/type/function/variable, a similar implementation must be found as a prototype. If that prototype carries a model name, replace it with `RWKV` or another case variant; otherwise keep the same name. Using branch statements without affecting existing functionality.

Process principle: Strictly follow <https://docs.vllm.ai/en/latest/contributing>. Any development step should follow the instructions in the official documentation.

For the computation flow of the RWKV7 model, it is necessary to find ways to release hardware performance as much as possible while ensuring a certain degree of maintainability.

Due to the characteristics of the RWKV7 model, such as O(1) complexity and no KV Cache, conventional optimization methods for Transformer-like models are not applicable, such as PageAttention, because RWKV can achieve completely static VRAM allocation.

We generally refer to other authoritative RWKV implementations, or the Kimi-k3 (with Kimi-Delta-Attention) implementation, for optimization.

This repository's support for rwkv7 inference should fully align with Albatross in numerical precision and throughput. Use the official vllm bench for speed measurement. If a custom bench is needed, it must be completed end-to-end in a real 7.2B model generation scenario with batch_size = {1, 4, 64, 320, 512}. Prefill speed = prompt length / first token latency; decode speed = completion length / (total generation time - first token latency). Implement asynchronous detokenization, so this part should not affect generation speed.

FlashRWKV (<https://github.com/rwkv-rs/FlashRWKV2>) is the RWKV community's authoritative operator implementation repository and provides a high-performance backend for this repository. This repository only imports, and does not develop, operator-related content. If precision and inference speed suffer from poor precision/slow inference due to errors in the FlashRWKV2 implementation, feedback should be given directly to the user; there is no need to cross the implementation boundary to perform fixes.

## Authoritative RWKV7 Implementations

(1) <https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py>
(2) <https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py>
(3) <https://github.com/BlinkDL/Albatross> -- authoritative low-level inference engine implementation repository (CUDA, for Pro6000, no scheduling, no varlen)
(4) <https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp> -- authoritative pretraining implementation repository (CUDA, for H100)
(5) <https://zhiyuan1i.github.io/posts/dplr-mathematics> -- mathematical principles of Diagonal Plus Low Rank (DPLR): parallel computation of explicit transition matrices
(6) **<https://github.com/rwkv-rs/transformers-rwkv>** -- authoritative RWKV Huggingface Transformers adaptation repository (with Rust tokenizer, 10x faster than the Python implementation)

## RWKV7 Weights

General weight naming convention: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
For example: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: architecture version, such as rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: data version, such as g1a, g1b... (The further back in the alphabet, the better)
param_size: parameter scale, only 0.1b, 0.4b, 1.5b(often used in RL), 2.9b, 7.2b(often used in the infer test), 13.3b
(1) <https://huggingface.co/BlinkDL/rwkv7-g1/tree/main> -- authoritative weight Release source (update every month)
(2) <https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main> -- authoritative weight Test source (updated irregularly)
(3) <https://huggingface.co/rwkv-rs/rwkv7-g1-st> -- authoritative weight Release source (for transformers)
After conversion to safetensor format, they are fixed in the `~/Weights/RWKV/hf` directory on `rwkv-sha-pro6000x8`; do not download them repeatedly.
