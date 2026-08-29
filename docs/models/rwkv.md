# RWKV-7

vLLM serves RWKV-7 artifacts with the native `RwkvForCausalLM`
implementation. The artifact must use the canonical `model_type="rwkv"`
configuration, safetensors weight names, tokenizer, and chat template from
[`transformers-rwkv`](https://github.com/rwkv-rs/transformers-rwkv).

RWKV execution requires CUDA and the public fused operators and recurrent-state
provider in FlashRWKV2. The vLLM implementation does not include a CUDA, Triton,
or PyTorch fallback.

## Install dependencies

Create and activate the project `.venv`, install vLLM, and then install the
pinned RWKV dependencies:

```bash
uv pip install -r requirements/rwkv.txt
```

The requirements file fixes the validated `transformers-rwkv` and
`tokenizers-rwkv` commits and `FlashRWKV2==0.1.0a11`.

## Serve a model

RWKV uses FP16 weights and FP16 recurrent state by default. Chunked prefill is
required, and prefix caching uses vLLM's state-aligned cache blocks.

```bash
vllm serve <model> \
  --dtype float16 \
  --mamba-ssm-cache-dtype float16 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

Use FP32 recurrent state with FP16 operator input and output (FP32IO16) when
additional state precision is required:

```bash
vllm serve <model> \
  --dtype float16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --enable-prefix-caching
```

RWKV defaults to `CompilationMode.NONE` and full CUDA Graph capture so that
TorchDynamo does not decompose the external fused operators. Use
`--enforce-eager` only for diagnostics.

Continuous batching, asynchronous scheduling, prefix sharing and copy-on-write,
structured outputs, and the OpenAI-compatible Chat and Responses APIs use the
standard vLLM paths.

## Pipeline parallelism

Single GPU and two-stage pipeline parallelism are supported. For PP2, run one
stage per GPU:

```bash
vllm serve <model> \
  --dtype float16 \
  --pipeline-parallel-size 2 \
  --enable-chunked-prefill
```

The first stage owns embedding and LN0. The second stage owns the final norm and
LM head. `hidden_states`, `residual`, and `v_first` are transferred between the
stages.

## Tool calling

The canonical artifact chat template emits `**Tool Call:**` followed by fenced
JSON. Load that template directly from the artifact and select the RWKV parser:

```bash
vllm serve <model> \
  --dtype float16 \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser rwkv
```

The parser supports streaming and non-streaming requests with automatic,
required, or named tool choice. Do not pass a copied chat template.

## Unsupported features

RWKV rejects tensor parallelism, decode or prefill context parallelism,
speculative decoding, quantized weights, vLLM LoRA adapters, KV cache
offloading/connectors, ReplaySSM, stochastic cache rounding, and Mamba cache
mode `all`. Pipeline parallel sizes other than one or two are also rejected.
