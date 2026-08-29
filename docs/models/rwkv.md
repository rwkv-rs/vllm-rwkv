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

## Decoding defaults

RWKV samples through FlashRWKV2 Rapid-Sampling. The model artifact must publish
three standalone generation configs:

| Request mode | Config file | temperature | top_p | top_k | presence | frequency | decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Think | `generation_config.json` | 0.96 | 0.76 | 32 | 1.0 | 0.1 | 0.988 |
| Fake Think | `fake_think_generation_config.json` | 1.0 | 0.28 | 32 | 0.0 | 0.0 | 1.0 |
| Tools | `tools_generation_config.json` | 0.96 | 0.76 | 32 | 0.0 | 0.0 | 1.0 |

The default template mode selects Open Think. Setting
`chat_template_kwargs.rwkv_generation_prompt` to `fake_think` selects Fake
Think. Supplying tools always selects the Tools profile, regardless of the
thinking mode. Explicit request sampling fields override the selected profile.

All six controls are executed by
`infer_sampling_six_parameter_forward_varlen`; `frequency_penalty` is the
Rapid-Sampling additive increment and `penalty_decay` decays its per-request
token state. `--generation-config vllm` explicitly disables artifact profiles.

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
mode `all`. Multiplicative `repetition_penalty`, `min_p`, processed logprobs,
sampling masks, and trace replay are not supported by Rapid-Sampling. Pipeline
parallel sizes other than one or two are also rejected.
