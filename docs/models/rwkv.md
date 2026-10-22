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
`tokenizers-rwkv` commits and `FlashRWKV2==0.1.0a16`.

## Serve a model

RWKV uses BF16 weights, activations, and shift state with FP32 accumulation,
WKV recurrent state, and head logits. Chunked prefill is required. Prefix
caching is disabled by default, so each request updates one recurrent state.

```bash
vllm serve <model> \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

RWKV defaults to `CompilationMode.NONE` and full CUDA Graph capture so that
TorchDynamo does not decompose the external fused operators. Use
`--enforce-eager` only for diagnostics.

To opt into prefix sharing and copy-on-write, pass `--enable-prefix-caching`.
This uses vLLM's state-aligned cache blocks and copies recurrent state at block
boundaries.

Continuous batching, asynchronous scheduling, structured outputs, and the
OpenAI-compatible Chat and Responses APIs use the standard vLLM paths.

## Production deployment

The `RWKV production` workflow uses `tools/deploy_rwkv_production.sh` to build
each source revision in its own release directory and `.venv`. Both model
hosts retain their existing g1l artifacts and served model names. No weight
conversion or download is required for the a16 upgrade.

The deployment commands use BF16 weights, FP32 WKV state, prefix caching off,
and a 1024-token batch budget. Full CUDA Graphs use that token budget as their
maximum capture size. The per-GPU sequence limits account for the larger WKV
state. The small-model host has 24 GB GPUs and the large-model host has 48 GB
GPUs:

| Model | GPUs | Max sequences per GPU | BF16 weights + active state (GiB) |
| --- | ---: | ---: | ---: |
| g1l 1.5B | 1 | 1024 | 15.03 |
| g1l 2.9B | 1 | 512 | 15.65 |
| g1l 7.2B | 2, data parallel | 128 | 17.47 |
| g1l 13.3B | 4, data parallel | 256 | 40.20 |

The memory column is a lower bound from the installed artifacts. For `L`
layers, hidden size `C`, and head size `D`, each active sequence requires
`L * C * (4 * D + 4)` bytes for FP32 WKV state and the two BF16 shift states.
Model warmup must also accommodate CUDA Graphs, activations, and workspaces
within the configured memory budget. Data-parallel replicas each hold the full
model and their own state pool.

Stage the new release before switching services. The deployment script
snapshots the current release, unit files, enabled services, and active services
before activation; failed health checks restore that snapshot. Keep the old
release and its `.venv` intact for rollback. A push to `rwkv` triggers the
workflow. Its `models` target updates both model hosts, and a deployment-script
change also selects the router component. Deployment activates the small-model
host, the large-model host, then the frontend router. The router retains the
backends' g1l model IDs and the existing tunnel ports 28001 through 28004.

## Decoding defaults

RWKV samples through FlashRWKV2 Rapid-Sampling. The model artifact must publish
three standalone generation configs:

| Request mode | Config file | temperature | top_p | top_k | presence | frequency | decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Open Think | `generation_config.json` | 0.96 | 0.76 | 32 | 1.0 | 0.1 | 0.988 |
| Fake Think | `fake_think_generation_config.json` | 1.0 | 0.28 | 32 | 0.0 | 0.0 | 1.0 |
| Tools | `tools_generation_config.json` | 0.96 | 0.76 | 32 | 0.0 | 0.0 | 1.0 |

These must be standalone `GenerationConfig` documents. Omit
`_from_model_config` or set it to `false`; otherwise Transformers rebuilds the
configuration from the model config and drops the RWKV-only penalty fields.

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
  --dtype bfloat16 \
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
  --dtype bfloat16 \
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
