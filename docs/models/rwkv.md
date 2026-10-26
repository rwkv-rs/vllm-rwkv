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

The requirements file fixes the `transformers-rwkv`, `tokenizers-rwkv`, and
FlashRWKV2 commits. FlashRWKV2 must expose the public `prepare_tuning` and
`get_tuning_cache_dir` APIs in addition to its recurrent-state provider.

## Serve a model

RWKV uses BF16 weights, activations, and shift state with FP32 accumulation,
WKV recurrent state, and head logits. Chunked prefill is required. Prefix
caching is disabled by default, so each request updates one recurrent state.
The default FP32 recurrence follows the training precision contract.

```bash
vllm serve <model> \
  --dtype bfloat16 \
  --mamba-ssm-cache-dtype float32 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching
```

For optional lower-memory inference, explicitly pass
`--mamba-ssm-cache-dtype float16`. This retains BF16 token inputs/outputs and
shift state, FP32 arithmetic and head logits, and rounds the WKV state to FP16
after every token. It follows a different numerical trajectory from FP32
training and can change logits and generated tokens. Prefill and decode use
the same per-token state rounding. This option does not change training
precision. Production uses this FP16 state option for higher throughput.

RWKV defaults to `CompilationMode.NONE` and full CUDA Graph capture so that
TorchDynamo does not decompose the external fused operators. Use
`--enforce-eager` only for diagnostics.

To opt into prefix sharing and copy-on-write, pass `--enable-prefix-caching`.
This uses vLLM's state-aligned cache blocks and copies recurrent state at block
boundaries.

Continuous batching, asynchronous scheduling, structured outputs, and the
OpenAI-compatible Chat and Responses APIs use the standard vLLM paths.

## Kernel preparation

FlashRWKV2 stores verified tuning profiles and native builds under
`~/.cache/flashrwkv2`. Set `FLASHRWKV2_CACHE_DIR` before starting the process to
choose another persistent root. The tuning root is independent of the virtual
environment, release directory, `TORCH_HOME`, and `XDG_CACHE_HOME`. TileLang
0.1.15 keeps its separate JIT cache at `~/.tilelang/cache`.

`VLLM_RWKV_TUNING_MODE` controls preparation before CUDA Graph capture:

- `inspect` (default) logs actual dispatch coverage and permits untuned defaults.
- `require` loads verified selections and fails on the first missing profile,
  with the operator, full key, and cache root in the diagnostic.
- `tune` calls FlashRWKV2 to verify and tune only missing keys, then requires
  complete coverage before capture. Use this explicitly during preparation.

Preparation uses the loaded model weights, configured state dtype, and each
actual graph shape, including the separate head/sampling graphs. Model tuning
uses fully active metadata and startup state slots, which are reset before
capture. The report records each selected configuration, fingerprint, device
class, profile path, and hit/missing/tuned/revalidated status. Same-class GPUs
and new worker processes reuse profiles through FlashRWKV2's locking and
compatibility checks.

Run a preparation process with the same model and serving arguments as the
target service, including its sequence limit, token budget, and capture sizes:

```bash
FLASHRWKV2_CACHE_DIR=/path/to/persistent/flashrwkv2 \
VLLM_RWKV_TUNING_MODE=tune \
timeout 7200s .venv/bin/vllm serve <model> <serving-arguments>
```

After preparation, restart with `VLLM_RWKV_TUNING_MODE=require` to verify loading
and capture in a fresh process. Eager workloads can use FlashRWKV2's public
`prepare_tuning` with representative forward calls and disposable state.

For a legacy namespace, add
`VLLM_RWKV_TUNING_LEGACY_DIR=/path/to/tuning/<legacy-namespace>` to the explicit
`tune` run. FlashRWKV2 revalidates matching old configurations against the
current operators before publishing them; it tunes missing shapes normally.
Preserve the original namespace until the new process passes coverage checks.

## Production deployment

The `RWKV production` workflow uses `tools/deploy_rwkv_production.sh` to build
each source revision in its own release directory and `.venv`. Both model
hosts retain their existing g1l artifacts and served model names. No weight
conversion or download is required for the tuning cache update.

Model services set `FLASHRWKV2_CACHE_DIR=/opt/rwkv/vllm/cache/flashrwkv2` and
`VLLM_RWKV_TUNING_MODE=require`. `prepare_flashrwkv2` checks the native extension
under that root through the public cache API. Each service checks actual model
and sampling dispatch coverage before capturing its graphs, and deployment
health probes must pass before activation succeeds. Prepare or revalidate
profiles for the new revision before deployment. Native builds retain their
own ABI/toolchain compatibility keys; vLLM compilation caches remain local to
each release. TileLang retains its default `~/.tilelang/cache` directory.

The deployment commands use BF16 weights, FP16 WKV state, prefix caching off,
and a 1024-token batch budget. Full CUDA Graphs use that token budget as their
maximum capture size. The per-GPU sequence limits are selected from serving
throughput measurements and the available recurrent-state slots. The small-model
host has 24 GB GPUs and the large-model host has 48 GB GPUs:

| Model | GPUs | Max sequences per GPU | Active state per sequence (MiB) |
| --- | ---: | ---: | ---: |
| g1l 1.5B | 1 | 256 | 6.19 |
| g1l 2.9B | 1 | 1024 | 10.31 |
| g1l 7.2B | 2, data parallel | 256 | 16.50 |
| g1l 13.3B | 4, data parallel | 256 | 31.45 |

For `L` layers, hidden size `C`, and head size `D`, each active sequence requires
`L * C * (2 * D + 4)` bytes for FP16 WKV state and the two BF16 shift states.
Model warmup must also accommodate CUDA Graphs, activations, and workspaces.
Data-parallel replicas each hold the full model and their own state pool.
The scheduler allocates its state pool from the profiled GPU memory budget,
so `--max-num-seqs` limits active sequences rather than total allocated slots.

Before publishing tuning profiles, the 2026-10-26 serving sweep used the installed
g1l checkpoints, FlashRWKV2 a16, 128 random input tokens, greedy sampling, and
streaming completions. Request
count and client concurrency both equaled the per-GPU batch size times the
number of data-parallel replicas. Output throughput includes prefill and decode.

| Model | Output tokens/request | Batch per GPU | Output tokens/s | Runs |
| --- | ---: | ---: | ---: | ---: |
| 1.5B | 1024 | 256 | 15,514 [14,774–16,319] | 3 |
| 1.5B | 1024 | 512 | 13,457 [12,011–14,463] | 3 |
| 2.9B | 1024 | 512 | 9,324 [9,324–9,337] | 3 |
| 2.9B | 1024 | 1024 | 9,453 [9,453–9,453] | 2 |
| 7.2B DP2 | 512 | 256 | 5,019 [5,017–5,021] | 2 |
| 7.2B DP2 | 512 | 320 | 4,887 [4,884–4,890] | 2 |
| 13.3B DP4 | 512 | 256 | 4,617 [4,617–4,618] | 2 |
| 13.3B DP4 | 512 | 448 | 4,402 | 1 |

Repeated results show the median and [minimum–maximum]. Every measured request
completed successfully. The selected limits favor throughput: larger batches
reduced long-output throughput for 1.5B, 7.2B, and 13.3B. The 2.9B result remained
slightly higher at 1024 than at 512. The 13.3B allocation supplied approximately
476 FP16 state slots per GPU, including the scheduler's reserved slot.

The profile deployment comparison used 128 input and 1024 output tokens for all
four models, with the same sampling settings and seed 20261026. Each model ran
once before and once after deployment. All requests completed successfully.
The deployed profiles cover the per-GPU decode batch and 1024-token prefill:

| Model | Batch per GPU | Requests/concurrency | Before output tokens/s | After output tokens/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.5B | 256 | 256 | 13,248 | 14,477 | +9.3% |
| 2.9B | 1024 | 1024 | 9,444 | 10,681 | +13.1% |
| 7.2B DP2 | 256 | 512 | 4,599 | 10,753 | +133.8% |
| 13.3B DP4 | 256 | 1024 | 4,335 | 7,359 | +69.8% |

[PR #80](https://github.com/rwkv-rs/vllm-rwkv/pull/80) records the baseline and
deployed revisions, raw artifact locations, latency measurements, and
production validation. New processes using the production installations
verified actual dispatch hits for all 50 published profiles: 36 on the
small-model host and 14 on the 13.3B host.

Reproduce the 7.2B DP2 profile comparison on its model host:

```bash
ulimit -n 65536
timeout 600s .venv/bin/vllm bench serve \
  --backend openai --base-url http://127.0.0.1:18003 \
  --endpoint /v1/completions --model rwkv7-g1l-7.2b \
  --tokenizer /srv/rwkv/models/rwkv-g1l-7b-temp-3797 \
  --dataset-name random --random-input-len 128 --random-output-len 1024 \
  --num-prompts 512 --max-concurrency 512 --ignore-eos \
  --temperature 0 --seed 20261026 --disable-tqdm \
  --extra-body '{"top_k":1,"presence_penalty":0,"frequency_penalty":0,"penalty_decay":1}'
```

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

Single GPU and two-stage pipeline parallelism are supported. On the production
RTX 4090 host, the 7.2B model with FP16 WKV state measured 2,268 tokens/s on one
GPU at batch 128 and 4,507 tokens/s with DP2 at batch 128 per GPU. At batch 256
per GPU, one GPU measured 2,519 tokens/s and DP2 measured 5,019 tokens/s.
Both comparisons used 128 input / 512 output tokens and the serving benchmark
above; DP2 achieved approximately 1.99 times single-GPU aggregate throughput.

For PP2, run one stage per GPU:

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

## Unmerged LoRA and NoRA adapters

RL trainers that keep one PEFT LoRA or NoRA adapter unmerged can serve it
directly. Enable it with the `rwkv_lora` config attribute:

```bash
vllm serve <model> \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --hf-overrides '{"rwkv_lora": {"r": 256, "lora_alpha": 256, "use_nora": true, "target_modules": ["r_proj", "k_proj", "v_proj", "o_proj", "lm_head"]}}'
```

`r`, `lora_alpha`, `use_nora`, and `target_modules` are all required. The scale
is `lora_alpha / r`. `target_modules` is a subset of `r_proj`, `k_proj`,
`v_proj`, `o_proj`, `mlp.key`, `mlp.value`, and `lm_head`, with PEFT suffix
matching. Each targeted projection gets zero-initialized FP32 parameters in
PEFT layout, `lora_A.weight` `[r, in_features]` and `lora_B.weight`
`[out_features, r]`, named after the base weight, for example
`model.layers.3.linear_attn.r_proj.lora_A.weight`,
`model.layers.3.mlp.value.lora_B.weight`, and `lm_head.lora_A.weight`.
`mlp.value` keeps the checkpoint layout, `[r, intermediate_size]` and
`[hidden_size, r]`, although its base weight is transposed after loading.
Pipeline stages create adapters only for the projections they own.

`load_weights` copies adapter tensors into the existing FP32 storage, so
captured CUDA Graphs stay valid across updates. The raw adapters, scale, and
NoRA flag are passed to the FlashRWKV2 projection operators, which apply NoRA
normalization from the raw A matrix. These are the operators and arguments of
the `transformers-rwkv` forward with an active unmerged PEFT adapter. Without
`rwkv_lora`, the model has no adapter parameters and passes no adapter
arguments.

## Unsupported features

RWKV rejects tensor parallelism, decode or prefill context parallelism,
speculative decoding, quantized weights, vLLM LoRA adapters (`--enable-lora`),
KV cache offloading/connectors, ReplaySSM, stochastic cache rounding, and Mamba
cache mode `all`. Multiplicative `repetition_penalty`, `min_p`, processed logprobs,
sampling masks, and trace replay are not supported by Rapid-Sampling. Pipeline
parallel sizes other than one or two are also rejected.
