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
`tokenizers-rwkv` commits and `FlashRWKV2==0.1.0a13`.

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

## Reuse recurrent state across chat requests

RWKV State refs let a client continue a chat without sending the complete
conversation on every request. The ref stores a process-local recurrent-state
snapshot; it does not consume a scheduler request slot. Enable the cache with a
positive byte limit before starting the server:

```bash
VLLM_RWKV_STATE_CACHE_MAX_BYTES=$((4 * 1024 * 1024 * 1024)) \
  vllm serve <model> \
  --dtype float16 \
  --enable-chunked-prefill
```

Ask the server to generate an unguessable ref for the initial turn:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "messages": [{"role": "user", "content": "Remember that my build is 42."}],
    "vllm_xargs": {"rwkv_state_write_ref": "auto"}
  }'
```

The normal response and the terminal streaming chunk include
`rwkv_state_ref`. To continue, send only newly appended messages, restore that
ref, and write a new ref:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "messages": [{"role": "user", "content": "What is my build number?"}],
    "vllm_xargs": {
      "rwkv_state_read_ref": "<previous-rwkv_state_ref>",
      "rwkv_state_write_ref": "auto"
    }
  }'
```

For the canonical artifact template, the server inserts the inter-turn
boundary before each restored delta. A naturally stopped bot turn contributes
its pending stop token followed by a newline; a length- or repetition-limited
turn is closed with `✿` followed by a newline. The `assistant` and
`function_calling` prompt styles use their corresponding newline separators.

State refs are immutable. Keep the old ref to branch a conversation, clone it
without copying its snapshot, or delete it when it is no longer needed:

```bash
curl http://localhost:8000/v1/rwkv/state/capabilities
curl http://localhost:8000/v1/rwkv/state/<state-ref>
curl -X POST http://localhost:8000/v1/rwkv/state/<state-ref>/clone \
  -H "Content-Type: application/json" \
  -d '{"target_ref": "branch-1"}'
curl -X DELETE http://localhost:8000/v1/rwkv/state/<state-ref>
```

Each State request supports one text-token input and one sampled sequence.
State refs are held only by the serving process, are lost on restart, and do
not currently support data parallelism. A continuation's new prompt plus
output must still fit `--max-model-len`; the ref removes replay of earlier
turns, not the per-request limit. Prefix-cache reads and writes are disabled
when a State ref is restored so the two state sources cannot be mixed or leak
into ordinary prefix-cache entries.

Quiesce active requests before reloading or transferring model weights. These
operations invalidate existing State refs and pending read/write reservations
before changing the target model, including when the weight update later fails.
Updating a separate speculative draft model does not invalidate target-model refs.

Treat refs as bearer capabilities. Prefer `"auto"`, do not expose the State
management endpoints without the same access controls as generation, and drop
unused refs to release memory. State reuse extends computational history but
does not by itself establish model quality beyond the model's evaluated
context distribution.

Snapshot restoration preserves the provider's state representation. It does not
guarantee bitwise equivalence between different prefill chunk sizes: provider
rounding and reduction order can still affect logits and generated tokens. Check
the intended workload with the equivalence benchmark before relying on state reuse.

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
