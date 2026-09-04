# RWKV source build

This checkout keeps vLLM's unrestricted `full` build as the default. The
explicit reduced `rwkv` CUDA profile targets dense RWKV7 raw `.pth`
checkpoints and supports the built-in RWKV tokenizer, text
chat/completions, streaming, stop handling, Prometheus metrics, rapid sampling,
and sleep/wake GPU-memory release through the CuMem allocator with TP=1, PP=1,
and DP=1.

Install the explicit dependency set and disable dependency/build isolation when
installing the source tree:

```bash
uv venv --python 3.12 .venv-rwkv
uv pip install --python .venv-rwkv/bin/python -r requirements/rwkv.txt
VLLM_BUILD_PROFILE=rwkv \
VLLM_TARGET_DEVICE=cuda \
uv pip install --python .venv-rwkv/bin/python \
  --no-deps --no-build-isolation --editable .
```

When building a copied source snapshot without `.git` metadata, also set a
deterministic `SETUPTOOLS_SCM_PRETEND_VERSION` (for example, the source
revision's release version). A normal Git checkout does not need this override.

`--no-deps` avoids resolving a second time after the explicit requirements
install. The selected RWKV artifact records that same reduced set in its
package metadata and receives an `rwkv` version label, so dependency-consistency
checks remain meaningful without a second dependency resolution.
`--no-build-isolation` reuses the build tools declared by
`requirements/rwkv.txt` and avoids resolving an unrelated `pyproject.toml` build
environment. The generated `vllm/_build_profile.json` records the profile,
native targets, external projects, architecture, weight format, device,
runner, and TP/PP/DP boundaries that were actually configured.

| Capability | Supported |
| --- | --- |
| Dense RWKV7 raw `.pth` | yes |
| Built-in RWKV tokenizer and text OpenAI API | yes |
| Rapid sampler first-use JIT | yes |
| Sleep mode and CuMem allocator | yes |
| Other architectures or weight formats | no |
| Quantization, multimodal, speculative decoding, LoRA | no |
| Responses, Anthropic, generative-scoring, MCP routes | no |
| Structured-output constraints | no |
| SageMaker container-standard routes | no |
| TP, PP, or DP greater than one; Ray | no |
| Generic stable/MoE operators and attention external projects | no |
| Rust frontend | no |

Unsupported configurations fail during `VllmConfig` validation before engine
workers start. This checkout rejects precompiled non-RWKV extensions rather
than relabeling them as reduced artifacts.

`VLLM_BUILD_PROFILE` defaults to `full`, preserving the upstream platform,
extension, and Rust frontend build paths. Set it to `rwkv` only when producing
the capability-limited artifact described above.

## Reuse recurrent state across chat turns

RWKV State refs let clients continue a conversation without replaying its full
token history. A ref is an immutable, process-local recurrent-state snapshot;
it does not occupy a scheduler request slot. Set a positive byte limit before
starting the server:

```bash
VLLM_RWKV_STATE_CACHE_MAX_BYTES=$((4 * 1024 * 1024 * 1024)) \
  vllm serve <model> --dtype float16 --enable-chunked-prefill
```

Ask the server to create the initial ref:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "messages": [{"role": "user", "content": "Remember build 42."}],
    "vllm_xargs": {"rwkv_state_write_ref": "auto"}
  }'
```

The response, or the terminal streaming chunk, contains `rwkv_state_ref`. Send
only the next message while reading the prior ref and writing a new one:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model>",
    "messages": [{"role": "user", "content": "What build did I give you?"}],
    "vllm_xargs": {
      "rwkv_state_read_ref": "<previous-ref>",
      "rwkv_state_write_ref": "auto"
    }
  }'
```

The server preserves the unprocessed terminal token, inserts the native RWKV
turn boundary, and removes the repeated BOS token from each continuation delta.
State-restored requests bypass prefix-cache reads and writes so recurrent state
from the two mechanisms cannot be mixed.

Refs can be inspected, cloned without copying their tensor storage, and dropped:

```bash
curl http://localhost:8000/v1/rwkv/state/capabilities
curl http://localhost:8000/v1/rwkv/state/<state-ref>
curl -X POST http://localhost:8000/v1/rwkv/state/<state-ref>/clone \
  -H "Content-Type: application/json" \
  -d '{"target_ref": "branch-1"}'
curl -X DELETE http://localhost:8000/v1/rwkv/state/<state-ref>
```

Each State request supports one token input and one sampled sequence. Refs are
lost on restart and currently reject data parallelism. Treat them as bearer
capabilities, prefer server-generated `"auto"` refs, and delete unused refs to
release GPU memory. A continuation delta plus its output must still fit
`--max-model-len`; the ref removes history replay but does not change the
per-request limit.

Compare deterministic top-1 output against full-token replay before running
longer quality evaluations. The benchmark warms both paths before collecting
latency:

```bash
.venv/bin/python benchmarks/rwkv7/benchmark_stateful_chat.py \
  --input-file benchmarks/rwkv7/stateful_chat_example.json \
  --url http://127.0.0.1:8000 \
  --require-exact-match \
  --output rwkv-state-results.json
```

For clean comparison runs, use:

```bash
tools/rwkv_profile/measure_profile.sh rwkv /path/to/clean/source /path/to/new/output
```

Each output contains dependency and build logs, resolved distributions,
dependency consistency output, the native target manifest, and size/time
metrics.
