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

For clean comparison runs, use:

```bash
tools/rwkv_profile/measure_profile.sh rwkv /path/to/clean/source /path/to/new/output
```

Each output contains dependency and build logs, resolved distributions,
dependency consistency output, the native target manifest, and size/time
metrics.
