# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Model evaluation results when the change affects output, accuracy, or serving.
    - Clear statement that AI assistance was used.
- Before opening a PR (drafts included) or requesting re-review, run the [`pr-checklist`](.agents/skills/pr-checklist/SKILL.md) skill and address its findings.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# Start with precompiled artifacts for an editable install:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

For C/C++ or CUDA changes, follow the
[incremental compilation workflow](docs/contributing/incremental_build.md) to
configure and perform incremental builds.

### Tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies (use cuda.in on non-x86_64):
uv pip install -r requirements/test/cuda.in

# Run a specific test file:
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

When adding tests:

- **Design before you write.** Answer four questions first: what is the module
  for, what is its I/O contract, what failure am I guarding against, and what is
  the cheapest level that catches it (unit over integration over e2e)?
- **Reuse before create.** Extend existing test files, `conftest.py` fixtures, and
  helpers; add a new file only when no nearby suite fits.
- **Test behavior with intent.** Assert observable outcomes through public APIs;
  state why in the name or docstring. Skip trivial wiring; flaky tests are worse
  than no tests.
- **Keep it minimal.** One behavior per test and the smallest setup that
  triggers it; if the test diff dwarfs the code change, cut scope.
- **No one-off kernel benchmarks in `tests/`.** Put kernel perf work in
  `benchmarks/kernels/`; prove correctness in existing pytest suites.
- **Run model evals for model-affecting changes.** Search `tests/evals/` or use
  `vllm bench` and include results in the PR — do not wait for reviewers to ask.

For model-specific requirements, see
[`docs/contributing/model/tests.md`](docs/contributing/model/tests.md).

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Coding style guidelines

- Match existing code style
- Minimize use of comments. Eliminate comments which are redundant, preferring legible and self-documenting code. When used, keep docstrings and comments brief and direct.
- Assume the reader is familiar with vLLM.

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`):

```text
Your commit message here

Co-authored-by: Agent Name Here
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

Security reviewers should start with [`SECURITY.md`](SECURITY.md),
[`docs/usage/security.md`](docs/usage/security.md), and
[`docs/contributing/vulnerability_management.md`](docs/contributing/vulnerability_management.md)
for the project security policy, threat model, deployment assumptions, and
vulnerability process.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.

## Core Objectives

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

## Directory Conventions

When adding files, you must ask the user.

`setup.py` and `pyproject.toml` define the build and packaging contract. Do not commit generated files such as `build/`, `artifacts/`, cache directories, or `.egg-info`.

## Env

Use uv to manage the dedicated local and remote environment ./.venv. Using other environments is strictly prohibited to avoid environment pollution issues.

## Machine for Testing and Benchmarking

```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/vllm-rwkv
```

Use git to sync your changes instead of rsync.

## Machine for Deployment

url: <https://vllm.rwkvos.com>

```bash
ssh rwkv-szx-4090x4-ip129
ssh rwkv-szx-4090dx4-ip157
ssh rwkv-alicloud2
```

[rwkv-szx-4090dx4-ip157]
GPU0-GPU3: 13.3B_bsz320 \* 4 (full_bsz=1280)
[rwkv-szx-4090x4-ip129]
GPU0: 1.5b_bsz1024
GPU1: 2.9b_bsz1024
GPU2 + GPU3: 7.2B_bsz256 \* 2 (full_bsz=512)

(bsz: batch size per device)
