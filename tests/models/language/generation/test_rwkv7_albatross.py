# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RWKV7 dynamic correctness checks against the Albatross reference path.

These tests are opt-in because they require an Albatross checkout, its legacy
`.pth` checkpoint, CUDA, and the matching standard Hugging Face vLLM artifact.
Use `tests/models/language/generation/run_rwkv7_albatross.sh` to load
the local environment and fail fast when required model paths are missing. They
default to eager and decode CUDAGraph vLLM execution. RWKV7 does not support
torch.compile. The CUDAGraph mode is limited to the Albatross-style fixed-buffer
decode path. These tests intentionally do not test registry/import presence.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pytest
import regex as re
import requests
import torch

from vllm import SamplingParams
from vllm.config.compilation import CUDAGraphMode
from vllm.tokenizers.registry import get_tokenizer
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.transformers_utils.config import get_config

TOP_K = int(os.environ.get("RWKV7_ALBATROSS_TOP_K", "20"))
LOGPROB_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOGPROB_ATOL", "0.20"))
LOSS_MEAN_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOSS_MEAN_ATOL", "0.03"))
LOSS_MAX_ATOL = float(os.environ.get("RWKV7_ALBATROSS_LOSS_MAX_ATOL", "0.25"))
LOGIT_MAX_ABS = float(os.environ.get("RWKV7_ALBATROSS_LOGIT_MAX_ABS", "0.1"))
LOGIT_REL_L2 = float(os.environ.get("RWKV7_ALBATROSS_LOGIT_REL_L2", "0.0025"))
LOGIT_COSINE = float(os.environ.get("RWKV7_ALBATROSS_LOGIT_COSINE", "0.9999995"))
ALBATROSS_REFERENCE_NAME = "albatross"
ALBATROSS_REFERENCE_IMPL = "faster3a_2607"
ALBATROSS_REFERENCE_REVISION = "ee3308f6922e59f2166c7fac3c5a192340a2b48e"
ALBATROSS_REFERENCE_TREE = "84a3324a2d138ea0ea83f10c710ae7c382388a6c"
ALBATROSS_REFERENCE_TRACKED_FILES = 9
MAX_MODEL_LEN = int(os.environ.get("RWKV7_ALBATROSS_MAX_MODEL_LEN", "1024"))
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("RWKV7_ALBATROSS_GPU_MEMORY_UTILIZATION", "0.70")
)
ENABLE_FLASHINFER_AUTOTUNE = (
    os.environ.get("RWKV7_ALBATROSS_ENABLE_FLASHINFER_AUTOTUNE", "0") == "1"
)
RWKV7_RUNNER_ENV = {
    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_USE_RAPID_SAMPLER": "1",
}
EXECUTION_MODE_NAMES = tuple(
    mode.strip()
    for mode in os.environ.get(
        "RWKV7_ALBATROSS_EXECUTION_MODES", "eager,cudagraph"
    ).split(",")
    if mode.strip()
)

PROMPTS = [
    {
        "name": "english",
        "text": "User: Explain why recurrent state matters in RWKV.\nAssistant:",
    },
    {
        "name": "chinese",
        "text": "用户：用两句话解释 RWKV 的 state 为什么不能串请求。\n助手：",
    },
    {
        "name": "code",
        "text": (
            "def rwkv_state_debug(requests):\n    for request in requests:\n        "
        ),
    },
    {
        "name": "math",
        "text": "Question: If x + 2 = 7, what is x?\nAnswer:",
    },
    {
        "name": "long_context",
        "text": "RWKV state alignment check. " * 80 + "\nConclusion:",
    },
]

MMLU_STYLE_CASES = [
    {
        "name": "math_choice",
        "prompt": (
            "User: You are a very talented expert in elementary math. "
            "Answer this question:\n"
            "What is 2 + 2?\n"
            "A. 3\n"
            "B. 4\n"
            "C. 5\n"
            "D. 6\n\n"
            "Assistant: The answer is"
        ),
        "choices": [" A", " B", " C", " D"],
    },
    {
        "name": "science_choice",
        "prompt": (
            "User: You are a very talented expert in science. "
            "Answer this question:\n"
            "Which object orbits the Earth?\n"
            "A. The Moon\n"
            "B. Mars\n"
            "C. The Sun\n"
            "D. Venus\n\n"
            "Assistant: The answer is"
        ),
        "choices": [" A", " B", " C", " D"],
    },
]


@dataclass(frozen=True)
class AlbatrossSettings:
    root: Path
    impl_dir: Path
    checkpoint: Path
    vllm_model: str
    max_model_len: int
    revision: str | None = None
    tree: str | None = None
    tracked_files: int | None = None


@dataclass(frozen=True)
class AlbatrossReference:
    name: str
    revision: str
    tree: str
    tracked_files: int


@dataclass(frozen=True)
class ParallelSettings:
    tensor_parallel_size: int
    pipeline_parallel_size: int


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _parallel_settings_from_env() -> ParallelSettings:
    return ParallelSettings(
        tensor_parallel_size=_positive_int_from_env(
            "RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", 1
        ),
        pipeline_parallel_size=_positive_int_from_env(
            "RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", 1
        ),
    )


PARALLEL_SETTINGS = _parallel_settings_from_env()


@dataclass(frozen=True)
class ExecutionMode:
    name: str
    enforce_eager: bool
    env: Mapping[str, str]


def _execution_modes() -> tuple[Any, ...]:
    modes: list[Any] = []
    for mode in EXECUTION_MODE_NAMES:
        if mode == "eager":
            modes.append(
                pytest.param(
                    ExecutionMode(
                        name=mode,
                        enforce_eager=True,
                        env={},
                    ),
                    id=mode,
                )
            )
        elif mode in ("none", "cudagraph"):
            modes.append(
                pytest.param(
                    ExecutionMode(
                        name=mode,
                        enforce_eager=False,
                        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
                    ),
                    id=mode,
                )
            )
        else:
            msg = (
                "RWKV7_ALBATROSS_EXECUTION_MODES only supports "
                f"'eager', 'none', and 'cudagraph', got {mode!r}"
            )
            raise ValueError(msg)
    if not modes:
        raise ValueError("RWKV7_ALBATROSS_EXECUTION_MODES cannot be empty")
    return tuple(modes)


@pytest.fixture(
    scope="module",
    params=_execution_modes(),
)
def rwkv7_execution_mode(request: pytest.FixtureRequest) -> ExecutionMode:
    return cast(ExecutionMode, request.param)


@contextmanager
def _temporary_env(overrides: Mapping[str, str | None]) -> Iterator[None]:
    old_values = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in old_values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _require_cuda_devices(parallel_settings: ParallelSettings) -> None:
    required = (
        parallel_settings.tensor_parallel_size
        * parallel_settings.pipeline_parallel_size
    )
    available = torch.accelerator.device_count()
    if available < required:
        pytest.fail(
            "RWKV7 Albatross alignment requires at least "
            f"{required} CUDA devices for tensor_parallel_size="
            f"{parallel_settings.tensor_parallel_size} and "
            f"pipeline_parallel_size={parallel_settings.pipeline_parallel_size}; "
            f"only {available} CUDA device(s) are visible."
        )


def _albatross_reference(
    root: Path,
    impl_dir: Path,
) -> AlbatrossReference | None:
    revision_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if revision_result.returncode == 0:
        try:
            relative_impl = impl_dir.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"Albatross implementation must be inside its repository: {impl_dir}"
            ) from exc
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                str(relative_impl),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if status_result.returncode != 0:
            raise RuntimeError(
                "Cannot inspect Albatross reference status: "
                f"{status_result.stderr.strip()}"
            )
        if status_result.stdout.strip():
            raise RuntimeError(
                "Albatross reference implementation is dirty: "
                f"{relative_impl}\n{status_result.stdout.rstrip()}"
            )
        tree_result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                f"HEAD:{relative_impl}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        files_result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-tree",
                "-r",
                "--name-only",
                "HEAD",
                "--",
                str(relative_impl),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        revision = revision_result.stdout.strip()
        tree = tree_result.stdout.strip()
        if (
            tree_result.returncode == 0
            and files_result.returncode == 0
            and re.fullmatch(r"[0-9a-f]{40}", revision)
            and re.fullmatch(r"[0-9a-f]{40,64}", tree)
        ):
            tracked_files = len(
                [line for line in files_result.stdout.splitlines() if line]
            )
            return AlbatrossReference(
                name=ALBATROSS_REFERENCE_NAME,
                revision=revision,
                tree=tree,
                tracked_files=tracked_files,
            )
    marker = impl_dir / ".helicopter-dev-reference"
    if marker.is_file():
        fields: dict[str, str] = {}
        for line in marker.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                fields[key] = value
        name = fields.get("reference", "")
        revision = fields.get("revision", "")
        tree = fields.get("tree", "")
        tracked_files_text = fields.get("tracked_files", "")
        if (
            re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name)
            and re.fullmatch(r"[0-9a-f]{40}", revision)
            and re.fullmatch(r"[0-9a-f]{40,64}", tree)
            and tracked_files_text.isdigit()
        ):
            return AlbatrossReference(
                name=name,
                revision=revision,
                tree=tree,
                tracked_files=int(tracked_files_text),
            )
        raise RuntimeError(f"Invalid Albatross reference marker: {marker}")
    return None


@pytest.fixture(scope="module")
def rwkv7_albatross_settings() -> AlbatrossSettings:
    if (
        not torch.accelerator.is_available()
        or torch.accelerator.current_accelerator().type != "cuda"
    ):
        pytest.skip("CUDA is required for RWKV7 Albatross alignment tests")
    _require_cuda_devices(PARALLEL_SETTINGS)
    if os.environ.get("VLLM_RWKV7_WKV_MODE", "fp32io16") != "fp32io16":
        pytest.skip("RWKV7 Albatross alignment requires VLLM_RWKV7_WKV_MODE=fp32io16")

    root = Path(
        os.environ.get(
            "ALBATROSS_ROOT",
            str(Path.home() / "Projects/MachineLearning/albatross"),
        )
    ).expanduser()
    impl = os.environ.get("ALBATROSS_IMPL", ALBATROSS_REFERENCE_IMPL)
    impl_dir = root / impl
    checkpoint = Path(os.environ.get("ALBATROSS_PTH", "")).expanduser()
    vllm_model_env = os.environ.get("VLLM_RWKV7_MODEL", "")
    parsed_vllm_model = urlparse(vllm_model_env)
    is_vllm_model_url = parsed_vllm_model.scheme in ("http", "https")
    vllm_model = Path(vllm_model_env).expanduser()

    if not impl_dir.is_dir():
        pytest.skip(f"Albatross implementation directory not found: {impl_dir}")
    if not checkpoint.is_file():
        pytest.skip("Set ALBATROSS_PTH to a matching RWKV7 .pth checkpoint")
    if not vllm_model_env:
        pytest.skip("Set VLLM_RWKV7_MODEL to the matching vLLM checkpoint")
    if is_vllm_model_url or parsed_vllm_model.path.endswith(".pth"):
        pytest.fail(
            "VLLM_RWKV7_MODEL must be a local HF artifact directory or HF repo ID, "
            "not a raw checkpoint or direct file URL"
        )
    is_hf_repo_id = (
        not vllm_model.exists()
        and not vllm_model.is_absolute()
        and len(vllm_model_env.split("/")) == 2
    )
    if not is_hf_repo_id and not vllm_model.is_dir():
        pytest.skip(f"vLLM Hugging Face artifact not found: {vllm_model}")

    hf_config = get_config(vllm_model_env, trust_remote_code=False)
    source_sha256 = getattr(hf_config, "rwkv_source_sha256", None)
    with checkpoint.open("rb") as source:
        checkpoint_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if source_sha256 != checkpoint_sha256:
        pytest.fail(
            "RWKV7 HF artifact source digest does not match the Albatross "
            f"checkpoint: artifact={source_sha256} checkpoint={checkpoint_sha256}"
        )
    artifact_context = int(getattr(hf_config, "max_position_embeddings", 0))
    if artifact_context < MAX_MODEL_LEN:
        pytest.fail(
            "RWKV7 HF artifact context is smaller than the requested alignment "
            f"window: artifact={artifact_context} requested={MAX_MODEL_LEN}"
        )
    artifact_tokenizer = get_tokenizer(vllm_model_env, tokenizer_mode="rwkv")
    reference_tokenizer = RWKVTokenizer()
    for prompt in PROMPTS:
        text = prompt["text"]
        if artifact_tokenizer.encode(
            text, add_special_tokens=False
        ) != reference_tokenizer.encode(text, add_special_tokens=False):
            pytest.fail(f"RWKV7 HF tokenizer mismatch for prompt {prompt['name']!r}")

    if impl != ALBATROSS_REFERENCE_IMPL:
        pytest.fail(
            "Albatross reference implementation mismatch: "
            f"expected={ALBATROSS_REFERENCE_IMPL} actual={impl}"
        )
    reference = _albatross_reference(root, impl_dir)
    if reference is None:
        pytest.fail(f"Albatross reference provenance is unavailable: {impl_dir}")
    assert reference is not None
    expected_revision = os.environ.get(
        "ALBATROSS_REVISION",
        ALBATROSS_REFERENCE_REVISION,
    )
    if reference.name != ALBATROSS_REFERENCE_NAME:
        pytest.fail(
            "Albatross reference name mismatch: "
            f"expected={ALBATROSS_REFERENCE_NAME} actual={reference.name}"
        )
    if reference.revision != expected_revision:
        pytest.fail(
            "Albatross reference revision mismatch: "
            f"expected={expected_revision} actual={reference.revision}"
        )
    if reference.tree != ALBATROSS_REFERENCE_TREE:
        pytest.fail(
            "Albatross 2607 tree mismatch: "
            f"expected={ALBATROSS_REFERENCE_TREE} actual={reference.tree}"
        )
    if reference.tracked_files != ALBATROSS_REFERENCE_TRACKED_FILES:
        pytest.fail(
            "Albatross 2607 tracked-file count mismatch: "
            f"expected={ALBATROSS_REFERENCE_TRACKED_FILES} "
            f"actual={reference.tracked_files}"
        )

    return AlbatrossSettings(
        root=root,
        impl_dir=impl_dir,
        checkpoint=checkpoint,
        vllm_model=vllm_model_env if is_hf_repo_id else str(vllm_model),
        max_model_len=MAX_MODEL_LEN,
        revision=reference.revision,
        tree=reference.tree,
        tracked_files=reference.tracked_files,
    )


@pytest.fixture(scope="module")
def albatross_oracle(
    rwkv7_albatross_settings: AlbatrossSettings,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    work_dir = tmp_path_factory.mktemp("rwkv7_albatross")
    request_path = work_dir / "request.json"
    response_path = work_dir / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "prompts": PROMPTS,
                "mmlu_style_cases": MMLU_STYLE_CASES,
                "max_new_tokens": 32,
                "top_k": TOP_K,
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _albatross_oracle_code(),
            str(rwkv7_albatross_settings.impl_dir),
            str(rwkv7_albatross_settings.checkpoint),
            str(request_path),
            str(response_path),
        ],
        cwd=rwkv7_albatross_settings.impl_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Albatross oracle failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    oracle = json.loads(response_path.read_text(encoding="utf-8"))
    oracle["reference"] = {
        "implementation": rwkv7_albatross_settings.impl_dir.name,
        "revision": rwkv7_albatross_settings.revision,
        "tree": rwkv7_albatross_settings.tree,
        "tracked_files": rwkv7_albatross_settings.tracked_files,
    }
    return oracle


def _albatross_oracle_code() -> str:
    return textwrap.dedent(
        r"""
        import json
        import sys

        import torch
        import torch.nn.functional as F
        from vllm.tokenizers.rwkv import RWKVTokenizer

        impl_dir, checkpoint, request_path, response_path = sys.argv[1:5]
        sys.path.insert(0, impl_dir)

        import rwkv7_fast_v3a as v3a

        with open(request_path, "r", encoding="utf-8") as f:
            request = json.load(f)

        v3a.MODEL_PATH = checkpoint
        v3a.WKV_MODE = "fp32io16"
        v3a.EMB_DEVICE = "cpu"
        v3a.RKV_MODE = "off"
        v3a.CMIX_SPARSE = "no-fc"
        v3a.LOWRANK_WEIGHT = "both"
        v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups("none")
        v3a.load_extensions(v3a.WKV_MODE)
        model = v3a.RWKV7()
        tokenizer = RWKVTokenizer()
        token_device = "cpu" if model.emb_cpu else "cuda"

        def encode(text):
            return tokenizer.encode(text, add_special_tokens=False)

        def top_logprobs(logits, top_k):
            logprobs = F.log_softmax(logits.float(), dim=-1)
            values, ids = torch.topk(logprobs, k=top_k)
            return [
                [int(token_id), float(value)]
                for token_id, value in zip(ids.detach().cpu(), values.detach().cpu())
            ]

        def prompt_losses(ids):
            if len(ids) < 2:
                return []
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids[:-1], dtype=torch.long, device=token_device
            ).view(1, -1)
            targets = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
            logits = model.forward_all_logits(tokens, state).squeeze(0).float()
            return [
                float(x)
                for x in F.cross_entropy(logits, targets, reduction="none")
                .detach()
                .cpu()
            ]

        def greedy(ids, max_new_tokens):
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            out = []
            top1_gaps = []
            for _ in range(max_new_tokens):
                top2 = torch.topk(logits.float(), k=2)
                token_id = int(top2.indices[0].item())
                out.append(token_id)
                top1_gaps.append(float((top2.values[0] - top2.values[1]).cpu()))
                token = torch.tensor(
                    [[token_id]], dtype=torch.long, device=token_device
                )
                logits = model.forward(token, state).view(-1)
            return out, top1_gaps

        def choice_logprobs(logits, choice_ids):
            logprobs = F.log_softmax(logits.float(), dim=-1)
            return [
                [int(token_id), float(logprobs[int(token_id)].detach().cpu())]
                for token_id in choice_ids
            ]

        cases = []
        for item in request["prompts"]:
            ids = encode(item["text"])
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            losses = prompt_losses(ids)
            top2_logits = torch.topk(logits.float(), k=2).values
            greedy_token_ids, greedy_top1_gaps = greedy(
                ids, request["max_new_tokens"]
            )
            cases.append(
                {
                    "name": item["name"],
                    "prompt": item["text"],
                    "prompt_token_ids": ids,
                    "next_logits": logits.detach().float().cpu().tolist(),
                    "next_top1_gap": float(
                        (top2_logits[0] - top2_logits[1]).detach().cpu()
                    ),
                    "next_top_logprobs": top_logprobs(logits, request["top_k"]),
                    "greedy_token_ids": greedy_token_ids,
                    "greedy_top1_gaps": greedy_top1_gaps,
                    "prompt_losses": losses,
                    "prompt_mean_loss": (
                        sum(losses) / len(losses) if losses else 0.0
                    ),
                }
            )

        mmlu_cases = []
        for item in request["mmlu_style_cases"]:
            ids = encode(item["prompt"])
            choice_token_ids = [encode(choice) for choice in item["choices"]]
            if not all(len(choice) == 1 for choice in choice_token_ids):
                raise RuntimeError(f"MMLU choices must be single tokens: {item}")
            choice_token_ids = [choice[0] for choice in choice_token_ids]
            state = model.zero_state(1)
            tokens = torch.tensor(
                ids, dtype=torch.long, device=token_device
            ).view(1, -1)
            logits = model.forward(tokens, state).view(-1)
            choices = choice_logprobs(logits, choice_token_ids)
            pred = max(choices, key=lambda item: item[1])[0]
            mmlu_cases.append(
                {
                    "name": item["name"],
                    "prompt_token_ids": ids,
                    "choice_token_ids": choice_token_ids,
                    "choice_logprobs": choices,
                    "pred_token_id": pred,
                }
            )

        with open(response_path, "w", encoding="utf-8") as f:
            json.dump({"cases": cases, "mmlu_style_cases": mmlu_cases}, f)
        """
    )


def _vllm_outputs(
    vllm_runner,
    settings: AlbatrossSettings,
    execution_mode: ExecutionMode,
    prompt_token_ids: list[list[int]],
    *,
    max_tokens: int,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    logprob_token_ids: list[int] | None = None,
    enable_chunked_prefill: bool = True,
    max_num_batched_tokens: int | None = None,
) -> list[Any]:
    runner_env = {
        **execution_mode.env,
        **RWKV7_RUNNER_ENV,
        # This is a test-harness model input, not a registered vLLM option.
        "VLLM_RWKV7_MODEL": None,
    }
    kwargs: dict[str, Any] = {
        "enforce_eager": execution_mode.enforce_eager,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "kernel_config": {
            "enable_flashinfer_autotune": ENABLE_FLASHINFER_AUTOTUNE,
        },
        "max_model_len": settings.max_model_len,
        "enable_chunked_prefill": enable_chunked_prefill,
        "logprobs_mode": "raw_logprobs",
        "tensor_parallel_size": PARALLEL_SETTINGS.tensor_parallel_size,
        "pipeline_parallel_size": PARALLEL_SETTINGS.pipeline_parallel_size,
    }
    if PARALLEL_SETTINGS.pipeline_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "mp"
        kwargs["compilation_config"] = {
            "cudagraph_mode": CUDAGraphMode.NONE,
            "cudagraph_capture_sizes": [],
        }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = max_num_batched_tokens
    if execution_mode.name == "none":
        kwargs["compilation_config"] = {
            "cudagraph_mode": CUDAGraphMode.NONE,
            "cudagraph_capture_sizes": [],
        }
    elif (
        execution_mode.name == "cudagraph"
        and PARALLEL_SETTINGS.pipeline_parallel_size == 1
    ):
        kwargs["compilation_config"] = {
            "cudagraph_capture_sizes": sorted({1, len(prompt_token_ids)}),
        }

    with (
        _temporary_env(runner_env),
        vllm_runner(settings.vllm_model, **kwargs) as runner,
    ):
        llm = runner.get_llm()
        sampling_params = SamplingParams(
            temperature=1.0,
            top_k=1,
            ignore_eos=True,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
        )
        return llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_token_ids],
            sampling_params=sampling_params,
        )


def _worker_direct_next_logits(
    worker: Any,
    prompt_token_ids: list[list[int]],
) -> dict[str, Any]:
    """Return full next-token logits through the production FP32 boundary."""
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None:
        raise RuntimeError("worker.model_runner.model is unavailable")
    if getattr(model, "tp_size", 1) != 1:
        raise RuntimeError("direct Albatross logits alignment requires TP=1")
    if getattr(model, "start_layer", 0) != 0 or getattr(
        model, "end_layer", 0
    ) != getattr(model, "total_num_layers", 0):
        raise RuntimeError("direct Albatross logits alignment requires PP=1")
    if getattr(model, "wkv_mode", None) != "fp32io16":
        raise RuntimeError(
            "direct Albatross logits alignment requires WKV_MODE=fp32io16"
        )
    if getattr(model, "wkv_state_dtype", None) != torch.float32:
        raise RuntimeError("fp32io16 must keep the WKV state in FP32")
    if getattr(model, "allow_fp16_accumulation", None) is not False:
        raise RuntimeError("fp32io16 must disable FP16 GEMM accumulation")

    head_weight = model.z["head.weight"]
    if head_weight.dtype != torch.float16 or not head_weight.is_cuda:
        raise RuntimeError("RWKV7 head weights must remain CUDA FP16")

    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for ids in prompt_token_ids:
            tokens = torch.tensor(ids, dtype=torch.long, device="cuda").view(1, -1)
            state = model.zero_state(1)
            hidden = model.forward_tokens(tokens, state)
            if hidden.dtype != torch.float16 or not hidden.is_cuda:
                raise RuntimeError("RWKV7 hidden states must remain CUDA FP16")
            logits = model.compute_logits(hidden)
            if logits is None:
                raise RuntimeError("RWKV7 direct logits unexpectedly returned None")
            logits = logits.squeeze(0)
            expected_shape = (model.vocab_size,)
            if (
                logits.shape != expected_shape
                or logits.dtype != torch.float32
                or not logits.is_cuda
            ):
                raise RuntimeError(
                    "RWKV7 logits contract mismatch: "
                    f"shape={tuple(logits.shape)} dtype={logits.dtype} "
                    f"device={logits.device}; expected CUDA FP32 {expected_shape}"
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError("RWKV7 direct logits contain non-finite values")
            outputs.append(logits.detach().cpu())

    return {
        "wkv_mode": model.wkv_mode,
        "wkv_state_dtype": str(model.wkv_state_dtype),
        "allow_fp16_accumulation": model.allow_fp16_accumulation,
        "head_weight_dtype": str(head_weight.dtype),
        "logits_dtype": str(outputs[0].dtype) if outputs else "torch.float32",
        "logits": outputs,
    }


@pytest.fixture(scope="module")
def vllm_direct_logits(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    albatross_oracle: dict[str, Any],
) -> dict[str, Any]:
    if (
        ParallelSettings(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        != PARALLEL_SETTINGS
    ):
        pytest.skip("direct Albatross logits alignment requires TP=1 and PP=1")

    cases = albatross_oracle["cases"]
    prompt_token_ids = [case["prompt_token_ids"] for case in cases]
    runner_env = {
        **RWKV7_RUNNER_ENV,
        "VLLM_RWKV7_WKV_MODE": "fp32io16",
        # This is a test-harness model input, not a registered vLLM option.
        "VLLM_RWKV7_MODEL": None,
    }
    with (
        _temporary_env(runner_env),
        vllm_runner(
            rwkv7_albatross_settings.vllm_model,
            skip_tokenizer_init=True,
            enforce_eager=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=rwkv7_albatross_settings.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=rwkv7_albatross_settings.max_model_len,
            enable_chunked_prefill=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            compilation_config={
                "cudagraph_mode": CUDAGraphMode.NONE,
                "cudagraph_capture_sizes": [],
            },
        ) as runner,
    ):
        worker_results = runner.get_llm().collective_rpc(
            _worker_direct_next_logits,
            args=(prompt_token_ids,),
        )

    if len(worker_results) != 1:
        raise RuntimeError(
            f"expected one direct-alignment worker, got {len(worker_results)}"
        )
    return cast(dict[str, Any], worker_results[0])


def _sorted_logprobs(
    sample_logprobs: Mapping[int, Any],
) -> list[tuple[int, float]]:
    return sorted(
        (
            (int(token_id), float(logprob.logprob))
            for token_id, logprob in sample_logprobs.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )


def _assert_next_token_logprobs_close(
    *,
    case: dict[str, Any],
    vllm_logprobs: Mapping[int, Any],
) -> None:
    albatross_top = [
        (int(token_id), float(value)) for token_id, value in case["next_top_logprobs"]
    ]
    vllm_top = _sorted_logprobs(vllm_logprobs)

    assert vllm_top[0][0] == albatross_top[0][0], (
        f"{case['name']}: top-1 mismatch: "
        f"albatross={albatross_top[:5]} vllm={vllm_top[:5]}"
    )

    albatross_top10 = {token_id for token_id, _ in albatross_top[:10]}
    vllm_top_ids = {token_id for token_id, _ in vllm_top}
    missing = albatross_top10 - vllm_top_ids
    assert not missing, (
        f"{case['name']}: vLLM logprobs missed Albatross top tokens: "
        f"missing={sorted(missing)} albatross={albatross_top[:10]} "
        f"vllm={vllm_top[:10]}"
    )

    vllm_by_id = dict(vllm_top)
    for token_id, expected in albatross_top[:10]:
        actual = vllm_by_id[token_id]
        assert abs(actual - expected) <= LOGPROB_ATOL, (
            f"{case['name']}: logprob mismatch for token {token_id}: "
            f"albatross={expected:.6f} vllm={actual:.6f}"
        )


def test_rwkv7_direct_fp32_logits_match_albatross(
    albatross_oracle: dict[str, Any],
    vllm_direct_logits: dict[str, Any],
) -> None:
    assert vllm_direct_logits["wkv_mode"] == "fp32io16"
    assert vllm_direct_logits["wkv_state_dtype"] == "torch.float32"
    assert vllm_direct_logits["allow_fp16_accumulation"] is False
    assert vllm_direct_logits["head_weight_dtype"] == "torch.float16"
    assert vllm_direct_logits["logits_dtype"] == "torch.float32"

    metrics = []
    for case, actual in zip(
        albatross_oracle["cases"],
        vllm_direct_logits["logits"],
        strict=True,
    ):
        expected = torch.tensor(case["next_logits"], dtype=torch.float32)
        actual = cast(torch.Tensor, actual).float()
        assert actual.shape == expected.shape
        difference = actual - expected
        max_abs = float(difference.abs().max())
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(expected).clamp_min(
                torch.finfo(torch.float32).tiny
            )
        )
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual.double(),
                expected.double(),
                dim=0,
            )
        )
        top1_matches = int(actual.argmax()) == int(expected.argmax())
        metrics.append(
            {
                "name": case["name"],
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "cosine": cosine,
                "top1_matches": top1_matches,
            }
        )
        assert top1_matches, f"{case['name']}: direct FP32 logits top-1 mismatch"
        assert max_abs <= LOGIT_MAX_ABS, (
            f"{case['name']}: max logits error {max_abs:.6f} > {LOGIT_MAX_ABS:.6f}"
        )
        assert relative_l2 <= LOGIT_REL_L2, (
            f"{case['name']}: relative L2 {relative_l2:.6e} > {LOGIT_REL_L2:.6e}"
        )
        assert cosine >= LOGIT_COSINE, (
            f"{case['name']}: cosine {cosine:.9f} < {LOGIT_COSINE:.9f}"
        )

    print(
        json.dumps(
            {
                "albatross_reference": albatross_oracle["reference"],
                "rwkv7_direct_fp32_logits_alignment": metrics,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def test_albatross_reference_reads_remote_marker(tmp_path: Path) -> None:
    impl_dir = tmp_path / ALBATROSS_REFERENCE_IMPL
    impl_dir.mkdir()
    (impl_dir / ".helicopter-dev-reference").write_text(
        "\n".join(
            (
                f"reference={ALBATROSS_REFERENCE_NAME}",
                f"revision={ALBATROSS_REFERENCE_REVISION}",
                f"tree={ALBATROSS_REFERENCE_TREE}",
                f"tracked_files={ALBATROSS_REFERENCE_TRACKED_FILES}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    reference = _albatross_reference(tmp_path, impl_dir)

    assert reference == AlbatrossReference(
        name=ALBATROSS_REFERENCE_NAME,
        revision=ALBATROSS_REFERENCE_REVISION,
        tree=ALBATROSS_REFERENCE_TREE,
        tracked_files=ALBATROSS_REFERENCE_TRACKED_FILES,
    )


def test_albatross_reference_rejects_dirty_local_tree(tmp_path: Path) -> None:
    root = tmp_path / "albatross"
    impl_dir = root / ALBATROSS_REFERENCE_IMPL
    impl_dir.mkdir(parents=True)
    source = impl_dir / "reference.py"
    source.write_text("VALUE = 2607\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "reference",
        ],
        check=True,
    )

    clean = _albatross_reference(root, impl_dir)
    assert clean is not None
    assert clean.tracked_files == 1

    source.write_text("VALUE = 'dirty'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reference implementation is dirty"):
        _albatross_reference(root, impl_dir)


def test_rwkv7_parallel_settings_default_to_single_rank(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", raising=False)
    monkeypatch.delenv("RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", raising=False)

    assert _parallel_settings_from_env() == ParallelSettings(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def test_rwkv7_parallel_settings_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_ALBATROSS_TENSOR_PARALLEL_SIZE", "2")
    monkeypatch.setenv("RWKV7_ALBATROSS_PIPELINE_PARALLEL_SIZE", "3")

    assert _parallel_settings_from_env() == ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    )


def test_rwkv7_vllm_outputs_passes_parallel_settings_to_runner(monkeypatch) -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model-hf",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(
        name="none",
        enforce_eager=False,
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
    )
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    monkeypatch.setattr(sys.modules[__name__], "PARALLEL_SETTINGS", parallel_settings)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setenv("VLLM_USE_RAPID_SAMPLER", "0")
    monkeypatch.setenv("VLLM_RWKV7_MODEL", "/test-only/model-hf")
    runner_calls: list[dict[str, Any]] = []
    sampling_calls: list[Any] = []

    class FakeLLM:
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params):
            sampling_calls.append(sampling_params)
            return [(prompts, sampling_params)]

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_llm(self) -> FakeLLM:
            return FakeLLM()

    def fake_vllm_runner(model: str, **kwargs: Any) -> FakeRunner:
        runner_calls.append(
            {
                "model": model,
                "kwargs": kwargs,
                "use_v2": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
                "use_rapid": os.environ.get("VLLM_USE_RAPID_SAMPLER"),
                "model_input_env": os.environ.get("VLLM_RWKV7_MODEL"),
            }
        )
        return FakeRunner()

    _vllm_outputs(
        fake_vllm_runner,
        settings,
        execution_mode,
        [[1, 2, 3]],
        max_tokens=1,
    )

    assert runner_calls[0]["kwargs"]["tensor_parallel_size"] == 2
    assert runner_calls[0]["kwargs"]["pipeline_parallel_size"] == 1
    assert runner_calls[0]["kwargs"]["logprobs_mode"] == "raw_logprobs"
    assert runner_calls[0]["use_v2"] == "1"
    assert runner_calls[0]["use_rapid"] == "1"
    assert runner_calls[0]["model_input_env"] is None
    assert sampling_calls[0].temperature == 1.0
    assert sampling_calls[0].top_k == 1
    assert sampling_calls[0].ignore_eos is True
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert os.environ["VLLM_USE_RAPID_SAMPLER"] == "0"
    assert os.environ["VLLM_RWKV7_MODEL"] == "/test-only/model-hf"


def test_rwkv7_vllm_outputs_disables_cudagraph_for_pipeline_parallel(
    monkeypatch,
) -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model-hf",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(
        name="cudagraph",
        enforce_eager=False,
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
    )
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    monkeypatch.setattr(sys.modules[__name__], "PARALLEL_SETTINGS", parallel_settings)
    runner_calls: list[dict[str, Any]] = []

    class FakeLLM:
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params):
            return [(prompts, sampling_params)]

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_llm(self) -> FakeLLM:
            return FakeLLM()

    def fake_vllm_runner(model: str, **kwargs: Any) -> FakeRunner:
        runner_calls.append({"model": model, "kwargs": kwargs})
        return FakeRunner()

    _vllm_outputs(
        fake_vllm_runner,
        settings,
        execution_mode,
        [[1, 2, 3]],
        max_tokens=1,
    )

    kwargs = runner_calls[0]["kwargs"]
    assert kwargs["distributed_executor_backend"] == "mp"
    assert kwargs["compilation_config"] == {
        "cudagraph_mode": CUDAGraphMode.NONE,
        "cudagraph_capture_sizes": [],
    }


def test_rwkv7_vllm_outputs_captures_requested_decode_batch(
    monkeypatch,
) -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model-hf",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(
        name="cudagraph",
        enforce_eager=False,
        env={"VLLM_USE_BREAKABLE_CUDAGRAPH": "0"},
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "PARALLEL_SETTINGS",
        ParallelSettings(tensor_parallel_size=1, pipeline_parallel_size=1),
    )
    runner_calls: list[dict[str, Any]] = []

    class FakeLLM:
        def generate(self, prompts: list[dict[str, list[int]]], sampling_params):
            return [(prompts, sampling_params)]

    class FakeRunner:
        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def get_llm(self) -> FakeLLM:
            return FakeLLM()

    def fake_vllm_runner(model: str, **kwargs: Any) -> FakeRunner:
        runner_calls.append({"model": model, "kwargs": kwargs})
        return FakeRunner()

    _vllm_outputs(
        fake_vllm_runner,
        settings,
        execution_mode,
        [[1], [2], [3]],
        max_tokens=1,
    )

    assert runner_calls[0]["kwargs"]["compilation_config"] == {
        "cudagraph_capture_sizes": [1, 3],
    }


def test_rwkv7_server_args_include_parallel_settings() -> None:
    settings = AlbatrossSettings(
        root=Path("/unused"),
        impl_dir=Path("/unused/impl"),
        checkpoint=Path("/unused/model.pth"),
        vllm_model="/unused/model-hf",
        max_model_len=128,
    )
    execution_mode = ExecutionMode(name="cudagraph", enforce_eager=False, env={})
    parallel_settings = ParallelSettings(
        tensor_parallel_size=2,
        pipeline_parallel_size=3,
    )

    server_args = _server_args(settings, execution_mode, parallel_settings, port=1234)

    assert server_args[server_args.index("--tensor-parallel-size") + 1] == "2"
    assert server_args[server_args.index("--pipeline-parallel-size") + 1] == "3"


def test_rwkv7_next_token_logprobs_match_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        logprobs=TOP_K,
    )

    for case, output in zip(cases, outputs):
        sample = output.outputs[0]
        assert case["next_top1_gap"] > 0, (
            f"{case['name']}: rapid top-k=1 argmax equivalence requires "
            "a unique maximum"
        )
        assert list(sample.token_ids) == [case["greedy_token_ids"][0]]
        assert sample.logprobs is not None
        _assert_next_token_logprobs_close(
            case=case,
            vllm_logprobs=cast(Mapping[int, Any], sample.logprobs[0]),
        )


def test_rwkv7_greedy_generation_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=32,
    )

    for case, output in zip(cases, outputs):
        sample = output.outputs[0]
        actual = list(sample.token_ids)
        assert min(case["greedy_top1_gaps"]) > 0, (
            f"{case['name']}: rapid top-k=1 argmax equivalence requires "
            "a unique maximum at every generated position"
        )
        mismatch_index = next(
            (
                index
                for index, (expected_token, actual_token) in enumerate(
                    zip(case["greedy_token_ids"], actual)
                )
                if expected_token != actual_token
            ),
            None,
        )
        mismatch_detail = ""
        if mismatch_index is not None:
            mismatch_detail = (
                f"\nfirst_mismatch={mismatch_index}"
                "\nreference_top1_gap="
                f"{case['greedy_top1_gaps'][mismatch_index]}"
            )
        assert actual == case["greedy_token_ids"], (
            f"{case['name']}: greedy token mismatch\n"
            f"albatross={case['greedy_token_ids']}\nvllm={actual}"
            f"{mismatch_detail}"
        )


def test_rwkv7_prompt_loss_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        prompt_logprobs=0,
    )

    for case, output in zip(cases, outputs):
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert prompt_logprobs[0] is None
        actual_losses = []
        for token_id, logprob_by_token in zip(
            case["prompt_token_ids"][1:],
            prompt_logprobs[1:],
        ):
            assert logprob_by_token is not None
            actual_losses.append(-float(logprob_by_token[int(token_id)].logprob))

        expected_losses = [float(x) for x in case["prompt_losses"]]
        assert len(actual_losses) == len(expected_losses)
        actual_mean = sum(actual_losses) / len(actual_losses)
        expected_mean = case["prompt_mean_loss"]
        assert abs(actual_mean - expected_mean) <= LOSS_MEAN_ATOL, (
            f"{case['name']}: mean loss mismatch: "
            f"albatross={expected_mean:.6f} vllm={actual_mean:.6f}"
        )
        max_diff = max(
            abs(actual - expected)
            for actual, expected in zip(actual_losses, expected_losses)
        )
        assert max_diff <= LOSS_MAX_ATOL, (
            f"{case['name']}: per-position loss diff too high: max_diff={max_diff:.6f}"
        )


def test_rwkv7_mmlu_style_choice_logits_match_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["mmlu_style_cases"]
    choice_token_ids = cases[0]["choice_token_ids"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in cases],
        max_tokens=1,
        logprobs=len(choice_token_ids),
        logprob_token_ids=choice_token_ids,
    )

    for case, output in zip(cases, outputs):
        sample = output.outputs[0]
        assert sample.logprobs is not None
        vllm_logprobs = {
            int(token_id): float(logprob.logprob)
            for token_id, logprob in sample.logprobs[0].items()
        }
        expected = {
            int(token_id): float(logprob)
            for token_id, logprob in case["choice_logprobs"]
        }
        actual_pred = max(
            (token_id for token_id in case["choice_token_ids"]),
            key=lambda token_id: vllm_logprobs[token_id],
        )
        assert actual_pred == case["pred_token_id"], (
            f"{case['name']}: MMLU-style prediction mismatch: "
            f"albatross={case['pred_token_id']} vllm={actual_pred}"
        )
        for token_id, expected_logprob in expected.items():
            actual_logprob = vllm_logprobs[token_id]
            assert abs(actual_logprob - expected_logprob) <= LOGPROB_ATOL, (
                f"{case['name']}: choice logprob mismatch for token {token_id}: "
                f"albatross={expected_logprob:.6f} vllm={actual_logprob:.6f}"
            )


def test_rwkv7_chunked_prefill_matches_albatross(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"]
    long_cases = [case for case in cases if case["name"] == "long_context"]
    outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in long_cases],
        max_tokens=1,
        logprobs=TOP_K,
        enable_chunked_prefill=True,
        max_num_batched_tokens=64,
    )

    for case, output in zip(long_cases, outputs):
        sample = output.outputs[0]
        assert list(sample.token_ids) == [case["greedy_token_ids"][0]]
        assert sample.logprobs is not None
        _assert_next_token_logprobs_close(
            case=case,
            vllm_logprobs=cast(Mapping[int, Any], sample.logprobs[0]),
        )


def test_rwkv7_continuous_batching_preserves_albatross_outputs(
    vllm_runner,
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    cases = albatross_oracle["cases"][:3]
    target_case = cases[0]
    solo_outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [target_case["prompt_token_ids"]],
        max_tokens=16,
    )
    batched_outputs = _vllm_outputs(
        vllm_runner,
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        [case["prompt_token_ids"] for case in reversed(cases)],
        max_tokens=16,
    )

    solo_actual = list(solo_outputs[0].outputs[0].token_ids)
    assert solo_actual == target_case["greedy_token_ids"][:16]
    for case, output in zip(reversed(cases), batched_outputs):
        actual = list(output.outputs[0].token_ids)
        expected = case["greedy_token_ids"][:16]
        assert actual == expected
        if case["name"] == target_case["name"]:
            assert actual == solo_actual


def _server_args(
    settings: AlbatrossSettings,
    execution_mode: ExecutionMode,
    parallel_settings: ParallelSettings,
    port: int,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        settings.vllm_model,
        "--port",
        str(port),
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--tensor-parallel-size",
        str(parallel_settings.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(parallel_settings.pipeline_parallel_size),
        "--trust-remote-code",
    ]
    if parallel_settings.pipeline_parallel_size > 1:
        args.extend(["--distributed-executor-backend", "mp"])
    if execution_mode.enforce_eager:
        args.append("--enforce-eager")
    if ENABLE_FLASHINFER_AUTOTUNE:
        args.append("--enable-flashinfer-autotune")
    else:
        args.append("--no-enable-flashinfer-autotune")
    return args


def test_rwkv7_openai_server_matches_albatross_greedy(
    rwkv7_albatross_settings: AlbatrossSettings,
    rwkv7_execution_mode: ExecutionMode,
    albatross_oracle: dict[str, Any],
) -> None:
    if os.environ.get("RWKV7_RUN_SERVER_ALIGNMENT") != "1":
        pytest.skip("Set RWKV7_RUN_SERVER_ALIGNMENT=1 to run the server test")

    case = albatross_oracle["cases"][0]
    port = _free_port()
    server_args = _server_args(
        rwkv7_albatross_settings,
        rwkv7_execution_mode,
        PARALLEL_SETTINGS,
        port,
    )
    with tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8", prefix="rwkv7-vllm-server-", suffix=".log"
    ) as server_log:
        server_log_path = Path(server_log.name)
        with _temporary_env(rwkv7_execution_mode.env):
            env = os.environ.copy()
            env.setdefault("VLLM_RWKV7_WKV_MODE", "fp32io16")
            env.update(RWKV7_RUNNER_ENV)
            env.pop("VLLM_RWKV7_MODEL", None)
            process = subprocess.Popen(
                server_args,
                env=env,
                text=True,
                stdout=server_log,
                stderr=subprocess.STDOUT,
            )
        try:
            _wait_for_server(port, process, server_log_path)
            response = requests.post(
                f"http://127.0.0.1:{port}/v1/completions",
                json={
                    "model": rwkv7_albatross_settings.vllm_model,
                    "prompt": case["prompt"],
                    "temperature": 1,
                    "top_k": 1,
                    "ignore_eos": True,
                    "max_tokens": 16,
                    "logprobs": 1,
                    "return_token_ids": True,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()
            token_ids = data["choices"][0]["token_ids"]
            assert token_ids == case["greedy_token_ids"][:16]
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(port: int, process: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 180
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = _read_server_log(log_path)
            raise RuntimeError(f"vLLM server exited early:\n{output}")
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(2)
    output = _read_server_log(log_path)
    raise TimeoutError(f"Timed out waiting for vLLM server:\n{output}")


def _read_server_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"<failed to read vLLM server log {log_path}: {exc}>"
