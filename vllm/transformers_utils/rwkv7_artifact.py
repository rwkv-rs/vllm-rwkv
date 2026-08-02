# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-time migration of legacy RWKV7 checkpoints to Hugging Face artifacts."""

import hashlib
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import regex as re
import torch
from safetensors.torch import save_file
from transformers import GenerationConfig

from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.transformers_utils.configs.rwkv7 import (
    STANDARD_RWKV7_ARCHITECTURE,
    RWKV7Config,
    rwkv7_internal_to_hf_weight_name,
    rwkv7_is_legacy_low_rank_weight_name,
    rwkv7_legacy_checkpoint_weight_shapes,
    validate_rwkv7_hf_artifact_config,
)

_LEGACY_RWKV7_ARCHITECTURE = "RWKV7ForCausalLM"
_RWKV7_G1_SPECS = {
    "0.1b": (12, 768),
    "0.4b": (24, 1024),
    "1.5b": (24, 2048),
    "2.9b": (32, 2560),
    "7.2b": (32, 4096),
    "13.3b": (61, 4096),
}
_RWKV7_G1_FILENAME_RE = re.compile(
    r"^rwkv7-g1[a-z]-"
    r"(?P<size>0\.1b|0\.4b|1\.5b|2\.9b|7\.2b|13\.3b)-"
    r"(?P<date>\d{8})-ctx(?P<context>\d+)\.pth$",
    re.IGNORECASE,
)


def _load_migration_config(
    checkpoint: Path,
    config_path: Path | None,
) -> RWKV7Config:
    config_path = config_path or checkpoint.with_name("config.json")
    if config_path.is_file():
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid RWKV7 migration config: {config_path}"
            ) from error
        if not isinstance(values, dict):
            raise ValueError(f"Invalid RWKV7 migration config: {config_path}")
        if values.get("model_type") != "rwkv7" or values.get("architectures") not in (
            [_LEGACY_RWKV7_ARCHITECTURE],
            [STANDARD_RWKV7_ARCHITECTURE],
        ):
            raise ValueError(
                "Invalid RWKV7 migration config identity: expected model_type="
                "'rwkv7' and a legacy or standard RWKV7 causal-LM architecture."
            )
        values = dict(values)
        values.pop("model_type", None)
        values["architectures"] = [STANDARD_RWKV7_ARCHITECTURE]
        config = RWKV7Config(**values)
    else:
        match = _RWKV7_G1_FILENAME_RE.match(checkpoint.name)
        if match is None:
            raise ValueError(
                "Cannot infer RWKV7 config from legacy checkpoint "
                f"{checkpoint.name!r}; "
                "provide a config.json sidecar."
            )
        num_layers, hidden_size = _RWKV7_G1_SPECS[match.group("size").lower()]
        config = RWKV7Config(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            max_position_embeddings=int(match.group("context")),
        )
    validate_rwkv7_hf_artifact_config(config)
    return config


def _load_legacy_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("RWKV7 legacy checkpoint must contain a tensor mapping.")
    weights: dict[str, torch.Tensor] = {}
    for name, value in loaded.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(
                "RWKV7 legacy checkpoint must map string names directly to tensors."
            )
        if name in weights:
            raise ValueError(
                f"RWKV7 legacy checkpoint contains duplicate key {name!r}."
            )
        weights[name] = value.detach().cpu()
    return weights


def _validate_and_map_weights(
    weights: Mapping[str, torch.Tensor],
    config: RWKV7Config,
) -> dict[str, torch.Tensor]:
    expected = rwkv7_legacy_checkpoint_weight_shapes(config)
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            "RWKV7 legacy checkpoint key mismatch before migration: "
            f"missing={missing}, unexpected={unexpected}"
        )

    standard: dict[str, torch.Tensor] = {}
    for internal_name, shape in expected.items():
        tensor = weights[internal_name]
        if tuple(tensor.shape) != shape:
            raise ValueError(
                "RWKV7 legacy checkpoint shape mismatch for "
                f"{internal_name}: expected {shape}, got {tuple(tensor.shape)}"
            )
        hf_name = rwkv7_internal_to_hf_weight_name(internal_name)
        if hf_name is None:
            raise ValueError(
                f"RWKV7 legacy checkpoint key has no HF mapping: {internal_name!r}"
            )
        standard[hf_name] = (
            tensor.transpose(0, 1).contiguous()
            if rwkv7_is_legacy_low_rank_weight_name(internal_name)
            else tensor.contiguous()
        )
    return standard


def convert_rwkv7_pth_to_hf_artifact(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Convert one legacy ``.pth`` checkpoint into one strict HF artifact.

    This is the only supported legacy checkpoint entry. Runtime model loading
    consumes the resulting config, tokenizer files, and safetensors directly.
    The destination must not exist so a failed migration cannot partially
    overwrite an artifact that may already be in service.
    """
    checkpoint = Path(checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    resolved_config_path = (
        Path(config_path).expanduser().resolve() if config_path is not None else None
    )
    if checkpoint.suffix != ".pth" or not checkpoint.is_file():
        raise ValueError(
            f"RWKV7 migration requires an existing .pth file: {checkpoint}"
        )
    if output_dir.exists():
        raise FileExistsError(
            f"RWKV7 HF artifact destination already exists: {output_dir}"
        )

    config = _load_migration_config(checkpoint, resolved_config_path)
    standard_weights = _validate_and_map_weights(
        _load_legacy_weights(checkpoint),
        config,
    )
    with checkpoint.open("rb") as source:
        config.rwkv_source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    config.rwkv_artifact_format_version = 1

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.migration-",
        dir=output_dir.parent,
    ) as temporary_dir:
        artifact = Path(temporary_dir)
        config.save_pretrained(artifact)
        GenerationConfig.from_model_config(config).save_pretrained(artifact)
        RWKVTokenizer().save_pretrained(artifact)
        save_file(
            standard_weights,
            artifact / "model.safetensors",
            metadata={"format": "pt"},
        )
        artifact.rename(output_dir)
    return output_dir
