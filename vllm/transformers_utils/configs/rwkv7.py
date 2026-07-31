# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from huggingface_hub import hf_hub_download
from transformers import PretrainedConfig

from vllm.transformers_utils.config_parser_base import ConfigParserBase

BLINKDL_RWKV7_G1_REPO = "BlinkDL/rwkv7-g1"


@dataclass(frozen=True)
class _RWKV7G1Spec:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    decay_rank: int
    in_context_learning_rank: int
    value_residual_rank: int
    gate_rank: int


_RWKV7_G1_SPECS = {
    "rwkv7-g1d-0.1b-20260129-ctx8192.pth": _RWKV7G1Spec(12, 768, 3072, 64, 64, 32, 128),
    "rwkv7-g1d-0.4b-20260210-ctx8192.pth": _RWKV7G1Spec(
        24, 1024, 4096, 64, 64, 32, 128
    ),
    "rwkv7-g1f-1.5b-20260419-ctx8192.pth": _RWKV7G1Spec(
        24, 2048, 8192, 96, 96, 64, 256
    ),
    "rwkv7-g1f-2.9b-20260420-ctx8192.pth": _RWKV7G1Spec(
        32, 2560, 10240, 96, 96, 64, 320
    ),
    "rwkv7-g1f-7.2b-20260414-ctx8192.pth": _RWKV7G1Spec(
        32, 4096, 16384, 128, 128, 96, 480
    ),
    "rwkv7-g1f-13.3b-20260415-ctx8192.pth": _RWKV7G1Spec(
        61, 4096, 16384, 192, 192, 128, 384
    ),
    "rwkv7-g1g-1.5b-20260526-ctx8192.pth": _RWKV7G1Spec(
        24, 2048, 8192, 96, 96, 64, 256
    ),
    "rwkv7-g1g-2.9b-20260526-ctx8192.pth": _RWKV7G1Spec(
        32, 2560, 10240, 96, 96, 64, 320
    ),
    "rwkv7-g1g-7.2b-20260523-ctx8192.pth": _RWKV7G1Spec(
        32, 4096, 16384, 128, 128, 96, 480
    ),
    "rwkv7-g1g-13.3b-20260523-ctx8192.pth": _RWKV7G1Spec(
        61, 4096, 16384, 192, 192, 128, 384
    ),
    "rwkv7-g1h-1.5b-20260710-ctx10240.pth": _RWKV7G1Spec(
        24, 2048, 8192, 96, 96, 64, 256
    ),
    "rwkv7-g1h-7.2b-20260710-ctx10240.pth": _RWKV7G1Spec(
        32, 4096, 16384, 128, 128, 96, 480
    ),
}

_RWKV7_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "head_size",
    "num_hidden_layers",
    "max_position_embeddings",
    "decay_rank",
    "in_context_learning_rank",
    "value_residual_rank",
    "gate_rank",
)

_RWKV7_CONTEXT_RE = re.compile(r"-ctx(?P<context>\d+)\.pth$")


@dataclass(frozen=True)
class RWKV7PthSource:
    filename: str
    local_path: Path | None = None
    repo_id: str | None = None
    revision: str | None = None


class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"
    architectures = ["RWKV7ForCausalLM"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        head_size: int = 64,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 8192,
        decay_rank: int = 96,
        in_context_learning_rank: int = 96,
        value_residual_rank: int = 64,
        gate_rank: int = 256,
        **kwargs,
    ):
        values = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "head_size": head_size,
            "num_hidden_layers": num_hidden_layers,
            "max_position_embeddings": max_position_embeddings,
            "decay_rank": decay_rank,
            "in_context_learning_rank": in_context_learning_rank,
            "value_residual_rank": value_residual_rank,
            "gate_rank": gate_rank,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values.values()
        ):
            raise ValueError("RWKV7 config values must be positive integers.")
        if head_size != 64:
            raise ValueError("RWKV7 currently requires head_size=64.")
        if hidden_size % head_size != 0:
            raise ValueError("RWKV7 hidden_size must be divisible by head_size.")

        kwargs.setdefault("architectures", self.architectures)
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = hidden_size // head_size
        self.max_position_embeddings = max_position_embeddings
        self.decay_rank = decay_rank
        self.in_context_learning_rank = in_context_learning_rank
        self.value_residual_rank = value_residual_rank
        self.gate_rank = gate_rank


class RWKV7PthConfigParser(ConfigParserBase):
    supports_hf_quant_config = False

    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        del trust_remote_code, revision, code_revision, kwargs
        config = build_rwkv7_config_from_pth(model)
        if config is None:
            raise ValueError(f"Not a supported RWKV7 raw .pth source: {model!s}")
        return config.to_dict(), config


def try_parse_rwkv7_pth_source(model: str | Path) -> RWKV7PthSource | None:
    model_str = str(model)
    path = Path(model_str)
    if path.is_file() and path.suffix == ".pth":
        return RWKV7PthSource(filename=path.name, local_path=path)

    parsed = urlparse(model_str)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
            return None
        path = Path(unquote(parsed.path))
        if path.is_absolute() and path.is_file() and path.suffix == ".pth":
            return RWKV7PthSource(filename=path.name, local_path=path)
        return None
    if parsed.scheme not in {"http", "https"}:
        return None
    if parsed.netloc != "huggingface.co":
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5:
        return None
    repo_id = "/".join(parts[:2])
    marker = parts[2]
    if repo_id != BLINKDL_RWKV7_G1_REPO or marker not in {"blob", "resolve"}:
        return None

    revision = parts[3]
    filename = "/".join(parts[4:])
    if not filename.endswith(".pth"):
        return None
    return RWKV7PthSource(filename=filename, repo_id=repo_id, revision=revision)


def is_rwkv7_pth_source(model: str | Path) -> bool:
    """Return whether auto config detection can identify an RWKV-7 checkpoint."""

    source = try_parse_rwkv7_pth_source(model)
    if source is None:
        return False
    if Path(source.filename).name in _RWKV7_G1_SPECS:
        return True
    if source.local_path is None:
        return False

    config_path = source.local_path.parent / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(raw_config, dict) and raw_config.get("model_type") == "rwkv7"


def build_rwkv7_config_from_pth(model: str | Path) -> RWKV7Config | None:
    source = try_parse_rwkv7_pth_source(model)
    if source is None:
        return None

    filename = Path(source.filename).name
    spec = _RWKV7_G1_SPECS.get(filename)
    if source.local_path is not None:
        config_path = source.local_path.parent / "config.json"
        if config_path.is_file():
            try:
                raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid RWKV7 checkpoint config: {config_path}"
                ) from error
            if not isinstance(raw_config, dict):
                raise ValueError(f"Invalid RWKV7 checkpoint config: {config_path}")
            if raw_config.get("model_type") != "rwkv7" or raw_config.get(
                "architectures"
            ) != ["RWKV7ForCausalLM"]:
                raise ValueError(f"Invalid RWKV7 checkpoint config: {config_path}")
            if spec is not None:
                config = _build_rwkv7_config_from_spec(filename, spec)
                mismatched = [
                    key
                    for key in _RWKV7_CONFIG_FIELDS
                    if key in raw_config and raw_config[key] != getattr(config, key)
                ]
                if mismatched:
                    raise ValueError(
                        f"Invalid RWKV7 checkpoint config: {config_path}; "
                        "official checkpoint fields do not match the catalog: "
                        f"{', '.join(mismatched)}"
                    )
                return config

            missing = [key for key in _RWKV7_CONFIG_FIELDS if key not in raw_config]
            if missing:
                raise ValueError(
                    f"Invalid RWKV7 checkpoint config: {config_path}; "
                    f"missing required fields: {', '.join(missing)}"
                )
            try:
                return RWKV7Config(
                    **{key: raw_config[key] for key in _RWKV7_CONFIG_FIELDS}
                )
            except ValueError as error:
                raise ValueError(
                    f"Invalid RWKV7 checkpoint config: {config_path}"
                ) from error

    if spec is None:
        raise ValueError(
            f"Unsupported RWKV7 raw .pth checkpoint: {filename}. "
            "Expected an audited BlinkDL/rwkv7-g1 checkpoint filename or a "
            "local checkpoint with a complete config.json sidecar."
        )
    return _build_rwkv7_config_from_spec(filename, spec)


def _build_rwkv7_config_from_spec(filename: str, spec: _RWKV7G1Spec) -> RWKV7Config:
    match = _RWKV7_CONTEXT_RE.search(filename)
    assert match is not None
    return RWKV7Config(
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.num_hidden_layers,
        max_position_embeddings=int(match.group("context")),
        decay_rank=spec.decay_rank,
        in_context_learning_rank=spec.in_context_learning_rank,
        value_residual_rank=spec.value_residual_rank,
        gate_rank=spec.gate_rank,
    )


def download_rwkv7_pth_source(
    source: RWKV7PthSource,
    *,
    cache_dir: str | None,
    revision: str | None,
) -> Path:
    if source.local_path is not None:
        return source.local_path
    if source.repo_id is None:
        raise ValueError(f"Cannot download non-Hugging Face RWKV7 source: {source}")
    resolved = hf_hub_download(
        repo_id=source.repo_id,
        filename=source.filename,
        revision=revision or source.revision,
        cache_dir=cache_dir,
    )
    return Path(resolved)
