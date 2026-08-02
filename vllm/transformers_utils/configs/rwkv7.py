# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import regex as re
from transformers import PretrainedConfig

from vllm.transformers_utils.repo_utils import hf_api

BLINKDL_RWKV7_G1_REPO = "BlinkDL/rwkv7-g1"
STANDARD_RWKV7_ARCHITECTURE = "Rwkv7ForCausalLM"
LEGACY_PTH_RWKV7_ARCHITECTURE = "RWKV7ForCausalLM"

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
        head_size: int = 64,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 8192,
        **kwargs,
    ):
        kwargs.setdefault("architectures", self.architectures)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = hidden_size // head_size
        self.max_position_embeddings = max_position_embeddings


def _rwkv7_config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def validate_rwkv7_hf_artifact_config(
    config: object, *, allow_legacy_pth: bool = False
) -> None:
    required_values = (
        "vocab_size",
        "hidden_size",
        "head_size",
        "num_hidden_layers",
        "max_position_embeddings",
    )
    architectures = _rwkv7_config_value(config, "architectures")
    allowed_architectures = [[STANDARD_RWKV7_ARCHITECTURE]]
    if allow_legacy_pth:
        allowed_architectures.append([LEGACY_PTH_RWKV7_ARCHITECTURE])
    values = {name: _rwkv7_config_value(config, name) for name in required_values}
    valid_values = all(
        not isinstance(value, bool) and isinstance(value, int) and value > 0
        for value in values.values()
    )
    divisible_head_size = valid_values and (
        int(values["hidden_size"]) % int(values["head_size"]) == 0
    )
    if (
        _rwkv7_config_value(config, "model_type") != "rwkv7"
        or architectures not in allowed_architectures
        or not valid_values
        or not divisible_head_size
    ):
        raise ValueError(
            "Invalid RWKV7 Hugging Face artifact config: expected model_type="
            f"'rwkv7', architectures={allowed_architectures}, positive integer "
            "vocab_size/hidden_size/head_size/num_hidden_layers/"
            "max_position_embeddings, and hidden_size divisible by head_size."
        )


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


def build_rwkv7_config_from_pth(model: str | Path) -> RWKV7Config | None:
    source = try_parse_rwkv7_pth_source(model)
    if source is None:
        return None

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
            try:
                validate_rwkv7_hf_artifact_config(raw_config, allow_legacy_pth=True)
            except ValueError as error:
                raise ValueError(
                    f"Invalid RWKV7 checkpoint config: {config_path}"
                ) from error
            return RWKV7Config(
                vocab_size=raw_config["vocab_size"],
                hidden_size=raw_config["hidden_size"],
                head_size=raw_config["head_size"],
                num_hidden_layers=raw_config["num_hidden_layers"],
                max_position_embeddings=raw_config["max_position_embeddings"],
            )

    filename = Path(source.filename).name
    match = _RWKV7_G1_FILENAME_RE.match(filename)
    if match is None:
        raise ValueError(
            f"Unsupported RWKV7 raw .pth checkpoint: {filename}. "
            "Expected a BlinkDL/rwkv7-g1 filename such as "
            "rwkv7-g1g-1.5b-20260526-ctx8192.pth."
        )

    size = match.group("size").lower()
    num_layers, hidden_size = _RWKV7_G1_SPECS[size]
    return RWKV7Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        max_position_embeddings=int(match.group("context")),
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
    resolved = hf_api().hf_hub_download(
        repo_id=source.repo_id,
        filename=source.filename,
        revision=revision or source.revision,
        cache_dir=cache_dir,
    )
    return Path(resolved)
