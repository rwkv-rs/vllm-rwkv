# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned Transformers-RWKV runtime provenance."""

import importlib.util
import json
from functools import cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from vllm.transformers_utils.rwkv7_runtime_contract import (
    FLA_RWKV_REPOSITORY,
    FLA_RWKV_REVISION,
    FLASH_RWKV_REPOSITORY,
    FLASH_RWKV_REVISION,
    canonicalize_github_repository_url,
)

TRANSFORMERS_RWKV_DISTRIBUTION = "transformers"
TRANSFORMERS_RWKV_REPOSITORY = "https://github.com/rwkv-rs/transformers-rwkv.git"
TRANSFORMERS_RWKV_REVISION = "2bf8c01b7993f6b351678418664bb7bd798d0c71"
TRANSFORMERS_RWKV_REQUIREMENT = (
    f"{TRANSFORMERS_RWKV_DISTRIBUTION} @ "
    f"git+{TRANSFORMERS_RWKV_REPOSITORY}@{TRANSFORMERS_RWKV_REVISION}"
)

_RWKV7_CONFIG_MODULE = "transformers.models.rwkv7.configuration_rwkv7"
_RWKV7_MODEL_MODULE = "transformers.models.rwkv7.modeling_rwkv7"


def _validate_rwkv7_operator_provenance(
    provenance: object,
) -> dict[str, Any]:
    if not isinstance(provenance, dict):
        raise RuntimeError("vLLM RWKV7 operator provenance must be an object")
    raw_fla_repository = provenance.get("repository")
    raw_flash_repository = provenance.get("flash_rwkv_repository")
    try:
        fla_repository = canonicalize_github_repository_url(raw_fla_repository)
        flash_repository = canonicalize_github_repository_url(raw_flash_repository)
    except ValueError:
        fla_repository = None
        flash_repository = None
    if (
        fla_repository != canonicalize_github_repository_url(FLA_RWKV_REPOSITORY)
        or provenance.get("revision") != FLA_RWKV_REVISION
        or flash_repository != canonicalize_github_repository_url(FLASH_RWKV_REPOSITORY)
        or provenance.get("flash_rwkv_revision") != FLASH_RWKV_REVISION
    ):
        raise RuntimeError(
            "vLLM RWKV7 operator provenance mismatch: "
            f"FLA={raw_fla_repository!r}@{provenance.get('revision')!r}, "
            "FlashRWKV="
            f"{raw_flash_repository!r}@{provenance.get('flash_rwkv_revision')!r}; "
            f"expected FLA={FLA_RWKV_REPOSITORY!r}@{FLA_RWKV_REVISION!r}, "
            "FlashRWKV="
            f"{FLASH_RWKV_REPOSITORY!r}@{FLASH_RWKV_REVISION!r}."
        )
    return provenance


def _module_origin(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(
            f"Pinned Transformers-RWKV does not provide {module_name!r}. "
            f"Install `{TRANSFORMERS_RWKV_REQUIREMENT}`."
        )
    return Path(spec.origin).resolve()


@cache
def validate_transformers_rwkv7_runtime_provenance() -> dict[str, Any]:
    """Fail closed unless RWKV7 resolves from the pinned Transformers fork."""
    try:
        distribution = importlib_metadata.distribution(TRANSFORMERS_RWKV_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"vLLM RWKV7 requires `{TRANSFORMERS_RWKV_REQUIREMENT}`."
        ) from error

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError(
            "vLLM RWKV7 rejects registry or ordinary upstream Transformers; "
            f"install `{TRANSFORMERS_RWKV_REQUIREMENT}`."
        )
    try:
        direct_url = json.loads(direct_url_text)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "vLLM RWKV7 Transformers direct_url.json is invalid; "
            f"reinstall `{TRANSFORMERS_RWKV_REQUIREMENT}`."
        ) from error

    vcs_info = direct_url.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        raise RuntimeError(
            "vLLM RWKV7 requires a pinned Git Transformers install; "
            f"install `{TRANSFORMERS_RWKV_REQUIREMENT}`."
        )
    raw_repository = direct_url.get("url")
    try:
        repository = canonicalize_github_repository_url(raw_repository)
    except ValueError:
        repository = None
    requested_revision = str(vcs_info.get("requested_revision", "")).lower()
    resolved_revision = str(vcs_info.get("commit_id", "")).lower()
    if repository != TRANSFORMERS_RWKV_REPOSITORY or (
        requested_revision != TRANSFORMERS_RWKV_REVISION
        or resolved_revision != TRANSFORMERS_RWKV_REVISION
    ):
        raise RuntimeError(
            "vLLM RWKV7 Transformers provenance mismatch: "
            f"repository={raw_repository!r}, requested={requested_revision!r}, "
            f"resolved={resolved_revision!r}; install "
            f"`{TRANSFORMERS_RWKV_REQUIREMENT}`."
        )

    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.rwkv7 import Rwkv7Config, Rwkv7ForCausalLM
    from transformers.models.rwkv7.modeling_rwkv7 import (
        validate_rwkv7_runtime_provenance,
    )

    expected_config_origin = Path(
        str(
            distribution.locate_file("transformers/models/rwkv7/configuration_rwkv7.py")
        )
    ).resolve()
    expected_model_origin = Path(
        str(distribution.locate_file("transformers/models/rwkv7/modeling_rwkv7.py"))
    ).resolve()
    if (
        Rwkv7Config.__module__ != _RWKV7_CONFIG_MODULE
        or _module_origin(_RWKV7_CONFIG_MODULE) != expected_config_origin
    ):
        raise RuntimeError(
            "vLLM RWKV7 AutoConfig did not resolve Rwkv7Config from the "
            "pinned rwkv-rs Transformers distribution."
        )
    if (
        Rwkv7ForCausalLM.__module__ != _RWKV7_MODEL_MODULE
        or _module_origin(_RWKV7_MODEL_MODULE) != expected_model_origin
    ):
        raise RuntimeError(
            "vLLM RWKV7 AutoModel did not resolve Rwkv7ForCausalLM from the "
            "pinned rwkv-rs Transformers distribution."
        )

    auto_config = AutoConfig.for_model("rwkv7")
    auto_model = AutoModelForCausalLM._model_mapping[type(auto_config)]
    if type(auto_config) is not Rwkv7Config or auto_model is not Rwkv7ForCausalLM:
        raise RuntimeError(
            "vLLM RWKV7 AutoConfig/AutoModel registration does not resolve "
            "the pinned rwkv-rs RWKV7 implementation."
        )

    operator_provenance = _validate_rwkv7_operator_provenance(
        validate_rwkv7_runtime_provenance()
    )
    return {
        "distribution": TRANSFORMERS_RWKV_DISTRIBUTION,
        "distribution_version": distribution.version,
        "repository": TRANSFORMERS_RWKV_REPOSITORY,
        "revision": TRANSFORMERS_RWKV_REVISION,
        "config_module": Rwkv7Config.__module__,
        "causal_lm_module": Rwkv7ForCausalLM.__module__,
        "operator_runtime": operator_provenance,
    }


__all__ = [
    "TRANSFORMERS_RWKV_DISTRIBUTION",
    "TRANSFORMERS_RWKV_REPOSITORY",
    "TRANSFORMERS_RWKV_REQUIREMENT",
    "TRANSFORMERS_RWKV_REVISION",
    "canonicalize_github_repository_url",
    "validate_transformers_rwkv7_runtime_provenance",
]
