# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.rwkv7 import (
    RWKV7_COMPRESSED_TENSORS_VERSION,
    RWKV7_NVFP4_CONSUMER_SEMANTIC_REVISION,
    RWKV7_NVFP4_W4A16_CONSUMER,
    RWKV7_SUPPORTED_PACKED_FORMATS,
    RWKV7Config,
    rwkv7_checkpoint_weight_shapes,
    rwkv7_internal_to_hf_weight_name,
    rwkv7_is_legacy_low_rank_weight_name,
    rwkv7_legacy_checkpoint_weight_shapes,
    validate_rwkv7_hf_artifact_config,
    validate_rwkv7_quantization_artifact_metadata,
)
from vllm.transformers_utils.rwkv7_artifact import (
    convert_rwkv7_pth_to_hf_artifact,
)


def _standard_rwkv7_hf_config(**overrides: object) -> SimpleNamespace:
    values = {
        "architectures": ["Rwkv7ForCausalLM"],
        "model_type": "rwkv7",
        "vocab_size": 65536,
        "hidden_size": 64,
        "head_size": 64,
        "num_hidden_layers": 2,
        "context_length": 321,
        "intermediate_size": 128,
        "embedding_layer_norm_fused": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _nvfp4_targets(candidate: str, num_hidden_layers: int = 2) -> list[str]:
    if candidate in {"nvfp4-w4a4", "nvfp4-w4a16"}:
        return [
            f"model.blocks.{layer_id}.ffn.{name}"
            for layer_id in range(num_hidden_layers)
            for name in ("key", "value")
        ]
    assert candidate == "nvfp4-w4a16-protection-ablation"
    targets = [
        *(
            f"model.blocks.0.att.{name}"
            for name in ("w1", "w2", "a1", "a2", "g1", "g2")
        ),
        *(f"model.blocks.0.att.{name}" for name in ("receptance", "key", "output")),
    ]
    targets.extend(
        target
        for layer_id in range(1, num_hidden_layers)
        for target in (
            *(
                f"model.blocks.{layer_id}.att.{name}"
                for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
            ),
            *(
                f"model.blocks.{layer_id}.att.{name}"
                for name in ("receptance", "key", "value", "output")
            ),
        )
    )
    return targets


def _nvfp4_consumer_config(
    candidate: str = "nvfp4-w4a16-protection-ablation",
) -> SimpleNamespace:
    config = _standard_rwkv7_hf_config(vocab_size=32)
    targets = _nvfp4_targets(candidate, config.num_hidden_layers)
    target_set = set(targets)
    linear_modules = ["head"]
    normalization_modules = ["model.ln_out", "model.blocks.0.ln0"]
    low_rank_modules: list[str] = []
    for layer_id in range(config.num_hidden_layers):
        attention = f"model.blocks.{layer_id}.att"
        linear_modules.extend(
            f"{attention}.{name}" for name in ("receptance", "key", "value", "output")
        )
        layer_low_rank = (
            ("w1", "w2", "a1", "a2", "g1", "g2")
            if layer_id == 0
            else ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
        )
        low_rank_modules.extend(f"{attention}.{name}" for name in layer_low_rank)
        linear_modules.extend(f"{attention}.{name}" for name in layer_low_rank)
        linear_modules.extend(
            f"model.blocks.{layer_id}.ffn.{name}" for name in ("key", "value")
        )
        normalization_modules.extend(
            (
                f"model.blocks.{layer_id}.ln1",
                f"model.blocks.{layer_id}.ln2",
                f"{attention}.ln_x",
            )
        )
    protected_linear = [name for name in linear_modules if name not in target_set]
    protected_embedding = ["model.embeddings"]
    protected_normalization = normalization_modules
    protected_modules = [
        *protected_linear,
        *protected_embedding,
        *protected_normalization,
    ]
    module_parameter_keys = {
        *(f"{name}.weight" for name in linear_modules),
        "model.embeddings.weight",
        *(
            parameter_name
            for name in normalization_modules
            for parameter_name in (f"{name}.weight", f"{name}.bias")
        ),
    }
    expected_keys = set(rwkv7_checkpoint_weight_shapes(config))
    protected_state = sorted(expected_keys - module_parameter_keys)
    protected_parameter_keys = [
        *(f"{name}.weight" for name in protected_linear),
        "model.embeddings.weight",
        *(
            parameter_name
            for name in protected_normalization
            for parameter_name in (f"{name}.weight", f"{name}.bias")
        ),
        *protected_state,
    ]
    quantized_low_rank = [name for name in targets if name in low_rank_modules]
    consumer = (
        "vllm-rwkv-nvfp4-w4a4" if candidate == "nvfp4-w4a4" else "vllm-rwkv-nvfp4-w4a16"
    )
    target_schema = (
        "rwkv7-nvfp4-critical-high-v1"
        if candidate in {"nvfp4-w4a4", "nvfp4-w4a16"}
        else "rwkv7-nvfp4-protection-ablation-no-ffn-v1"
    )
    metadata = {
        "schema_version": 3,
        "candidate": candidate,
        "target_policy": {
            "policy": "rwkv7",
            "policy_version": 2,
            "model_type": "rwkv7",
            "base_model_prefix": "model",
            "selection": {"kind": "module", "names": targets},
            "protections": [
                {"kind": "module", "names": protected_modules},
                {"kind": "tensor", "names": protected_state},
            ],
            "recipe": {
                "schema_version": 1,
                "candidate": candidate,
                "targets": targets,
                "framework_versions": {
                    "compressed_tensors": RWKV7_COMPRESSED_TENSORS_VERSION,
                    "transformers": "5.15.0.dev0",
                },
            },
        },
        "vllm": {
            "architecture": "Rwkv7ForCausalLM",
            "model_type": "rwkv7",
            "source_format": "standard_hf",
            "load_format": "safetensors",
            "quant_method": "compressed-tensors",
            "quantization_format": "nvfp4-pack-quantized",
            "target_schema_version": 1,
            "target_schema": target_schema,
            "num_hidden_layers": config.num_hidden_layers,
            "quantized_target_fqns_digest": hashlib.sha256(
                json.dumps(targets, separators=(",", ":")).encode()
            ).hexdigest(),
            "consumer_capabilities": [
                "transformers-rwkv-compressed-tensors",
                consumer,
            ],
            "vllm_consumer_requirement": consumer,
            "vllm_consumer_revision": RWKV7_NVFP4_CONSUMER_SEMANTIC_REVISION,
            "legacy_pth_direct_load": False,
            "quantization_target_type": "Linear",
            "linear_weight_suffix": "weight",
            "linear_weight_layout": "out-in",
            "quantized_modules": targets,
            "quantized_weight_names": [f"{name}.weight" for name in targets],
            "low_rank_linear_modules": sorted(low_rank_modules),
            "quantized_low_rank_modules": quantized_low_rank,
            "protected_v_first_linear_modules": sorted(
                name for name in protected_linear if name.endswith((".v1", ".v2"))
            ),
            "layer_zero_v_first_producer": "model.blocks.0.att.value",
            "protected_linear_modules": protected_linear,
            "protected_embedding_modules": protected_embedding,
            "protected_normalization_modules": protected_normalization,
            "protected_state_tensors": protected_state,
            "protected_modules": protected_modules,
            "protected_tensors": protected_state,
            "protected_parameter_keys": protected_parameter_keys,
        },
    }
    weights = {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": False,
        "scale_dtype": "float8_e4m3fn",
    }
    input_activations = (
        {
            **weights,
            "dynamic": "local",
        }
        if candidate == "nvfp4-w4a4"
        else None
    )
    config.rwkv7_quantization_metadata = metadata
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
    )

    serialized_weights = QuantizationArgs(**weights)
    serialized_input_activations = (
        QuantizationArgs(**input_activations) if input_activations is not None else None
    )
    config.quantization_config = QuantizationConfig(
        config_groups={
            "group_0": QuantizationScheme(
                targets=["Linear"],
                weights=serialized_weights,
                input_activations=serialized_input_activations,
                format="nvfp4-pack-quantized",
            )
        },
        ignore=protected_linear,
        format="nvfp4-pack-quantized",
        quantization_status="compressed",
    ).model_dump(mode="json")
    return config


def _quantized_rwkv7_config() -> SimpleNamespace:
    return _nvfp4_consumer_config("nvfp4-w4a4")


def _write_legacy_checkpoint(checkpoint: Path) -> dict[str, torch.Tensor]:
    config = _standard_rwkv7_hf_config()
    weights = {}
    for index, (name, shape) in enumerate(
        rwkv7_legacy_checkpoint_weight_shapes(config).items(),
        start=1,
    ):
        if rwkv7_is_legacy_low_rank_weight_name(name):
            numel = math.prod(shape)
            weights[name] = (
                torch.arange(numel, dtype=torch.float32)
                .reshape(shape)
                .div_(max(1, numel))
                .add_(index)
                .to(torch.float16)
            )
        else:
            weights[name] = torch.full(shape, index / 128, dtype=torch.float16)
    torch.save(weights, checkpoint)
    checkpoint.with_name("config.json").write_text(
        json.dumps(
            {
                **vars(config),
                "architectures": ["RWKV7ForCausalLM"],
                "bos_token_id": 0,
                "eos_token_id": 0,
                "pad_token_id": 0,
            }
        ),
        encoding="utf-8",
    )
    return weights


def test_standard_rwkv7_hf_artifact_config_is_accepted() -> None:
    validate_rwkv7_hf_artifact_config(_standard_rwkv7_hf_config())


def test_rwkv7_explicit_projection_ranks_drive_checkpoint_shapes() -> None:
    config = _standard_rwkv7_hf_config(
        hidden_size=2048,
        head_size=64,
        num_hidden_layers=2,
        intermediate_size=8192,
        decay_low_rank_dim=96,
        a_low_rank_dim=96,
        v_low_rank_dim=64,
        gate_low_rank_dim=256,
    )

    shapes = rwkv7_checkpoint_weight_shapes(config)

    assert shapes["model.blocks.0.att.w1.weight"] == (96, 2048)
    assert shapes["model.blocks.0.att.a1.weight"] == (96, 2048)
    assert shapes["model.blocks.0.att.g1.weight"] == (256, 2048)
    assert shapes["model.blocks.1.att.v1.weight"] == (64, 2048)


def test_rwkv7_schema_v3_quantization_metadata_is_validated() -> None:
    config = _quantized_rwkv7_config()

    assert validate_rwkv7_quantization_artifact_metadata(config) == (
        "nvfp4-pack-quantized"
    )
    assert frozenset({"nvfp4-pack-quantized"}) == RWKV7_SUPPORTED_PACKED_FORMATS
    validate_rwkv7_hf_artifact_config(config)


def test_rwkv7_nvfp4_consumer_revision_pins_all_three_candidates() -> None:
    assert RWKV7_NVFP4_CONSUMER_SEMANTIC_REVISION == (
        "72ad9109e63e00b67f8707c791a9f43783eff4b1"
    )


def test_rwkv7_w4a16_baseline_is_distinct_weight_only_candidate() -> None:
    baseline = _nvfp4_consumer_config("nvfp4-w4a16")
    ablation = _nvfp4_consumer_config("nvfp4-w4a16-protection-ablation")

    assert validate_rwkv7_quantization_artifact_metadata(baseline) == (
        "nvfp4-pack-quantized"
    )
    baseline_metadata = baseline.rwkv7_quantization_metadata["vllm"]
    ablation_metadata = ablation.rwkv7_quantization_metadata["vllm"]
    assert baseline.rwkv7_quantization_metadata["candidate"] == "nvfp4-w4a16"
    assert baseline_metadata["target_schema"] == "rwkv7-nvfp4-critical-high-v1"
    assert baseline_metadata["vllm_consumer_requirement"] == (
        RWKV7_NVFP4_W4A16_CONSUMER
    )
    assert (
        baseline.quantization_config["config_groups"]["group_0"]["input_activations"]
        is None
    )
    assert all(".ffn." in target for target in baseline_metadata["quantized_modules"])
    assert not any(
        ".ffn." in target for target in ablation_metadata["quantized_modules"]
    )
    assert (
        baseline_metadata["quantized_modules"] != ablation_metadata["quantized_modules"]
    )


@pytest.mark.parametrize(
    "candidate",
    ("nvfp4-w4a16", "nvfp4-w4a16-protection-ablation"),
)
def test_rwkv7_w4a16_accepts_standard_compressed_tensors_serialization(
    candidate: str,
) -> None:
    config = _nvfp4_consumer_config(candidate)
    vllm_metadata = config.rwkv7_quantization_metadata["vllm"]
    group = config.quantization_config["config_groups"]["group_0"]

    assert group["targets"] == ["Linear"]
    assert set(config.quantization_config["ignore"]) == set(
        vllm_metadata["protected_linear_modules"]
    )
    assert set(config.quantization_config["ignore"]).isdisjoint(
        vllm_metadata["protected_embedding_modules"]
    )
    assert set(config.quantization_config["ignore"]).isdisjoint(
        vllm_metadata["protected_normalization_modules"]
    )
    config.quantization_config["ignore"].reverse()

    assert validate_rwkv7_quantization_artifact_metadata(config) == (
        "nvfp4-pack-quantized"
    )


def test_rwkv7_w4a16_baseline_rejects_activation_quantization() -> None:
    config = _nvfp4_consumer_config("nvfp4-w4a16")
    group = config.quantization_config["config_groups"]["group_0"]
    group["input_activations"] = {
        **group["weights"],
        "dynamic": "local",
    }

    with pytest.raises(ValueError, match="W4A16 requires input_activations=None"):
        validate_rwkv7_quantization_artifact_metadata(config)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("schema", "schema_version"),
        ("identity", "model/loader identity"),
        ("weight_key", r"standard \*\.weight Linear keys"),
        ("category_overlap", "categories must be disjoint"),
        ("category_incomplete", "categories are incomplete"),
        ("target_policy", "target policy disagrees"),
        ("candidate", "candidate must be"),
        ("consumer", "exact candidate consumer"),
        ("consumer_capability", "exact candidate consumer"),
        ("framework_version", "compressed-tensors version"),
        ("scheme_targets", r"exactly \['Linear'\]"),
        ("group_format", "config_group format"),
        ("output_activations", "output_activations=None"),
        ("ignore_missing", "protected Linear inventory"),
        ("ignore_extra", "protected Linear inventory"),
        ("weight_scheme", "weight quantization arguments"),
        ("scheme", "dynamic-local"),
        ("digest", "FQN digest"),
        ("normalization_bias", "parameter key inventory is incomplete"),
        ("unknown_protected", "close the whole model"),
    ],
)
def test_invalid_rwkv7_quantization_metadata_fails_closed(
    case: str,
    match: str,
) -> None:
    config = _quantized_rwkv7_config()
    metadata = deepcopy(config.rwkv7_quantization_metadata)
    vllm_metadata = metadata["vllm"]
    assert isinstance(vllm_metadata, dict)

    if case == "schema":
        metadata["schema_version"] = 2
    elif case == "identity":
        vllm_metadata["quantization_target_type"] = "Parameter"
    elif case == "weight_key":
        weight_names = list(vllm_metadata["quantized_weight_names"])
        weight_names[0] = weight_names[0].removesuffix(".weight") + ".weight_packed"
        vllm_metadata["quantized_weight_names"] = weight_names
    elif case == "category_overlap":
        embedding_modules = list(vllm_metadata["protected_embedding_modules"])
        embedding_modules.append("head")
        vllm_metadata["protected_embedding_modules"] = embedding_modules
    elif case == "category_incomplete":
        vllm_metadata["protected_modules"] = list(vllm_metadata["protected_modules"])[
            :-1
        ]
    elif case == "target_policy":
        target_policy = metadata["target_policy"]
        assert isinstance(target_policy, dict)
        selection = target_policy["selection"]
        assert isinstance(selection, dict)
        selection["names"] = list(selection["names"])[1:]
    elif case == "candidate":
        metadata["candidate"] = "nvfp4-w4a16-unknown"
    elif case == "consumer":
        vllm_metadata["vllm_consumer_revision"] = (
            "6df9bfb41d7ec091fc9e13210ca4249389518114"
        )
    elif case == "consumer_capability":
        vllm_metadata["consumer_capabilities"] = [
            "transformers-rwkv-compressed-tensors"
        ]
    elif case == "framework_version":
        target_policy = metadata["target_policy"]
        assert isinstance(target_policy, dict)
        target_policy["recipe"]["framework_versions"]["compressed_tensors"] = "0.17.0"
    elif case == "scheme_targets":
        config.quantization_config["config_groups"]["group_0"]["targets"] = list(
            vllm_metadata["quantized_modules"]
        )
    elif case == "group_format":
        config.quantization_config["config_groups"]["group_0"]["format"] = (
            "pack-quantized"
        )
    elif case == "output_activations":
        group = config.quantization_config["config_groups"]["group_0"]
        group["output_activations"] = deepcopy(group["weights"])
    elif case == "ignore_missing":
        config.quantization_config["ignore"] = list(
            config.quantization_config["ignore"]
        )[1:]
    elif case == "ignore_extra":
        config.quantization_config["ignore"] = [
            *config.quantization_config["ignore"],
            vllm_metadata["protected_embedding_modules"][0],
        ]
    elif case == "weight_scheme":
        config.quantization_config["config_groups"]["group_0"]["weights"][
            "group_size"
        ] = 32
    elif case == "scheme":
        config.quantization_config["config_groups"]["group_0"]["input_activations"] = (
            None
        )
    elif case == "digest":
        vllm_metadata["quantized_target_fqns_digest"] = "0" * 64
    elif case == "normalization_bias":
        vllm_metadata["protected_parameter_keys"].remove("model.ln_out.bias")
    elif case == "unknown_protected":
        unknown = "model.unknown_state"
        vllm_metadata["protected_state_tensors"].append(unknown)
        vllm_metadata["protected_parameter_keys"].append(unknown)
    else:
        raise AssertionError(f"unknown test case: {case}")

    config.rwkv7_quantization_metadata = metadata
    with pytest.raises(ValueError, match=match):
        validate_rwkv7_quantization_artifact_metadata(config)


def test_rwkv7_compressed_tensors_requires_schema_v3_metadata() -> None:
    config = _standard_rwkv7_hf_config(
        quantization_config={
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "format": "pack-quantized",
        }
    )

    with pytest.raises(ValueError, match="missing rwkv7_quantization_metadata"):
        validate_rwkv7_hf_artifact_config(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_type": "rwkv"},
        {"architectures": None},
        {"architectures": ["RWKV7ForCausalLM"]},
        {"architectures": ["AutoModelForCausalLM"]},
        {"hidden_size": 0},
        {"head_size": 96},
        {"num_hidden_layers": True},
        {"context_length": None},
    ],
)
def test_invalid_rwkv7_hf_artifact_config_fails_closed(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Invalid RWKV7 Hugging Face artifact"):
        validate_rwkv7_hf_artifact_config(_standard_rwkv7_hf_config(**overrides))


def test_runtime_loader_rejects_raw_pth_directory(tmp_path: Path) -> None:
    (tmp_path / "model.pth").touch()
    loader: Any = object.__new__(DefaultModelLoader)
    setattr(
        loader,
        "load_config",
        SimpleNamespace(
            load_format="auto",
            download_dir=None,
            ignore_patterns=None,
        ),
    )

    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            subfolder=None,
            revision=None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


@pytest.mark.parametrize(
    "hf_config",
    [
        SimpleNamespace(
            model_type="llama",
            rwkv7_quantization_metadata={"schema_version": 3},
        ),
        _standard_rwkv7_hf_config(),
    ],
    ids=("non-rwkv", "rwkv-without-metadata"),
)
def test_default_loader_rwkv7_quantization_preflight_is_otherwise_a_noop(
    hf_config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda _config: None)
    monkeypatch.setattr(
        loader,
        "get_all_weights",
        lambda _config, _model: iter(()),
    )
    model = SimpleNamespace(load_weights=lambda _weights: None)
    model_config = SimpleNamespace(hf_config=hf_config, quantization=None)

    loader.load_weights(model, model_config)


def test_default_loader_rejects_invalid_rwkv7_consumer_before_weight_iterator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(
        loader,
        "_init_ep_weight_filter",
        lambda _config: pytest.fail("EP initialization ran before RWKV7 preflight"),
    )
    monkeypatch.setattr(
        loader,
        "get_all_weights",
        lambda _config, _model: pytest.fail(
            "weight iterator opened before RWKV7 preflight"
        ),
    )
    config = _quantized_rwkv7_config()
    config.rwkv7_quantization_metadata["vllm"]["vllm_consumer_revision"] = None
    model_config = SimpleNamespace(hf_config=config, quantization="compressed-tensors")

    with pytest.raises(ValueError, match="exact candidate consumer"):
        loader.load_weights(SimpleNamespace(), model_config)


def test_rwkv7_rejects_invalid_consumer_before_quantized_linear_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.models.rwkv7 as rwkv7_module

    config = _quantized_rwkv7_config()
    config.rwkv7_quantization_metadata["vllm"]["vllm_consumer_revision"] = None
    model = object.__new__(RWKV7ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.quant_config = SimpleNamespace(get_name=lambda: "compressed-tensors")
    model.tp_size = 1
    model._quantized_linears = {}
    monkeypatch.setattr(
        rwkv7_module,
        "ReplicatedLinear",
        lambda **_kwargs: pytest.fail(
            "quantized tensor/kernel initialization ran before RWKV7 preflight"
        ),
    )

    with pytest.raises(ValueError, match="exact candidate consumer"):
        model._initialize_quantized_linears()


def _exercise_nvfp4_consumer_candidate(
    candidate: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4,
    )

    kernel_calls: list[nn.Module] = []
    process_calls: list[nn.Module] = []

    class FunctionalKernel:
        @staticmethod
        def input_quant_key():
            return None

        @staticmethod
        def process_weights_after_loading(layer: nn.Module) -> None:
            layer.backend_weights_processed = True
            process_calls.append(layer)

        @staticmethod
        def apply_weights(
            *,
            layer: nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert bias is None
            assert layer.backend_weights_processed
            kernel_calls.append(layer)
            return x.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], layer.output_size)

    monkeypatch.setattr(
        compressed_tensors_w4a4_nvfp4,
        "init_nvfp4_linear_kernel",
        lambda **_kwargs: FunctionalKernel(),
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    config = _nvfp4_consumer_config(candidate)
    validate_rwkv7_hf_artifact_config(config)
    quant_config = CompressedTensorsConfig.from_config(
        deepcopy(config.quantization_config)
    )
    model: Any = object.__new__(RWKV7ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.quant_config = quant_config
    model.tp_size = 1
    model._quantized_linears = {}
    model._initialize_quantized_linears()

    targets = config.rwkv7_quantization_metadata["vllm"]["quantized_modules"]
    expected_target_count = (
        9 + 12 * (config.num_hidden_layers - 1)
        if candidate == "nvfp4-w4a16-protection-ablation"
        else 2 * config.num_hidden_layers
    )
    assert len(targets) == expected_target_count
    assert set(model._quantized_linears) == set(targets)
    parameter_identities: dict[str, int] = {}
    artifact_tensors: dict[str, torch.Tensor] = {}
    for index, target in enumerate(targets, start=1):
        linear = model.get_submodule(target)
        expected_parameter_names = {
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
        }
        if candidate == "nvfp4-w4a4":
            expected_parameter_names.add("input_global_scale")
        assert set(dict(linear.named_parameters(recurse=False))) == (
            expected_parameter_names
        )
        assert linear.params_dtype == torch.float16
        assert linear.scheme.use_a16 == (candidate != "nvfp4-w4a4")
        assert not hasattr(linear, "weight")
        for parameter_name, parameter in linear.named_parameters(recurse=False):
            full_name = f"{target}.{parameter_name}"
            parameter_identities[full_name] = id(parameter)
            parameter.data.zero_()
            fill_value = (
                index
                if parameter.dtype == torch.uint8
                else (2 if parameter_name.endswith("global_scale") else 1)
            )
            artifact_tensors[full_name] = torch.full(
                tuple(parameter.shape),
                fill_value,
                dtype=parameter.dtype,
            )

    expected_shapes = rwkv7_checkpoint_weight_shapes(config)
    quantized_dense_names = {f"{target}.weight" for target in targets}
    for index, (name, shape) in enumerate(expected_shapes.items(), start=1):
        if name in quantized_dense_names:
            continue
        artifact_tensors[name] = torch.full(
            shape,
            (index % 7 + 1) / 16,
            dtype=torch.float16,
        )
    protected_value = artifact_tensors["model.blocks.0.att.value.weight"].clone()
    protected_state = artifact_tensors["model.blocks.0.ffn.x_k"].clone()

    artifact = tmp_path / "rwkv7-nvfp4"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(vars(config)),
        encoding="utf-8",
    )
    save_file(artifact_tensors, artifact / "model.safetensors")

    setattr(model, "_preprocess_weights", lambda _weights: None)
    setattr(
        model,
        "_commit_preprocessed_weights",
        lambda weights: setattr(model, "z", weights),
    )
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda _config: None)
    model_config: Any = SimpleNamespace(
        hf_config=config,
        quantization="compressed-tensors",
        model=str(artifact),
        revision=None,
        dtype=torch.float16,
    )

    loader.load_weights(model, model_config)

    for full_name, identity in parameter_identities.items():
        target, parameter_name = full_name.rsplit(".", 1)
        parameter = dict(model.get_submodule(target).named_parameters(recurse=False))[
            parameter_name
        ]
        assert id(parameter) == identity
        torch.testing.assert_close(parameter, artifact_tensors[full_name])
        assert not hasattr(model.get_submodule(target), "weight")
    checkpoint_weight_names: set[str] = model.checkpoint_weight_names or set()
    assert quantized_dense_names.isdisjoint(checkpoint_weight_names)
    for target in targets:
        internal_name = model._checkpoint_name_to_internal(f"{target}.weight")
        assert internal_name not in model.z
    torch.testing.assert_close(
        model.z["blocks.0.att.value.weight"],
        protected_value,
    )
    torch.testing.assert_close(model.z["blocks.0.ffn.x_k"], protected_state)

    process_weights_after_loading(model, model_config, torch.device("cpu"))
    assert process_calls == [model.get_submodule(target) for target in targets]
    for target in targets:
        linear = model.get_submodule(target)
        assert not hasattr(linear, "weight_packed")
        assert id(linear.weight) == parameter_identities[f"{target}.weight_packed"]
        assert id(linear.weight_scale) == parameter_identities[f"{target}.weight_scale"]
        torch.testing.assert_close(
            linear.weight_global_scale,
            torch.tensor(0.5, dtype=torch.float32),
        )
        if candidate == "nvfp4-w4a4":
            torch.testing.assert_close(
                linear.input_global_scale,
                torch.tensor(0.5, dtype=torch.float32),
            )
            torch.testing.assert_close(
                linear.input_global_scale_inv,
                torch.tensor(2, dtype=torch.float32),
            )

    if candidate in {"nvfp4-w4a4", "nvfp4-w4a16"}:
        key = model._apply_quantized_linear(
            "blocks.0.ffn.key",
            torch.ones((2, 64), dtype=torch.float16),
        )
        assert key is not None and key.shape == (2, 128)
        value = model._apply_quantized_linear("blocks.0.ffn.value", key)
        assert value is not None and value.shape == (2, 64)
        assert kernel_calls == [
            model.get_submodule("model.blocks.0.ffn.key"),
            model.get_submodule("model.blocks.0.ffn.value"),
        ]
    else:
        output = model._apply_quantized_linear(
            "blocks.1.att.w1",
            torch.ones((2, 64), dtype=torch.float16),
        )
        assert output is not None and output.shape == (2, 32)
        r, k, v = cast(Any, model).project_att_rkv(
            *(torch.ones((2, 64), dtype=torch.float16) for _ in range(3)),
            "blocks.1.att.",
            2,
        )
        assert r.shape == k.shape == v.shape == (2, 64)
        att_output = model._apply_quantized_linear(
            "blocks.1.att.output",
            torch.ones((2, 64), dtype=torch.float16),
        )
        assert att_output is not None and att_output.shape == (2, 64)
        assert kernel_calls == [
            model.get_submodule("model.blocks.1.att.w1"),
            model.get_submodule("model.blocks.1.att.receptance"),
            model.get_submodule("model.blocks.1.att.key"),
            model.get_submodule("model.blocks.1.att.value"),
            model.get_submodule("model.blocks.1.att.output"),
        ]


def test_default_loader_w4a4_ffn_candidate_uses_real_ct_process_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_nvfp4_consumer_candidate("nvfp4-w4a4", tmp_path, monkeypatch)


def test_default_loader_w4a16_ffn_candidate_uses_real_ct_process_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_nvfp4_consumer_candidate("nvfp4-w4a16", tmp_path, monkeypatch)


def test_default_loader_w4a16_ablation_uses_real_ct_process_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_nvfp4_consumer_candidate(
        "nvfp4-w4a16-protection-ablation", tmp_path, monkeypatch
    )


def _unpack_nvfp4_for_reference(packed: torch.Tensor) -> torch.Tensor:
    low = (packed & 0b10000000) | ((packed & 0b01110000) >> 2)
    low = low.view(torch.float8_e4m3fn).to(torch.float16) * 64
    high_bits = packed << 4
    high = (high_bits & 0b10000000) | ((high_bits & 0b01110000) >> 2)
    high = high.view(torch.float8_e4m3fn).to(torch.float16) * 64
    return torch.cat((high.unsqueeze(2), low.unsqueeze(2)), dim=2).flatten(1)


@pytest.mark.parametrize(
    "candidate",
    ("nvfp4-w4a4", "nvfp4-w4a16", "nvfp4-w4a16-protection-ablation"),
)
def test_default_loader_nvfp4_candidate_reaches_native_gpu_kernel(
    candidate: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_library = os.environ.get("VLLM_RWKV_NVFP4_TEST_LIBRARY")
    if not native_library:
        pytest.skip("VLLM_RWKV_NVFP4_TEST_LIBRARY is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.ops.load_library(native_library)

    import vllm.model_executor.kernels.linear.nvfp4.cutlass as cutlass_module
    import vllm.model_executor.layers.quantization.utils.marlin_utils as marlin_utils
    import vllm.model_executor.parameter as parameter_module
    from vllm.model_executor.kernels.linear.nvfp4.base import (
        NvFp4LinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
        CutlassNvFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.nvfp4.marlin import (
        MarlinNvFp4LinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4,
    )

    use_a16 = candidate != "nvfp4-w4a4"
    kernel_type = MarlinNvFp4LinearKernel if use_a16 else CutlassNvFp4LinearKernel

    def native_kernel(**_kwargs: object) -> object:
        # A source checkout without the full vllm._C package reports an
        # unspecified platform. The reduced extension loaded above owns the
        # actual ops, so construct the production kernel class without only its
        # package-level platform probe; no process/apply method is replaced.
        kernel = object.__new__(kernel_type)
        kernel.config = NvFp4LinearLayerConfig()
        return kernel

    monkeypatch.setattr(
        compressed_tensors_w4a4_nvfp4,
        "init_nvfp4_linear_kernel",
        native_kernel,
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    monkeypatch.setattr(
        marlin_utils,
        "num_compute_units",
        lambda _device_id: properties.multi_processor_count,
    )

    config = _nvfp4_consumer_config(candidate)
    validate_rwkv7_hf_artifact_config(config)
    quant_config = CompressedTensorsConfig.from_config(
        deepcopy(config.quantization_config)
    )
    model: Any = object.__new__(RWKV7ForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.quant_config = quant_config
    model.tp_size = 1
    model._quantized_linears = {}
    with torch.device(device):
        model._initialize_quantized_linears()

    targets = config.rwkv7_quantization_metadata["vllm"]["quantized_modules"]
    artifact_tensors: dict[str, torch.Tensor] = {}
    torch.manual_seed(20260802)
    for target in targets:
        linear = model.get_submodule(target)
        for parameter_name, parameter in linear.named_parameters(recurse=False):
            full_name = f"{target}.{parameter_name}"
            if parameter_name == "weight_packed":
                value = torch.randint(0, 256, parameter.shape, dtype=torch.uint8)
            elif parameter_name == "weight_scale":
                value = (torch.rand(parameter.shape) * 0.25).to(torch.float8_e4m3fn)
            elif parameter_name == "weight_global_scale":
                value = torch.full(parameter.shape, 64.0, dtype=torch.float32)
            elif parameter_name == "input_global_scale":
                value = torch.full(parameter.shape, 2.0, dtype=torch.float32)
            else:  # pragma: no cover - the exact schema is asserted above
                raise AssertionError(parameter_name)
            artifact_tensors[full_name] = value

    expected_shapes = rwkv7_checkpoint_weight_shapes(config)
    quantized_dense_names = {f"{target}.weight" for target in targets}
    for name, shape in expected_shapes.items():
        if name not in quantized_dense_names:
            artifact_tensors[name] = torch.zeros(shape, dtype=torch.float16)

    artifact = tmp_path / candidate
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(vars(config)),
        encoding="utf-8",
    )
    save_file(artifact_tensors, artifact / "model.safetensors")
    setattr(model, "_preprocess_weights", lambda _weights: None)
    setattr(
        model,
        "_commit_preprocessed_weights",
        lambda weights: setattr(model, "z", weights),
    )
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    monkeypatch.setattr(loader, "_init_ep_weight_filter", lambda _config: None)
    model_config: Any = SimpleNamespace(
        hf_config=config,
        quantization="compressed-tensors",
        model=str(artifact),
        revision=None,
        dtype=torch.float16,
    )
    loader.load_weights(model, model_config)

    runtime_target = (
        "model.blocks.1.att.key"
        if candidate == "nvfp4-w4a16-protection-ablation"
        else "model.blocks.0.ffn.key"
    )
    linear = model.get_submodule(runtime_target)
    packed_before = linear.weight_packed.detach().clone()
    scales_before = linear.weight_scale.detach().clone()
    input_value = torch.randn(
        8,
        linear.input_size_per_partition,
        device=device,
        dtype=torch.float16,
    )
    reference = None
    if use_a16:
        reference_weight = _unpack_nvfp4_for_reference(packed_before)
        reference_weight *= scales_before.to(torch.float16).repeat_interleave(16, dim=1)
        reference = input_value @ (reference_weight / 64).t()

    quant_calls: list[dict[str, object]] = []
    real_scaled_fp4_quant = cast(Any, cutlass_module.scaled_fp4_quant)

    def record_real_scaled_fp4_quant(
        input_tensor: torch.Tensor,
        input_global_scale: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed, block_scale = real_scaled_fp4_quant(
            input_tensor,
            input_global_scale,
            **kwargs,
        )
        quant_calls.append(
            {
                "backend": kwargs.get("backend"),
                "swizzled": kwargs.get("is_sf_swizzled_layout"),
                "scale": block_scale.float().clone(),
            }
        )
        return packed, block_scale

    monkeypatch.setattr(
        cutlass_module,
        "scaled_fp4_quant",
        record_real_scaled_fp4_quant,
    )
    process_weights_after_loading(model, model_config, device)
    runtime_name = runtime_target.removeprefix("model.")
    output = model._apply_quantized_linear(runtime_name, input_value)
    assert output is not None
    torch.accelerator.synchronize()
    assert torch.isfinite(output).all()

    assert type(linear.scheme.kernel) is kernel_type
    if use_a16:
        assert not quant_calls
        assert hasattr(linear, "workspace")
        assert reference is not None
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    else:
        second_output = model._apply_quantized_linear(runtime_name, input_value * 0.5)
        assert second_output is not None
        torch.accelerator.synchronize()
        assert len(quant_calls) == 2
        assert all(call["backend"] == "cutlass" for call in quant_calls)
        assert all(call["swizzled"] is True for call in quant_calls)
        assert not torch.equal(quant_calls[0]["scale"], quant_calls[1]["scale"])
        torch.testing.assert_close(
            linear.input_global_scale_inv,
            torch.tensor(2.0, device=device),
        )
        torch.testing.assert_close(
            linear.alpha,
            torch.tensor(0.5 / 64, device=device),
        )


def test_fresh_process_rejects_invalid_rwkv7_consumer_before_file_open(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rwkv7-packed"
    artifact.mkdir()
    config = _quantized_rwkv7_config()
    config.rwkv7_quantization_metadata["vllm"]["vllm_consumer_revision"] = None
    (artifact / "config.json").write_text(
        json.dumps(vars(config)),
        encoding="utf-8",
    )
    (artifact / "model.safetensors").write_bytes(b"not a safetensors file")
    code = r"""
import json
import sys
from types import SimpleNamespace

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.transformers_utils.configs.rwkv7 import RWKV7Config

artifact = sys.argv[1]
with open(f"{artifact}/config.json", encoding="utf-8") as config_file:
    config = RWKV7Config(**json.load(config_file))
loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
try:
    loader.load_weights(
        object(),
        SimpleNamespace(hf_config=config, quantization="compressed-tensors"),
    )
except (RuntimeError, ValueError) as error:
    print(error)
else:
    raise AssertionError("packed RWKV7 artifact unexpectedly reached weight loading")
"""

    completed = subprocess.run(
        [sys.executable, "-c", code, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert "exact candidate consumer capability and revision" in completed.stdout


def test_pth_conversion_is_lossless_and_strictly_reloadable_in_fresh_process(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pth"
    artifact = tmp_path / "model-hf"
    source_weights = _write_legacy_checkpoint(checkpoint)
    source_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_tokenizer = RWKVTokenizer()
    prompt = "User: 你好, RWKV\n\nAssistant:"
    expected_token_ids = source_tokenizer.encode(prompt)

    converted = convert_rwkv7_pth_to_hf_artifact(checkpoint, artifact)

    assert converted == artifact
    assert not list(artifact.glob("*.pth"))
    assert {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "rwkv_vocab_v20230424.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
    }.issubset(path.name for path in artifact.iterdir())
    published = load_file(artifact / "model.safetensors")
    assert len(published) == len(source_weights)
    for internal_name, source_weight in source_weights.items():
        hf_name = rwkv7_internal_to_hf_weight_name(internal_name)
        assert hf_name is not None
        expected_weight = (
            source_weight.transpose(0, 1)
            if rwkv7_is_legacy_low_rank_weight_name(internal_name)
            else source_weight
        )
        torch.testing.assert_close(published[hf_name], expected_weight, rtol=0, atol=0)
    expected_shapes = rwkv7_checkpoint_weight_shapes(_standard_rwkv7_hf_config())
    assert {name: tuple(weight.shape) for name, weight in published.items()} == (
        expected_shapes
    )
    for projection in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
        assert f"model.blocks.1.att.{projection}.weight" in published
        assert f"model.blocks.1.att.{projection}" not in published

    checkpoint.unlink()
    checkpoint.with_name("config.json").unlink()
    code = r"""
import json
import sys

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
from vllm.model_executor.models.rwkv7 import RWKV7ForCausalLM
from vllm.tokenizers.registry import get_tokenizer
from vllm.transformers_utils.config import get_config, try_get_generation_config

artifact = sys.argv[1]
config = get_config(artifact, trust_remote_code=False)
tokenizer = get_tokenizer(artifact, tokenizer_mode="rwkv")
generation_config = try_get_generation_config(artifact, trust_remote_code=False)
loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
_, files, use_safetensors = loader._prepare_weights(
    artifact,
    subfolder=None,
    revision=None,
    fall_back_to_pt=False,
    allow_patterns_overrides=None,
)
model = object.__new__(RWKV7ForCausalLM)
model.config = config
internal, loaded = model._normalize_checkpoint_weights(
    safetensors_weights_iterator(files, use_tqdm_on_load=False)
)
print(json.dumps({
    "architectures": config.architectures,
    "context": config.max_position_embeddings,
    "generation_eos": generation_config.eos_token_id,
    "source_sha256": config.rwkv_source_sha256,
    "token_ids": tokenizer.encode("User: 你好, RWKV\n\nAssistant:"),
    "use_safetensors": use_safetensors,
    "loaded_count": len(loaded),
    "internal_count": len(internal),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(artifact)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    )
    assert completed.returncode == 0, completed.stderr
    reloaded = json.loads(completed.stdout.splitlines()[-1])

    assert reloaded == {
        "architectures": ["Rwkv7ForCausalLM"],
        "context": 321,
        "generation_eos": 0,
        "source_sha256": source_digest,
        "token_ids": expected_token_ids,
        "use_safetensors": True,
        "loaded_count": len(source_weights),
        "internal_count": len(source_weights),
    }


def test_converter_never_overwrites_existing_artifact(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pth"
    _write_legacy_checkpoint(checkpoint)
    artifact = tmp_path / "model-hf"
    artifact.mkdir()

    with pytest.raises(FileExistsError, match="destination already exists"):
        convert_rwkv7_pth_to_hf_artifact(checkpoint, artifact)


def test_rwkv7_config_serializes_standard_architecture(tmp_path: Path) -> None:
    config = RWKV7Config(
        hidden_size=64,
        head_size=64,
        num_hidden_layers=1,
        max_position_embeddings=321,
    )
    config.save_pretrained(tmp_path)
    serialized = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))

    reloaded = get_config(tmp_path, trust_remote_code=False)

    assert serialized["context_length"] == 321
    assert "max_position_embeddings" not in serialized
    assert reloaded.architectures == ["Rwkv7ForCausalLM"]
    assert reloaded.context_length == 321
    assert reloaded.max_position_embeddings == 321
