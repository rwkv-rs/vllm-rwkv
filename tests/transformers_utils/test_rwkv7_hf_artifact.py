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

import pytest
import torch
from safetensors.torch import load_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.rwkv7 import (
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


def _rwkv7_quantization_metadata(
    packed_format: str = "pack-quantized",
) -> dict[str, object]:
    quantized_low_rank = [
        f"model.blocks.0.att.{name}" for name in ("w1", "w2", "a1", "a2", "g1", "g2")
    ]
    protected_v_first = [
        "model.blocks.1.att.v1",
        "model.blocks.1.att.v2",
    ]
    quantized_modules = [
        "model.blocks.0.ffn.key",
        "model.blocks.0.ffn.value",
        *quantized_low_rank,
    ]
    protected_linear = [
        "head",
        "model.blocks.0.att.value",
        *protected_v_first,
    ]
    protected_embedding = ["model.embeddings"]
    protected_normalization = ["model.blocks.0.ln0", "model.ln_out"]
    protected_modules = [
        *protected_linear,
        *protected_embedding,
        *protected_normalization,
    ]
    protected_state = ["model.blocks.0.ffn.x_k"]
    return {
        "schema_version": 3,
        "target_policy": {"selection": {"names": quantized_modules}},
        "vllm": {
            "architecture": "Rwkv7ForCausalLM",
            "model_type": "rwkv7",
            "source_format": "standard_hf",
            "load_format": "safetensors",
            "quant_method": "compressed-tensors",
            "quantization_format": packed_format,
            "legacy_pth_direct_load": False,
            "quantization_target_type": "Linear",
            "linear_weight_suffix": "weight",
            "linear_weight_layout": "out-in",
            "quantized_modules": quantized_modules,
            "quantized_weight_names": [f"{name}.weight" for name in quantized_modules],
            "low_rank_linear_modules": sorted(
                [*quantized_low_rank, *protected_v_first]
            ),
            "quantized_low_rank_modules": quantized_low_rank,
            "protected_v_first_linear_modules": protected_v_first,
            "layer_zero_v_first_producer": "model.blocks.0.att.value",
            "protected_linear_modules": protected_linear,
            "protected_embedding_modules": protected_embedding,
            "protected_normalization_modules": protected_normalization,
            "protected_state_tensors": protected_state,
            "protected_modules": protected_modules,
            "protected_tensors": protected_state,
        },
    }


def _quantized_rwkv7_config(
    packed_format: str = "pack-quantized",
) -> SimpleNamespace:
    return _standard_rwkv7_hf_config(
        rwkv7_quantization_metadata=_rwkv7_quantization_metadata(packed_format),
        quantization_config={
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "format": packed_format,
        },
    )


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


def test_rwkv7_schema_v3_quantization_metadata_is_validated() -> None:
    config = _quantized_rwkv7_config()

    assert validate_rwkv7_quantization_artifact_metadata(config) == ("pack-quantized")
    assert frozenset() == RWKV7_SUPPORTED_PACKED_FORMATS
    validate_rwkv7_hf_artifact_config(config)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("schema", "schema_version"),
        ("identity", "model/loader identity"),
        ("weight_key", r"standard \*\.weight Linear keys"),
        ("category_overlap", "categories must be disjoint"),
        ("category_incomplete", "categories are incomplete"),
        ("target_policy", "target policy disagrees"),
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
    loader = object.__new__(DefaultModelLoader)
    loader.load_config = SimpleNamespace(
        load_format="auto",
        download_dir=None,
        ignore_patterns=None,
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


@pytest.mark.parametrize(
    "packed_format",
    ["pack-quantized", "nvfp4-pack-quantized"],
)
def test_default_loader_rejects_rwkv7_packed_format_before_weight_iterator(
    packed_format: str,
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
    model_config = SimpleNamespace(
        hf_config=_quantized_rwkv7_config(packed_format),
        quantization="compressed-tensors",
    )

    with pytest.raises(RuntimeError, match=packed_format):
        loader.load_weights(SimpleNamespace(), model_config)


def test_fresh_process_rejects_rwkv7_packed_format_before_file_open(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "rwkv7-packed"
    artifact.mkdir()
    config = _quantized_rwkv7_config()
    (artifact / "config.json").write_text(
        json.dumps(vars(config)),
        encoding="utf-8",
    )
    (artifact / "model.safetensors").write_bytes(b"not a safetensors file")
    code = r"""
import sys
from types import SimpleNamespace

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.transformers_utils.config import get_config

artifact = sys.argv[1]
config = get_config(artifact, trust_remote_code=False)
loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
try:
    loader.load_weights(
        object(),
        SimpleNamespace(hf_config=config, quantization="compressed-tensors"),
    )
except RuntimeError as error:
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
    assert "pack-quantized" in completed.stdout
    assert "currently consumes only dense standard HF *.weight tensors" in (
        completed.stdout
    )
    assert "export an unquantized standard HF safetensors artifact" in (
        completed.stdout
    )


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
