# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

import vllm.model_executor.model_loader as model_loader
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.sharded_state_loader import ShardedStateLoader


@register_model_loader("custom_load_format")
class CustomModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig) -> None:
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        pass


def test_register_model_loader():
    load_config = LoadConfig(load_format="custom_load_format")
    assert isinstance(get_model_loader(load_config), CustomModelLoader)


@pytest.mark.parametrize(
    ("load_format", "expected_loader"),
    [
        ("auto", DefaultModelLoader),
        ("hf", DefaultModelLoader),
        ("safetensors", DefaultModelLoader),
        ("sharded_state", ShardedStateLoader),
    ],
)
def test_builtin_model_loader_dispatch(load_format, expected_loader):
    assert isinstance(
        get_model_loader(LoadConfig(load_format=load_format)),
        expected_loader,
    )


def test_builtin_loader_targets_are_public_exports():
    assert set(model_loader._BUILTIN_LOADERS.values()) <= set(
        model_loader._PUBLIC_EXPORTS.values()
    )


@pytest.mark.parametrize("materialize_builtin_first", [False, True])
def test_custom_loader_overrides_builtin(
    monkeypatch,
    materialize_builtin_first,
):
    monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", {})
    if materialize_builtin_first:
        assert isinstance(
            get_model_loader(LoadConfig(load_format="hf")),
            DefaultModelLoader,
        )

    register_model_loader("hf")(CustomModelLoader)

    assert isinstance(
        get_model_loader(LoadConfig(load_format="hf")),
        CustomModelLoader,
    )


def test_invalid_model_loader():
    with pytest.raises(ValueError):

        @register_model_loader("invalid_load_format")
        class InValidModelLoader:
            pass


def test_default_loader_rejects_zero_num_threads():
    # num_threads=0 used to fail late in ThreadPoolExecutor ("max_workers must be > 0").
    with pytest.raises(ValueError, match="num_threads"):
        DefaultModelLoader(
            LoadConfig(
                model_loader_extra_config={
                    "enable_multithread_load": True,
                    "num_threads": 0,
                }
            )
        )


def test_default_loader_rejects_multithread_with_non_lazy_strategy():
    # The multi-thread loader ignores safetensors_load_strategy; reject the
    # combination instead of silently dropping the requested strategy.
    with pytest.raises(ValueError, match="does not support"):
        DefaultModelLoader(
            LoadConfig(
                safetensors_load_strategy="torchao",
                model_loader_extra_config={"enable_multithread_load": True},
            )
        )


def test_default_loader_explicit_safetensors_does_not_misread_pt(tmp_path):
    # Explicit safetensors must not fall back to a .pt and open it as safetensors.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        loader._prepare_weights(
            str(tmp_path),
            None,
            None,
            fall_back_to_pt=True,
            allow_patterns_overrides=None,
        )


def test_default_loader_hf_still_falls_back_to_pt(tmp_path):
    # Control: load_format="hf" still picks up .pt weights via fallback.
    (tmp_path / "model.pt").write_bytes(b"\x00\x00\x00\x00")
    loader = DefaultModelLoader(LoadConfig(load_format="hf"))
    _, files, use_safetensors = loader._prepare_weights(
        str(tmp_path),
        None,
        None,
        fall_back_to_pt=True,
        allow_patterns_overrides=None,
    )
    assert use_safetensors is False
    assert any(f.endswith("model.pt") for f in files)
