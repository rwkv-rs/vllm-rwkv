# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import torch

from vllm.config import LoadConfig
from vllm.model_executor.model_loader import get_model_loader, rwkv_pth_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.rwkv_pth_loader import RWKV7PthModelLoader

_CHECKPOINT_NAME = "rwkv7-g1h-1.5b-20260710-ctx10240.pth"


def test_rwkv7_pth_loader_registry_resolves_builtin() -> None:
    loader = get_model_loader(
        LoadConfig(load_format="rwkv_pth", use_tqdm_on_load=False)
    )

    assert isinstance(loader, RWKV7PthModelLoader)


def test_rwkv7_pth_loader_prepares_local_single_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / _CHECKPOINT_NAME
    checkpoint.touch()
    loader = RWKV7PthModelLoader(LoadConfig(load_format="rwkv_pth"))

    folder, files, use_safetensors = loader._prepare_weights(
        str(checkpoint),
        subfolder=None,
        revision=None,
        fall_back_to_pt=False,
        allow_patterns_overrides=None,
    )

    assert folder == str(tmp_path)
    assert files == [str(checkpoint)]
    assert not use_safetensors


def test_rwkv7_pth_loader_downloads_trusted_hf_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / _CHECKPOINT_NAME
    checkpoint.touch()
    loader = RWKV7PthModelLoader(LoadConfig(load_format="rwkv_pth"))
    monkeypatch.setattr(
        rwkv_pth_loader,
        "download_rwkv7_pth_source",
        lambda source, **kwargs: checkpoint,
    )

    _, files, _ = loader._prepare_weights(
        f"https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/{_CHECKPOINT_NAME}",
        subfolder=None,
        revision=None,
        fall_back_to_pt=False,
        allow_patterns_overrides=None,
    )

    assert files == [str(checkpoint)]


def test_rwkv7_pth_loader_rejects_untrusted_source() -> None:
    loader = RWKV7PthModelLoader(LoadConfig(load_format="rwkv_pth"))

    with pytest.raises(ValueError, match="requires a supported RWKV-7"):
        loader._prepare_weights(
            "https://example.com/model.pth",
            subfolder=None,
            revision=None,
            fall_back_to_pt=False,
            allow_patterns_overrides=None,
        )


def test_rwkv7_pth_loader_reads_raw_state_dict(tmp_path: Path) -> None:
    checkpoint = tmp_path / _CHECKPOINT_NAME
    expected = torch.arange(4, dtype=torch.float32)
    torch.save({"weight": expected}, checkpoint)
    loader = RWKV7PthModelLoader(
        LoadConfig(load_format="rwkv_pth", use_tqdm_on_load=False)
    )

    weights = dict(
        loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_or_path=str(checkpoint),
                revision=None,
            )
        )
    )

    torch.testing.assert_close(weights["weight"], expected)
