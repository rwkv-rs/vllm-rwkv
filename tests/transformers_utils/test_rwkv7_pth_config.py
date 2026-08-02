# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.transformers_utils.config import get_config, get_hf_image_processor_config
from vllm.transformers_utils.configs import rwkv7
from vllm.transformers_utils.configs.rwkv7 import (
    try_parse_rwkv7_pth_source,
    validate_rwkv7_hf_artifact_config,
)


def _standard_rwkv7_hf_config(**overrides: object) -> SimpleNamespace:
    values = {
        "architectures": ["Rwkv7ForCausalLM"],
        "model_type": "rwkv7",
        "vocab_size": 65536,
        "hidden_size": 2048,
        "head_size": 64,
        "num_hidden_layers": 24,
        "max_position_embeddings": 8192,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_standard_rwkv7_hf_artifact_config_is_accepted() -> None:
    validate_rwkv7_hf_artifact_config(_standard_rwkv7_hf_config())


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
        {"max_position_embeddings": None},
    ],
)
def test_invalid_rwkv7_hf_artifact_config_fails_closed(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Invalid RWKV7 Hugging Face artifact"):
        validate_rwkv7_hf_artifact_config(_standard_rwkv7_hf_config(**overrides))


def test_rwkv7_pth_url_builds_config_from_blinkdl_filename():
    config = get_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
        trust_remote_code=False,
    )

    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.model_type == "rwkv7"
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24
    assert config.head_size == 64
    assert config.vocab_size == 65536
    assert config.max_position_embeddings == 8192


def test_rwkv7_local_pth_builds_config_from_filename(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-2.9b-20260526-ctx8192.pth"
    checkpoint.touch()

    for model in (checkpoint, checkpoint.as_uri()):
        config = get_config(model, trust_remote_code=False)
        assert (config.hidden_size, config.num_hidden_layers) == (2560, 32)
    source = try_parse_rwkv7_pth_source(checkpoint.as_uri())
    assert source is not None
    assert source.local_path == checkpoint
    assert try_parse_rwkv7_pth_source(f"file://remote{checkpoint}") is None
    assert try_parse_rwkv7_pth_source("https://example.com/model.pth") is None


def test_unknown_rwkv7_pth_filename_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "rwkv_lm.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="Unsupported RWKV7 raw .pth checkpoint"):
        get_config(checkpoint, trust_remote_code=False)


def test_local_rwkv7_checkpoint_uses_sidecar_config(tmp_path: Path):
    checkpoint = tmp_path / "rwkv_lm.pth"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["RWKV7ForCausalLM"],
                "model_type": "rwkv7",
                "vocab_size": 65536,
                "hidden_size": 2048,
                "head_size": 64,
                "num_hidden_layers": 24,
                "max_position_embeddings": 10240,
            }
        ),
        encoding="utf-8",
    )

    config = get_config(checkpoint, trust_remote_code=False)

    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.model_type == "rwkv7"
    assert config.vocab_size == 65536
    assert config.hidden_size == 2048
    assert config.head_size == 64
    assert config.num_hidden_layers == 24
    assert config.max_position_embeddings == 10240


def test_rwkv7_pth_url_has_no_image_processor_config():
    config = get_hf_image_processor_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
    )

    assert config == {}


def test_rwkv7_pth_download_uses_shared_hf_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    calls: list[dict[str, object]] = []

    class FakeHfApi:
        def hf_hub_download(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return str(checkpoint)

    monkeypatch.setattr(rwkv7, "hf_api", lambda: FakeHfApi())
    source = rwkv7.RWKV7PthSource(
        filename="weights/model.pth",
        repo_id="BlinkDL/rwkv7-g1",
        revision="main",
    )

    resolved = rwkv7.download_rwkv7_pth_source(
        source,
        cache_dir="/cache",
        revision="release",
    )

    assert resolved == checkpoint
    assert calls == [
        {
            "repo_id": "BlinkDL/rwkv7-g1",
            "filename": "weights/model.pth",
            "revision": "release",
            "cache_dir": "/cache",
        }
    ]
