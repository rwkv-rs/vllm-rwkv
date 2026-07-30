# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.rwkv7 import try_parse_rwkv7_pth_source


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


def test_rwkv7_source_parser_rejects_untrusted_locations(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()

    source = try_parse_rwkv7_pth_source(checkpoint.as_uri())
    assert source is not None
    assert source.local_path == checkpoint
    assert try_parse_rwkv7_pth_source(f"file://remote{checkpoint}") is None
    assert try_parse_rwkv7_pth_source("https://example.com/model.pth") is None
    assert (
        try_parse_rwkv7_pth_source(
            "https://huggingface.co/other/repo/blob/main/model.pth"
        )
        is None
    )


def test_unknown_rwkv7_pth_filename_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "rwkv_lm.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="Unsupported RWKV7 raw .pth checkpoint"):
        get_config(
            checkpoint,
            trust_remote_code=False,
            config_format="rwkv_pth",
        )


def test_non_rwkv_pth_is_not_claimed_by_auto_detection(tmp_path: Path):
    checkpoint = tmp_path / "other_model.pth"
    checkpoint.touch()

    with pytest.raises(ValueError) as exc_info:
        get_config(checkpoint, trust_remote_code=False)

    assert "Unsupported RWKV7 raw .pth checkpoint" not in str(exc_info.value)


def test_local_rwkv7_checkpoint_uses_valid_sidecar_config(tmp_path: Path):
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


def test_rwkv7_config_uses_common_hf_overrides(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-1.5b-20260526-ctx8192.pth"
    checkpoint.touch()

    config = get_config(
        checkpoint,
        trust_remote_code=False,
        hf_overrides_kw={"max_position_embeddings": 4096},
    )

    assert config.max_position_embeddings == 4096


@pytest.mark.parametrize(
    "sidecar",
    [
        {"model_type": "rwkv7"},
        {
            "architectures": ["RWKV7ForCausalLM"],
            "model_type": "rwkv7",
            "vocab_size": 65536,
            "hidden_size": 2049,
            "head_size": 64,
            "num_hidden_layers": 24,
            "max_position_embeddings": 10240,
        },
    ],
)
def test_invalid_rwkv7_sidecar_config_fails_closed(tmp_path: Path, sidecar):
    checkpoint = tmp_path / "rwkv_lm.pth"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid RWKV7 checkpoint config"):
        get_config(checkpoint, trust_remote_code=False)
