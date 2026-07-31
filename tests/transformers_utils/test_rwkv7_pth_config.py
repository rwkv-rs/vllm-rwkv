# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.rwkv7 import (
    RWKV7Config,
    try_parse_rwkv7_pth_source,
)


def test_rwkv7_pth_url_builds_config_from_blinkdl_filename():
    config = get_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
        trust_remote_code=False,
    )

    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.model_type == "rwkv7"
    assert config.hidden_size == 2048
    assert config.intermediate_size == 8192
    assert config.num_hidden_layers == 24
    assert config.head_size == 64
    assert config.vocab_size == 65536
    assert config.max_position_embeddings == 8192
    assert config.decay_rank == 96
    assert config.in_context_learning_rank == 96
    assert config.value_residual_rank == 64
    assert config.gate_rank == 256
    assert config.head_dtype == "float32"


def test_rwkv7_head_dtype_can_follow_model_dtype():
    config = RWKV7Config(head_dtype="model")

    assert config.head_dtype == "model"


def test_rwkv7_local_pth_builds_config_from_filename(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-2.9b-20260526-ctx8192.pth"
    checkpoint.touch()

    for model in (checkpoint, checkpoint.as_uri()):
        config = get_config(model, trust_remote_code=False)
        assert (config.hidden_size, config.num_hidden_layers) == (2560, 32)
        assert config.intermediate_size == 10240
        assert (
            config.decay_rank,
            config.in_context_learning_rank,
            config.value_residual_rank,
            config.gate_rank,
        ) == (96, 96, 64, 320)


def test_rwkv7_catalog_matches_audited_checkpoint_shapes():
    fixture_path = (
        Path(__file__).parent / "fixtures" / "rwkv7_g1_checkpoint_shapes.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["repo_id"] == "BlinkDL/rwkv7-g1"
    assert fixture["torch_version"] == "2.11.0+cu130"
    checkpoints = fixture["checkpoints"]

    for checkpoint in checkpoints:
        assert len(checkpoint["revision"]) == 40
        assert checkpoint["lfs_oid"] == f"sha256:{checkpoint['sha256']}"
        assert checkpoint["etag"] == checkpoint["sha256"]
        assert checkpoint["size"] > 0
        config = get_config(
            "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
            f"{checkpoint['filename']}",
            trust_remote_code=False,
        )
        embedding_shape = checkpoint["emb.weight"]
        decay_shape = checkpoint["blocks.0.att.w1"]
        learning_shape = checkpoint["blocks.0.att.a1"]
        value_shape = checkpoint["blocks.0.att.v1"]
        gate_shape = checkpoint["blocks.0.att.g1"]
        ffn_shape = checkpoint["blocks.0.ffn.key.weight"]

        assert config.vocab_size == embedding_shape[0]
        assert config.hidden_size == embedding_shape[1]
        assert config.num_hidden_layers == checkpoint["num_hidden_layers"]
        assert decay_shape[0] == config.hidden_size
        assert learning_shape[0] == config.hidden_size
        assert value_shape[0] == config.hidden_size
        assert gate_shape[0] == config.hidden_size
        assert config.decay_rank == decay_shape[1]
        assert config.in_context_learning_rank == learning_shape[1]
        assert config.value_residual_rank == value_shape[1]
        assert config.gate_rank == gate_shape[1]
        assert config.intermediate_size == ffn_shape[0]
        assert ffn_shape[1] == config.hidden_size


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


@pytest.mark.parametrize(
    "filename",
    [
        "rwkv7-g1z-1.5b-20260526-ctx8192.pth",
        "rwkv7-g1g-1.5b-20990101-ctx8192.pth",
    ],
)
def test_unaudited_rwkv7_suffix_or_date_fails_closed(tmp_path: Path, filename: str):
    checkpoint = tmp_path / filename
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
                "intermediate_size": 8192,
                "head_size": 64,
                "num_hidden_layers": 24,
                "max_position_embeddings": 10240,
                "decay_rank": 96,
                "in_context_learning_rank": 96,
                "value_residual_rank": 64,
                "gate_rank": 256,
            }
        ),
        encoding="utf-8",
    )

    config = get_config(checkpoint, trust_remote_code=False)

    assert config.architectures == ["RWKV7ForCausalLM"]
    assert config.model_type == "rwkv7"
    assert config.vocab_size == 65536
    assert config.hidden_size == 2048
    assert config.intermediate_size == 8192
    assert config.head_size == 64
    assert config.num_hidden_layers == 24
    assert config.max_position_embeddings == 10240
    assert config.decay_rank == 96
    assert config.in_context_learning_rank == 96
    assert config.value_residual_rank == 64
    assert config.gate_rank == 256
    assert config.head_dtype == "float32"


def test_official_checkpoint_legacy_sidecar_uses_audited_shape_catalog(
    tmp_path: Path,
):
    checkpoint = tmp_path / "rwkv7-g1h-1.5b-20260710-ctx10240.pth"
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

    assert config.intermediate_size == 8192
    assert (
        config.decay_rank,
        config.in_context_learning_rank,
        config.value_residual_rank,
        config.gate_rank,
    ) == (96, 96, 64, 256)


def test_official_checkpoint_sidecar_conflict_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1h-1.5b-20260710-ctx10240.pth"
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
                "decay_rank": 128,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="official checkpoint fields do not match the catalog: decay_rank",
    ):
        get_config(checkpoint, trust_remote_code=False)


def test_custom_checkpoint_incomplete_sidecar_lists_missing_fields(tmp_path: Path):
    checkpoint = tmp_path / "custom-rwkv7.pth"
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

    with pytest.raises(
        ValueError,
        match=(
            "missing required fields: intermediate_size, decay_rank, "
            "in_context_learning_rank, value_residual_rank, gate_rank"
        ),
    ):
        get_config(checkpoint, trust_remote_code=False)


def test_rwkv7_config_pretrained_config_round_trip():
    config = get_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-13.3b-20260523-ctx8192.pth",
        trust_remote_code=False,
    )

    restored = RWKV7Config.from_dict(config.to_dict())

    assert restored.to_dict() == config.to_dict()


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
            "intermediate_size": 8192,
            "head_size": 64,
            "num_hidden_layers": 24,
            "max_position_embeddings": 10240,
            "decay_rank": 96,
            "in_context_learning_rank": 96,
            "value_residual_rank": 64,
            "gate_rank": 256,
        },
        {
            "architectures": ["RWKV7ForCausalLM"],
            "model_type": "rwkv7",
            "vocab_size": 65536,
            "hidden_size": 2048,
            "intermediate_size": 8192,
            "head_size": 32,
            "num_hidden_layers": 24,
            "max_position_embeddings": 10240,
            "decay_rank": 96,
            "in_context_learning_rank": 96,
            "value_residual_rank": 64,
            "gate_rank": 256,
        },
    ],
)
def test_invalid_rwkv7_sidecar_config_fails_closed(tmp_path: Path, sidecar):
    checkpoint = tmp_path / "rwkv_lm.pth"
    checkpoint.touch()
    (tmp_path / "config.json").write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid RWKV7 checkpoint config"):
        get_config(checkpoint, trust_remote_code=False)
