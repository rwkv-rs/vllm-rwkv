# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from vllm.transformers_utils.config import get_config, get_hf_image_processor_config
from vllm.transformers_utils.configs.rwkv7 import (
    RWKV7_DEFAULT_MAX_NUM_SEQS,
    RWKV7Config,
    resolve_rwkv7_default_max_num_seqs,
    try_parse_rwkv7_pth_source,
)


@pytest.mark.parametrize(
    ("size", "layers", "hidden_size"),
    [
        ("1.5b", 24, 2048),
        ("2.9b", 32, 2560),
        ("7.2b", 32, 4096),
        ("13.3b", 61, 4096),
    ],
)
@pytest.mark.parametrize("memory_gib", [24, 32, 48, 96])
@pytest.mark.parametrize("wkv_mode", ["fp16", "fp32io16"])
def test_rwkv7_batch_default_matrix(size, layers, hidden_size, memory_gib, wkv_mode):
    expected = RWKV7_DEFAULT_MAX_NUM_SEQS[size][memory_gib][wkv_mode]
    config = RWKV7Config(num_hidden_layers=layers, hidden_size=hidden_size)
    if expected is None:
        with pytest.raises(ValueError, match="unsupported"):
            resolve_rwkv7_default_max_num_seqs(config, memory_gib * (1 << 30), wkv_mode)
    else:
        assert (
            resolve_rwkv7_default_max_num_seqs(config, memory_gib * (1 << 30), wkv_mode)
            == expected
        )


def test_rwkv7_batch_default_requires_explicit_override_for_unknown_inputs():
    with pytest.raises(ValueError, match="model configuration"):
        resolve_rwkv7_default_max_num_seqs(
            RWKV7Config(num_hidden_layers=12, hidden_size=768),
            24 * (1 << 30),
            "fp16",
        )
    with pytest.raises(ValueError, match="GPU memory"):
        resolve_rwkv7_default_max_num_seqs(
            RWKV7Config(num_hidden_layers=24, hidden_size=2048),
            80 * (1 << 30),
            "fp16",
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
    assert try_parse_rwkv7_pth_source(checkpoint.as_uri()).local_path == checkpoint
    assert try_parse_rwkv7_pth_source(f"file://remote{checkpoint}") is None
    assert try_parse_rwkv7_pth_source("https://example.com/model.pth") is None


def test_unknown_rwkv7_pth_filename_fails_closed(tmp_path: Path):
    checkpoint = tmp_path / "rwkv7-g1g-custom-ctx8192.pth"
    checkpoint.touch()

    with pytest.raises(ValueError, match="Unsupported RWKV7 raw .pth checkpoint"):
        get_config(checkpoint, trust_remote_code=False)


def test_rwkv7_pth_url_has_no_image_processor_config():
    config = get_hf_image_processor_config(
        "https://huggingface.co/BlinkDL/rwkv7-g1/blob/main/"
        "rwkv7-g1g-1.5b-20260526-ctx8192.pth",
    )

    assert config == {}
