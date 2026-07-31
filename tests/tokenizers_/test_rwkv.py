# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.resources import files

import pytest
from transformers import BatchEncoding

from vllm.renderers.hf import HfRenderer
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry, get_tokenizer
from vllm.tokenizers.rwkv import RWKVTokenizer


@pytest.fixture(scope="module")
def tokenizer() -> RWKVTokenizer:
    loaded = TokenizerRegistry.load_tokenizer("rwkv", "BlinkDL/rwkv7-g1")
    assert isinstance(loaded, RWKVTokenizer)
    return loaded


def test_rwkv_tokenizer_registry_and_packaged_vocab():
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")
    loaded = get_tokenizer(
        files("vllm.tokenizers"),
        tokenizer_mode="rwkv",
    )
    vocab = files("vllm.tokenizers").joinpath("assets", "rwkv_vocab_v20230424.txt")

    assert tokenizer_cls is RWKVTokenizer
    assert isinstance(loaded, RWKVTokenizer)
    assert RENDERER_REGISTRY.load_renderer_cls("rwkv") is HfRenderer
    assert vocab.is_file()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", [33155, 40213]),
        ("你好", [10464, 11685]),
        (" 42", [3515]),
    ],
)
def test_rwkv_tokenizer_matches_world_vocab_without_automatic_bos(
    tokenizer: RWKVTokenizer,
    text: str,
    expected: list[int],
):
    assert tokenizer.encode(text) == expected
    assert tokenizer.encode(text, add_special_tokens=False) == expected
    assert tokenizer.decode(expected) == text
    assert tokenizer.num_special_tokens_to_add() == 0


def test_rwkv_tokenizer_round_trips_arbitrary_bytes(tokenizer: RWKVTokenizer):
    source = bytes(range(256))
    token_ids = tokenizer.encode_bytes(source)

    assert tokenizer.decode_bytes(token_ids) == source


def test_rwkv_tokenizer_replaces_invalid_utf8(tokenizer: RWKVTokenizer):
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_exposes_lossless_byte_tokens(tokenizer: RWKVTokenizer):
    raw_byte_token = tokenizer.convert_ids_to_tokens([129])[0]
    chinese_ids = tokenizer.encode("你好")

    assert raw_byte_token == "\x80"
    assert tokenizer.convert_tokens_to_ids(raw_byte_token) == 129
    assert tokenizer.get_vocab()[raw_byte_token] == 129
    assert tokenizer.convert_tokens_to_string([raw_byte_token]) == "\ufffd"
    assert (
        tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(chinese_ids)
        )
        == "你好"
    )


def test_rwkv_tokenizer_pads_unused_logits_ids(tokenizer: RWKVTokenizer):
    assert tokenizer.vocab_size == 65536
    assert tokenizer.max_token_id == 65535
    assert tokenizer.decode([65530, 65535]) == "\ufffd\ufffd"


def test_rwkv_tokenizer_batch_and_truncation(tokenizer: RWKVTokenizer):
    encoded = tokenizer(["Hello world", " 42"])

    assert isinstance(encoded, BatchEncoding)
    assert encoded["input_ids"] == [[33155, 40213], [3515]]
    assert encoded["attention_mask"] == [[1, 1], [1]]
    assert tokenizer.encode("Hello world", truncation=True, max_length=1) == [40213]
    assert tokenizer.encode("Hello world", truncation=True, max_length=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        tokenizer.encode("Hello world", truncation=True, max_length=-1)

    right = RWKVTokenizer.from_pretrained(
        "BlinkDL/rwkv7-g1",
        truncation_side="right",
    )
    assert right.encode("Hello world", truncation=True, max_length=1) == [33155]
    assert right.encode("Hello world", truncation=True, max_length=0) == []


def test_rwkv_tokenizer_metadata_and_chat_boundary(tokenizer: RWKVTokenizer):
    assert tokenizer.is_fast is False
    assert tokenizer.name_or_path == "BlinkDL/rwkv7-g1"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert tokenizer.all_special_ids == [0]

    with pytest.raises(
        NotImplementedError,
        match="does not define a chat template",
    ):
        tokenizer.apply_chat_template([{"role": "user", "content": "Hello"}])
