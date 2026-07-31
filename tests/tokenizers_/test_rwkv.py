# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.resources import files

import pytest
from transformers import BatchEncoding

from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.renderers.rwkv import RwkvRenderer
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
    assert RENDERER_REGISTRY.load_renderer_cls("rwkv") is RwkvRenderer
    assert vocab.is_file()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", [33155, 40213]),
        ("你好", [10464, 11685]),
        (" 42", [3515]),
    ],
)
def test_rwkv_tokenizer_matches_world_vocab_with_automatic_bos(
    tokenizer: RWKVTokenizer,
    text: str,
    expected: list[int],
):
    assert tokenizer.encode(text) == [0, *expected]
    assert tokenizer.encode(text, add_special_tokens=False) == expected
    assert tokenizer.decode(expected) == text
    assert tokenizer.num_special_tokens_to_add() == 1


def test_rwkv_tokenizer_round_trips_arbitrary_bytes(tokenizer: RWKVTokenizer):
    source = bytes(range(256))
    token_ids = tokenizer.encode_bytes(source)

    assert tokenizer.decode_bytes(token_ids) == source


def test_rwkv_tokenizer_replaces_invalid_utf8(tokenizer: RWKVTokenizer):
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_exposes_lossless_byte_tokens(tokenizer: RWKVTokenizer):
    raw_byte_token = tokenizer.convert_ids_to_tokens([129])[0]
    chinese_ids = tokenizer.encode("你好", add_special_tokens=False)

    assert raw_byte_token == "\x80"
    assert tokenizer.convert_tokens_to_ids(raw_byte_token) == 129
    assert tokenizer.get_vocab()[raw_byte_token] == 129
    assert tokenizer.convert_tokens_to_string([raw_byte_token]) == "\ufffd"
    assert (
        tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(chinese_ids))
        == "你好"
    )


def test_rwkv_tokenizer_pads_unused_logits_ids(tokenizer: RWKVTokenizer):
    assert tokenizer.vocab_size == 65536
    assert tokenizer.max_token_id == 65535
    assert tokenizer.decode([65530, 65535]) == "\ufffd\ufffd"


def test_rwkv_tokenizer_batch_and_truncation(tokenizer: RWKVTokenizer):
    encoded = tokenizer(["Hello world", " 42"])

    assert isinstance(encoded, BatchEncoding)
    assert encoded["input_ids"] == [[0, 33155, 40213], [0, 3515]]
    assert encoded["attention_mask"] == [[1, 1, 1], [1, 1]]
    assert tokenizer.encode("Hello world", truncation=True, max_length=2) == [0, 40213]
    assert tokenizer.encode("Hello world", truncation=True, max_length=1) == [0]
    with pytest.raises(ValueError, match="require one token"):
        tokenizer.encode("Hello world", truncation=True, max_length=0)
    assert (
        tokenizer.encode(
            "Hello world",
            truncation=True,
            max_length=0,
            add_special_tokens=False,
        )
        == []
    )
    with pytest.raises(ValueError, match="non-negative"):
        tokenizer.encode("Hello world", truncation=True, max_length=-1)

    right = RWKVTokenizer.from_pretrained(
        "BlinkDL/rwkv7-g1",
        truncation_side="right",
    )
    assert right.encode("Hello world", truncation=True, max_length=1) == [0]
    with pytest.raises(ValueError, match="require one token"):
        right.encode("Hello world", truncation=True, max_length=0)


def test_rwkv_tokenizer_metadata_and_default_chat_template(
    tokenizer: RWKVTokenizer,
):
    assert tokenizer.is_fast is False
    assert tokenizer.name_or_path == "BlinkDL/rwkv7-g1"
    assert tokenizer.bos_token_id == 0
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 0
    assert tokenizer.all_special_ids == [0]

    messages = [{"role": "user", "content": "Hello"}]
    assert tokenizer.apply_chat_template(messages) == "User✿Hello✿"
    assert (
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        == "User✿Hello✿\nBot✿<think"
    )
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    assert token_ids[0] == 0
    assert token_ids[1] != 0


@pytest.mark.parametrize(
    ("prompt_template", "expected"),
    [
        ("\nBot✿", "User✿Hello✿\nBot✿<think"),
        ("\n\nAssistant: ", "User: Hello\n\nAssistant: <think"),
    ],
)
def test_rwkv_native_prompt_styles(
    tokenizer: RWKVTokenizer,
    prompt_template: str,
    expected: str,
):
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            add_generation_prompt=True,
            rwkv_prompt_template=prompt_template,
        )
        == expected
    )


def test_rwkv_native_continue_final_message(tokenizer: RWKVTokenizer):
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Partial answer"},
    ]

    assert (
        tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
        )
        == "User✿Hello✿\nBot✿Partial answer"
    )
    assert (
        tokenizer.apply_chat_template(
            messages,
            continue_final_message=True,
            rwkv_prompt_template="\n\nAssistant: ",
        )
        == "User: Hello\n\nAssistant: Partial answer"
    )
    with pytest.raises(ValueError, match="cannot both be true"):
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            continue_final_message=True,
        )


def test_rwkv_tools_select_function_calling_template(tokenizer: RWKVTokenizer):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Read it"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object"},
                },
            }
        ],
        add_generation_prompt=True,
    )

    assert "### `read`" in prompt
    assert "\n### User\nRead it\n### Assistant\n<think" in prompt


def test_rwkv_function_calling_template_golden(tokenizer: RWKVTokenizer):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Read it"},
            {
                "role": "assistant",
                "content": "I'll read it",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"a.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"content":"hello"}'},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        add_generation_prompt=True,
    )

    assert prompt == "\n".join(
        [
            "### System",
            "You are helpful",
            "### `read`",
            "**Description:** Read a file",
            "**Parameters:**",
            "```json",
            "{",
            '  "type": "object",',
            '  "properties": {',
            '    "path": {',
            '      "type": "string"',
            "    }",
            "  }",
            "}",
            "```",
            "To call one of these tools, write exactly this format:",
            "**Tool Call:**",
            "```json",
            '{"name": "tool_name", "arguments": {"key": "value"}}',
            "```",
            "Do not invent tool call IDs or write tool outputs yourself.",
            "### User",
            "Read it",
            "### Assistant",
            "I'll read it",
            "**Tool Call:**",
            "```json",
            "{",
            '  "name": "read",',
            '  "arguments": {',
            '    "path": "a.txt"',
            "  }",
            "}",
            "```",
            "### Tool Output",
            "```json",
            "{",
            '  "content": "hello"',
            "}",
            "```",
            "### Assistant",
            "<think",
        ]
    )


def test_rwkv_fake_think_generation_prompt(tokenizer: RWKVTokenizer):
    assert tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
        rwkv_generation_prompt="fake_think",
    ).endswith("Bot✿<think></think")


def test_rwkv_chat_ignores_non_text_structured_content(tokenizer: RWKVTokenizer):
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                    {"text": "using words"},
                ],
            }
        ]
    )

    assert prompt == "User✿Describe this\nusing words✿"
    assert "image_url" not in prompt


def test_rwkv_custom_jinja_template_remains_supported(tokenizer: RWKVTokenizer):
    messages = [{"role": "user", "content": "Hello"}]
    template = "{{ messages[0]['content'] }}!"

    assert (
        tokenizer.apply_chat_template(
            messages,
            chat_template=template,
        )
        == "Hello!"
    )
    assert tokenizer.apply_chat_template(
        messages,
        chat_template=template,
        tokenize=True,
    ) == tokenizer.encode("Hello!")
    assert (
        tokenizer.apply_chat_template(
            messages,
            chat_template=template,
            tokenize=True,
        ).count(0)
        == 1
    )
