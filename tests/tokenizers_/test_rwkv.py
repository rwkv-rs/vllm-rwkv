# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers.registry import TokenizerRegistry, get_tokenizer
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.tokenizers.rwkv_defaults import (
    RWKV_PROMPT_TEMPLATE_ASSISTANT,
    RWKV_PROMPT_TEMPLATE_BOT,
    RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING,
    ensure_rwkv_prompt_bos_token,
    resolve_rwkv_prompt_template,
)


def test_rwkv_tokenizer_matches_world_vocab_golden_ids():
    tokenizer = RWKVTokenizer()

    assert tokenizer.encode("Hello world", add_special_tokens=False) == [33155, 40213]
    assert tokenizer.encode("你好", add_special_tokens=False) == [10464, 11685]
    assert tokenizer.encode(" 42", add_special_tokens=False) == [3515]
    assert tokenizer.encode("Hello world") == [0, 33155, 40213]
    assert tokenizer.num_special_tokens_to_add() == 1
    assert tokenizer.decode([33155, 40213]) == "Hello world"
    assert tokenizer.decode([0, 33155, 40213]) == "Hello world"
    assert tokenizer.decode([10464, 11685]) == "你好"


def test_rwkv_tokenizer_left_truncation_can_drop_bos_when_budget_is_too_small():
    tokenizer = RWKVTokenizer()

    assert tokenizer.encode("Hello world", truncation=True, max_length=2) == [
        33155,
        40213,
    ]
    encoded = tokenizer("Hello world", truncation=True, max_length=2)
    assert encoded["input_ids"] == [33155, 40213]
    assert encoded["attention_mask"] == [1, 1]


def test_rwkv_tokenizer_decode_replaces_invalid_utf8_tokens():
    tokenizer = RWKVTokenizer()

    assert tokenizer.decode([129]) == "\ufffd"
    assert tokenizer.decode([196]) == "\ufffd"
    assert tokenizer.decode([256]) == "\ufffd"
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_pads_unused_logits_ids():
    tokenizer = RWKVTokenizer()

    assert tokenizer.vocab_size == 65536
    assert tokenizer.max_token_id == 65535
    assert tokenizer.decode([65530, 65535]) == "\ufffd\ufffd"


def test_rwkv_tokenizer_exposes_lossless_byte_vocab_tokens():
    tokenizer = RWKVTokenizer()

    raw_byte_token = tokenizer.convert_ids_to_tokens([129])[0]
    assert raw_byte_token == "\x80"
    assert tokenizer.convert_tokens_to_ids(raw_byte_token) == 129
    assert tokenizer.get_vocab()["\x80"] == 129
    assert tokenizer.get_vocab()["\x81"] == 130


def test_rwkv_tokenizer_reports_slow_until_offsets_are_supported():
    tokenizer = RWKVTokenizer()

    assert tokenizer.is_fast is False


def test_rwkv_tokenizer_exposes_cached_metadata(tmp_path):
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")
    RWKVTokenizer().save_pretrained(tmp_path)
    tokenizer = tokenizer_cls.from_pretrained(tmp_path)

    assert tokenizer.name_or_path == str(tmp_path)
    cached_max_chars = tokenizer.max_chars_per_token
    tokenizer.idx2token.append(b"x" * (cached_max_chars + 1))
    assert tokenizer.max_chars_per_token == cached_max_chars


def test_rwkv_tokenizer_save_pretrained_round_trip_uses_artifact_vocab(tmp_path):
    source = RWKVTokenizer()

    saved = source.save_pretrained(tmp_path)
    reloaded = get_tokenizer(tmp_path, tokenizer_mode="rwkv")

    assert {path.name for path in tmp_path.iterdir()} == {
        "rwkv_vocab_v20230424.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
    }
    assert set(saved) == {str(path) for path in tmp_path.iterdir()}
    assert reloaded.name_or_path == str(tmp_path)
    assert reloaded.encode("Hello 你好") == source.encode("Hello 你好")


def test_rwkv_tokenizer_downloads_vocab_from_remote_hf_artifact(
    tmp_path,
    monkeypatch,
):
    source = RWKVTokenizer()
    calls = []

    class FakeHfApi:
        def hf_hub_download(self, **kwargs):
            calls.append(kwargs)
            return source.name_or_path

    monkeypatch.setattr("vllm.tokenizers.rwkv.hf_api", lambda: FakeHfApi())

    tokenizer = RWKVTokenizer.from_pretrained(
        "rwkv-rs/model-hf",
        revision="key-contract",
        download_dir=str(tmp_path),
    )

    assert tokenizer.encode("Hello") == source.encode("Hello")
    assert calls == [
        {
            "repo_id": "rwkv-rs/model-hf",
            "filename": "rwkv_vocab_v20230424.txt",
            "revision": "key-contract",
            "cache_dir": str(tmp_path),
            "token": None,
        }
    ]


def test_rwkv_chat_template_renders_basic_dialogue_from_training_template():
    tokenizer = RWKVTokenizer()

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Continue"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == (
        "System✿You are concise.✿\nUser✿Hello✿\nBot✿Hi✿\nUser✿Continue✿\nBot✿<think"
    )


@pytest.mark.parametrize(
    ("prompt_template", "expected"),
    [
        (RWKV_PROMPT_TEMPLATE_BOT, "User✿Hi✿\nBot✿<think"),
        (
            RWKV_PROMPT_TEMPLATE_ASSISTANT,
            "User: Hi\n\nAssistant: <think",
        ),
    ],
)
def test_rwkv_chat_template_and_stop_style_render_together(
    prompt_template: str,
    expected: str,
):
    tokenizer = RWKVTokenizer()

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        rwkv_prompt_template=prompt_template,
    )

    assert rendered == expected


@pytest.mark.parametrize(
    ("prompt_template", "stop"),
    [
        (RWKV_PROMPT_TEMPLATE_BOT, "✿"),
        (RWKV_PROMPT_TEMPLATE_ASSISTANT, "\nUser:"),
        (RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING, "\n### User"),
    ],
)
def test_rwkv_prompt_templates_own_their_stop_string(
    prompt_template: str,
    stop: str,
):
    assert resolve_rwkv_prompt_template(prompt_template=prompt_template).stop == stop


def test_rwkv_prompt_bos_normalization_preserves_a_truncated_length_budget():
    assert ensure_rwkv_prompt_bos_token([0, 0, 11]) == [0, 11]
    assert ensure_rwkv_prompt_bos_token(
        [11, 12, 13],
        max_length=3,
        truncate_from_left=True,
    ) == [0, 12, 13]


@pytest.mark.parametrize(
    "prompt_template",
    [
        RWKV_PROMPT_TEMPLATE_BOT,
        RWKV_PROMPT_TEMPLATE_ASSISTANT,
        RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING,
    ],
)
def test_rwkv_chat_template_tokenizes_with_exactly_one_leading_bos_eos(
    prompt_template: str,
):
    tokenizer = RWKVTokenizer()

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: Hi\nAnswer:"}],
        tokenize=False,
        add_generation_prompt=True,
        rwkv_prompt_template=prompt_template,
    )
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: Hi\nAnswer:"}],
        tokenize=True,
        add_generation_prompt=True,
        rwkv_prompt_template=prompt_template,
    )

    assert isinstance(token_ids, list)
    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[1] != tokenizer.bos_token_id
    assert tokenizer.decode(token_ids) == rendered
    without_special_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=True,
        add_generation_prompt=True,
        add_special_tokens=False,
        rwkv_prompt_template=prompt_template,
    )
    assert without_special_tokens[0] == tokenizer.bos_token_id
    assert without_special_tokens[1] != tokenizer.bos_token_id


def test_rwkv_native_chat_preserves_dataset_prompt_verbatim():
    tokenizer = RWKVTokenizer()
    dapo_prompt = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n"
        "Question: How many positive integers are less than 2024?\n"
        "Answer:\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": dapo_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == f"User✿{dapo_prompt}✿\nBot✿<think"


def test_rwkv_chat_template_can_render_fake_think_generation_prompt():
    tokenizer = RWKVTokenizer()

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        rwkv_generation_prompt="fake_think",
    )

    assert rendered == "User✿Hi✿\nBot✿<think></think"


def test_rwkv_chat_template_preserves_raw_user_content():
    tokenizer = RWKVTokenizer()
    problem = "已知 x+y=3, 求 x。"

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == f"User✿{problem}✿\nBot✿<think"


def test_rwkv_chat_template_renders_tools_and_tool_outputs_from_training_template():
    tokenizer = RWKVTokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Use tools carefully.\nExplain gaps."},
            {"role": "user", "content": "Weather in Paris?\nUse Celsius."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"temperature": 21, "unit": "celsius"}',
            },
        ],
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == (
        "### System\n"
        "Use tools carefully.\n"
        "Explain gaps.\n"
        "### `get_weather`\n"
        "**Description:** Get the weather for a city.\n"
        "**Parameters:**\n"
        "```json\n"
        "{\n"
        '  "type": "object",\n'
        '  "properties": {\n'
        '    "city": {\n'
        '      "type": "string"\n'
        "    }\n"
        "  },\n"
        '  "required": [\n'
        '    "city"\n'
        "  ]\n"
        "}\n"
        "```\n"
        "To call one of these tools, write exactly this format:\n"
        "**Tool Call:**\n"
        "```json\n"
        '{"name": "tool_name", "arguments": {"key": "value"}}\n'
        "```\n"
        "Do not invent tool call IDs or write tool outputs yourself.\n"
        "### User\n"
        "Weather in Paris?\n"
        "Use Celsius.\n"
        "### Assistant\n"
        "**Tool Call:**\n"
        "```json\n"
        "{\n"
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "city": "Paris"\n'
        "  }\n"
        "}\n"
        "```\n"
        "### Tool Output\n"
        "```json\n"
        "{\n"
        '  "temperature": 21,\n'
        '  "unit": "celsius"\n'
        "}\n"
        "```\n"
        "### Assistant\n"
        "<think"
    )
