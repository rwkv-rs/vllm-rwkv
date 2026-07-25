# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.registry import TokenizerRegistry, get_tokenizer


def test_rwkv_tokenizer_matches_world_vocab_golden_ids():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.encode("Hello world", add_special_tokens=False) == [33155, 40213]
    assert tokenizer.encode("你好", add_special_tokens=False) == [10464, 11685]
    assert tokenizer.encode(" 42", add_special_tokens=False) == [3515]
    assert tokenizer.encode("Hello world") == [0, 33155, 40213]
    assert tokenizer.num_special_tokens_to_add() == 1
    assert tokenizer.decode([33155, 40213]) == "Hello world"
    assert tokenizer.decode([0, 33155, 40213]) == "Hello world"
    assert tokenizer.decode([10464, 11685]) == "你好"


def test_rwkv_tokenizer_left_truncation_can_drop_bos_when_budget_is_too_small():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.encode("Hello world", truncation=True, max_length=2) == [
        33155,
        40213,
    ]
    encoded = tokenizer("Hello world", truncation=True, max_length=2)
    assert encoded["input_ids"] == [33155, 40213]
    assert encoded["attention_mask"] == [1, 1]


def test_rwkv_tokenizer_decode_replaces_invalid_utf8_tokens():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.decode([129]) == "\ufffd"
    assert tokenizer.decode([196]) == "\ufffd"
    assert tokenizer.decode([256]) == "\ufffd"
    assert tokenizer.decode([129, 196, 256]) == "\ufffd\ufffd\ufffd"


def test_rwkv_tokenizer_pads_unused_logits_ids():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.vocab_size == 65536
    assert tokenizer.max_token_id == 65535
    assert tokenizer.decode([65530, 65535]) == "\ufffd\ufffd"


def test_rwkv_tokenizer_exposes_lossless_byte_vocab_tokens():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    raw_byte_token = tokenizer.convert_ids_to_tokens([129])[0]
    assert raw_byte_token == "\x80"
    assert tokenizer.convert_tokens_to_ids(raw_byte_token) == 129
    assert tokenizer.get_vocab()["\x80"] == 129
    assert tokenizer.get_vocab()["\x81"] == 130


def test_rwkv_tokenizer_reports_slow_until_offsets_are_supported():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    assert tokenizer.is_fast is False


def test_rwkv_tokenizer_exposes_cached_metadata():
    tokenizer_cls = TokenizerRegistry.load_tokenizer_cls("rwkv")
    tokenizer = tokenizer_cls.from_pretrained("BlinkDL/rwkv7-g1")

    assert tokenizer.name_or_path == "BlinkDL/rwkv7-g1"
    cached_max_chars = tokenizer.max_chars_per_token
    tokenizer.idx2token.append(b"x" * (cached_max_chars + 1))
    assert tokenizer.max_chars_per_token == cached_max_chars


def test_rwkv_chat_template_renders_basic_dialogue_from_training_template():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "\r\n  You are concise.\r\n\r\nNo fluff.  "},
            {"role": "user", "content": "  Hello\r\n\r\nworld  "},
            {"role": "assistant", "content": "  Hi\r\n\r\nthere  "},
            {"role": "user", "content": "  Continue\r\nplease  "},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == (
        "System: You are concise.\nNo fluff.\n\n"
        "User: Hello\nworld\n\n"
        "Assistant: Hi\nthere\n\n"
        "User: Continue\nplease\n\n"
        "Assistant: <think"
    )


def test_rwkv_chat_template_tokenizes_rendered_prompt():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: Hi\nAnswer:"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Question: Hi\nAnswer:"}],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert isinstance(token_ids, list)
    assert token_ids[0] == tokenizer.bos_token_id
    assert tokenizer.decode(token_ids) == rendered
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi"}],
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
        )[0]
        != tokenizer.bos_token_id
    )


def test_rwkv_native_chat_unwraps_only_paired_user_shells(monkeypatch):
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")
    dapo_problem = "How many positive integers are less than 2024?"
    dapo_prompt = (
        "Solve the following math problem step by step. The last line of your "
        "response should be of the form Answer: $Answer (without quotes) where "
        "$Answer is the answer to the problem.\n"
        f"{dapo_problem}\n"
        'Remember to put your answer on its own line after "Answer:".'
    )
    paired = (
        (dapo_prompt, dapo_problem),
        ("Question: 1+1?\nAnswer:", "1+1?"),
        (
            "Intro\nQuestion: Pick.\nA. One\nB. Two\nAnswer:",
            "Intro\nPick.\nA. One\nB. Two",
        ),
        ("Question:\nState the result.\nAnswer:", "State the result."),
    )  # noqa: E501
    unchanged = (
        'Solve this. End with "ANSWER: $ANSWER".',
        "Write at least two words.",
        "plain\nAnswer:",
        "Question: missing answer",
        "Question:foo\nAnswer:",
        "Question: one\nQuestion: two\nAnswer:",
        "Question 1: numbered\nAnswer:",
        "Keep Question: inline\nAnswer:",
        "Question: wrong case\nANSWER:",
        "Question: nonempty\nAnswer: value",
    )  # noqa: E501
    render = lambda content: tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )  # noqa: E501
    for content, expected in paired + tuple((text, text) for text in unchanged):
        assert render(content) == f"User: {expected}\n\nAssistant: <think"
    dialogue = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Question: system\nAnswer:"},
            {"role": "user", "content": paired[1][0]},
            {"role": "assistant", "content": "Question: gold\nAnswer:"},
            {"role": "user", "content": paired[2][0]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )  # noqa: E501
    assert (
        dialogue
        == "System: Question: system\nAnswer:\n\nUser: 1+1?\n\nAssistant: Question: gold\nAnswer:\n\nUser: Intro\nPick.\nA. One\nB. Two\n\nAssistant: <think"  # noqa: E501
    )  # noqa: E501
    tool_chat = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Question: call?\nAnswer:"},
            {"role": "tool", "content": "Question: tool\nAnswer:"},
        ],
        tools=[{"function": {"name": "noop", "parameters": {}}}],
        tokenize=False,
        add_generation_prompt=True,
    )  # noqa: E501
    assert (
        "### User\ncall?\n### Tool Output" in tool_chat
        and '"Question: tool\\nAnswer:"' in tool_chat
    )  # noqa: E501
    seen = {}
    monkeypatch.setattr(
        tokenizer.apply_chat_template.__globals__["hf_chat_utils"],
        "render_jinja_template",
        lambda conversation, **_: (
            seen.update(conversation=conversation) or (["sentinel"], [])
        ),
    )
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": paired[1][0]}],
            chat_template="custom",
            tokenize=False,
        )
        == "sentinel"
        and seen["conversation"][0]["content"] == paired[1][0]
    )  # noqa: E501, E702


def test_rwkv_chat_template_can_render_fake_think_generation_prompt():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        rwkv_generation_prompt="fake_think",
    )

    assert rendered == "User: Hi\n\nAssistant: <think></think"


def test_rwkv_chat_template_preserves_raw_user_content():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")
    problem = "已知 x+y=3, 求 x。"

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert rendered == f"User: {problem}\n\nAssistant: <think"


def test_rwkv_chat_template_renders_tools_and_tool_outputs_from_training_template():
    tokenizer = get_tokenizer("BlinkDL/rwkv7-g1", tokenizer_mode="rwkv")
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
            {"role": "system", "content": "  Use tools carefully.\n\nExplain gaps. "},
            {"role": "user", "content": " Weather in Paris?\r\n\r\nUse Celsius. "},
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
