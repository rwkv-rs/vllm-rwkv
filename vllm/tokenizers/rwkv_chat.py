# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

RWKV_NATIVE_CHAT_TEMPLATE = "{# RWKV native chat template #}"
RWKV_BOS_EOS_TOKEN_ID = 0
RWKV_PROMPT_TEMPLATE_BOT = "\nBot✿"
RWKV_PROMPT_TEMPLATE_ASSISTANT = "\n\nAssistant: "
RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING = "\n### Assistant"
RWKV_DEFAULT_PROMPT_TEMPLATE = RWKV_PROMPT_TEMPLATE_BOT
RWKV_GENERATION_PROMPT_OPEN_THINK = "open_think"
RWKV_GENERATION_PROMPT_FAKE_THINK = "fake_think"
RWKV_GENERATION_PROMPT_MODES = (
    RWKV_GENERATION_PROMPT_OPEN_THINK,
    RWKV_GENERATION_PROMPT_FAKE_THINK,
)


@dataclass(frozen=True)
class RWKVPromptTemplate:
    name: str
    style: Literal["bot", "assistant", "function_calling"]
    stop: str


RWKV_PROMPT_TEMPLATES = {
    RWKV_PROMPT_TEMPLATE_BOT: RWKVPromptTemplate(RWKV_PROMPT_TEMPLATE_BOT, "bot", "✿"),
    RWKV_PROMPT_TEMPLATE_ASSISTANT: RWKVPromptTemplate(
        RWKV_PROMPT_TEMPLATE_ASSISTANT, "assistant", "\nUser:"
    ),
    RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING: RWKVPromptTemplate(
        RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING, "function_calling", "\n### User"
    ),
}


def resolve_rwkv_prompt_template(
    *,
    prompt_template: str | None = None,
    messages: Sequence[Any] = (),
    tools: Sequence[dict[str, Any]] = (),
) -> RWKVPromptTemplate:
    has_tool_history = any(
        _field(message, "role", "") == "tool"
        or bool(_field(message, "tool_calls", None))
        for message in messages
    )
    name = (
        RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING
        if tools or has_tool_history
        else prompt_template or RWKV_DEFAULT_PROMPT_TEMPLATE
    )
    try:
        return RWKV_PROMPT_TEMPLATES[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported RWKV prompt template: {name!r}. "
            f"Expected one of {tuple(RWKV_PROMPT_TEMPLATES)!r}."
        ) from error


def ensure_rwkv_prompt_bos(
    token_ids: Sequence[int],
    *,
    max_length: int | None = None,
    truncate_from_left: bool = False,
) -> list[int]:
    first_non_bos = 0
    while (
        first_non_bos < len(token_ids)
        and token_ids[first_non_bos] == RWKV_BOS_EOS_TOKEN_ID
    ):
        first_non_bos += 1
    normalized = [RWKV_BOS_EOS_TOKEN_ID, *token_ids[first_non_bos:]]
    if max_length is None or len(normalized) <= max_length:
        return normalized
    if max_length < 1:
        raise ValueError("RWKV prompts require one token for BOS/EOS token 0.")
    if truncate_from_left:
        return (
            [RWKV_BOS_EOS_TOKEN_ID, *normalized[-(max_length - 1) :]]
            if (max_length > 1)
            else [RWKV_BOS_EOS_TOKEN_ID]
        )
    raise ValueError(
        f"RWKV prompt with required BOS/EOS has {len(normalized)} tokens, "
        f"exceeding max_length={max_length}."
    )


def render_rwkv_chat(
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool,
    continue_final_message: bool = False,
    generation_prompt: str = RWKV_GENERATION_PROMPT_OPEN_THINK,
    prompt_template: str | None = None,
) -> str:
    if add_generation_prompt and continue_final_message:
        raise ValueError(
            "continue_final_message and add_generation_prompt cannot both be true."
        )
    if generation_prompt not in RWKV_GENERATION_PROMPT_MODES:
        raise ValueError(
            f"Unsupported RWKV generation prompt mode: {generation_prompt!r}. "
            f"Expected one of {RWKV_GENERATION_PROMPT_MODES!r}."
        )
    template = resolve_rwkv_prompt_template(
        prompt_template=prompt_template,
        messages=messages,
        tools=tools or (),
    )
    if template.style == "function_calling":
        return _render_tool_chat(
            messages,
            tools or [],
            add_generation_prompt=add_generation_prompt,
            generation_prompt=generation_prompt,
        )
    rendered = _render_plain_chat(
        messages,
        template,
        add_generation_prompt=add_generation_prompt,
        generation_prompt=generation_prompt,
    )
    if continue_final_message and template.style == "bot":
        return rendered.removesuffix("✿")
    return rendered


def _generation_prompt(mode: str) -> str:
    return "<think" if mode == RWKV_GENERATION_PROMPT_OPEN_THINK else "<think></think"


def _render_plain_chat(
    messages: list[Any],
    template: RWKVPromptTemplate,
    *,
    add_generation_prompt: bool,
    generation_prompt: str,
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = _field(message, "role", "")
        content = stringify_rwkv_content(_field(message, "content", ""))
        if role == "system":
            label = "System"
        elif role == "user":
            label = "User"
        elif role == "assistant":
            label = "Bot" if template.style == "bot" else "Assistant"
        else:
            raise ValueError(f"Unsupported RWKV chat message role: {role!r}")
        if template.style == "bot":
            rendered.append(f"{label}✿{content}✿")
        else:
            rendered.append(f"{label}: {content}" if content else f"{label}:")

    if add_generation_prompt:
        label = "Bot" if template.style == "bot" else "Assistant"
        separator = "✿" if template.style == "bot" else ": "
        rendered.append(f"{label}{separator}{_generation_prompt(generation_prompt)}")
    return ("\n" if template.style == "bot" else "\n\n").join(rendered)


def _render_tool_chat(
    messages: list[Any],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    generation_prompt: str,
) -> str:
    lines: list[str] = []
    pending_system: list[str] = []
    for message in messages:
        role = _field(message, "role", "")
        content = stringify_rwkv_content(_field(message, "content", ""))
        if role == "system":
            pending_system.append(content)
            continue

        if pending_system or (tools and not lines):
            lines.append("### System")
            lines.extend(item for item in pending_system if item)
            lines.extend(_render_tool_definitions(tools))
            pending_system.clear()

        if role == "user":
            lines.extend(["### User", content])
        elif role == "assistant":
            lines.append("### Assistant")
            if content:
                lines.append(content)
            for tool_call in _field(message, "tool_calls", []) or []:
                function = _tool_function(tool_call)
                payload = {
                    "name": _field(function, "name", ""),
                    "arguments": _json_value(_field(function, "arguments", {}) or {}),
                }
                lines.extend(["**Tool Call:**", "```json", _json_text(payload), "```"])
        elif role == "tool":
            lines.extend(
                ["### Tool Output", "```json", _json_text(_json_value(content)), "```"]
            )
        else:
            raise ValueError(f"Unsupported RWKV chat message role: {role!r}")

    if pending_system:
        lines.append("### System")
        lines.extend(item for item in pending_system if item)
        lines.extend(_render_tool_definitions(tools))
    if add_generation_prompt:
        lines.extend(["### Assistant", _generation_prompt(generation_prompt)])
    return "\n".join(lines)


def _render_tool_definitions(tools: list[dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for tool in tools:
        function = _tool_function(tool)
        rendered.append(f"### `{_field(function, 'name', '')}`")
        if description := _field(function, "description", ""):
            rendered.append(f"**Description:** {stringify_rwkv_content(description)}")
        rendered.extend(
            [
                "**Parameters:**",
                "```json",
                _json_text(_field(function, "parameters", {}) or {}),
                "```",
            ]
        )
    if rendered:
        rendered.extend(
            [
                "To call one of these tools, write exactly this format:",
                "**Tool Call:**",
                "```json",
                '{"name": "tool_name", "arguments": {"key": "value"}}',
                "```",
                "Do not invent tool call IDs or write tool outputs yourself.",
            ]
        )
    return rendered


def stringify_rwkv_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, indent=2)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _tool_function(tool: Any) -> Any:
    return tool.get("function", tool) if isinstance(tool, dict) else tool.function


def _field(value: Any, name: str, default: Any = None) -> Any:
    return (
        value.get(name, default)
        if isinstance(value, dict)
        else getattr(value, name, default)
    )
