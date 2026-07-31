# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec

from vllm.transformers_utils.configs.rwkv7 import build_rwkv7_config_from_pth

RWKV_NATIVE_CHAT_TEMPLATE = "{# RWKV native chat template #}"
RWKV_TOOL_CALL_PARSER = "rwkv"
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
class RWKVPromptTemplateSpec:
    name: str
    style: Literal["bot", "assistant", "function_calling"]
    stop: str


RWKV_PROMPT_TEMPLATES = {
    RWKV_PROMPT_TEMPLATE_BOT: RWKVPromptTemplateSpec(
        name=RWKV_PROMPT_TEMPLATE_BOT,
        style="bot",
        stop="✿",
    ),
    RWKV_PROMPT_TEMPLATE_ASSISTANT: RWKVPromptTemplateSpec(
        name=RWKV_PROMPT_TEMPLATE_ASSISTANT,
        style="assistant",
        stop="\nUser:",
    ),
    RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING: RWKVPromptTemplateSpec(
        name=RWKV_PROMPT_TEMPLATE_FUNCTION_CALLING,
        style="function_calling",
        stop="\n### User",
    ),
}


def resolve_rwkv_prompt_template(
    *,
    prompt_template: str | None = None,
    messages: Sequence[Any] = (),
    tools: Sequence[Any] = (),
) -> RWKVPromptTemplateSpec:
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


def stringify_rwkv_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
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


def ensure_rwkv_prompt_bos_token(
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
        raise ValueError("RWKV prompts require at least one token for BOS/EOS token 0.")
    if truncate_from_left:
        tail_length = max_length - 1
        return (
            [RWKV_BOS_EOS_TOKEN_ID]
            if tail_length == 0
            else [RWKV_BOS_EOS_TOKEN_ID, *normalized[-tail_length:]]
        )
    raise ValueError(
        f"RWKV prompt with required BOS/EOS has {len(normalized)} tokens, "
        f"exceeding max_length={max_length}; prompt insertion never truncates."
    )


def render_rwkv_chat_template(
    messages: list[Any],
    tools: list[dict[str, Any]] | None = None,
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str = RWKV_GENERATION_PROMPT_OPEN_THINK,
    rwkv_prompt_template: str | None = None,
) -> str:
    _check_generation_prompt(rwkv_generation_prompt)
    prompt_template = resolve_rwkv_prompt_template(
        prompt_template=rwkv_prompt_template,
        messages=messages,
        tools=tools or (),
    )
    if prompt_template.style == "function_calling":
        return _render_tool_chat(
            messages,
            tools or [],
            add_generation_prompt=add_generation_prompt,
            rwkv_generation_prompt=rwkv_generation_prompt,
        )
    return _render_plain_chat(
        messages,
        prompt_template,
        add_generation_prompt=add_generation_prompt,
        rwkv_generation_prompt=rwkv_generation_prompt,
    )


def is_rwkv_model_config(model_config: Any) -> bool:
    if _is_rwkv_tokenizer_mode(getattr(model_config, "tokenizer_mode", None)):
        return True
    hf_config = getattr(model_config, "hf_config", None)
    return getattr(hf_config, "model_type", None) == "rwkv7"


def is_rwkv_model_arg(model: str | Path | None) -> bool:
    if model is None:
        return False
    try:
        return build_rwkv7_config_from_pth(model) is not None
    except ValueError:
        return False


def resolve_rwkv_tool_parser(
    *,
    tool_parser: str | None,
    enable_auto_tools: bool,
    model_config: Any | None = None,
    tokenizer_mode: str | None = None,
    model: str | Path | None = None,
) -> str | None:
    if tool_parser is not None or not enable_auto_tools:
        return tool_parser
    if model_config is not None and is_rwkv_model_config(model_config):
        return RWKV_TOOL_CALL_PARSER
    if _is_rwkv_tokenizer_mode(tokenizer_mode) or is_rwkv_model_arg(model):
        return RWKV_TOOL_CALL_PARSER
    return tool_parser


def apply_rwkv_sampling_stops(
    sampling_params: dict[str, Any],
    model_config: Any,
    *,
    prompt_template: RWKVPromptTemplateSpec | None = None,
) -> None:
    if not is_rwkv_model_config(model_config):
        return
    stops, stop_token_ids = _resolve_rwkv_stops(
        stop=sampling_params.get("stop"),
        stop_token_ids=sampling_params.get("stop_token_ids"),
        prompt_template=prompt_template,
        detokenize=bool(sampling_params.get("detokenize", True)),
    )
    sampling_params["stop"] = stops
    sampling_params["stop_token_ids"] = stop_token_ids
    sampling_params["ignore_eos"] = False


def resolve_rwkv_sampling_params(
    sampling_params: Any,
    model_config: Any,
    *,
    prompt_template: RWKVPromptTemplateSpec | None = None,
) -> Any:
    if not is_rwkv_model_config(model_config):
        return sampling_params

    def resolve(item: Any) -> Any:
        if not hasattr(item, "stop_token_ids"):
            if prompt_template is not None:
                raise ValueError(
                    "RWKV native chat templates require text stop strings, "
                    "which beam search does not support."
                )
            return msgspec.structs.replace(item, ignore_eos=False)
        stops, stop_token_ids = _resolve_rwkv_stops(
            stop=item.stop,
            stop_token_ids=item.stop_token_ids,
            prompt_template=prompt_template,
            detokenize=item.detokenize,
        )
        return msgspec.structs.replace(
            item,
            ignore_eos=False,
            _all_stop_token_ids=set(item.all_stop_token_ids) | set(stop_token_ids),
            stop=stops,
            stop_token_ids=stop_token_ids,
        )

    if isinstance(sampling_params, list):
        return [resolve(item) for item in sampling_params]
    if isinstance(sampling_params, tuple):
        return tuple(resolve(item) for item in sampling_params)
    return resolve(sampling_params)


def resolve_rwkv_chat_sampling_params(
    sampling_params: Any,
    model_config: Any,
    *,
    prompt_templates: Sequence[RWKVPromptTemplateSpec | None],
) -> Any:
    if not is_rwkv_model_config(model_config):
        return sampling_params
    if isinstance(sampling_params, (list, tuple)):
        if len(sampling_params) != len(prompt_templates):
            raise ValueError(
                "The number of RWKV chat sampling params must match the number "
                f"of conversations: {len(sampling_params)} != {len(prompt_templates)}."
            )
        resolved = [
            resolve_rwkv_sampling_params(
                item,
                model_config,
                prompt_template=prompt_template,
            )
            for item, prompt_template in zip(
                sampling_params, prompt_templates, strict=True
            )
        ]
        return tuple(resolved) if isinstance(sampling_params, tuple) else resolved
    resolved = [
        resolve_rwkv_sampling_params(
            sampling_params,
            model_config,
            prompt_template=prompt_template,
        )
        for prompt_template in prompt_templates
    ]
    return resolved[0] if len(resolved) == 1 else resolved


def _render_plain_chat(
    messages: list[Any],
    prompt_template: RWKVPromptTemplateSpec,
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str,
) -> str:
    rendered: list[str] = []
    for message in messages:
        role = _field(message, "role", "")
        content = stringify_rwkv_message_content(_field(message, "content", ""))
        label = _plain_role_label(role, prompt_template)
        if prompt_template.style == "bot":
            rendered.append(f"{label}✿{content}✿")
        else:
            rendered.append(f"{label}: {content}" if content else f"{label}:")

    if add_generation_prompt:
        generation_prompt = _generation_prompt_text(rwkv_generation_prompt)
        if prompt_template.style == "bot":
            rendered.append(f"Bot✿{generation_prompt}")
        else:
            rendered.append(f"Assistant: {generation_prompt}")

    separator = "\n" if prompt_template.style == "bot" else "\n\n"
    return separator.join(rendered)


def _render_tool_chat(
    messages: list[Any],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    rwkv_generation_prompt: str,
) -> str:
    lines: list[str] = []
    pending_system: list[str] = []

    for message in messages:
        role = _field(message, "role", "")
        content = stringify_rwkv_message_content(_field(message, "content", ""))

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
                name = _field(function, "name", "")
                arguments = _field(function, "arguments", {}) or {}
                payload = {
                    "name": name,
                    "arguments": _json_value(arguments),
                }
                lines.extend(["**Tool Call:**", "```json", _json_text(payload), "```"])
        elif role == "tool":
            payload = _json_value(content)
            lines.extend(["### Tool Output", "```json", _json_text(payload), "```"])
        else:
            raise ValueError(f"Unsupported RWKV chat message role: {role!r}")

    if pending_system:
        lines.append("### System")
        lines.extend(item for item in pending_system if item)
        lines.extend(_render_tool_definitions(tools))

    if add_generation_prompt:
        lines.append("### Assistant")
        lines.append(_generation_prompt_text(rwkv_generation_prompt))

    return "\n".join(lines)


def _check_generation_prompt(mode: str) -> None:
    if mode not in RWKV_GENERATION_PROMPT_MODES:
        raise ValueError(
            "Unsupported RWKV generation prompt mode: "
            f"{mode!r}. Expected one of {RWKV_GENERATION_PROMPT_MODES!r}."
        )


def _generation_prompt_text(mode: str) -> str:
    if mode == RWKV_GENERATION_PROMPT_OPEN_THINK:
        return "<think"
    if mode == RWKV_GENERATION_PROMPT_FAKE_THINK:
        return "<think></think"
    _check_generation_prompt(mode)
    raise AssertionError("unreachable")


def _render_tool_definitions(tools: list[dict[str, Any]]) -> list[str]:
    rendered: list[str] = []
    for tool in tools:
        function = _tool_function(tool)
        name = _field(function, "name", "")
        description = _field(function, "description", "") or ""
        parameters = _field(function, "parameters", {}) or {}
        rendered.append(f"### `{name}`")
        if description:
            rendered.append(
                f"**Description:** {stringify_rwkv_message_content(description)}"
            )
        rendered.append("**Parameters:**")
        rendered.append("```json")
        rendered.append(_json_text(parameters))
        rendered.append("```")
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


def _json_text(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, indent=2)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _plain_role_label(
    role: str,
    prompt_template: RWKVPromptTemplateSpec,
) -> str:
    if role == "system":
        return "System"
    if role == "user":
        return "User"
    if role == "assistant":
        return "Bot" if prompt_template.style == "bot" else "Assistant"
    raise ValueError(f"Unsupported RWKV chat message role: {role!r}")


def _resolve_rwkv_stops(
    *,
    stop: str | Sequence[str] | None,
    stop_token_ids: Sequence[int] | None,
    prompt_template: RWKVPromptTemplateSpec | None,
    detokenize: bool,
) -> tuple[list[str], list[int]]:
    user_stops = [stop] if isinstance(stop, str) else list(stop or ())
    required_stops = (
        [prompt_template.stop] if prompt_template is not None and detokenize else []
    )
    stops = list(dict.fromkeys([*required_stops, *user_stops]))
    token_ids = list(
        dict.fromkeys(
            [
                RWKV_BOS_EOS_TOKEN_ID,
                *(int(token_id) for token_id in stop_token_ids or ()),
            ]
        )
    )
    return stops, token_ids


def _tool_function(tool: Any) -> Any:
    if isinstance(tool, dict):
        return tool.get("function", tool)
    return getattr(tool, "function", tool)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _is_rwkv_tokenizer_mode(tokenizer_mode: Any) -> bool:
    return isinstance(tokenizer_mode, str) and tokenizer_mode.lower() == "rwkv"
