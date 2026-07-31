# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.entrypoints.offline_utils import OfflineInferenceMixin
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs import tokens_input
from vllm.renderers.hf import HfRenderer, safe_apply_chat_template
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.renderers.params import ChatParams
from vllm.renderers.rwkv import RwkvRenderer
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers.rwkv import RWKVTokenizer
from vllm.tokenizers.rwkv_chat import RWKV_NATIVE_CHAT_TEMPLATE
from vllm.v1.engine.input_processor import InputProcessor


def test_rwkv_renderer_preserves_exactly_one_bos_during_left_truncation():
    renderer = object.__new__(RwkvRenderer)

    assert renderer.normalize_prompt_token_ids([]) == [0]
    assert renderer.normalize_prompt_token_ids([0, 0, 1, 2]) == [0, 1, 2]
    assert renderer.normalize_prompt_token_ids(
        [0, 1, 2, 3],
        max_length=3,
    ) == [0, 2, 3]


def test_rwkv_renderer_resolves_native_chat_stops():
    renderer = object.__new__(RwkvRenderer)
    messages = [{"role": "user", "content": "Hello"}]

    assert renderer.get_chat_stop(messages, ChatParams()) == "✿"
    assert (
        renderer.get_chat_stop(
            messages,
            ChatParams(
                chat_template=RWKV_NATIVE_CHAT_TEMPLATE,
                chat_template_kwargs={"rwkv_prompt_template": "\n\nAssistant: "},
            ),
        )
        == "\nUser:"
    )
    assert (
        renderer.get_chat_stop(
            messages,
            ChatParams(
                chat_template="{{ messages[0]['content'] }}",
            ),
        )
        is None
    )


def test_hf_rendering_pipeline_uses_rwkv_native_template():
    tokenizer = RWKVTokenizer()
    model_config = SimpleNamespace(
        trust_remote_code=False,
        hf_config=SimpleNamespace(model_type="rwkv7"),
    )
    conversation = [{"role": "user", "content": "Hello"}]

    prompt = safe_apply_chat_template(
        model_config,
        tokenizer,
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        rwkv_prompt_template="\n\nAssistant: ",
    )
    token_ids = safe_apply_chat_template(
        model_config,
        tokenizer,
        conversation,
        tokenize=True,
        add_generation_prompt=True,
    )

    assert prompt == "User: Hello\n\nAssistant: <think"
    assert token_ids[0] == 0
    assert token_ids[1] != 0


def test_rwkv_renderer_tool_history_selects_function_stop():
    renderer = object.__new__(RwkvRenderer)
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"function": {}}]}
    ]

    assert renderer.get_chat_stop(messages, ChatParams()) == "\n### User"
    assert renderer.default_tool_parser == "rwkv"
    assert renderer.resolve_ignore_eos(True) is False


def _make_input_processor(renderer):
    processor = object.__new__(InputProcessor)
    processor.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_size_local=1,
            local_engines_only=False,
        )
    )
    processor.model_config = SimpleNamespace(max_model_len=128)
    processor.renderer = renderer
    processor.generation_config_fields = {}
    processor._validate_params = lambda *args: None
    processor._validate_lora = lambda *args: None
    processor._validate_model_inputs = lambda *args: None
    return processor


def test_input_processor_enforces_rwkv_eos_policy(monkeypatch):
    from vllm.v1.engine import input_processor as input_processor_module

    monkeypatch.setattr(
        input_processor_module.current_platform,
        "validate_request",
        lambda *args: None,
    )
    rwkv_renderer = object.__new__(RwkvRenderer)
    rwkv_renderer.tokenizer = RWKVTokenizer()
    rwkv_processor = _make_input_processor(rwkv_renderer)
    requested = SamplingParams(ignore_eos=True, max_tokens=1)

    processed = rwkv_processor.process_inputs(
        "rwkv-request",
        tokens_input([0, 1]),
        requested,
        ("generate",),
    )

    assert requested.ignore_eos is True
    assert processed.sampling_params is not None
    assert processed.sampling_params.ignore_eos is False
    assert processed.sampling_params.eos_token_id == 0
    assert 0 in processed.sampling_params.all_stop_token_ids

    hf_renderer = object.__new__(HfRenderer)
    hf_renderer.tokenizer = SimpleNamespace(eos_token_id=7)
    hf_processor = _make_input_processor(hf_renderer)
    other = hf_processor.process_inputs(
        "hf-request",
        tokens_input([7, 1]),
        SamplingParams(ignore_eos=True, max_tokens=1),
        ("generate",),
    )

    assert other.sampling_params is not None
    assert other.sampling_params.ignore_eos is True
    assert other.sampling_params.eos_token_id is None
    assert 7 in other.sampling_params.all_stop_token_ids


def test_rwkv_renderer_supplies_auto_tool_parser():
    renderer = object.__new__(RwkvRenderer)
    online = OnlineRenderer(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="rwkv7"),
            model="rwkv.pth",
        ),
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        enable_auto_tools=True,
    )

    assert online.parser is not None
    assert online.tool_parser_name == "rwkv"
    assert online.parser.tool_parser_cls.__name__ == "RWKVToolParser"


def test_rwkv_default_tool_parser_requires_auto_tools():
    renderer = object.__new__(RwkvRenderer)
    online = OnlineRenderer(
        SimpleNamespace(
            hf_config=SimpleNamespace(model_type="rwkv7"),
            model="rwkv.pth",
        ),
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )

    assert online.tool_parser_name is None
    assert online.parser is None


def test_other_renderers_still_require_explicit_auto_tool_parser():
    renderer = object.__new__(HfRenderer)

    with pytest.raises(TypeError, match="requires --tool-call-parser"):
        OnlineRenderer(
            SimpleNamespace(
                hf_config=SimpleNamespace(model_type="llama"),
                model="llama",
            ),
            renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )


class _OnlineRwkvRenderer(RwkvRenderer):
    async def render_chat_async(self, *args, **kwargs):
        return ([{"role": "user", "content": "Hello"}],), (tokens_input([0, 1]),)


@pytest.mark.asyncio
async def test_online_rwkv_chat_merges_native_stop():
    renderer = object.__new__(_OnlineRwkvRenderer)
    renderer.tokenizer = RWKVTokenizer()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="rwkv7"),
        model="rwkv.pth",
        multimodal_config=None,
        enable_prompt_embeds=False,
        max_model_len=128,
    )
    online = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        stop=["user-stop"],
    )

    result = await online.preprocess_chat(
        request,
        request.messages,
        default_template=None,
        default_template_content_format="auto",
        default_template_kwargs=None,
    )

    assert result[1][0]["prompt_token_ids"] == [0, 1]
    assert request.stop == ["✿", "user-stop"]


@pytest.mark.asyncio
async def test_online_rwkv_native_chat_rejects_beam_search():
    renderer = object.__new__(_OnlineRwkvRenderer)
    renderer.tokenizer = RWKVTokenizer()
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="rwkv7"),
        model="rwkv.pth",
        multimodal_config=None,
        enable_prompt_embeds=False,
        max_model_len=128,
    )
    online = OnlineRenderer(
        model_config,
        renderer,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "Hello"}],
        use_beam_search=True,
    )

    with pytest.raises(ValueError, match="text stop string"):
        await online.preprocess_chat(
            request,
            request.messages,
            default_template=None,
            default_template_content_format="auto",
            default_template_kwargs=None,
        )


class _CaptureOfflineParams(OfflineInferenceMixin):
    def __init__(self):
        self.renderer = object.__new__(RwkvRenderer)
        self.model_config = SimpleNamespace(enable_prompt_embeds=False)
        self.params = None

    def _render_and_add_requests(self, *, params, **kwargs):
        self.params = params
        return []


def test_offline_rwkv_chat_merges_native_stop_without_mutating_caller():
    offline = _CaptureOfflineParams()
    original = SamplingParams(stop=["user-stop"])

    offline._add_chat_requests(
        messages=[{"role": "user", "content": "Hello"}],
        params=original,
    )

    assert original.stop == ["user-stop"]
    assert offline.params[0].stop == ["✿", "user-stop"]
    assert offline.params[0].output_text_buffer_length == len("user-stop") - 1


def test_offline_rwkv_chat_rebuilds_stop_buffer_length():
    offline = _CaptureOfflineParams()

    offline._add_chat_requests(
        messages=[{"role": "user", "content": "Hello"}],
        params=SamplingParams(),
        chat_template_kwargs={"rwkv_prompt_template": "\n\nAssistant: "},
    )

    assert offline.params[0].stop == ["\nUser:"]
    assert offline.params[0].output_text_buffer_length == len("\nUser:") - 1

    offline._add_chat_requests(
        messages=[
            {"role": "user", "content": "Use a tool"},
            {"role": "assistant", "content": None, "tool_calls": [{"function": {}}]},
        ],
        params=SamplingParams(),
    )

    assert offline.params[0].stop == ["\n### User"]
    assert offline.params[0].output_text_buffer_length == len("\n### User") - 1


def test_offline_rwkv_chat_does_not_add_text_stop_without_detokenization():
    offline = _CaptureOfflineParams()

    offline._add_chat_requests(
        messages=[{"role": "user", "content": "Hello"}],
        params=SamplingParams(detokenize=False),
    )

    assert offline.params[0].stop == []
    assert offline.params[0].output_text_buffer_length == 0


def test_offline_rwkv_native_chat_rejects_beam_search():
    offline = _CaptureOfflineParams()

    with pytest.raises(ValueError, match="text stop string"):
        offline._add_chat_requests(
            messages=[{"role": "user", "content": "Hello"}],
            params=BeamSearchParams(beam_width=2, max_tokens=4),
        )
