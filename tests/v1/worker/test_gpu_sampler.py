# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import SamplingStates


def _new_rapid_sampler(
    *,
    rapid_penalties: torch.Tensor,
) -> Sampler:
    sampler = object.__new__(Sampler)
    sampler.rapid_penalties = rapid_penalties
    sampler.rapid_penalty_native_fallback = np.zeros(
        rapid_penalties.shape[0], dtype=bool
    )
    sampler.request_requires_rapid = np.zeros(rapid_penalties.shape[0], dtype=bool)
    sampler.req_states = SimpleNamespace(
        max_num_reqs=rapid_penalties.shape[0],
        vocab_size=rapid_penalties.shape[1],
        device=rapid_penalties.device,
    )
    return sampler


def test_worker_sampler_accepts_frequency_penalty_for_rapid_sampling():
    rapid_penalties = torch.full((1, 8), 3.0)
    sampler = _new_rapid_sampler(rapid_penalties=rapid_penalties)
    sampler.use_rapid = True
    sampler.require_rapid = False
    sampler.rapid_penalty_native_fallback[0] = True
    for name in (
        "sampling_states",
        "penalties_state",
        "logit_bias_state",
        "bad_words_state",
        "logprob_token_ids_state",
    ):
        setattr(sampler, name, SimpleNamespace(add_request=lambda *args: None))

    sampler.add_request(0, 0, SamplingParams(frequency_penalty=0.5))

    assert torch.count_nonzero(rapid_penalties) == 0
    assert not sampler.rapid_penalty_native_fallback[0]


def test_worker_sampler_rejects_repetition_penalty_when_rapid_is_required():
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = True
    sampler.rapid_penalty_native_fallback = np.zeros(1, dtype=bool)

    with pytest.raises(RuntimeError, match="does not support repetition_penalty"):
        sampler.add_request(0, 0, SamplingParams(repetition_penalty=1.2))


@pytest.mark.parametrize(
    ("sampling_params", "match"),
    [
        (SamplingParams(temperature=0.0), "greedy requests"),
        (SamplingParams(seed=7), "per-request seeds"),
        (SamplingParams(min_p=0.1), "min_p"),
        (SamplingParams(repetition_penalty=1.2), "repetition_penalty"),
        (
            SamplingParams(frequency_penalty=0.2),
            "speculative expanded logits",
        ),
    ],
)
def test_required_rapid_incompatibility_fails_at_request_boundary(
    sampling_params,
    match,
):
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = True
    sampler.num_speculative_tokens = 2
    sampler.req_states = SimpleNamespace(vocab_size=8)
    sampler.logprobs_mode = "processed_logprobs"

    with pytest.raises(RuntimeError, match=match):
        sampler.validate_sampling_params(sampling_params)


@pytest.mark.parametrize(
    ("logprobs_mode", "sampling_params", "match"),
    [
        (
            "processed_logprobs",
            SamplingParams(logprobs=2),
            "only supports the sampled token",
        ),
        (
            "processed_logprobs",
            SamplingParams(logprob_token_ids=[1]),
            "only supports the sampled token",
        ),
        (
            "processed_logits",
            SamplingParams(logprobs=0),
            "cannot return exact processed_logits",
        ),
    ],
)
def test_required_rapid_rejects_inexact_processed_logprob_contract(
    logprobs_mode,
    sampling_params,
    match,
):
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = True
    sampler.num_speculative_tokens = 1
    sampler.req_states = SimpleNamespace(vocab_size=8)
    sampler.logprobs_mode = logprobs_mode

    with pytest.raises(RuntimeError, match=match):
        sampler.validate_sampling_params(sampling_params)


def test_required_rapid_accepts_exact_sampled_processed_logprob_contract():
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = True
    sampler.num_speculative_tokens = 1
    sampler.req_states = SimpleNamespace(vocab_size=8)
    sampler.logprobs_mode = "processed_logprobs"

    sampler.validate_sampling_params(SamplingParams(logprobs=0))


def test_structured_output_request_marks_rapid_as_non_fallback():
    sampler = _new_rapid_sampler(rapid_penalties=torch.zeros((1, 8)))
    sampler.use_rapid = True
    sampler.require_rapid = False
    sampler.num_speculative_tokens = 1
    for name in (
        "sampling_states",
        "penalties_state",
        "logit_bias_state",
        "bad_words_state",
        "logprob_token_ids_state",
    ):
        setattr(sampler, name, SimpleNamespace(add_request=lambda *args: None))

    sampler.add_request(
        0,
        0,
        SamplingParams(structured_outputs=StructuredOutputsParams(regex="a(b|c)")),
    )

    assert sampler.request_requires_rapid.tolist() == [True]


def test_structured_output_rejects_rapid_incompatibility_at_request_boundary():
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = False
    sampler.num_speculative_tokens = 1
    sampler.req_states = SimpleNamespace(vocab_size=8)
    sampler.logprobs_mode = "processed_logprobs"

    sampling_params = SamplingParams(
        min_p=0.1,
        structured_outputs=StructuredOutputsParams(regex="a(b|c)"),
    )
    with pytest.raises(RuntimeError, match="min_p"):
        sampler.validate_sampling_params(sampling_params)


def test_model_runner_validates_sampler_before_request_state_mutation():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.is_last_pp_rank = True
    calls = []

    class RejectingSampler:
        def validate_sampling_params(self, _sampling_params):
            calls.append("validate")
            raise RuntimeError("request-scoped rapid incompatibility")

    runner.sampler = RejectingSampler()
    runner._remove_request = lambda _req_id: pytest.fail(
        "request state was mutated before sampler validation"
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id="structured-request",
                prompt_token_ids=[1],
                prefill_token_ids=[1],
                sampling_params=SamplingParams(
                    structured_outputs=StructuredOutputsParams(regex="a+")
                ),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="request-scoped rapid incompatibility"):
        GPUModelRunner.add_requests(runner, scheduler_output)

    assert calls == ["validate"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_sampling_states_can_collapse_uniform_rapid_params():
    states = SamplingStates(max_num_reqs=4, vocab_size=8)
    for req_idx in range(3):
        states.add_request(
            req_idx,
            SamplingParams(temperature=1.0, top_k=4, top_p=0.28),
        )
    states.apply_staged_writes()

    device = states.temperature.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1, 2])

    temperatures = states.get_temperatures(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )
    top_k, top_p = states.get_top_k_top_p(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert temperatures == pytest.approx(1.0)
    assert top_k == 4
    assert top_p == pytest.approx(0.28)

    vector_top_k, vector_top_p = states.get_top_k_top_p(
        expanded_idx_mapping,
        idx_mapping_np,
    )

    assert vector_top_k is not None and vector_top_k.shape == (3,)
    assert vector_top_p is not None and vector_top_p.shape == (3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_penalties_state_can_collapse_uniform_rapid_params():
    req_states = SimpleNamespace(
        max_num_reqs=4,
        vocab_size=8,
        device=torch.device("cuda"),
    )
    state = PenaltiesState(req_states)
    state.presence_penalty.np[:3] = 0.2
    state.frequency_penalty.np[:3] = 0.1
    state.penalty_decay.np[:3] = 0.95
    state.presence_penalty.copy_to_uva()
    state.frequency_penalty.copy_to_uva()
    state.penalty_decay.copy_to_uva()

    device = state.presence_penalty.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1, 2])

    presence, frequency, decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert presence == pytest.approx(0.2)
    assert frequency == pytest.approx(0.1)
    assert decay == pytest.approx(0.95)

    vector_presence, vector_frequency, vector_decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
    )

    assert vector_presence.shape == (3,)
    assert vector_frequency.shape == (3,)
    assert vector_decay.shape == (3,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA tensors require CUDA")
def test_penalties_state_keeps_mixed_rapid_params_as_vectors():
    req_states = SimpleNamespace(
        max_num_reqs=4,
        vocab_size=8,
        device=torch.device("cuda"),
    )
    state = PenaltiesState(req_states)
    state.presence_penalty.np[:2] = [0.2, 0.3]
    state.frequency_penalty.np[:2] = 0.1
    state.penalty_decay.np[:2] = 0.95
    state.presence_penalty.copy_to_uva()
    state.frequency_penalty.copy_to_uva()
    state.penalty_decay.copy_to_uva()

    device = state.presence_penalty.gpu.device
    expanded_idx_mapping = torch.tensor([0, 1], dtype=torch.int32, device=device)
    idx_mapping_np = np.array([0, 1])

    presence, frequency, decay = state.rapid_penalty_params(
        expanded_idx_mapping,
        idx_mapping_np,
        scalar_if_uniform=True,
    )

    assert isinstance(presence, torch.Tensor)
    assert isinstance(frequency, torch.Tensor)
    assert isinstance(decay, torch.Tensor)
    assert presence.shape == (2,)


@pytest.mark.parametrize(
    "processed_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_rapid_sampler_returns_kernel_processed_logprob(monkeypatch, processed_dtype):
    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = False
    sampler.use_flashinfer = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.rapid_penalties = None
    sampler.rapid_penalty_native_fallback = np.zeros(1, dtype=bool)
    sampler.request_requires_rapid = np.zeros(1, dtype=bool)

    class FakeSamplingStates:
        def get_temperatures(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return 1.0

        def get_top_k_top_p(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return None, None

        def any_greedy(self, idx_mapping_np):
            return False

        def any_explicit_seed(self, idx_mapping_np):
            return False

        def any_min_p(self, idx_mapping_np):
            return False

    class FakePenaltiesState:
        def any_repetition_penalty(self, idx_mapping_np):
            return False

        def rapid_penalty_mask(self, idx_mapping_np):
            return np.zeros(len(idx_mapping_np), dtype=bool)

    sampler.sampling_states = FakeSamplingStates()
    sampler.penalties_state = FakePenaltiesState()

    sampling_param_calls = []
    first_processed_logits = torch.full((1, 4), 1.0, dtype=processed_dtype)

    def fake_apply_sampling_params(*args, **kwargs):
        sampling_param_calls.append(kwargs)
        return first_processed_logits

    sampler.apply_sampling_params = fake_apply_sampling_params
    monkeypatch.setattr(
        sampler_module, "rapid_sample_input_supported", lambda logits: True
    )

    def fake_rapid_sample(logits, top_k, top_p, temperatures, return_logprobs):
        assert logits.dtype == torch.float32
        return torch.tensor([3]), torch.tensor([-0.75])

    monkeypatch.setattr(sampler_module, "rapid_sample", fake_rapid_sample)

    sampled, processed_logits, sampled_logprobs, sampled_only_fast_path = (
        sampler.sample(
            logits=torch.zeros((1, 4)),
            expanded_idx_mapping=torch.tensor([0]),
            idx_mapping_np=np.array([0]),
            pos=torch.tensor([0]),
            input_ids=torch.tensor([0]),
            expanded_local_pos=torch.tensor([0]),
            return_logprobs=True,
            sampled_only_logprobs=True,
        )
    )

    assert sampled.tolist() == [3]
    assert sampled_logprobs.tolist() == [-0.75]
    assert sampled_only_fast_path is True
    assert processed_logits.dtype == torch.float32
    assert torch.equal(processed_logits, first_processed_logits.float())
    if processed_dtype == torch.float32:
        assert processed_logits is first_processed_logits
    assert len(sampling_param_calls) == 1
    assert sampling_param_calls[0]["skip_top_k_top_p"] is True
    assert sampling_param_calls[0]["skip_temperature"] is True


def test_rapid_sampler_output_uses_kernel_sampled_logprob(monkeypatch):
    sampler = object.__new__(Sampler)
    sampler.compute_nans = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.sampling_states = SimpleNamespace(
        max_num_logprobs=lambda idx_mapping_np: 0,
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        max_num_token_ids=lambda idx_mapping_np: 0,
    )
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([0])),
    )
    sampler.sample = lambda *args, **kwargs: (
        torch.tensor([3]),
        torch.zeros((1, 4)),
        torch.tensor([-0.75]),
        True,
    )

    monkeypatch.setattr(
        sampler_module,
        "compute_topk_scores",
        lambda *args, **kwargs: pytest.fail(
            "sampled-only rapid path must not recompute native logprobs"
        ),
    )
    monkeypatch.setattr(
        sampler_module,
        "compute_token_ranks",
        lambda logits, token_ids: torch.tensor([2]),
    )
    monkeypatch.setattr(
        sampler_module,
        "get_num_sampled_and_rejected",
        lambda *args, **kwargs: (torch.tensor([1]), torch.tensor([0])),
    )

    input_batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        cu_num_logits_np=np.array([1]),
        expanded_local_pos=torch.tensor([0]),
        positions=torch.tensor([0]),
        logits_indices=torch.tensor([0]),
        input_ids=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        cu_num_logits=torch.tensor([0, 1]),
        idx_mapping=torch.tensor([0]),
        num_reqs=1,
    )

    output = sampler(torch.zeros((1, 4)), input_batch)

    assert output.sampled_token_ids.tolist() == [[3]]
    assert output.logprobs_tensors is not None
    assert output.logprobs_tensors.logprob_token_ids.tolist() == [[3]]
    assert output.logprobs_tensors.logprobs.tolist() == [[-0.75]]
    assert output.logprobs_tensors.selected_token_ranks.tolist() == [2]


def test_rapid_sampled_only_output_preserves_expanded_row_layout(monkeypatch):
    sampler = object.__new__(Sampler)
    sampler.compute_nans = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.sampling_states = SimpleNamespace(
        max_num_logprobs=lambda idx_mapping_np: 0,
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        max_num_token_ids=lambda idx_mapping_np: 0,
    )
    sampler.req_states = SimpleNamespace(
        prefill_len=SimpleNamespace(gpu=torch.tensor([0, 0])),
    )
    sampler.sample = lambda *args, **kwargs: (
        torch.tensor([1, 2, 3]),
        torch.zeros((3, 4)),
        torch.tensor([-0.25, -0.5, -0.75]),
        True,
    )
    monkeypatch.setattr(
        sampler_module,
        "compute_topk_scores",
        lambda *args, **kwargs: pytest.fail(
            "expanded sampled-only path must not recompute native logprobs"
        ),
    )
    monkeypatch.setattr(
        sampler_module,
        "compute_token_ranks",
        lambda logits, token_ids: torch.tensor([1, 2, 3]),
    )
    monkeypatch.setattr(
        sampler_module,
        "get_num_sampled_and_rejected",
        lambda *args, **kwargs: (torch.tensor([2, 1]), torch.tensor([0, 0])),
    )
    input_batch = SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0, 0, 1]),
        idx_mapping_np=np.array([0, 1]),
        cu_num_logits_np=np.array([0, 2, 3]),
        expanded_local_pos=torch.tensor([0, 1, 0]),
        positions=torch.tensor([0, 1, 0]),
        logits_indices=torch.tensor([0, 1, 2]),
        input_ids=torch.tensor([0, 0, 0]),
        seq_lens=torch.tensor([2, 1]),
        cu_num_logits=torch.tensor([0, 2, 3]),
        idx_mapping=torch.tensor([0, 1]),
        num_reqs=2,
    )

    output = sampler(torch.zeros((3, 4)), input_batch)

    assert output.sampled_token_ids.tolist() == [[1], [2], [3]]
    assert output.logprobs_tensors is not None
    assert output.logprobs_tensors.logprobs.shape == (3, 1)
    assert output.logprobs_tensors.selected_token_ranks.shape == (3,)
    assert output.logprobs_tensors.cu_num_generated_tokens == [0, 2, 3]


@pytest.mark.parametrize(
    ("require_rapid", "structured_output_requires_rapid", "raises"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
def test_rapid_sampler_native_fallback_is_forbidden_when_required(
    monkeypatch,
    require_rapid,
    structured_output_requires_rapid,
    raises,
):
    monkeypatch.delenv("VLLM_USE_RAPID_SAMPLER", raising=False)

    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.use_flashinfer = False
    sampler.logprobs_mode = "raw_logprobs"
    sampler.rapid_penalties = None
    sampler.rapid_penalty_native_fallback = np.zeros(1, dtype=bool)
    sampler.use_fp64_gumbel = False
    sampler.require_rapid = require_rapid
    sampler.request_requires_rapid = np.array([structured_output_requires_rapid])

    top_k_calls = []

    class FakeSamplingStates:
        temperature = SimpleNamespace(gpu=torch.tensor([1.0]))
        seeds = SimpleNamespace(gpu=torch.tensor([0]))

        def get_temperatures(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            assert scalar_if_uniform is True
            return 1.0

        def get_top_k_top_p(
            self,
            expanded_idx_mapping,
            idx_mapping_np,
            *,
            scalar_if_uniform=False,
        ):
            top_k_calls.append(scalar_if_uniform)
            return None, None

        def any_greedy(self, idx_mapping_np):
            return False

        def any_explicit_seed(self, idx_mapping_np):
            return False

        def any_min_p(self, idx_mapping_np):
            return False

    class FakePenaltiesState:
        def any_repetition_penalty(self, idx_mapping_np):
            return False

        def rapid_penalty_mask(self, idx_mapping_np):
            return np.zeros(len(idx_mapping_np), dtype=bool)

    sampler.sampling_states = FakeSamplingStates()
    sampler.penalties_state = FakePenaltiesState()

    sampling_param_calls = []
    rapid_processed_logits = torch.full((1, 4), 1.0)
    native_processed_logits = torch.full((1, 4), 2.0)

    def fake_apply_sampling_params(*args, **kwargs):
        sampling_param_calls.append(kwargs)
        return (
            rapid_processed_logits
            if len(sampling_param_calls) == 1
            else native_processed_logits
        )

    sampler.apply_sampling_params = fake_apply_sampling_params
    monkeypatch.setattr(
        sampler_module, "rapid_sample_input_supported", lambda logits: False
    )
    monkeypatch.setattr(
        sampler_module,
        "rapid_sample",
        lambda *args, **kwargs: pytest.fail("unsupported input should use native"),
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *args, **kwargs: torch.tensor([2]),
    )

    sample = lambda: sampler.sample(
        logits=torch.zeros((1, 4)),
        expanded_idx_mapping=torch.tensor([0]),
        idx_mapping_np=np.array([0]),
        pos=torch.tensor([0]),
        input_ids=torch.tensor([0]),
        expanded_local_pos=torch.tensor([0]),
    )

    if raises:
        with pytest.raises(RuntimeError, match="rapid-sampling requires"):
            sample()
        return

    sampled, processed_logits, sampled_logprobs, sampled_only_fast_path = sample()

    assert sampled.tolist() == [2]
    assert sampled_logprobs is None
    assert sampled_only_fast_path is False
    assert processed_logits is native_processed_logits
    assert top_k_calls == [True, False]
    assert len(sampling_param_calls) == 2
    assert sampling_param_calls[0]["skip_top_k_top_p"] is True
    assert sampling_param_calls[0]["skip_temperature"] is True
    assert sampling_param_calls[1].get("skip_top_k_top_p", False) is False
    assert sampling_param_calls[1].get("skip_temperature", False) is False


@pytest.mark.parametrize("constraint", ["json", "regex", "choice"])
def test_xgrammar_mask_and_rapid_params_share_streaming_distribution(
    monkeypatch,
    constraint,
):
    xgr = pytest.importorskip("xgrammar")
    from vllm.v1.structured_output.utils import choice_as_grammar
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    vocab = [
        b"a",
        b"b",
        b"c",
        b"d",
        b'"',
        b"{",
        b"}",
        b":",
        b",",
        b" ",
        b"0",
        b"1",
        b"[",
        b"]",
        b"true",
        b"<eos>",
    ]
    eos_token_id = 15
    tokenizer_info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[eos_token_id],
    )
    compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=False)
    if constraint == "json":
        ctx = compiler.compile_json_schema(
            '{"type":"string","enum":["ab","ac","ad"]}',
            any_whitespace=False,
        )
        prefix = [4, 0]
    elif constraint == "regex":
        ctx = compiler.compile_regex("a(b|c|d)")
        prefix = [0]
    else:
        ctx = compiler.compile_grammar(choice_as_grammar(["ab", "ac", "ad"]))
        prefix = [0]
    matcher = xgr.GrammarMatcher(ctx)
    for token_id in prefix:
        assert matcher.accept_token(token_id)

    sampler = object.__new__(Sampler)
    sampler.use_rapid = True
    sampler.require_rapid = True
    sampler.use_flashinfer = False
    sampler.logprobs_mode = "processed_logprobs"
    sampler.use_fp64_gumbel = False
    sampler.rapid_penalties = torch.zeros((1, len(vocab)), dtype=torch.float32)
    sampler.rapid_penalties[0, 1] = 0.5
    sampler.rapid_penalty_native_fallback = np.zeros(1, dtype=bool)
    sampler.request_requires_rapid = np.ones(1, dtype=bool)

    class SamplingStates:
        temperature = SimpleNamespace(gpu=torch.tensor([0.5]))
        seeds = SimpleNamespace(gpu=torch.tensor([0]))

        def get_temperatures(self, *args, **kwargs):
            assert kwargs["scalar_if_uniform"] is True
            return 0.5

        def get_top_k_top_p(self, *args, **kwargs):
            assert kwargs["scalar_if_uniform"] is True
            return 2, 0.9

        def any_greedy(self, _idx_mapping_np):
            return False

        def any_explicit_seed(self, _idx_mapping_np):
            return False

        def any_min_p(self, _idx_mapping_np):
            return False

    class PenaltiesState:
        def rapid_penalty_mask(self, _idx_mapping_np):
            return np.ones(1, dtype=bool)

        def rapid_penalty_params(self, *args, **kwargs):
            assert kwargs["scalar_if_uniform"] is True
            return 0.25, 0.5, 0.75

        def any_repetition_penalty(self, _idx_mapping_np):
            return False

    sampler.sampling_states = SamplingStates()
    sampler.penalties_state = PenaltiesState()
    sampler.apply_sampling_params = lambda logits, *args, **kwargs: logits.clone()

    monkeypatch.setattr(
        sampler_module, "rapid_sample_input_supported", lambda _logits: True
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *args, **kwargs: pytest.fail("grammar request bypassed rapid sampler"),
    )

    rapid_calls = []

    def rapid_sample(
        logits,
        top_k,
        top_p,
        *,
        temperatures,
        penalties,
        presence_penalties,
        frequency_penalties,
        penalty_decays,
        penalty_indices,
        return_logprobs,
    ):
        rapid_calls.append(logits.clone())
        assert top_k == 2
        assert top_p == pytest.approx(0.9)
        assert temperatures == pytest.approx(0.5)
        assert presence_penalties == pytest.approx(0.25)
        assert frequency_penalties == pytest.approx(0.5)
        assert penalty_decays == pytest.approx(0.75)
        assert penalty_indices.tolist() == [0]
        assert penalties is sampler.rapid_penalties
        assert return_logprobs is True

        allowed = torch.isfinite(logits[0]).nonzero().flatten().tolist()
        if len(rapid_calls) == 1:
            assert allowed == [1, 2, 3]
            adjusted = torch.tensor(
                [
                    (logits[0, 1] - penalties[0, 1]) / temperatures,
                    (logits[0, 2] - penalties[0, 2]) / temperatures,
                ]
            )
            sampled_logprob = adjusted.log_softmax(dim=0)[1]
            penalties[0, 2] = 7.0
            return torch.tensor([2]), sampled_logprob.view(1)

        expected_terminal = 4 if constraint == "json" and len(rapid_calls) == 2 else 15
        assert allowed == [expected_terminal]
        assert penalties[0, 2].item() == pytest.approx(7.0)
        return torch.tensor([expected_terminal]), torch.tensor([0.0])

    monkeypatch.setattr(sampler_module, "rapid_sample", rapid_sample)

    sampled_steps = []

    class RapidSamplerAdapter:
        def __call__(self, masked_logits, _input_batch):
            sampled, _processed, sampled_logprobs, _fast_path = sampler.sample(
                masked_logits,
                expanded_idx_mapping=torch.tensor([0]),
                idx_mapping_np=np.array([0]),
                pos=torch.tensor([0]),
                input_ids=torch.tensor([0]),
                expanded_local_pos=torch.tensor([0]),
                return_logprobs=True,
                sampled_only_logprobs=True,
            )
            assert sampled_logprobs is not None
            sampled_steps.append((sampled.item(), sampled_logprobs.item()))
            return SimpleNamespace(
                num_sampled=torch.tensor([1]),
                num_rejected=torch.tensor([0]),
            )

    class StructuredOutputsWorker:
        def apply_grammar_bitmask(
            self,
            logits,
            _input_batch,
            request_ids,
            grammar_bitmask,
        ):
            assert request_ids == ["structured-request"]
            xgr.apply_token_bitmask_inplace(
                logits,
                torch.from_numpy(grammar_bitmask),
                vocab_size=len(vocab),
                backend="cpu",
            )

    class Model:
        def compute_sampling_logits(self, *_args):
            logits = torch.full((1, len(vocab)), 100.0, dtype=torch.float32)
            logits[0, 1:4] = torch.tensor([3.0, 2.0, 1.0])
            return logits

    runner = object.__new__(GPUModelRunner)
    runner.model = Model()
    runner.structured_outputs_worker = StructuredOutputsWorker()
    runner.sampler = RapidSamplerAdapter()
    runner.rejection_sampler = None
    runner.speculator = None
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0]),
        num_draft_tokens=0,
    )

    def grammar_output():
        bitmask = xgr.allocate_token_bitmask(1, len(vocab))
        matcher.fill_next_token_bitmask(bitmask, 0)
        return SimpleNamespace(
            structured_output_request_ids=["structured-request"],
            grammar_bitmask=bitmask.numpy(),
        )

    GPUModelRunner.sample(runner, torch.empty(0), input_batch, grammar_output())
    assert matcher.accept_token(sampled_steps[-1][0])
    GPUModelRunner.sample(runner, torch.empty(0), input_batch, grammar_output())
    assert matcher.accept_token(sampled_steps[-1][0])
    if constraint == "json":
        GPUModelRunner.sample(runner, torch.empty(0), input_batch, grammar_output())
        assert matcher.accept_token(sampled_steps[-1][0])

    expected_tokens = [2, 4, eos_token_id] if constraint == "json" else [2, 15]
    assert [token_id for token_id, _ in sampled_steps] == expected_tokens
    assert sampled_steps[0][1] < 0
    assert all(logprob == pytest.approx(0.0) for _, logprob in sampled_steps[1:])
    assert len(rapid_calls) == len(expected_tokens)
