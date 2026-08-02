# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import PROCESSED_LOGPROBS_MODES, LogprobsMode
from vllm.sampling_params import RAPID_PENALTY_DECAY_DEFAULT, SamplingParams
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
    rapid_sample,
    rapid_sample_input_supported,
    rapid_sampler_supported,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, get_num_sampled_and_rejected
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import (
    LogprobTokenIdsState,
    compute_token_ranks,
    compute_topk_scores,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates
from vllm.v1.worker.gpu.states import RequestState


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
        use_fp64_gumbel: bool = False,
    ):
        self.logprobs_mode = logprobs_mode
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.
        self.use_fp64_gumbel = use_fp64_gumbel

        self.req_states = req_states
        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(req_states)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.bad_words_state = BadWordsState(req_states)
        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)
        self.num_speculative_tokens = num_speculative_tokens
        self.use_rapid = rapid_sampler_supported()
        self.use_flashinfer = not self.use_rapid and flashinfer_sampler_supported()
        self.require_rapid = False
        self.rapid_penalties: torch.Tensor | None = None
        self.rapid_penalty_native_fallback = np.zeros(max_num_reqs, dtype=bool)
        self.request_requires_rapid = np.zeros(max_num_reqs, dtype=bool)

    def validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        """Reject request-scoped rapid incompatibilities before state mutation."""
        rapid_explicitly_enabled = (
            envs.is_set("VLLM_USE_RAPID_SAMPLER") and envs.VLLM_USE_RAPID_SAMPLER
        )
        rapid_required = (
            self.require_rapid
            or rapid_explicitly_enabled
            or (self.use_rapid and sampling_params.structured_outputs is not None)
        )
        if not self.use_rapid:
            if self.require_rapid:
                raise RuntimeError("RWKV7 requires rapid-sampling on CUDA.")
            if sampling_params.penalty_decay != RAPID_PENALTY_DECAY_DEFAULT:
                raise RuntimeError(
                    "penalty_decay is only supported when rapid-sampling is enabled."
                )
            return
        if not rapid_required:
            return

        if sampling_params.temperature == 0.0:
            raise RuntimeError(
                "rapid-sampling does not support greedy requests. "
                "Set temperature to a positive value."
            )
        if sampling_params.seed is not None:
            raise RuntimeError(
                "rapid-sampling does not support per-request seeds. Remove seed."
            )
        if sampling_params.min_p != 0.0:
            raise RuntimeError("rapid-sampling does not support min_p. Set min_p=0.0.")
        if sampling_params.repetition_penalty != 1.0:
            raise RuntimeError(
                "rapid-sampling does not support repetition_penalty. "
                "Set repetition_penalty=1.0."
            )

        num_logprobs = sampling_params.num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == "processed_logits":
                raise RuntimeError(
                    "rapid-sampling cannot return exact processed_logits. "
                    "Use processed_logprobs with logprobs=0 for the exact "
                    "sampled-token distribution."
                )
            if self.logprobs_mode == "processed_logprobs" and num_logprobs != 0:
                raise RuntimeError(
                    "rapid-sampling only supports the sampled token for exact "
                    "processed_logprobs. Set logprobs=0 and do not request "
                    "logprob_token_ids."
                )

        vocab_size = self.req_states.vocab_size
        if vocab_size <= 0 or vocab_size > 1048576 or vocab_size % 4 != 0:
            raise RuntimeError(
                "rapid-sampling requires vocab size in (0, 1048576] and divisible by 4."
            )
        rapid_penalty_active = (
            sampling_params.presence_penalty != 0.0
            or sampling_params.frequency_penalty != 0.0
            or sampling_params.penalty_decay != RAPID_PENALTY_DECAY_DEFAULT
        )
        if rapid_penalty_active and self.num_speculative_tokens > 1:
            raise RuntimeError(
                "rapid-sampling penalties do not support speculative expanded "
                "logits. Disable speculative decoding."
            )

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        self.validate_sampling_params(sampling_params)
        self.rapid_penalty_native_fallback[req_idx] = False
        self.request_requires_rapid[req_idx] = bool(
            self.require_rapid
            or (envs.is_set("VLLM_USE_RAPID_SAMPLER") and envs.VLLM_USE_RAPID_SAMPLER)
            or (self.use_rapid and sampling_params.structured_outputs is not None)
        )
        use_rapid_penalty = self.use_rapid and (
            sampling_params.presence_penalty != 0.0
            or sampling_params.frequency_penalty != 0.0
            or sampling_params.penalty_decay != RAPID_PENALTY_DECAY_DEFAULT
        )
        if self.use_rapid and (self.rapid_penalties is not None or use_rapid_penalty):
            self._ensure_rapid_penalties()[req_idx].zero_()
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)
        self.bad_words_state.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

    def _ensure_rapid_penalties(self) -> torch.Tensor:
        if self.rapid_penalties is None:
            self.rapid_penalties = torch.zeros(
                (
                    self.req_states.max_num_reqs,
                    self.req_states.vocab_size,
                ),
                dtype=torch.float32,
                device=self.req_states.device,
            )
        return self.rapid_penalties

    def _mark_rapid_penalty_native_fallback(self, idx_mapping_np: np.ndarray) -> None:
        rapid_penalty_mask = self.penalties_state.rapid_penalty_mask(idx_mapping_np)
        self.rapid_penalty_native_fallback[idx_mapping_np[rapid_penalty_mask]] = True

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes()
        self.logit_bias_state.apply_staged_writes()
        self.bad_words_state.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        expanded_idx_mapping = input_batch.expanded_idx_mapping
        idx_mapping_np = input_batch.idx_mapping_np
        cu_num_logits_np = input_batch.cu_num_logits_np
        expanded_local_pos = input_batch.expanded_local_pos
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]

        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None

        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        max_per_req_token_ids = self.logprob_token_ids_state.max_num_token_ids(
            idx_mapping_np
        )
        return_logprobs = max_num_logprobs != NO_LOGPROBS or max_per_req_token_ids > 0
        sampled_only_logprobs = max_num_logprobs == 0 and max_per_req_token_ids == 0

        (
            sampled,
            processed_logits,
            rapid_sampled_logprobs,
            rapid_sampled_only_fast_path,
        ) = self.sample(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            return_logprobs=return_logprobs,
            sampled_only_logprobs=sampled_only_logprobs,
        )

        if return_logprobs:
            if self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
                logits = processed_logits
            expanded_logits = logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            if rapid_sampled_only_fast_path:
                assert rapid_sampled_logprobs is not None
                logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=sampled.unsqueeze(-1),
                    logprobs=rapid_sampled_logprobs.unsqueeze(-1),
                    selected_token_ranks=compute_token_ranks(
                        processed_logits,
                        sampled,
                    ),
                    cu_num_generated_tokens=cu_num_logits,
                )
            else:
                if self.logprobs_mode in ("processed_logprobs", "processed_logits"):
                    logits = processed_logits
                num_logprobs = (
                    max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
                )
                logprobs_tensors = compute_topk_scores(
                    logits,
                    num_logprobs,
                    sampled,
                    cu_num_logits,
                    logprob_token_ids_state=self.logprob_token_ids_state,
                    expanded_idx_mapping=input_batch.expanded_idx_mapping,
                    max_per_req_token_ids=max_per_req_token_ids,
                    logits_mode=self.logprobs_mode
                    in ("raw_logits", "processed_logits"),
                )
            if rapid_sampled_logprobs is not None and not rapid_sampled_only_fast_path:
                assert (
                    rapid_sampled_logprobs.shape
                    == logprobs_tensors.logprobs[:, 0].shape
                )
                logprobs_tensors.logprobs[:, 0].copy_(rapid_sampled_logprobs)
        else:
            logprobs_tensors = None

        # 1 sampled token per request, except chunked-prefill requests
        # (seq_len < prefill_len) which aren't done prefilling and produce no
        # output token. num_rejected is always 0 here (one logit per request).
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            input_batch.seq_lens.new_ones(input_batch.num_reqs),
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sampler_output

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
        skip_penalties: bool = False,
        skip_temperature: bool = False,
        skip_min_p: bool = False,
    ) -> torch.Tensor:
        if not self._requires_logits_processing(idx_mapping_np):
            return logits

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(
            logits, expanded_idx_mapping, idx_mapping_np, pos
        )

        # Apply penalties in place.
        if not skip_penalties:
            self.penalties_state.apply_penalties(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                input_ids,
                expanded_local_pos,
            )

        # Apply bad words masking in place.
        self.bad_words_state.apply_bad_words(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
        )

        # Apply temperature in place.
        if not skip_temperature:
            self.sampling_states.apply_temperature(
                logits, expanded_idx_mapping, idx_mapping_np
            )

        # Apply min_p in place.
        if not skip_min_p:
            self.sampling_states.apply_min_p(
                logits, expanded_idx_mapping, idx_mapping_np
            )

        if skip_top_k_top_p:
            return logits

        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )

    def _requires_logits_processing(self, idx_mapping_np: np.ndarray) -> bool:
        if np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np]):
            return True
        if np.any(self.penalties_state.use_penalty[idx_mapping_np]):
            return True
        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):
            return True

        states = self.sampling_states
        temperatures = states.temperature.np[idx_mapping_np]
        if np.any((temperatures != 0.0) & (temperatures != 1.0)):
            return True
        if np.any(states.min_p.np[idx_mapping_np] != 0.0):
            return True
        if np.any(states.top_k.np[idx_mapping_np] != states.vocab_size):
            return True
        return bool(np.any(states.top_p.np[idx_mapping_np] != 1.0))

    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
        sampled_only_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        rapid_penalty_native_fallback = bool(
            np.any(self.rapid_penalty_native_fallback[idx_mapping_np])
        )
        use_rapid = self.use_rapid and not rapid_penalty_native_fallback
        rapid_sampler_forced = (
            self.require_rapid
            or (envs.is_set("VLLM_USE_RAPID_SAMPLER") and envs.VLLM_USE_RAPID_SAMPLER)
            or bool(np.any(self.request_requires_rapid[idx_mapping_np]))
        )
        if rapid_penalty_native_fallback and rapid_sampler_forced:
            raise RuntimeError(
                "rapid-sampling cannot resume a request after it falls back to "
                "native sampling with frequency or presence penalties."
            )
        if self.require_rapid and not use_rapid:
            raise RuntimeError("RWKV7 requires rapid-sampling on CUDA.")
        needs_processed_logprobs = (
            return_logprobs and self.logprobs_mode == "processed_logprobs"
        )
        rapid_sampled_logprobs = None
        rapid_sampled_only_fast_path = False

        def use_native_sampling_params():
            native_processed_logits = self.apply_sampling_params(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
            )
            native_top_k, native_top_p = self.sampling_states.get_top_k_top_p(
                expanded_idx_mapping, idx_mapping_np
            )
            return native_processed_logits, native_top_k, native_top_p

        processed_logits = self.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
            skip_penalties=use_rapid,
            skip_temperature=use_rapid,
            skip_min_p=use_rapid,
        )
        top_k, top_p = self.sampling_states.get_top_k_top_p(
            expanded_idx_mapping,
            idx_mapping_np,
            scalar_if_uniform=use_rapid,
        )
        rapid_penalty_active = False
        rapid_per_request_params = False
        rapid_penalty_params: (
            tuple[torch.Tensor | float, torch.Tensor | float, torch.Tensor | float]
            | None
        ) = None
        temperatures = None
        if use_rapid:
            rapid_penalty_mask = self.penalties_state.rapid_penalty_mask(idx_mapping_np)
            rapid_penalty_active = bool(np.any(rapid_penalty_mask))
            temperatures = self.sampling_states.get_temperatures(
                expanded_idx_mapping,
                idx_mapping_np,
                scalar_if_uniform=True,
            )
            if rapid_penalty_active:
                rapid_penalty_params = self.penalties_state.rapid_penalty_params(
                    expanded_idx_mapping,
                    idx_mapping_np,
                    scalar_if_uniform=True,
                )
            rapid_sampler_params: tuple[object, ...] = (temperatures, top_k, top_p)
            if rapid_penalty_params is not None:
                rapid_sampler_params += rapid_penalty_params
            rapid_per_request_params = any(
                isinstance(value, torch.Tensor) for value in rapid_sampler_params
            )
        use_flashinfer = self.use_flashinfer and not (
            # Don't use FI sampler if no requests use top_k/top_p, if there are
            # any greedy requests or per-request seeds, or if post-processed
            # logprobs need to be returned for any requests.
            (top_k is None and top_p is None)
            or (return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES)
            or self.sampling_states.any_greedy(idx_mapping_np)
            or self.sampling_states.any_explicit_seed(idx_mapping_np)
        )
        if use_rapid:
            # The rapid CUDA kernel consumes FP32 logits. Promote models that
            # emit FP16 or BF16 only at this sampling boundary; both source
            # formats are represented exactly in FP32.
            processed_logits = processed_logits.to(torch.float32)

            rapid_incompatibility = None
            if self.sampling_states.any_greedy(idx_mapping_np):
                rapid_incompatibility = (
                    "rapid-sampling does not support greedy requests. Set "
                    "VLLM_USE_RAPID_SAMPLER=0 to use the native greedy path."
                )
            elif self.sampling_states.any_explicit_seed(idx_mapping_np):
                rapid_incompatibility = (
                    "rapid-sampling does not support per-request seeds. Set "
                    "VLLM_USE_RAPID_SAMPLER=0 to use the native seeded path."
                )
            elif self.sampling_states.any_min_p(idx_mapping_np):
                rapid_incompatibility = (
                    "rapid-sampling does not support min_p. Set min_p=0.0."
                )
            elif not rapid_sample_input_supported(processed_logits):
                rapid_incompatibility = (
                    "rapid-sampling requires CUDA float32 logits with vocab size "
                    "in (0, 1048576] and divisible by 4. Set "
                    "VLLM_USE_RAPID_SAMPLER=0 to use another sampler."
                )
            elif self.penalties_state.any_repetition_penalty(idx_mapping_np):
                rapid_incompatibility = (
                    "rapid-sampling does not support repetition_penalty. "
                    "Set repetition_penalty=1.0."
                )
            elif (
                rapid_penalty_active
                and expanded_idx_mapping.shape[0] != idx_mapping_np.shape[0]
            ):
                rapid_incompatibility = (
                    "rapid-sampling penalties do not support speculative "
                    "expanded logits. Disable speculative decoding."
                )

            if rapid_incompatibility is not None:
                if rapid_sampler_forced:
                    raise RuntimeError(rapid_incompatibility)
                if rapid_penalty_active:
                    # The rapid kernel owns an accumulated penalty buffer. Once
                    # native sampling advances a request, that buffer cannot
                    # reconstruct its decayed history, so keep the affected
                    # requests on the native path for their remaining lifetime.
                    self._mark_rapid_penalty_native_fallback(idx_mapping_np)
                use_rapid = False
                rapid_penalty_active = False
                rapid_per_request_params = False
                rapid_penalty_params = None
                processed_logits, top_k, top_p = use_native_sampling_params()

        # Sample the next token.
        if use_rapid:
            assert temperatures is not None
            if rapid_penalty_active or rapid_per_request_params:
                if expanded_idx_mapping.shape[0] != idx_mapping_np.shape[0]:
                    raise RuntimeError(
                        "rapid-sampling penalties do not support speculative "
                        "expanded logits. Disable speculative decoding."
                    )
                rapid_penalties = self._ensure_rapid_penalties()
                if rapid_penalty_params is None:
                    presence_penalties = 0.0
                    frequency_penalties = 0.0
                    penalty_decays = RAPID_PENALTY_DECAY_DEFAULT
                else:
                    presence_penalties, frequency_penalties, penalty_decays = (
                        rapid_penalty_params
                    )
                rapid_result = rapid_sample(
                    processed_logits,
                    top_k,
                    top_p,
                    temperatures=temperatures,
                    penalties=rapid_penalties,
                    presence_penalties=presence_penalties,
                    frequency_penalties=frequency_penalties,
                    penalty_decays=penalty_decays,
                    penalty_indices=expanded_idx_mapping,
                    return_logprobs=needs_processed_logprobs,
                )
            else:
                rapid_result = rapid_sample(
                    processed_logits,
                    top_k,
                    top_p,
                    temperatures=temperatures,
                    return_logprobs=needs_processed_logprobs,
                )
            if needs_processed_logprobs:
                sampled, rapid_sampled_logprobs = rapid_result
            else:
                sampled = rapid_result
            sampled = sampled.to(torch.int64)
            if needs_processed_logprobs:
                rapid_sampled_only_fast_path = (
                    sampled_only_logprobs and not rapid_penalty_active
                )
                if not rapid_sampled_only_fast_path:
                    processed_logits = self.apply_sampling_params(
                        logits,
                        expanded_idx_mapping,
                        idx_mapping_np,
                        pos,
                        input_ids,
                        expanded_local_pos,
                    )
        elif use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
            sampled = gumbel_sample(
                processed_logits,
                expanded_idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                use_fp64=self.use_fp64_gumbel,
            )
        return (
            sampled,
            processed_logits,
            rapid_sampled_logprobs,
            rapid_sampled_only_fast_path,
        )
