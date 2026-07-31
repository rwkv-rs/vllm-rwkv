# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass
class UniformDecodeWavePlan:
    pending_decode_request_ids: set[str] = field(default_factory=set)
    block_decode: bool = False

    def allows_running_request(self, request: Request) -> bool:
        if self._is_live_decode_request(request):
            return (
                not self.block_decode
                and request.request_id in self.pending_decode_request_ids
            )
        return not self.pending_decode_request_ids

    def on_running_request_scheduled(
        self,
        request_or_request_id: Request | str,
    ) -> bool:
        request_id = self._request_id(request_or_request_id)
        if request_id not in self.pending_decode_request_ids:
            return False

        self.pending_decode_request_ids.remove(request_id)
        return not self.pending_decode_request_ids

    @staticmethod
    def _is_live_decode_request(request: Request) -> bool:
        return not request.is_prefill_chunk

    @staticmethod
    def _request_id(request_or_request_id: Request | str) -> str:
        if isinstance(request_or_request_id, str):
            return request_or_request_id
        return request_or_request_id.request_id


class UniformDecodeWavePolicy:
    @staticmethod
    def make_plan(
        *,
        running_requests: Sequence[Request],
        token_budget: int,
        current_step: int,
        max_model_len: int,
        num_sampled_tokens_per_step: int,
    ) -> UniformDecodeWavePlan:
        decode_wave: list[Request] = []
        has_lower_running_prefill = False

        for request in running_requests:
            if request.is_prefill_chunk:
                if not decode_wave:
                    has_lower_running_prefill = True
                continue
            if has_lower_running_prefill:
                return UniformDecodeWavePlan(block_decode=True)
            if current_step < request.next_decode_eligible_step:
                return UniformDecodeWavePlan(block_decode=True)
            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                return UniformDecodeWavePlan(block_decode=True)

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new_tokens = min(
                num_new_tokens,
                max_model_len
                - request.num_computed_tokens
                - num_sampled_tokens_per_step,
            )
            if num_new_tokens == 1:
                decode_wave.append(request)
            elif num_new_tokens == 0:
                return UniformDecodeWavePlan(block_decode=True)
            elif num_new_tokens > 1:
                raise ValueError(
                    "Uniform decode wave only supports one-token decode requests; "
                    f"request {request.request_id!r} has "
                    f"{num_new_tokens} schedulable tokens."
                )

        if len(decode_wave) > token_budget:
            raise ValueError(
                "Uniform decode wave requires scheduling all ready decode requests "
                "in one step. Increase max_num_scheduled_tokens or "
                "max_num_batched_tokens."
            )

        return UniformDecodeWavePlan(
            pending_decode_request_ids={request.request_id for request in decode_wave},
        )


__all__ = ["UniformDecodeWavePlan", "UniformDecodeWavePolicy"]
