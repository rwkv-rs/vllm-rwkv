# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import secrets
from collections.abc import Sequence
from types import ModuleType
from typing import Literal

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sampling_params import RAPID_PENALTY_DECAY_DEFAULT

logger = init_logger(__name__)

_RAPID_SAMPLER_MODULE: ModuleType | Literal[False] | None = None
_RAPID_SAMPLER_STATES: dict[tuple[int, int], torch.Tensor] = {}


def _load_rapid_sampler_module() -> ModuleType:
    global _RAPID_SAMPLER_MODULE
    if _RAPID_SAMPLER_MODULE is None:
        try:
            _RAPID_SAMPLER_MODULE = importlib.import_module("vllm._rapid_sampling")
        except ImportError:
            _RAPID_SAMPLER_MODULE = False
    if _RAPID_SAMPLER_MODULE is False:
        raise ImportError("vllm._rapid_sampling is not installed")
    return _RAPID_SAMPLER_MODULE


def rapid_sampler_supported() -> bool:
    """Decide whether the rapid-sampling CUDA sampler can be used."""
    if not current_platform.is_cuda():
        return False
    if not envs.VLLM_USE_RAPID_SAMPLER:
        return False

    capability = current_platform.get_device_capability()
    if capability is None or capability.major < 7:
        unsupported_reason = (
            "missing CUDA capability"
            if capability is None
            else f"unsupported compute capability {capability.as_version_str()}"
        )
        message = f"Rapid top-p/top-k sampling unavailable: {unsupported_reason}."
        if envs.is_set("VLLM_USE_RAPID_SAMPLER"):
            raise RuntimeError(
                f"{message} Unset VLLM_USE_RAPID_SAMPLER=1 to use another sampler."
            )
        logger.warning_once("%s Falling back to the native sampler.", message)
        return False

    try:
        _load_rapid_sampler_module()
    except ImportError as exc:
        message = "Rapid top-p/top-k sampling native extension is unavailable."
        if envs.is_set("VLLM_USE_RAPID_SAMPLER"):
            raise RuntimeError(
                f"{message} Build vllm._rapid_sampling or unset "
                "VLLM_USE_RAPID_SAMPLER=1."
            ) from exc
        logger.warning_once(
            "%s Falling back to the native sampler.",
            message,
        )
        return False

    logger.info_once("Using rapid-sampling for top-p & top-k sampling.")
    return True


def rapid_sample_input_supported(logits: torch.Tensor) -> bool:
    vocab_size = logits.shape[-1]
    return (
        logits.is_cuda
        and logits.dtype == torch.float32
        and vocab_size > 0
        and vocab_size <= 1048576
        and vocab_size % 4 == 0
    )


def scalar_parameter(value: torch.Tensor | int | float | None, default):
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    if value.numel() != 1:
        return None
    return value.reshape(-1)[0].item()


def _rapid_vector(
    value: torch.Tensor | int | float | None,
    batch_size: int,
    default,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), default, dtype=dtype, device=device)
    if isinstance(value, (int, float)):
        return torch.full((batch_size,), value, dtype=dtype, device=device)
    tensor = value
    if tensor.numel() == 1 and batch_size != 1:
        tensor = tensor.expand(batch_size)
    assert tensor.shape == (batch_size,), (
        f"rapid sampler parameter must have shape ({batch_size},), "
        f"got {tuple(tensor.shape)}"
    )
    return tensor.to(device=device, dtype=dtype).contiguous()


def _rapid_states(module, logits: torch.Tensor) -> torch.Tensor:
    batch_size = logits.shape[0] if logits.dim() >= 2 else 1
    device_idx = logits.device.index
    if device_idx is None:
        device_idx = torch.accelerator.current_device_index()
    key = (device_idx, batch_size)
    states = _RAPID_SAMPLER_STATES.get(key)
    if states is None or states.device != logits.device:
        seed = secrets.randbits(63)
        with torch.accelerator.device_index(device_idx):
            states = module.setup_rand(seed, batch_size)
        _RAPID_SAMPLER_STATES[key] = states
    return states


def _format_rapid_sample_result(
    result: Sequence[torch.Tensor],
    *,
    return_logprobs: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    sampled = result[0].view(-1)
    if return_logprobs:
        return sampled, result[1].view(-1)
    return sampled


def rapid_sample(
    logits: torch.Tensor,
    k: torch.Tensor | int | None,
    p: torch.Tensor | float | None,
    temperatures: torch.Tensor | float | None = None,
    penalties: torch.Tensor | None = None,
    presence_penalties: torch.Tensor | None = None,
    frequency_penalties: torch.Tensor | None = None,
    penalty_decays: torch.Tensor | None = None,
    penalty_indices: torch.Tensor | None = None,
    return_logprobs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sample with rapid CUDA and optionally return sampled-token logprobs.

    The optional logprob comes from the exact temperature/top-k/top-p support
    used by the sampling kernel. Recomputing it with a separate top-p
    implementation can exclude a sampled token at a threshold tie.
    """
    assert rapid_sample_input_supported(logits)

    logits = logits.contiguous()
    batch_size = logits.shape[0] if logits.dim() >= 2 else 1
    vocab_size = logits.shape[-1]
    if penalties is None:
        scalar_temperature = scalar_parameter(temperatures, 1.0)
        scalar_top_k = scalar_parameter(k, vocab_size)
        scalar_top_p = scalar_parameter(p, 1.0)
        if scalar_temperature is None or scalar_top_k is None or scalar_top_p is None:
            raise RuntimeError(
                "rapid-sampling without penalties only supports uniform scalar "
                "temperature/top_k/top_p. Use the native sampler for mixed "
                "per-request sampling parameters."
            )
        module = _load_rapid_sampler_module()
        states = _rapid_states(module, logits)
        return _format_rapid_sample_result(
            module.batch_sampling_temperature_topk_topp(
                logits,
                states,
                float(scalar_temperature),
                int(scalar_top_k),
                float(scalar_top_p),
                return_logprobs,
            ),
            return_logprobs=return_logprobs,
        )

    module = _load_rapid_sampler_module()
    states = _rapid_states(module, logits)

    assert penalties.device == logits.device and penalties.dtype == torch.float32
    assert (
        penalties.is_contiguous()
        and penalties.dim() == 2
        and penalties.shape[1] == vocab_size
    )
    scalar_temperature = scalar_parameter(temperatures, 1.0)
    scalar_top_k = scalar_parameter(k, vocab_size)
    scalar_top_p = scalar_parameter(p, 1.0)
    scalar_presence_penalty = scalar_parameter(presence_penalties, 0.0)
    scalar_repetition_penalty = scalar_parameter(frequency_penalties, 0.0)
    scalar_penalty_decay = scalar_parameter(penalty_decays, RAPID_PENALTY_DECAY_DEFAULT)
    scalar_params = (
        scalar_temperature,
        scalar_top_k,
        scalar_top_p,
        scalar_presence_penalty,
        scalar_repetition_penalty,
        scalar_penalty_decay,
    )
    per_request = any(value is None for value in scalar_params)
    if per_request and penalty_indices is None:
        penalty_indices = torch.arange(
            batch_size, dtype=torch.int32, device=logits.device
        )

    if penalty_indices is not None:
        penalty_indices = _rapid_vector(
            penalty_indices, batch_size, 0, torch.int32, logits.device
        )
        if per_request:
            return _format_rapid_sample_result(
                module.batch_sampling_repetition_temperature_topk_topp_per_request_indexed(
                    logits,
                    penalties,
                    penalty_indices,
                    states,
                    _rapid_vector(
                        presence_penalties,
                        batch_size,
                        0.0,
                        torch.float32,
                        logits.device,
                    ),
                    _rapid_vector(
                        frequency_penalties,
                        batch_size,
                        0.0,
                        torch.float32,
                        logits.device,
                    ),
                    _rapid_vector(
                        penalty_decays,
                        batch_size,
                        RAPID_PENALTY_DECAY_DEFAULT,
                        torch.float32,
                        logits.device,
                    ),
                    _rapid_vector(
                        temperatures,
                        batch_size,
                        1.0,
                        torch.float32,
                        logits.device,
                    ),
                    _rapid_vector(
                        k,
                        batch_size,
                        vocab_size,
                        torch.int32,
                        logits.device,
                    ),
                    _rapid_vector(
                        p,
                        batch_size,
                        1.0,
                        torch.float32,
                        logits.device,
                    ),
                    return_logprobs,
                ),
                return_logprobs=return_logprobs,
            )
        indexed_sampler = getattr(
            module,
            "batch_sampling_repetition_temperature_topk_topp_indexed",
            None,
        )
        if indexed_sampler is None:
            raise RuntimeError(
                "rapid-sampling indexed penalty kernel is unavailable; "
                "refusing the legacy gather/scatter path."
            )
        return _format_rapid_sample_result(
            indexed_sampler(
                logits,
                penalties,
                penalty_indices,
                states,
                float(scalar_presence_penalty),
                float(scalar_repetition_penalty),
                float(scalar_penalty_decay),
                float(scalar_temperature),
                int(scalar_top_k),
                float(scalar_top_p),
                return_logprobs,
            ),
            return_logprobs=return_logprobs,
        )

    assert penalties.shape[0] == batch_size

    return _format_rapid_sample_result(
        module.batch_sampling_repetition_temperature_topk_topp(
            logits,
            penalties,
            states,
            float(scalar_presence_penalty),
            float(scalar_repetition_penalty),
            float(scalar_penalty_decay),
            float(scalar_temperature),
            int(scalar_top_k),
            float(scalar_top_p),
            return_logprobs,
        ),
        return_logprobs=return_logprobs,
    )
