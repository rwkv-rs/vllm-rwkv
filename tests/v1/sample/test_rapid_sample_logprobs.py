# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.worker.gpu.sample.logprob import compute_token_ranks


def _rapid_cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7


@pytest.mark.skipif(
    not _rapid_cuda_available(),
    reason="Rapid sampler requires CUDA compute capability >= 7.",
)
@pytest.mark.parametrize(("temperature", "top_k"), [(1.0, -1), (0.7, 3)])
def test_rapid_sample_returns_same_distribution_logprob_at_top_p_tie(
    temperature: float,
    top_k: int,
):
    batch_size = 512
    probs = torch.tensor([0.4, 0.2, 0.2, 0.2], dtype=torch.float32, device="cuda")
    logits = probs.log().expand(batch_size, -1).contiguous()

    samples, sampled_logprobs = topk_topp_sampler.rapid_sample(
        logits,
        top_k,
        0.5,
        temperatures=temperature,
        return_logprobs=True,
    )

    transformed = probs.pow(1.0 / temperature)
    transformed /= transformed.sum()
    threshold = transformed[1]
    greater_mass = transformed[0]
    tie_compensation = min(
        float((0.5 - greater_mass) / (threshold * 3)),
        float((top_k - 1) / 3) if top_k > 0 else 1.0,
    )
    retained = transformed.clone()
    retained[1:] *= tie_compensation
    retained /= retained.sum()
    expected = retained[samples.long()].log()
    ranks = compute_token_ranks(logits, samples.to(torch.int64))
    expected_ranks = torch.tensor(
        [int((probs >= probs[token_id]).sum()) for token_id in samples.cpu()],
        dtype=torch.int64,
        device="cuda",
    )

    assert torch.isfinite(sampled_logprobs).all()
    assert torch.any(samples != 0), "fixture did not exercise a threshold-tie token"
    assert torch.allclose(sampled_logprobs, expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(ranks, expected_ranks)


@pytest.mark.skipif(
    not _rapid_cuda_available(),
    reason="Rapid sampler requires CUDA compute capability >= 7.",
)
def test_rapid_penalty_kernels_return_same_distribution_logprob():
    batch_size = 512
    probs = torch.tensor([0.4, 0.2, 0.2, 0.2], dtype=torch.float32, device="cuda")
    logits = probs.log().expand(batch_size, -1).contiguous()
    contiguous_penalties = torch.zeros_like(logits)
    indexed_penalties = torch.zeros_like(logits)
    penalty_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    kwargs = {
        "temperatures": 0.7,
        "presence_penalties": 0.0,
        "repetition_penalties": 0.0,
        "penalty_decays": 1.0,
        "return_logprobs": True,
    }

    contiguous, contiguous_logprobs = topk_topp_sampler.rapid_sample(
        logits, 3, 0.5, penalties=contiguous_penalties, **kwargs
    )
    indexed, indexed_logprobs = topk_topp_sampler.rapid_sample(
        logits,
        3,
        0.5,
        penalties=indexed_penalties,
        penalty_indices=penalty_indices,
        **kwargs,
    )

    transformed = probs.pow(1.0 / 0.7)
    transformed /= transformed.sum()
    threshold = transformed[1]
    tie_compensation = float((0.5 - transformed[0]) / (threshold * 3))
    retained = transformed.clone()
    retained[1:] *= tie_compensation
    retained /= retained.sum()

    assert torch.any(contiguous != 0), "fixture did not exercise a threshold-tie token"
    assert torch.any(indexed != 0), "fixture did not exercise an indexed tie token"
    assert torch.allclose(
        contiguous_logprobs, retained[contiguous.long()].log(), atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        indexed_logprobs, retained[indexed.long()].log(), atol=2e-5, rtol=2e-5
    )
