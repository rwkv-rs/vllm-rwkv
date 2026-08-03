# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contracts for the canonical packed-varlen RWKV7 WKV operators."""

import pytest
import torch

from vllm.model_executor.models.rwkv7_wkv_backend import (
    run_fla_rwkv7_recurrent_from_decay_logits,
)

HEAD_SIZE = 64
TWO_NEG_41 = 4.547473508864641e-13
NEXP_HALF_LOG2_E = -0.8750387749145276
NLOG2_E = -1.4426950408889634
ROT1 = 2654435769


@pytest.fixture(scope="module", autouse=True)
def rwkv7_ops_registered() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 custom op registration")
    try:
        import vllm.rwkv7_ops  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"RWKV7 extension is unavailable: {exc!r}")
    import vllm._custom_ops  # noqa: F401


def _case(
    lengths: tuple[int, ...],
    *,
    state_dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    batch_size = len(lengths)
    total_tokens = sum(lengths)
    slots = batch_size + 3
    query_start_loc = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    slot_indices = torch.arange(
        batch_size - 1,
        -1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    state = 0.02 * torch.randn(
        (slots, 1, HEAD_SIZE, HEAD_SIZE),
        device="cuda",
        dtype=state_dtype,
    )
    payload = tuple(
        0.02
        * torch.randn(
            (total_tokens, HEAD_SIZE),
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(6)
    )
    w0 = -2.0 + 0.02 * torch.randn(
        (HEAD_SIZE,),
        device="cuda",
        dtype=torch.float16,
    )
    elapsed = torch.arange(slots, device="cuda", dtype=torch.int32)
    return query_start_loc, slot_indices, state, *payload, w0, elapsed


def _rotator(phase: torch.Tensor) -> torch.Tensor:
    bits = (phase.to(torch.int64) * ROT1) & 0xFFFFFFFF
    signed = torch.where(bits >= 0x80000000, bits - 0x100000000, bits)
    return signed.float() * TWO_NEG_41


def _fp16_reference(
    query_start_loc: torch.Tensor,
    slot_indices: torch.Tensor,
    initial_state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w0: torch.Tensor,
    elapsed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    output = torch.empty_like(r)
    offsets = query_start_loc.cpu().tolist()
    slots = slot_indices.cpu().tolist()
    for request, slot in enumerate(slots):
        current = state[slot, 0].float()
        phase0 = int(elapsed[slot].item())
        for offset, token in enumerate(range(offsets[request], offsets[request + 1])):
            phase = torch.arange(
                phase0 + offset,
                phase0 + offset + HEAD_SIZE,
                device="cuda",
                dtype=torch.int64,
            )
            decay = (
                (
                    torch.exp2(
                        NEXP_HALF_LOG2_E
                        / (1.0 + torch.exp2(NLOG2_E * (w[token].float() + w0.float())))
                    )
                    - 1.0
                    + _rotator(phase)
                )
                .half()
                .float()
            )
            state_dot_a = (current * a[token].float()).sum(dim=1, keepdim=True)
            current = (
                (
                    current * (1.0 + decay)
                    + state_dot_a * b[token].float()
                    + v[token].float().unsqueeze(1) * k[token].float()
                )
                .half()
                .float()
            )
            output[token] = (current * r[token].float()).sum(dim=1).half()
        state[slot, 0] = current.to(state.dtype)
    return output, state


def _fp32_reference(
    query_start_loc: torch.Tensor,
    slot_indices: torch.Tensor,
    initial_state: torch.Tensor,
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state.clone()
    output = torch.empty_like(r)
    offsets = query_start_loc.cpu().tolist()
    slots = slot_indices.cpu().tolist()
    for request, slot in enumerate(slots):
        current = state[slot, 0]
        for token in range(offsets[request], offsets[request + 1]):
            decay = torch.exp2(
                NEXP_HALF_LOG2_E / (1.0 + torch.exp2(NLOG2_E * w[token].float()))
            )
            state_dot_a = (current * a[token].float()).sum(dim=1, keepdim=True)
            current = (
                current * decay
                + state_dot_a * b[token].float()
                + v[token].float().unsqueeze(1) * k[token].float()
            )
            output[token] = (current * r[token].float()).sum(dim=1).half()
        state[slot, 0] = current
    return output, state


def test_wkv_surface_contains_only_canonical_ops() -> None:
    assert hasattr(torch.ops.rwkv7_wkv_fp16_v2, "wkv")
    assert hasattr(torch.ops.rwkv7_wkv_fp32_v2, "wkv")
    for old_name in (
        "wkv_seq",
        "wkv_seq_slot",
        "wkv_seq_w0",
        "wkv_seq_w0_slot",
        "wkv_seq_grid2d",
        "wkv_seq_forced",
        "wkv_seq_varlen",
        "wkv_seq_w0_varlen",
    ):
        assert not hasattr(torch.ops.rwkv7_wkv_fp16_v2, old_name)
    for old_name in ("forward", "forward_slot", "forward_varlen"):
        assert not hasattr(torch.ops.rwkv7_wkv_fp32_v2, old_name)


def test_wkv_fp16_canonical_schema_and_fake_contract() -> None:
    query_start_loc, slot_indices, state, r, w, k, v, a, b, w0, elapsed = _case(
        (2, 3), state_dtype=torch.float16, seed=1
    )
    output = torch.empty_like(r)
    torch.library.opcheck(
        torch.ops.rwkv7_wkv_fp16_v2.wkv,
        (
            query_start_loc,
            slot_indices,
            state,
            r,
            w,
            w0,
            k,
            v,
            a,
            b,
            output,
            elapsed,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_wkv_fp32_canonical_schema_and_fake_contract() -> None:
    query_start_loc, slot_indices, state, r, w, k, v, a, b, _w0, _elapsed = _case(
        (2, 3), state_dtype=torch.float32, seed=2
    )
    output = torch.empty_like(r)
    torch.library.opcheck(
        torch.ops.rwkv7_wkv_fp32_v2.wkv,
        (
            query_start_loc,
            slot_indices,
            state,
            r,
            w,
            k,
            v,
            a,
            b,
            output,
        ),
        test_utils=("test_schema", "test_faketensor"),
    )


@pytest.mark.parametrize("lengths", [(1,), (2, 3), (1, 4, 2)])
def test_wkv_fp16_matches_float_reference(lengths: tuple[int, ...]) -> None:
    query_start_loc, slot_indices, state, r, w, k, v, a, b, w0, elapsed = _case(
        lengths, state_dtype=torch.float16, seed=10 + sum(lengths)
    )
    initial_state = state.clone()
    output = torch.empty_like(r)
    torch.ops.rwkv7_wkv_fp16_v2.wkv(
        query_start_loc,
        slot_indices,
        state,
        r,
        w,
        w0,
        k,
        v,
        a,
        b,
        output,
        elapsed,
    )
    expected_output, expected_state = _fp16_reference(
        query_start_loc,
        slot_indices,
        initial_state,
        r,
        w,
        k,
        v,
        a,
        b,
        w0,
        elapsed,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, expected_output, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state, expected_state, atol=2e-3, rtol=2e-3)
    untouched = torch.ones(state.size(0), dtype=torch.bool, device="cuda")
    untouched[slot_indices.long()] = False
    assert torch.equal(state[untouched], initial_state[untouched])


def test_wkv_fp32_matches_float_reference() -> None:
    query_start_loc, slot_indices, state, r, w, k, v, a, b, w0, _elapsed = _case(
        (1, 4, 2), state_dtype=torch.float32, seed=31
    )
    w_raw = (w + w0).half()
    initial_state = state.clone()
    output = torch.empty_like(r)
    torch.ops.rwkv7_wkv_fp32_v2.wkv(
        query_start_loc,
        slot_indices,
        state,
        r,
        w_raw,
        k,
        v,
        a,
        b,
        output,
    )
    expected_output, expected_state = _fp32_reference(
        query_start_loc,
        slot_indices,
        initial_state,
        r,
        w_raw,
        k,
        v,
        a,
        b,
    )
    torch.accelerator.synchronize()

    torch.testing.assert_close(output, expected_output, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", ["fp16", "fp32io16"])
@pytest.mark.parametrize("lengths", [(1,), (2, 3), (1, 4, 2)])
def test_fla_fused_raw_decay_matches_legacy_varlen_state_semantics(
    mode: str,
    lengths: tuple[int, ...],
) -> None:
    pytest.importorskip("fla.ops.rwkv7")
    state_dtype = torch.float16 if mode == "fp16" else torch.float32
    (
        query_start_loc,
        slot_indices,
        legacy_state,
        r,
        w,
        k,
        v,
        a,
        b,
        w0,
        elapsed,
    ) = _case(lengths, state_dtype=state_dtype, seed=70 + sum(lengths))
    initial_legacy_state = legacy_state.clone()
    legacy_output = torch.empty_like(r)
    if mode == "fp16":
        torch.ops.rwkv7_wkv_fp16_v2.wkv(
            query_start_loc,
            slot_indices,
            legacy_state,
            r,
            w,
            w0,
            k,
            v,
            a,
            b,
            legacy_output,
            elapsed,
        )
    else:
        combined_decay_logits = torch.ops.rwkv7_fast_ops_fp16.add_vec(
            HEAD_SIZE,
            w,
            w0,
        )
        torch.ops.rwkv7_wkv_fp32_v2.wkv(
            query_start_loc,
            slot_indices,
            legacy_state,
            r,
            combined_decay_logits,
            k,
            v,
            a,
            b,
            legacy_output,
        )

    canonical_state = initial_legacy_state.transpose(-1, -2).contiguous()
    fused_output = run_fla_rwkv7_recurrent_from_decay_logits(
        r.view(1, -1, 1, HEAD_SIZE),
        w.view(1, -1, 1, HEAD_SIZE),
        k.view(1, -1, 1, HEAD_SIZE),
        v.view(1, -1, 1, HEAD_SIZE),
        a.view(1, -1, 1, HEAD_SIZE),
        b.view(1, -1, 1, HEAD_SIZE),
        decay_bias=w0.view(1, HEAD_SIZE),
        elapsed_t=elapsed if mode == "fp16" else None,
        state_pool=canonical_state,
        cu_seqlens=query_start_loc,
        state_indices=slot_indices,
        mode=mode,
    ).view_as(r)
    torch.accelerator.synchronize()

    expected_canonical_state = legacy_state.transpose(-1, -2).contiguous()
    output_tolerance = 3e-3 if mode == "fp16" else 2e-3
    state_tolerance = 3e-3 if mode == "fp16" else 2e-5
    torch.testing.assert_close(
        fused_output,
        legacy_output,
        atol=output_tolerance,
        rtol=output_tolerance,
    )
    torch.testing.assert_close(
        canonical_state,
        expected_canonical_state,
        atol=state_tolerance,
        rtol=state_tolerance,
    )
    untouched = torch.ones(canonical_state.size(0), dtype=torch.bool, device="cuda")
    untouched[slot_indices.long()] = False
    assert torch.equal(
        canonical_state[untouched],
        initial_legacy_state.transpose(-1, -2)[untouched],
    )


def test_wkv_fp16_is_repeatable_and_graph_capturable() -> None:
    query_start_loc, slot_indices, initial_state, r, w, k, v, a, b, w0, elapsed = _case(
        (1, 3, 2), state_dtype=torch.float16, seed=41
    )

    def run(state: torch.Tensor, output: torch.Tensor) -> None:
        torch.ops.rwkv7_wkv_fp16_v2.wkv(
            query_start_loc,
            slot_indices,
            state,
            r,
            w,
            w0,
            k,
            v,
            a,
            b,
            output,
            elapsed,
        )

    first_state = initial_state.clone()
    first_output = torch.empty_like(r)
    second_state = initial_state.clone()
    second_output = torch.empty_like(r)
    run(first_state, first_output)
    run(second_state, second_output)
    torch.accelerator.synchronize()
    assert torch.equal(first_output, second_output)
    assert torch.equal(first_state, second_state)

    graph_state = initial_state.clone()
    graph_output = torch.empty_like(r)
    run(graph_state, graph_output)
    graph_state.copy_(initial_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(graph_state, graph_output)
    graph_state.copy_(initial_state)
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(graph_output, first_output)
    assert torch.equal(graph_state, first_state)


def test_wkv_canonical_rejects_invalid_inputs() -> None:
    query_start_loc, slot_indices, state, r, w, k, v, a, b, w0, elapsed = _case(
        (2, 3), state_dtype=torch.float16, seed=51
    )
    output = torch.empty_like(r)
    op = torch.ops.rwkv7_wkv_fp16_v2.wkv
    valid = (
        query_start_loc,
        slot_indices,
        state,
        r,
        w,
        w0,
        k,
        v,
        a,
        b,
        output,
        elapsed,
    )

    with pytest.raises(RuntimeError, match="query_start_loc must be int32"):
        op(query_start_loc.long(), *valid[1:])
    with pytest.raises(RuntimeError, match=r"query_start_loc must have shape \[B\+1\]"):
        op(query_start_loc[:-1], *valid[1:])
    with pytest.raises(RuntimeError, match="r must be fp16"):
        op(*valid[:3], r.float(), *valid[4:])
    with pytest.raises(RuntimeError, match="r must be contiguous"):
        noncontiguous = torch.empty(
            (r.size(1), r.size(0)),
            device="cuda",
            dtype=torch.float16,
        ).t()
        op(*valid[:3], noncontiguous, *valid[4:])
    with pytest.raises(RuntimeError, match="1..65535 requests"):
        op(
            torch.empty(65537, device="cuda", dtype=torch.int32),
            torch.empty(65536, device="cuda", dtype=torch.int32),
            *valid[2:],
        )
