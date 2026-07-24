# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.fixture(scope="module", autouse=True)
def rwkv7_ops_registered() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for RWKV7 custom op registration")
    try:
        import vllm.rwkv7_ops  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"RWKV7 extension is unavailable: {exc!r}")
    import vllm._custom_ops  # noqa: F401


@pytest.mark.parametrize(
    "name",
    [
        "wkv_seq_grid2d",
        "wkv_seq_slot_grid2d",
        "wkv_seq_w0_grid2d",
        "wkv_seq_w0_slot_grid2d",
        "wkv_seq_forced",
        "wkv_seq_slot_forced",
        "wkv_seq_w0_forced",
        "wkv_seq_w0_slot_forced",
        "wkv_seq_grid2d_forced",
        "wkv_seq_slot_grid2d_forced",
        "wkv_seq_w0_grid2d_forced",
        "wkv_seq_w0_slot_grid2d_forced",
    ],
)
def test_wkv_grid_routing_surface_is_registered(name: str) -> None:
    assert hasattr(torch.ops.rwkv7_wkv_fp16_v2, name)


def _payload(
    batch: int,
    time: int,
    hidden: int = 64,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        0.01
        * torch.randn(
            (batch, time, hidden),
            device="cuda",
            dtype=torch.float16,
        )
        for _ in range(6)
    )


def _opcheck_args(name: str) -> tuple:
    batch, time, slots = 2, 2, 5
    has_w0 = "_w0_" in name
    has_slots = "_slot_" in name
    forced = name.endswith("_forced")
    state_rows = slots if has_slots else batch
    state = torch.empty(
        (state_rows, 1, 64, 64),
        device="cuda",
        dtype=torch.float16,
    )
    payload = _payload(batch, time)
    y = torch.empty_like(payload[0])
    args: list[object] = [batch, time, 64, 1]
    if forced:
        args.append(0)
    args.extend((state, payload[0], payload[1]))
    if has_w0:
        args.append(torch.empty((64,), device="cuda", dtype=torch.float16))
    args.extend((*payload[2:], y))
    if has_slots:
        args.append(torch.tensor([3, 0], device="cuda", dtype=torch.int32))
    args.append(
        torch.empty(
            (state_rows,),
            device="cuda",
            dtype=torch.int32,
        )
    )
    return tuple(args)


@pytest.mark.parametrize(
    "name",
    [
        "wkv_seq_grid2d",
        "wkv_seq_slot_grid2d",
        "wkv_seq_w0_grid2d",
        "wkv_seq_w0_slot_grid2d",
        "wkv_seq_forced",
        "wkv_seq_slot_forced",
        "wkv_seq_w0_forced",
        "wkv_seq_w0_slot_forced",
        "wkv_seq_grid2d_forced",
        "wkv_seq_slot_grid2d_forced",
        "wkv_seq_w0_grid2d_forced",
        "wkv_seq_w0_slot_grid2d_forced",
    ],
)
def test_wkv_grid_routing_schema_and_fake_contract(name: str) -> None:
    torch.library.opcheck(
        getattr(torch.ops.rwkv7_wkv_fp16_v2, name),
        _opcheck_args(name),
        test_utils=("test_schema", "test_faketensor"),
    )


def _run_w0_slot(
    op: object,
    batch: int,
    time: int,
    state: torch.Tensor,
    payload: tuple[torch.Tensor, ...],
    w0: torch.Tensor,
    slot_indices: torch.Tensor,
    elapsed_t: torch.Tensor,
    mode: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_state = state.clone()
    y = torch.empty(
        (batch, time, 64),
        device=state.device,
        dtype=torch.float16,
    )
    prefix = (batch, time, 64, 1)
    if mode is not None:
        prefix += (mode,)
    op(
        *prefix,
        current_state,
        payload[0],
        payload[1],
        w0,
        *payload[2:],
        y,
        slot_indices,
        elapsed_t,
    )
    return y, current_state


def _run_w0_direct(
    op: object,
    batch: int,
    time: int,
    state: torch.Tensor,
    payload: tuple[torch.Tensor, ...],
    w0: torch.Tensor,
    elapsed_t: torch.Tensor,
    mode: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_state = state.clone()
    y = torch.empty(
        (batch, time, 64),
        device=state.device,
        dtype=torch.float16,
    )
    prefix = (batch, time, 64, 1)
    if mode is not None:
        prefix += (mode,)
    op(
        *prefix,
        current_state,
        payload[0],
        payload[1],
        w0,
        *payload[2:],
        y,
        elapsed_t,
    )
    return y, current_state


@pytest.mark.parametrize("batch", [2, 32, 96, 160])
def test_wkv_w0_slot_grid2d_matches_flat_for_t1_dispatches(batch: int) -> None:
    torch.manual_seed(20260725 + batch)
    slots = batch + 7
    slot_indices = torch.arange(
        batch - 1,
        -1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    state = 0.01 * torch.randn(
        (slots, 1, 64, 64),
        device="cuda",
        dtype=torch.float16,
    )
    payload = _payload(batch, 1)
    w0 = 0.01 * torch.randn((64,), device="cuda", dtype=torch.float16)
    elapsed_t = torch.arange(slots, device="cuda", dtype=torch.int32)

    flat_y, flat_state = _run_w0_slot(
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_slot,
        batch,
        1,
        state,
        payload,
        w0,
        slot_indices,
        elapsed_t,
    )
    grid_y, grid_state = _run_w0_slot(
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_slot_grid2d,
        batch,
        1,
        state,
        payload,
        w0,
        slot_indices,
        elapsed_t,
    )
    torch.accelerator.synchronize()

    assert torch.equal(grid_y, flat_y)
    assert torch.equal(grid_state, flat_state)


@pytest.mark.parametrize("mode", [0, 1], ids=["exact", "seq_v2"])
def test_wkv_w0_forced_grid2d_preserves_slot_mapping(mode: int) -> None:
    torch.manual_seed(20260731 + mode)
    batch, time, slots = 4, 4, 8
    slot_indices = torch.tensor([7, 2, 5, 0], device="cuda", dtype=torch.int32)
    state = 0.01 * torch.randn(
        (slots, 1, 64, 64),
        device="cuda",
        dtype=torch.float16,
    )
    payload = _payload(batch, time)
    w0 = 0.01 * torch.randn((64,), device="cuda", dtype=torch.float16)
    elapsed_t = torch.arange(slots, device="cuda", dtype=torch.int32)

    flat_y, flat_state = _run_w0_slot(
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_slot_forced,
        batch,
        time,
        state,
        payload,
        w0,
        slot_indices,
        elapsed_t,
        mode,
    )
    grid_y, grid_state = _run_w0_slot(
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_slot_grid2d_forced,
        batch,
        time,
        state,
        payload,
        w0,
        slot_indices,
        elapsed_t,
        mode,
    )

    compact_state = state[slot_indices.long()].clone()
    compact_elapsed = elapsed_t[slot_indices.long()].clone()
    direct_y, direct_state = _run_w0_direct(
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_w0_grid2d_forced,
        batch,
        time,
        compact_state,
        payload,
        w0,
        compact_elapsed,
        mode,
    )
    torch.accelerator.synchronize()

    assert torch.equal(grid_y, flat_y)
    assert torch.equal(grid_state, flat_state)
    assert torch.equal(grid_y, direct_y)
    assert torch.equal(grid_state[slot_indices.long()], direct_state)

    untouched = torch.ones(slots, device="cuda", dtype=torch.bool)
    untouched[slot_indices.long()] = False
    assert torch.equal(grid_state[untouched], state[untouched])


def test_wkv_forced_rejects_t1_and_unknown_mode() -> None:
    torch.manual_seed(20260725)
    batch, time = 2, 2
    state = torch.zeros(
        (batch, 1, 64, 64),
        device="cuda",
        dtype=torch.float16,
    )
    payload = _payload(batch, time)
    y = torch.empty_like(payload[0])
    elapsed_t = torch.zeros((batch,), device="cuda", dtype=torch.int32)
    op = torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_grid2d_forced

    with pytest.raises(RuntimeError, match="mode must be 0"):
        op(
            batch,
            time,
            64,
            1,
            2,
            state,
            *payload,
            y,
            elapsed_t,
        )

    with pytest.raises(RuntimeError, match="T>1 tuning-only"):
        op(
            batch,
            1,
            64,
            1,
            0,
            state,
            *(tensor[:, :1].contiguous() for tensor in payload),
            y[:, :1].contiguous(),
            elapsed_t,
        )


def test_wkv_grid2d_rejects_batch_above_cuda_grid_y_limit() -> None:
    state = torch.empty((1, 1, 64, 64), device="cuda", dtype=torch.float16)
    payload = tuple(
        torch.empty((1, 1, 64), device="cuda", dtype=torch.float16)
        for _ in range(6)
    )
    y = torch.empty_like(payload[0])
    elapsed_t = torch.empty((1,), device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="0 < B <= 65535"):
        torch.ops.rwkv7_wkv_fp16_v2.wkv_seq_grid2d(
            65536,
            1,
            64,
            1,
            state,
            *payload,
            y,
            elapsed_t,
        )
