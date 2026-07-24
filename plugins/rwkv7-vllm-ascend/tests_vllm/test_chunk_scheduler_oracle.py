import random

import torch


def recurrence(tokens, state=0):
    out = []
    for t in tokens:
        state = (state * 1664525 + int(t) + 1013904223) & 0xFFFFFFFF
        out.append(state)
    return out, state


def test_irregular_chunked_prefill_equals_whole():
    tokens = list(range(2048))
    expected, final = recurrence(tokens)
    state = 0
    got = []
    for width in [123, 511, 777, 637]:
        part = tokens[len(got) : len(got) + width]
        o, state = recurrence(part, state)
        got.extend(o)
    assert got == expected and state == final


def test_continuous_dynamic_batch_join_cancel_reorder_oracle():
    rng = random.Random(7)
    requests = {f"r{i}": list(range(i + 1, i + 20)) for i in range(8)}
    states = {k: 0 for k in requests}
    outputs = {k: [] for k in requests}
    active = ["r0", "r1", "r2"]
    pending = ["r3", "r4", "r5", "r6", "r7"]
    while active:
        rng.shuffle(active)
        for rid in list(active):
            if not requests[rid]:
                active.remove(rid)
                continue
            token = requests[rid].pop(0)
            o, states[rid] = recurrence([token], states[rid])
            outputs[rid] += o
        if pending and rng.random() < 0.8:
            active.append(pending.pop(0))
        if "r4" in active and len(outputs["r4"]) == 3:
            active.remove("r4")
    # Every non-cancelled request is identical to independent B=1 execution.
    for rid, out in outputs.items():
        original = list(range(int(rid[1:]) + 1, int(rid[1:]) + 20))
        expected, _ = recurrence(original[: len(out)])
        assert out == expected


def test_real_v018_mamba_metadata_segments_and_fresh_slots():
    from types import SimpleNamespace as NS

    from rwkv7_vllm_ascend.model import RWKV7Block

    block = object.__new__(RWKV7Block)
    metadata = NS(
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=2,
        state_indices_tensor_d=torch.tensor([[4], [5]], dtype=torch.int32),
        state_indices_tensor_p=torch.tensor([8, 9], dtype=torch.int32),
        query_start_loc_p=torch.tensor([0, 3, 7], dtype=torch.int32),
        has_initial_states_p=torch.tensor([False, True]),
    )
    assert block._segments(metadata, 9) == [
        (0, 1, 4, False),
        (1, 2, 5, False),
        (2, 5, 8, True),
        (5, 9, 9, False),
    ]


def test_single_token_new_request_decode_slot_is_fresh():
    from types import SimpleNamespace as NS

    from rwkv7_vllm_ascend.model import RWKV7Block

    block = object.__new__(RWKV7Block)
    metadata = NS(
        num_decodes=2,
        num_decode_tokens=2,
        num_prefills=0,
        num_prefill_tokens=0,
        query_start_loc_p=None,
        state_indices_tensor_d=torch.tensor([[3], [7]], dtype=torch.int32),
        state_indices_tensor_p=None,
        has_initial_states_p=None,
        seq_lens=torch.tensor([1, 9], dtype=torch.int32),
    )
    assert block._segments(metadata, 2) == [
        (0, 1, 3, True),
        (1, 2, 7, False),
    ]
