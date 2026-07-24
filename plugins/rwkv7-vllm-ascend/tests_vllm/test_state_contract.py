import pytest
import torch
from rwkv7_vllm_ascend.state import RWKV7StateCache, StateLayout


def cache(capacity=4):
    return RWKV7StateCache(StateLayout(2, 8, 2, 4), capacity)


def test_join_exit_reorder_compact_and_slot_reuse():
    c = cache()
    a, b, d = c.allocate("a"), c.allocate("b"), c.allocate("d")
    c.wkv[:, a].fill_(1)
    c.wkv[:, b].fill_(2)
    c.wkv[:, d].fill_(3)
    c.reorder([a, b, d], [2, 0, 1])
    assert [c.wkv[0, s, 0, 0, 0].item() for s in (a, b, d)] == [3, 1, 2]
    c.drop(b)
    e = c.allocate("e")
    assert e == b and torch.count_nonzero(c.wkv[:, e]) == 0
    mapping = c.compact()
    assert set(mapping) == {a, d, e}
    assert c._owners[:3] == ["a", "e", "d"]


def test_cancel_release_is_fail_closed_and_zeroizes():
    c = cache(1)
    s = c.allocate("request")
    c.att_x[:, s].fill_(7)
    c.release(s)
    assert torch.count_nonzero(c.att_x) == 0
    with pytest.raises(KeyError):
        c.select([s])
    with pytest.raises(KeyError):
        c.release(s)


def test_fork_offload_restore_and_chunk_continuation():
    c = cache()
    s = c.allocate("root")
    c.seen_tokens[s] = 128
    c.wkv[:, s].normal_()
    child = c.clone_or_fork(s, "child")
    assert torch.equal(c.wkv[:, s], c.wkv[:, child])
    snapshot = c.offload(child)
    c.wkv[:, child].zero_()
    c.restore(child, snapshot)
    assert torch.equal(c.wkv[:, s], c.wkv[:, child])
    c.seen_tokens[child] += 64
    assert c.seen_tokens[child] == 192 and c.seen_tokens[s] == 128


def test_capacity_duplicate_and_bad_reorder_rejected():
    c = cache(1)
    c.allocate("a")
    with pytest.raises(ValueError):
        c.allocate("a")
    with pytest.raises(RuntimeError):
        c.allocate("b")
    with pytest.raises(ValueError):
        c.reorder([0], [1])
