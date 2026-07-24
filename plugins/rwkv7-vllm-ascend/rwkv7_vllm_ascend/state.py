"""RWKV-7 recurrent-state contract shared by tests and serving adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateLayout:
    layers: int
    hidden_size: int
    num_heads: int
    head_size: int
    activation_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if self.layers <= 0 or self.hidden_size <= 0:
            raise ValueError("layers and hidden_size must be positive")
        if self.num_heads * self.head_size != self.hidden_size:
            raise ValueError("num_heads * head_size must equal hidden_size")


class RWKV7StateCache:
    """Reference slot cache with explicit lifecycle and fail-closed bounds checks.

    Production vLLM uses its MambaManager block table. This class defines the
    equivalent public contract and is used for oracle/parity tests.
    """

    def __init__(self, layout: StateLayout, capacity: int, *, device="cpu"):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.layout, self.capacity = layout, capacity
        L, C, H, N = (
            layout.layers,
            layout.hidden_size,
            layout.num_heads,
            layout.head_size,
        )
        self.wkv = torch.zeros(
            (L, capacity, H, N, N), dtype=torch.float32, device=device
        )
        self.att_x = torch.zeros(
            (L, capacity, C), dtype=layout.activation_dtype, device=device
        )
        self.ffn_x = torch.zeros(
            (L, capacity, C), dtype=layout.activation_dtype, device=device
        )
        self.seen_tokens = torch.zeros(capacity, dtype=torch.int64, device=device)
        self._owners: list[str | None] = [None] * capacity

    def allocate(self, request_id: str) -> int:
        if request_id in self._owners:
            raise ValueError(f"request already allocated: {request_id}")
        try:
            slot = self._owners.index(None)
        except ValueError as exc:
            raise RuntimeError("state cache exhausted") from exc
        self._owners[slot] = request_id
        self._zero(slot)
        return slot

    def _checked(self, slot: int) -> int:
        if slot < 0 or slot >= self.capacity or self._owners[slot] is None:
            raise KeyError(f"inactive state slot: {slot}")
        return slot

    def _zero(self, slot: int) -> None:
        self.wkv[:, slot].zero_()
        self.att_x[:, slot].zero_()
        self.ffn_x[:, slot].zero_()
        self.seen_tokens[slot].zero_()

    def select(self, slots: Iterable[int]):
        idx = torch.tensor(
            [self._checked(int(s)) for s in slots], device=self.wkv.device
        )
        return (
            self.wkv.index_select(1, idx),
            self.att_x.index_select(1, idx),
            self.ffn_x.index_select(1, idx),
            self.seen_tokens.index_select(0, idx),
        )

    gather = select

    def scatter(self, slots: Iterable[int], state) -> None:
        ids = [self._checked(int(s)) for s in slots]
        idx = torch.tensor(ids, device=self.wkv.device)
        wkv, att_x, ffn_x, seen = state
        if (
            wkv.shape[1] != len(ids)
            or att_x.shape[1] != len(ids)
            or ffn_x.shape[1] != len(ids)
            or seen.numel() != len(ids)
        ):
            raise ValueError("scatter batch does not match slot count")
        self.wkv.index_copy_(1, idx, wkv)
        self.att_x.index_copy_(1, idx, att_x)
        self.ffn_x.index_copy_(1, idx, ffn_x)
        self.seen_tokens.index_copy_(0, idx, seen)

    def reorder(self, slots: Iterable[int], order: Iterable[int]):
        slots, order = list(slots), list(order)
        if sorted(order) != list(range(len(slots))):
            raise ValueError("order must be a permutation")
        state = self.select(slots)
        perm = torch.tensor(order, device=self.wkv.device)
        reordered = (
            state[0].index_select(1, perm),
            state[1].index_select(1, perm),
            state[2].index_select(1, perm),
            state[3].index_select(0, perm),
        )
        self.scatter(slots, reordered)

    def clone_or_fork(self, source_slot: int, request_id: str) -> int:
        source_slot = self._checked(source_slot)
        dest = self.allocate(request_id)
        self.wkv[:, dest].copy_(self.wkv[:, source_slot])
        self.att_x[:, dest].copy_(self.att_x[:, source_slot])
        self.ffn_x[:, dest].copy_(self.ffn_x[:, source_slot])
        self.seen_tokens[dest].copy_(self.seen_tokens[source_slot])
        return dest

    def drop(self, slot: int) -> None:
        slot = self._checked(slot)
        self._zero(slot)
        self._owners[slot] = None

    release = drop

    def compact(self) -> dict[int, int]:
        active = [i for i, owner in enumerate(self._owners) if owner is not None]
        mapping = {old: new for new, old in enumerate(active)}
        if active:
            state = self.select(active)
            n = len(active)
            self.wkv[:, :n].copy_(state[0])
            self.att_x[:, :n].copy_(state[1])
            self.ffn_x[:, :n].copy_(state[2])
            self.seen_tokens[:n].copy_(state[3])
            owners = [self._owners[i] for i in active]
        else:
            n, owners = 0, []
        for i in range(n, self.capacity):
            self._zero(i)
        self._owners = owners + [None] * (self.capacity - n)
        return mapping

    def offload(self, slot: int):
        slot = self._checked(slot)
        return tuple(
            x.detach().cpu().clone()
            for x in (
                self.wkv[:, slot],
                self.att_x[:, slot],
                self.ffn_x[:, slot],
                self.seen_tokens[slot],
            )
        )

    def restore(self, slot: int, snapshot) -> None:
        slot = self._checked(slot)
        for dst, src in zip(
            (
                self.wkv[:, slot],
                self.att_x[:, slot],
                self.ffn_x[:, slot],
                self.seen_tokens[slot],
            ),
            snapshot,
        ):
            dst.copy_(src.to(dst.device, dst.dtype))
