# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import regex as re
import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.tasks import GenerationTask
from vllm.v1.core.sched.output import FinishedRequestData, NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)

DEFAULT_RWKV7_PREFIX_CACHE_CAPACITY = 8
RWKV_STATE_CACHE_PROTOCOL = "vllm-rwkv.state-cache.v1"
RWKV_STATE_READ_REF_ARG = "rwkv_state_read_ref"
RWKV_STATE_READ_LEASE_ARG = "rwkv_state_read_lease"
RWKV_STATE_WRITE_REF_ARG = "rwkv_state_write_ref"
_STATE_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_COMMIT_FINISH_REASONS = frozenset({"stop", "length", "repetition"})


@dataclass(frozen=True)
class RWKV7PrefixStateIdentity:
    """Compatibility identity for one recurrent prefix state."""

    token_ids: tuple[int, ...]
    model_artifact: str
    model_revision: str
    backend_provider: str
    backend_revision: str
    wkv_mode: str
    gemm_policy: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        if not self.token_ids:
            raise ValueError("RWKV7 prefix identity requires at least one token")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("RWKV7 prefix token ids must be non-negative")
        for field_name in (
            "model_artifact",
            "model_revision",
            "backend_provider",
            "backend_revision",
            "wkv_mode",
            "gemm_policy",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"RWKV7 prefix identity requires {field_name}")


@dataclass(frozen=True)
class RWKV7PrefixStateSnapshot:
    """A request-independent copy of all state needed to resume a prefix."""

    shift_state: torch.Tensor
    wkv_state: torch.Tensor
    elapsed: int

    @property
    def nbytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.shift_state, self.wkv_state)
        )

    def clone(self) -> "RWKV7PrefixStateSnapshot":
        return RWKV7PrefixStateSnapshot(
            shift_state=self.shift_state.clone(),
            wkv_state=self.wkv_state.clone(),
            elapsed=self.elapsed,
        )

    def is_identical(self, other: "RWKV7PrefixStateSnapshot") -> bool:
        return (
            self.elapsed == other.elapsed
            and self.shift_state.dtype == other.shift_state.dtype
            and self.shift_state.device == other.shift_state.device
            and torch.equal(self.shift_state, other.shift_state)
            and self.wkv_state.dtype == other.wkv_state.dtype
            and self.wkv_state.device == other.wkv_state.device
            and torch.equal(self.wkv_state, other.wkv_state)
        )


@dataclass(frozen=True)
class _RWKV7StateCacheEntry:
    snapshot: RWKV7PrefixStateSnapshot
    parent_ref: str | None
    processed_token_count: int
    pending_tail_token_ids: tuple[int, ...]
    finish_reason: str

    @property
    def nbytes(self) -> int:
        return self.snapshot.nbytes


class RWKV7PrefixStateCache:
    """Bounded LRU cache for recurrent state owned by the RWKV model state.

    Capacity is measured in snapshots. Stored values and cache hits are cloned,
    so a resumed request cannot mutate the cached state.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("RWKV7 prefix cache capacity must be positive")
        self.capacity = capacity
        self._entries: OrderedDict[
            RWKV7PrefixStateIdentity, RWKV7PrefixStateSnapshot
        ] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def get(
        self, identity: RWKV7PrefixStateIdentity
    ) -> RWKV7PrefixStateSnapshot | None:
        snapshot = self._entries.get(identity)
        if snapshot is not None:
            self._entries.move_to_end(identity)
            return snapshot.clone()
        self._reject_stale_identity(identity)
        return None

    def get_longest_prefix(
        self,
        identity: RWKV7PrefixStateIdentity,
        *,
        max_prefix_length: int,
    ) -> tuple[int, RWKV7PrefixStateSnapshot] | None:
        if max_prefix_length <= 0:
            return None

        best: RWKV7PrefixStateIdentity | None = None
        for cached in self._entries:
            prefix_length = len(cached.token_ids)
            if (
                prefix_length > max_prefix_length
                or identity.token_ids[:prefix_length] != cached.token_ids
            ):
                continue
            if cached.model_artifact != identity.model_artifact:
                continue
            if cached.model_revision != identity.model_revision:
                raise ValueError("RWKV7 prefix state has a stale model revision")
            if (
                cached.backend_provider != identity.backend_provider
                or cached.backend_revision != identity.backend_revision
                or cached.wkv_mode != identity.wkv_mode
                or cached.gemm_policy != identity.gemm_policy
            ):
                raise ValueError("RWKV7 prefix state has a stale backend configuration")
            if best is None or prefix_length > len(best.token_ids):
                best = cached

        if best is None:
            return None
        snapshot = self._entries[best]
        self._entries.move_to_end(best)
        return len(best.token_ids), snapshot.clone()

    def put(
        self,
        identity: RWKV7PrefixStateIdentity,
        snapshot: RWKV7PrefixStateSnapshot,
    ) -> None:
        existing = self._entries.get(identity)
        if existing is not None:
            if not existing.is_identical(snapshot):
                raise ValueError("RWKV7 prefix identity maps to conflicting state")
            self._entries.move_to_end(identity)
            return
        self._reject_stale_identity(identity)
        self._entries[identity] = snapshot.clone()
        if len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def _reject_stale_identity(self, identity: RWKV7PrefixStateIdentity) -> None:
        for cached in self._entries:
            if (
                cached.token_ids != identity.token_ids
                or cached.model_artifact != identity.model_artifact
            ):
                continue
            if cached.model_revision != identity.model_revision:
                raise ValueError("RWKV7 prefix state has a stale model revision")
            raise ValueError("RWKV7 prefix state has a stale backend configuration")


class RWKV7ModelState(ModelState):
    """Dense batched recurrent state for RWKV7."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        cfg = self.model_config.hf_config
        total_num_layers = int(
            getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
        )
        self.layer_offset = int(getattr(model, "start_layer", 0))
        self.num_layers = (
            int(getattr(model, "end_layer", total_num_layers)) - self.layer_offset
        )
        self.hidden_size = int(cfg.hidden_size)
        self.head_size = int(getattr(cfg, "head_size", 64))
        total_num_heads = int(
            getattr(
                cfg,
                "num_attention_heads",
                self.hidden_size // self.head_size,
            )
        )
        self.num_heads = int(getattr(model, "tp_num_heads", total_num_heads))
        wkv_dtype = getattr(model, "wkv_state_dtype", None)
        if wkv_dtype is None:
            wkv_dtype = (
                torch.float32
                if getattr(model, "wkv_mode", "fp16") == "fp32io16"
                else torch.float16
            )

        self.shift_state = torch.zeros(
            (self.num_layers, 2, self.max_num_reqs, self.hidden_size),
            dtype=torch.float16,
            device=device,
        )
        self.wkv_state = torch.zeros(
            (
                self.num_layers,
                self.max_num_reqs,
                self.num_heads,
                self.head_size,
                self.head_size,
            ),
            dtype=wkv_dtype,
            device=device,
        )
        self.elapsed = torch.zeros(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self.execution_idx_mapping = torch.arange(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.decode_query_start_loc = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device=device
        )
        self.decode_slot_indices = torch.empty(
            (self.max_num_reqs,), dtype=torch.int32, device=device
        )
        self.decode_token_positions = torch.empty(
            (self.max_num_reqs,), dtype=torch.long, device=device
        )
        # Maps request ids to stable RWKV state slots. A vLLM request index can
        # change after request metadata is condensed, but this slot must not.
        self.req_id_to_index: dict[str, int] = {}
        self.req_slot_owners: list[str | None] = [None] * self.max_num_reqs
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots: set[int] = set()
        self._prefill_req_slots: list[int] = []
        self._prefill_becomes_decode: list[bool] = []
        self._prefill_cache_targets: list[int] = []
        self.req_prompt_token_ids: dict[str, tuple[int, ...]] = {}
        self.req_prefix_cache_hit_lengths: dict[str, int] = {}
        self.prefix_cache_eligible_req_ids: set[str] = set()

        cache_config = getattr(vllm_config, "cache_config", None)
        prefix_cache_enabled = bool(
            cache_config is not None
            and getattr(cache_config, "rwkv_recurrent_prefix_caching", False)
        )
        self.prefix_state_cache = (
            RWKV7PrefixStateCache(capacity=DEFAULT_RWKV7_PREFIX_CACHE_CAPACITY)
            if prefix_cache_enabled
            else None
        )
        self._prefix_identity_fields = self._build_prefix_identity_fields()
        self._state_cache: dict[str, _RWKV7StateCacheEntry] = {}
        self._state_snapshot_refcounts: dict[int, int] = {}
        self._active_state_readers: dict[str, int] = {}
        self._prepared_state_reads_by_lease: dict[str, str] = {}
        self._prepared_state_bytes_by_ref: dict[str, int] = {}
        self._state_cache_bytes = 0
        self._reserved_state_cache_bytes = 0
        self._pending_write_ref_by_id: dict[str, str] = {}
        self._reserved_state_bytes_by_req_id: dict[str, int] = {}
        self._parent_state_ref_by_req_id: dict[str, str] = {}
        self._state_cache_max_bytes = int(envs.VLLM_RWKV_STATE_CACHE_MAX_BYTES)
        if self._state_cache_max_bytes < 0:
            raise ValueError("VLLM_RWKV_STATE_CACHE_MAX_BYTES must not be negative")
        if (
            self._state_cache_max_bytes
            and vllm_config.parallel_config.data_parallel_size > 1
        ):
            raise ValueError("RWKV State refs do not yet support data parallelism")

    def _reset_mappings(self) -> None:
        self.req_slot_owners = [None] * self.max_num_reqs
        self.req_slot_to_row = [-1] * self.max_num_reqs
        self.row_to_req_slot = [-1] * self.max_num_reqs
        self.free_rows = set(range(self.max_num_reqs))
        self.decode_req_slots = set()
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []
        self._prefill_cache_targets = []
        self.req_prompt_token_ids.clear()
        self.req_prefix_cache_hit_lengths.clear()
        self.prefix_cache_eligible_req_ids.clear()

    def _build_prefix_identity_fields(self) -> dict[str, str]:
        cfg = self.model_config.hf_config
        model_artifact = str(
            getattr(self.model_config, "model", None)
            or getattr(cfg, "name_or_path", None)
            or getattr(cfg, "_name_or_path", None)
            or type(self.model).__qualname__
        )
        model_revision = str(
            getattr(self.model_config, "revision", None)
            or getattr(cfg, "_commit_hash", None)
            or getattr(cfg, "rwkv_model_revision", None)
            or "runtime"
        )
        backend_provider = str(
            getattr(self.model, "rwkv_backend_provider", None) or "vllm-rwkv-native"
        )
        backend_revision = str(
            getattr(self.model, "rwkv_backend_revision", None)
            or getattr(cfg, "rwkv_backend_revision", None)
            or f"{type(self.model).__module__}.{type(self.model).__qualname__}"
        )
        wkv_mode = str(getattr(self.model, "wkv_mode", "fp16"))
        execution_profile = getattr(self.model, "execution_profile", None)
        gemm_policy = str(
            getattr(execution_profile, "gemm_accumulation_policy", None)
            or (
                "fp16"
                if getattr(self.model, "allow_fp16_accumulation", False)
                else "fp32"
            )
        )
        return {
            "model_artifact": model_artifact,
            "model_revision": model_revision,
            "backend_provider": backend_provider,
            "backend_revision": backend_revision,
            "wkv_mode": wkv_mode,
            "gemm_policy": gemm_policy,
        }

    def _prefix_identity(self, token_ids: tuple[int, ...]) -> RWKV7PrefixStateIdentity:
        return RWKV7PrefixStateIdentity(
            token_ids=token_ids,
            **self._prefix_identity_fields,
        )

    def _restore_row(
        self,
        row: int,
        snapshot: RWKV7PrefixStateSnapshot,
    ) -> None:
        shift_row = self.shift_state[:, :, row]
        wkv_row = self.wkv_state[:, row]
        if (
            snapshot.shift_state.shape != shift_row.shape
            or snapshot.shift_state.dtype != shift_row.dtype
            or snapshot.shift_state.device != shift_row.device
            or snapshot.wkv_state.shape != wkv_row.shape
            or snapshot.wkv_state.dtype != wkv_row.dtype
            or snapshot.wkv_state.device != wkv_row.device
            or snapshot.elapsed < 0
        ):
            raise ValueError("RWKV7 cached prefix state is incompatible with runtime")
        shift_row.copy_(snapshot.shift_state)
        wkv_row.copy_(snapshot.wkv_state)
        self.elapsed[row].fill_(snapshot.elapsed)

    def _snapshot_row(self, row: int) -> RWKV7PrefixStateSnapshot:
        if row < 0 or row >= self.max_num_reqs:
            raise ValueError(f"RWKV7 state row {row} is out of range")
        return RWKV7PrefixStateSnapshot(
            shift_state=self.shift_state[:, :, row].clone(),
            wkv_state=self.wkv_state[:, row].clone(),
            elapsed=int(self.elapsed[row].item()),
        )

    def _cache_row(self, req_slot: int, prefix_length: int) -> None:
        cache = self.prefix_state_cache
        req_id = self.req_slot_owners[req_slot]
        if (
            cache is None
            or req_id is None
            or req_id not in self.prefix_cache_eligible_req_ids
            or prefix_length <= 0
        ):
            return
        token_ids = self.req_prompt_token_ids[req_id]
        if prefix_length > len(token_ids):
            raise RuntimeError("RWKV7 prefix cache target exceeds prompt length")
        row = self.req_slot_to_row[req_slot]
        if row < 0:
            raise RuntimeError("RWKV7 prefix cache target has no resident state row")
        cache.put(
            self._prefix_identity(token_ids[:prefix_length]),
            RWKV7PrefixStateSnapshot(
                shift_state=self.shift_state[:, :, row].clone(),
                wkv_state=self.wkv_state[:, row].clone(),
                elapsed=prefix_length,
            ),
        )

    @property
    def state_snapshot_nbytes(self) -> int:
        shift_state = self.shift_state[:, :, 0]
        wkv_state = self.wkv_state[:, 0]
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (shift_state, wkv_state)
        )

    @staticmethod
    def _validate_state_ref(
        value: object,
        *,
        field: str,
        allow_empty: bool = False,
    ) -> str:
        if value is None and allow_empty:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"{field} must be a string")
        state_ref = value.strip()
        if not state_ref and allow_empty:
            return ""
        if not _STATE_REF_PATTERN.fullmatch(state_ref):
            raise ValueError(f"{field} is invalid")
        return state_ref

    def _state_ref_record(self, state_ref: str) -> dict[str, Any]:
        entry = self._state_cache[state_ref]
        return {
            "protocol": RWKV_STATE_CACHE_PROTOCOL,
            "state_ref": state_ref,
            "nbytes": entry.nbytes,
            "parent_ref": entry.parent_ref,
            "processed_token_count": entry.processed_token_count,
            "pending_tail_token_count": len(entry.pending_tail_token_ids),
            "finish_reason": entry.finish_reason,
            "state_dtype": str(entry.snapshot.wkv_state.dtype),
            "process_local": True,
            "durable": False,
            "scheduler_slots_reserved": 0,
        }

    def _state_ref_reader_count(self, state_ref: str) -> int:
        prepared = sum(
            prepared_ref == state_ref
            for prepared_ref in self._prepared_state_reads_by_lease.values()
        )
        return self._active_state_readers.get(state_ref, 0) + prepared

    def _reserve_state_ref(self) -> int:
        if self._state_cache_max_bytes == 0:
            raise RuntimeError(
                "RWKV State refs are disabled; set "
                "VLLM_RWKV_STATE_CACHE_MAX_BYTES to a positive byte limit"
            )
        required = self.state_snapshot_nbytes
        if (
            self._state_cache_bytes + self._reserved_state_cache_bytes + required
            > self._state_cache_max_bytes
        ):
            raise MemoryError(
                "RWKV State cache capacity exceeded before generation: "
                f"used={self._state_cache_bytes}, "
                f"reserved={self._reserved_state_cache_bytes}, "
                f"required={required}, limit={self._state_cache_max_bytes}"
            )
        self._reserved_state_cache_bytes += required
        return required

    def _store_state_ref(
        self,
        state_ref: str,
        entry: _RWKV7StateCacheEntry,
        *,
        reserved_nbytes: int = 0,
    ) -> dict[str, Any]:
        if state_ref in self._state_cache:
            raise ValueError(f"RWKV State ref already exists: {state_ref}")
        snapshot_key = id(entry.snapshot)
        required = 0 if snapshot_key in self._state_snapshot_refcounts else entry.nbytes
        unreserved = self._reserved_state_cache_bytes - reserved_nbytes
        if (
            self._state_cache_bytes + unreserved + required
            > self._state_cache_max_bytes
        ):
            raise MemoryError(
                "RWKV State cache capacity exceeded: "
                f"used={self._state_cache_bytes}, reserved={unreserved}, "
                f"required={required}, limit={self._state_cache_max_bytes}"
            )
        if reserved_nbytes:
            self._reserved_state_cache_bytes -= reserved_nbytes
        self._state_cache[state_ref] = entry
        self._state_cache_bytes += required
        self._state_snapshot_refcounts[snapshot_key] = (
            self._state_snapshot_refcounts.get(snapshot_key, 0) + 1
        )
        return self._state_ref_record(state_ref)

    def capture_state_ref(
        self,
        state_ref: str,
        row: int,
        *,
        parent_ref: str | None = None,
        processed_token_count: int = 0,
        pending_tail_token_ids: Sequence[int] = (),
        finish_reason: str = "snapshot",
        reserved_nbytes: int = 0,
    ) -> dict[str, Any]:
        state_ref = self._validate_state_ref(state_ref, field="state_ref")
        entry = _RWKV7StateCacheEntry(
            snapshot=self._snapshot_row(row),
            parent_ref=parent_ref,
            processed_token_count=processed_token_count,
            pending_tail_token_ids=tuple(pending_tail_token_ids),
            finish_reason=finish_reason,
        )
        return self._store_state_ref(state_ref, entry, reserved_nbytes=reserved_nbytes)

    def restore_state_ref(self, state_ref: str, row: int) -> None:
        state_ref = self._validate_state_ref(state_ref, field="state_ref")
        if state_ref not in self._state_cache:
            raise KeyError(f"Unknown RWKV State ref: {state_ref}")
        self._restore_row(row, self._state_cache[state_ref].snapshot)

    def state_cache_action(
        self,
        action: str,
        state_ref: str = "",
        target_ref: str = "",
    ) -> dict[str, Any]:
        if action == "capabilities":
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "supported": True,
                "enabled": self._state_cache_max_bytes > 0,
                "process_local": True,
                "durable": False,
                "scheduler_slots_reserved_per_ref": 0,
                "max_bytes": self._state_cache_max_bytes,
                "used_bytes": self._state_cache_bytes,
                "reserved_bytes": self._reserved_state_cache_bytes,
                "snapshot_nbytes": self.state_snapshot_nbytes,
                "state_count": len(self._state_cache),
                "prepared_read_count": len(self._prepared_state_reads_by_lease),
            }
        state_ref = self._validate_state_ref(state_ref, field="state_ref")
        if action == "inspect_if_exists":
            if state_ref not in self._state_cache:
                return {
                    "protocol": RWKV_STATE_CACHE_PROTOCOL,
                    "state_ref": state_ref,
                    "exists": False,
                }
            return {**self._state_ref_record(state_ref), "exists": True}
        if action == "prepare_read":
            read_lease = self._validate_state_ref(target_ref, field="read_lease")
            if state_ref not in self._state_cache:
                raise KeyError(f"Unknown RWKV State ref: {state_ref}")
            if read_lease in self._prepared_state_reads_by_lease:
                raise ValueError(f"RWKV State read lease already exists: {read_lease}")
            self._prepared_state_reads_by_lease[read_lease] = state_ref
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "read_lease": read_lease,
                "ready": True,
            }
        if action == "cancel_read":
            read_lease = self._validate_state_ref(target_ref, field="read_lease")
            canceled = self._prepared_state_reads_by_lease.get(read_lease) == state_ref
            if canceled:
                del self._prepared_state_reads_by_lease[read_lease]
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "read_lease": read_lease,
                "canceled": canceled,
            }
        if action == "prepare_write":
            if (
                state_ref in self._state_cache
                or state_ref in self._pending_write_ref_by_id.values()
                or state_ref in self._prepared_state_bytes_by_ref
            ):
                raise ValueError(f"RWKV State ref already exists: {state_ref}")
            reserved = self._reserve_state_ref()
            self._prepared_state_bytes_by_ref[state_ref] = reserved
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "ready": True,
                "reserved_bytes": reserved,
            }
        if action == "cancel_write":
            reserved = self._prepared_state_bytes_by_ref.pop(state_ref, 0)
            self._reserved_state_cache_bytes -= reserved
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "canceled": bool(reserved),
                "reserved_bytes": self._reserved_state_cache_bytes,
            }
        if action == "drop_if_exists" and state_ref not in self._state_cache:
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "dropped": False,
                "used_bytes": self._state_cache_bytes,
            }
        if state_ref not in self._state_cache:
            raise KeyError(f"Unknown RWKV State ref: {state_ref}")
        if action == "prepare_drop":
            if self._state_ref_reader_count(state_ref):
                raise ValueError(f"RWKV State ref is in use: {state_ref}")
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "ready": True,
            }
        if action == "prepare_clone":
            target_ref = self._validate_state_ref(target_ref, field="target_ref")
            if (
                target_ref in self._state_cache
                or target_ref in self._pending_write_ref_by_id.values()
                or target_ref in self._prepared_state_bytes_by_ref
            ):
                raise ValueError(f"RWKV State ref already exists: {target_ref}")
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "target_ref": target_ref,
                "ready": True,
            }
        if action == "inspect":
            return self._state_ref_record(state_ref)
        if action in {"drop", "drop_if_exists"}:
            if self._state_ref_reader_count(state_ref):
                raise ValueError(f"RWKV State ref is in use: {state_ref}")
            entry = self._state_cache.pop(state_ref)
            snapshot_key = id(entry.snapshot)
            refcount = self._state_snapshot_refcounts[snapshot_key] - 1
            if refcount:
                self._state_snapshot_refcounts[snapshot_key] = refcount
            else:
                del self._state_snapshot_refcounts[snapshot_key]
                self._state_cache_bytes -= entry.nbytes
            return {
                "protocol": RWKV_STATE_CACHE_PROTOCOL,
                "state_ref": state_ref,
                "dropped": True,
                "used_bytes": self._state_cache_bytes,
            }
        if action == "clone":
            target_ref = self._validate_state_ref(target_ref, field="target_ref")
            source = self._state_cache[state_ref]
            entry = _RWKV7StateCacheEntry(
                snapshot=source.snapshot,
                parent_ref=state_ref,
                processed_token_count=source.processed_token_count,
                pending_tail_token_ids=source.pending_tail_token_ids,
                finish_reason=source.finish_reason,
            )
            return self._store_state_ref(target_ref, entry)
        if action == "restore_metadata":
            entry = self._state_cache[state_ref]
            return {
                **self._state_ref_record(state_ref),
                "pending_tail_token_ids": list(entry.pending_tail_token_ids),
            }
        raise ValueError(f"unsupported RWKV State cache action: {action}")

    def _clear_state_cache(self) -> None:
        self._state_cache.clear()
        self._state_snapshot_refcounts.clear()
        self._active_state_readers.clear()
        self._prepared_state_reads_by_lease.clear()
        self._prepared_state_bytes_by_ref.clear()
        self._pending_write_ref_by_id.clear()
        self._reserved_state_bytes_by_req_id.clear()
        self._parent_state_ref_by_req_id.clear()
        self._state_cache_bytes = 0
        self._reserved_state_cache_bytes = 0

    def _state_slot_for_batch_entry(
        self,
        input_batch: InputBatch,
        batch_idx: int,
    ) -> int:
        req_ids = getattr(input_batch, "req_ids", None)
        if req_ids is None or batch_idx >= len(req_ids):
            raise RuntimeError("RWKV7 requires request ids for state lookup")
        req_id = req_ids[batch_idx]
        if req_id is None:
            raise RuntimeError("RWKV7 request id cannot be None")
        req_slot = self.req_id_to_index.get(req_id)
        if req_slot is None:
            raise RuntimeError(f"RWKV state for request id {req_id!r} missing")
        return req_slot

    @staticmethod
    def _is_contiguous_decode_context(
        decode_rows: list[int],
        decode_token_positions: list[int],
    ) -> bool:
        if not decode_rows:
            return False
        decode_len = len(decode_rows)
        if decode_rows != list(range(decode_len)):
            return False
        start = decode_token_positions[0]
        return decode_token_positions == list(range(start, start + decode_len))

    def _new_dummy_state_tensors(self, num_reqs: int) -> dict[str, torch.Tensor]:
        return {
            "shift_state": torch.zeros(
                (self.num_layers, 2, num_reqs, self.hidden_size),
                dtype=self.shift_state.dtype,
                device=self.device,
            ),
            "wkv_state": torch.zeros(
                (
                    self.num_layers,
                    num_reqs,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                ),
                dtype=self.wkv_state.dtype,
                device=self.device,
            ),
            "elapsed": torch.zeros(
                (num_reqs,),
                dtype=self.elapsed.dtype,
                device=self.device,
            ),
        }

    def _packed_prefill_inputs(
        self,
        prefill_ranges: list[tuple[int, int, int]],
        prefill_rows: list[int],
    ) -> dict[str, torch.Tensor]:
        if len(prefill_ranges) != len(prefill_rows):
            raise RuntimeError("RWKV7 prefill range/row metadata mismatch")
        query_offsets = [0]
        token_positions: list[int] = []
        req_id: list[int] = []
        for local_req, (_batch_idx, start, end) in enumerate(prefill_ranges):
            length = end - start
            if length <= 0:
                raise RuntimeError(
                    "RWKV7 packed prefill requires positive request lengths"
                )
            token_positions.extend(range(start, end))
            req_id.extend([local_req] * length)
            query_offsets.append(query_offsets[-1] + length)
        return {
            "rwkv_prefill_query_start_loc": torch.tensor(
                query_offsets,
                dtype=torch.int32,
                device=self.device,
            ),
            "rwkv_prefill_slot_indices": torch.tensor(
                prefill_rows,
                dtype=torch.int32,
                device=self.device,
            ),
            "rwkv_prefill_token_positions": torch.tensor(
                token_positions,
                dtype=torch.long,
                device=self.device,
            ),
            "rwkv_prefill_req_id": torch.tensor(
                req_id,
                dtype=torch.int32,
                device=self.device,
            ),
        }

    @staticmethod
    def _set_sampling_logits_fast_path(
        input_batch: InputBatch,
        enabled: bool,
    ) -> None:
        try:
            input_batch.rwkv_sampling_logits_contiguous = (  # type: ignore[attr-defined]
                enabled
            )
        except Exception:
            return

    def get_supported_generation_tasks(self) -> tuple[GenerationTask, ...]:
        return ("generate",)

    def get_v2_kernel_warmup_skip_reason(self) -> str | None:
        return "uniform recurrent decode waves do not support mixed warmup batches"

    def custom_sampler(self, sampler: Any) -> tuple[Any, None]:
        if not sampler.use_rapid:
            raise RuntimeError("RWKV7 requires rapid-sampling on CUDA.")
        sampler.require_rapid = True
        return sampler, None

    def sort_scheduled_req_ids(
        self,
        req_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        req_states: RequestState,
    ) -> list[str]:
        def key(item: tuple[int, str]) -> tuple[int, int, int]:
            order, req_id = item
            current_req_index = req_states.req_id_to_index.get(req_id)
            req_slot = self.req_id_to_index.get(req_id)
            if current_req_index is None or req_slot is None:
                return (num_scheduled_tokens[req_id], 1, order)
            is_prefilling = (
                req_states.num_computed_prefill_tokens[current_req_index]
                < req_states.prefill_len.np[current_req_index]
            )
            row = self.req_slot_to_row[req_slot]
            if (
                num_scheduled_tokens[req_id] == 1
                and not bool(is_prefilling)
                and row >= 0
            ):
                return (1, 0, row)
            return (num_scheduled_tokens[req_id], 1, order)

        return [req_id for _order, req_id in sorted(enumerate(req_ids), key=key)]

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        if req_index < 0 or req_index >= self.max_num_reqs:
            raise RuntimeError(f"RWKV7 request slot {req_index} is out of range")
        req_id = new_req_data.req_id
        if req_id in self.req_id_to_index:
            raise RuntimeError(f"RWKV7 request id {req_id!r} already owns state")
        current_owner = self.req_slot_owners[req_index]
        if current_owner is not None:
            raise RuntimeError(
                f"RWKV7 request slot {req_index} is already owned by {current_owner!r}"
            )
        if not self.free_rows:
            raise RuntimeError("RWKV7 state pool is full")

        sampling_params = new_req_data.sampling_params
        extra_args = sampling_params.extra_args if sampling_params is not None else None
        extra_args = extra_args or {}
        read_ref = self._validate_state_ref(
            extra_args.get(RWKV_STATE_READ_REF_ARG),
            field=RWKV_STATE_READ_REF_ARG,
            allow_empty=True,
        )
        read_lease = self._validate_state_ref(
            extra_args.get(RWKV_STATE_READ_LEASE_ARG),
            field=RWKV_STATE_READ_LEASE_ARG,
            allow_empty=True,
        )
        write_ref = self._validate_state_ref(
            extra_args.get(RWKV_STATE_WRITE_REF_ARG),
            field=RWKV_STATE_WRITE_REF_ARG,
            allow_empty=True,
        )
        if (read_ref or write_ref) and self._state_cache_max_bytes == 0:
            raise RuntimeError(
                "RWKV State refs are disabled; set "
                "VLLM_RWKV_STATE_CACHE_MAX_BYTES to a positive byte limit"
            )
        if read_lease and not read_ref:
            raise ValueError("rwkv_state_read_lease requires rwkv_state_read_ref")
        read_entry = self._state_cache.get(read_ref) if read_ref else None
        if read_ref:
            if read_entry is None:
                raise KeyError(f"Unknown RWKV State ref: {read_ref}")
            if read_lease:
                prepared_ref = self._prepared_state_reads_by_lease.get(read_lease)
                if prepared_ref != read_ref:
                    raise ValueError("RWKV State read lease is missing or mismatched")
            pending_tail = read_entry.pending_tail_token_ids
            prefill_token_ids = new_req_data.prefill_token_ids or ()
            if tuple(prefill_token_ids[: len(pending_tail)]) != pending_tail:
                raise ValueError(
                    "RWKV State restore input is missing its pending tail tokens"
                )
        if write_ref:
            if write_ref == read_ref:
                raise ValueError("RWKV State read/write refs must be different")
            if write_ref in self._state_cache or write_ref in (
                self._pending_write_ref_by_id.values()
            ):
                raise ValueError(f"RWKV State ref already exists: {write_ref}")

        prompt_token_ids = tuple(new_req_data.prompt_token_ids or ())
        prefix_state_cache = self.prefix_state_cache
        cache_eligible = bool(
            prefix_state_cache is not None
            and not read_ref
            and prompt_token_ids
            and (
                sampling_params is None
                or getattr(sampling_params, "prompt_logprobs", None) is None
            )
        )
        cache_hit_length = 0
        cached_snapshot = None
        if cache_eligible:
            assert prefix_state_cache is not None
            hit = prefix_state_cache.get_longest_prefix(
                self._prefix_identity(prompt_token_ids),
                max_prefix_length=len(prompt_token_ids) - 1,
            )
            if hit is not None:
                cache_hit_length, cached_snapshot = hit
        if new_req_data.num_computed_tokens > cache_hit_length:
            raise RuntimeError(
                "RWKV7 cannot restore scheduler-computed prefix state: "
                f"request has {new_req_data.num_computed_tokens} computed tokens "
                f"but recurrent cache restored {cache_hit_length}."
            )

        row = min(self.free_rows)
        if self.row_to_req_slot[row] != -1:
            raise RuntimeError(f"RWKV7 free state row {row} still has an owner")
        reserved = 0
        used_prepared_reservation = False
        if write_ref:
            reserved = self._prepared_state_bytes_by_ref.pop(write_ref, 0)
            used_prepared_reservation = bool(reserved)
            if not reserved:
                reserved = self._reserve_state_ref()
        self.free_rows.remove(row)
        self.req_id_to_index[req_id] = req_index
        self.req_slot_owners[req_index] = req_id
        self.req_slot_to_row[req_index] = row
        self.row_to_req_slot[row] = req_index
        self._zero_row(row)
        try:
            if read_entry is not None:
                self._restore_row(row, read_entry.snapshot)
            elif cached_snapshot is not None:
                self._restore_row(row, cached_snapshot)
        except Exception:
            self.req_id_to_index.pop(req_id)
            self.req_slot_owners[req_index] = None
            self.req_slot_to_row[req_index] = -1
            self.row_to_req_slot[row] = -1
            self.free_rows.add(row)
            self._zero_row(row)
            if reserved:
                if used_prepared_reservation:
                    self._prepared_state_bytes_by_ref[write_ref] = reserved
                else:
                    self._reserved_state_cache_bytes -= reserved
            raise
        self.req_prompt_token_ids[req_id] = prompt_token_ids
        self.req_prefix_cache_hit_lengths[req_id] = cache_hit_length
        if cache_eligible:
            self.prefix_cache_eligible_req_ids.add(req_id)
        if read_ref:
            if read_lease:
                del self._prepared_state_reads_by_lease[read_lease]
            self._active_state_readers[read_ref] = (
                self._active_state_readers.get(read_ref, 0) + 1
            )
            self._parent_state_ref_by_req_id[req_id] = read_ref
        if write_ref:
            self._pending_write_ref_by_id[req_id] = write_ref
            self._reserved_state_bytes_by_req_id[req_id] = reserved

    def remove_request(
        self,
        req_id: str,
        finished_data: FinishedRequestData | None = None,
    ) -> None:
        req_index = self.req_id_to_index.get(req_id)
        if req_index is None:
            return
        if self.req_slot_owners[req_index] != req_id:
            raise RuntimeError(
                f"RWKV7 request slot {req_index} has stale owner "
                f"{self.req_slot_owners[req_index]!r}, expected {req_id!r}"
            )
        row = self.req_slot_to_row[req_index]
        if row == -1:
            raise RuntimeError(f"RWKV state for request id {req_id!r} missing")
        write_ref = self._pending_write_ref_by_id.pop(req_id, "")
        reserved = self._reserved_state_bytes_by_req_id.pop(req_id, 0)
        parent_ref = self._parent_state_ref_by_req_id.pop(req_id, "")
        if parent_ref:
            active_readers = self._active_state_readers.get(parent_ref, 0) - 1
            if active_readers > 0:
                self._active_state_readers[parent_ref] = active_readers
            else:
                self._active_state_readers.pop(parent_ref, None)
        finish_reason = None if finished_data is None else finished_data.finish_reason
        if (
            write_ref
            and finished_data is not None
            and finish_reason in _COMMIT_FINISH_REASONS
            and reserved
        ):
            try:
                self.capture_state_ref(
                    write_ref,
                    row,
                    parent_ref=parent_ref or None,
                    processed_token_count=int(self.elapsed[row].item()),
                    pending_tail_token_ids=finished_data.pending_tail_token_ids,
                    finish_reason=finish_reason,
                    reserved_nbytes=reserved,
                )
            except Exception:
                self._reserved_state_cache_bytes -= reserved
                reserved = 0
                raise
            else:
                reserved = 0
        if reserved:
            self._reserved_state_cache_bytes -= reserved
        self.req_id_to_index.pop(req_id)
        if req_index in self.decode_req_slots:
            self._remove_decode_row(req_index, row)
        else:
            self._zero_row(row)
            self.req_slot_to_row[req_index] = -1
            self.row_to_req_slot[row] = -1
            self.req_slot_owners[req_index] = None
            self.free_rows.add(row)
        self.req_prompt_token_ids.pop(req_id, None)
        self.req_prefix_cache_hit_lengths.pop(req_id, None)
        self.prefix_cache_eligible_req_ids.discard(req_id)

    def _remove_decode_row(self, req_index: int, row: int) -> None:
        self._zero_row(row)
        self.decode_req_slots.remove(req_index)
        self.req_slot_to_row[req_index] = -1
        self.row_to_req_slot[row] = -1
        self.req_slot_owners[req_index] = None
        self.free_rows.add(row)

    def _mark_resident_row_decode(self, req_slot: int) -> int:
        current_row = self.req_slot_to_row[req_slot]
        if current_row == -1:
            raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
        self.decode_req_slots.add(req_slot)
        return current_row

    def _validate_decode_membership(self) -> None:
        for req_slot in self.decode_req_slots:
            owner = self.req_slot_owners[req_slot]
            if owner is None or self.req_id_to_index.get(owner) != req_slot:
                raise RuntimeError("RWKV7 decode request slot has a stale owner")
            row = self.req_slot_to_row[req_slot]
            if row < 0 or row >= self.max_num_reqs:
                raise RuntimeError("RWKV7 live decode resident row is out of range")
            if self.row_to_req_slot[row] != req_slot:
                raise RuntimeError("RWKV7 decode resident mapping is inconsistent")

    def _zero_row(self, row: int) -> None:
        self.shift_state[:, :, row].zero_()
        self.wkv_state[:, row].zero_()
        self.elapsed[row].zero_()

    def reset_after_weight_update(self) -> None:
        active_rows = self.max_num_reqs - len(self.free_rows)
        if active_rows:
            logger.warning(
                "Resetting RWKV7 state after weight update with %d active rows. "
                "The trainer should quiesce requests before updating weights.",
                active_rows,
            )
        self.shift_state.zero_()
        self.wkv_state.zero_()
        self.elapsed.zero_()
        if self.prefix_state_cache is not None:
            self.prefix_state_cache.clear()
        self._clear_state_cache()
        self.req_prefix_cache_hit_lengths = {
            req_id: 0 for req_id in self.req_id_to_index
        }
        # Level-2 sleep discards allocator-backed tensors. Rebuild the static
        # decode metadata that is not restored by checkpoint weight streaming.
        torch.arange(self.max_num_reqs, out=self.execution_idx_mapping)
        torch.arange(
            self.max_num_reqs + 1,
            out=self.decode_query_start_loc,
        )

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        self._set_sampling_logits_fast_path(input_batch, False)
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []
        self._prefill_cache_targets = []
        req_ids = getattr(input_batch, "req_ids", None)
        is_dummy_batch = (
            req_ids is not None
            and bool(req_ids)
            and all(req_id not in self.req_id_to_index for req_id in req_ids)
        )
        if is_dummy_batch:
            if not self.req_id_to_index:
                self._reset_mappings()
            query_start_loc_np = getattr(
                input_batch,
                "query_start_loc_np",
                None,
            )
            if query_start_loc_np is None:
                query_start_loc = input_batch.query_start_loc
                if (
                    not isinstance(query_start_loc, torch.Tensor)
                    or query_start_loc.device.type != "cpu"
                ):
                    raise RuntimeError("RWKV7 dummy prefill requires CPU query offsets")
                query_offsets = [int(offset) for offset in query_start_loc.tolist()]
            else:
                query_offsets = [int(offset) for offset in query_start_loc_np]
            prefill_ranges = [
                (batch_idx, query_offsets[batch_idx], query_offsets[batch_idx + 1])
                for batch_idx in range(input_batch.num_reqs)
            ]
            dummy_prefill_rows = list(range(input_batch.num_reqs))
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": self.execution_idx_mapping[: input_batch.num_reqs],
                "rwkv_prefill_token_ranges": prefill_ranges,
                "rwkv_prefill_rows": dummy_prefill_rows,
                **self._packed_prefill_inputs(prefill_ranges, dummy_prefill_rows),
                **self._new_dummy_state_tensors(input_batch.num_reqs),
            }

        query_start_loc_np = getattr(input_batch, "query_start_loc_np", None)
        if query_start_loc_np is None:
            raise RuntimeError("RWKV7 requires CPU query_start_loc metadata")
        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None)
        if is_prefilling_np is None:
            raise RuntimeError("RWKV7 requires CPU prefill metadata")

        batch_entries: list[tuple[int, int, bool, int, int]] = []
        for batch_idx in range(len(input_batch.idx_mapping_np)):
            req_slot = self._state_slot_for_batch_entry(input_batch, batch_idx)
            current_row = self.req_slot_to_row[req_slot]
            if current_row == -1:
                raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
            start = int(query_start_loc_np[batch_idx])
            end = int(query_start_loc_np[batch_idx + 1])
            query_len = end - start
            is_prefill = bool(is_prefilling_np[batch_idx]) or query_len > 1
            batch_entries.append((batch_idx, req_slot, is_prefill, start, end))

        decode_entries: list[tuple[int, int, int, int]] = []
        live_decode_req_slots = set(self.decode_req_slots)
        scheduled_decode_req_slots: set[int] = set()
        prefill_entries: list[tuple[int, int, int, bool, int, int, int]] = []
        for batch_idx, req_slot, is_prefill, start, end in batch_entries:
            current_row = self.req_slot_to_row[req_slot]
            if current_row == -1:
                raise RuntimeError(f"RWKV state for request slot {req_slot} missing")
            if is_prefill:
                if start >= end:
                    raise RuntimeError(
                        "RWKV7 fast prefill requires positive request lengths."
                    )
                num_computed_prefill = getattr(
                    input_batch, "num_computed_prefill_tokens_np", None
                )
                prefill_len = getattr(input_batch, "prefill_len_np", None)
                scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
                becomes_decode = False
                if (
                    num_computed_prefill is not None
                    and prefill_len is not None
                    and scheduled_tokens is not None
                ):
                    becomes_decode = int(num_computed_prefill[batch_idx]) + int(
                        scheduled_tokens[batch_idx]
                    ) >= int(prefill_len[batch_idx])
                effective_start = start
                cache_target = 0
                req_id = self.req_slot_owners[req_slot]
                cache_hit_length = (
                    self.req_prefix_cache_hit_lengths.get(req_id, 0)
                    if req_id is not None
                    else 0
                )
                if cache_hit_length:
                    if num_computed_prefill is None:
                        raise RuntimeError(
                            "RWKV7 prefix cache requires computed-token metadata"
                        )
                    computed = int(num_computed_prefill[batch_idx])
                    effective_start += min(
                        max(cache_hit_length - computed, 0), end - start
                    )
                if req_id in self.prefix_cache_eligible_req_ids:
                    if num_computed_prefill is None:
                        raise RuntimeError(
                            "RWKV7 prefix cache requires computed-token metadata"
                        )
                    cache_target = int(num_computed_prefill[batch_idx]) + (end - start)
                if effective_start < end:
                    prefill_entries.append(
                        (
                            batch_idx,
                            req_slot,
                            current_row,
                            becomes_decode,
                            effective_start,
                            end,
                            cache_target,
                        )
                    )
            else:
                decode_row = self._mark_resident_row_decode(req_slot)
                scheduled_decode_req_slots.add(req_slot)
                decode_entries.append((batch_idx, req_slot, decode_row, start))
        if scheduled_decode_req_slots:
            missing_decode_req_slots = (
                live_decode_req_slots - scheduled_decode_req_slots
            )
            if missing_decode_req_slots:
                raise RuntimeError(
                    "RWKV7 native decode requires scheduling all live decode "
                    "rows; missing request slots "
                    f"{sorted(missing_decode_req_slots)}"
                )
        if decode_entries:
            self._validate_decode_membership()
        scheduled_rows = [
            self.req_slot_to_row[req_slot]
            for _batch_idx, req_slot, _is_prefill, _start, _end in batch_entries
        ]
        idx_mapping = torch.tensor(
            scheduled_rows,
            dtype=torch.int32,
            device=self.device,
        )
        source_decode_rows = [
            row for _batch_idx, _req_slot, row, _start in decode_entries
        ]
        decode_token_positions = [
            start for _batch_idx, _req_slot, _row, start in decode_entries
        ]
        use_contiguous_decode = self._is_contiguous_decode_context(
            source_decode_rows,
            decode_token_positions,
        )
        if decode_entries and not use_contiguous_decode:
            decode_len = len(decode_entries)
            self.decode_slot_indices[:decode_len].copy_(
                torch.tensor(
                    source_decode_rows,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.decode_token_positions[:decode_len].copy_(
                torch.tensor(
                    decode_token_positions,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            slot_indices = self.decode_slot_indices[:decode_len]
            decode_token_position_tensor = self.decode_token_positions[:decode_len]
        elif decode_entries:
            slot_indices = None
            decode_token_position_tensor = decode_token_positions
        else:
            slot_indices = None
            decode_token_position_tensor = None
        decode_rows = source_decode_rows if decode_entries else []
        decode_context_size = len(decode_rows)
        decode_state_tensors = {
            "shift_state": self.shift_state,
            "wkv_state": self.wkv_state,
            "elapsed": self.elapsed,
        }

        if not prefill_entries:
            self._set_sampling_logits_fast_path(
                input_batch,
                bool(decode_entries and len(decode_entries) == input_batch.num_reqs),
            )
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                **decode_state_tensors,
                "rwkv_decode_batch_size": decode_context_size,
                "rwkv_decode_rows": decode_rows,
                "rwkv_decode_token_positions": decode_token_position_tensor,
                "rwkv_decode_query_start_loc": self.decode_query_start_loc[
                    : decode_context_size + 1
                ],
                "slot_indices": slot_indices,
            }

        prefill_rows: list[int] = []
        for (
            _batch_idx,
            req_slot,
            _decode_row,
            becomes_decode,
            _start,
            _end,
            cache_target,
        ) in prefill_entries:
            prefill_rows.append(self.req_slot_to_row[req_slot])
            self._prefill_req_slots.append(req_slot)
            self._prefill_becomes_decode.append(becomes_decode)
            self._prefill_cache_targets.append(cache_target)

        prefill_ranges = [
            (batch_idx, start, end)
            for (
                batch_idx,
                _req_slot,
                _row,
                _becomes_decode,
                start,
                end,
                _cache_target,
            ) in prefill_entries
        ]
        prefill_lengths = [end - start for _batch_idx, start, end in prefill_ranges]
        has_positive_prefill_lengths = all(length > 0 for length in prefill_lengths)
        if not has_positive_prefill_lengths:
            raise RuntimeError("RWKV7 fast prefill requires positive request lengths.")
        prefill_varlen_inputs = self._packed_prefill_inputs(
            prefill_ranges,
            prefill_rows,
        )
        if len(prefill_entries) == input_batch.num_reqs:
            return {
                "query_start_loc": input_batch.query_start_loc,
                "idx_mapping": idx_mapping,
                "rwkv_prefill_token_ranges": prefill_ranges,
                "rwkv_prefill_rows": prefill_rows,
                **prefill_varlen_inputs,
                "shift_state": self.shift_state,
                "wkv_state": self.wkv_state,
                "elapsed": self.elapsed,
            }
        mixed_inputs = {
            "query_start_loc": input_batch.query_start_loc,
            "idx_mapping": idx_mapping,
            **decode_state_tensors,
            "rwkv_decode_batch_size": decode_context_size,
            "rwkv_decode_rows": decode_rows,
            "rwkv_decode_token_positions": decode_token_position_tensor,
            "rwkv_decode_query_start_loc": self.decode_query_start_loc[
                : decode_context_size + 1
            ],
            "slot_indices": slot_indices,
            "rwkv_prefill_token_ranges": prefill_ranges,
            "rwkv_prefill_rows": prefill_rows,
            **prefill_varlen_inputs,
        }
        if decode_entries:
            mixed_inputs.update(
                {
                    "prefill_shift_state": self.shift_state,
                    "prefill_wkv_state": self.wkv_state,
                    "prefill_elapsed": self.elapsed,
                }
            )
        return mixed_inputs

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        if not self._prefill_req_slots:
            return
        for scratch_row, req_slot in enumerate(self._prefill_req_slots):
            self._cache_row(req_slot, self._prefill_cache_targets[scratch_row])
            if self._prefill_becomes_decode[scratch_row]:
                self._mark_resident_row_decode(req_slot)
        self._validate_decode_membership()
        self._prefill_req_slots = []
        self._prefill_becomes_decode = []
        self._prefill_cache_targets = []

    def has_pending_postprocess_state(self) -> bool:
        return bool(self._prefill_req_slots)

    @staticmethod
    def can_replay_full_cudagraph(model_inputs: dict[str, Any]) -> bool:
        """Return whether runtime decode matches the static contiguous graph."""
        decode_batch_size = model_inputs.get("rwkv_decode_batch_size")
        decode_rows = model_inputs.get("rwkv_decode_rows")
        positions = model_inputs.get("positions")
        if (
            not isinstance(decode_batch_size, int)
            or decode_batch_size <= 0
            or not isinstance(decode_rows, list)
            or not isinstance(positions, torch.Tensor)
        ):
            return False
        return (
            model_inputs.get("slot_indices") is None
            and decode_batch_size == positions.shape[0]
            and decode_rows == list(range(decode_batch_size))
        )

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        lengths = torch.full(
            (num_reqs,),
            num_tokens // num_reqs,
            dtype=torch.int32,
            device="cpu",
        )
        lengths[: num_tokens % num_reqs] += 1
        query_start_loc = torch.empty((num_reqs + 1,), dtype=torch.int32)
        query_start_loc[0] = 0
        query_start_loc[1:] = lengths.cumsum(dim=0)
        idx_mapping = self.execution_idx_mapping[:num_reqs]
        # Full CUDAGraph replay binds captured state pointers, so decode capture
        # uses resident buffers. Request-level dummy profiling uses scratch.
        state_tensors = {
            "shift_state": self.shift_state,
            "wkv_state": self.wkv_state,
            "elapsed": self.elapsed,
        }
        if num_tokens == num_reqs:
            return {
                "query_start_loc": query_start_loc,
                "idx_mapping": idx_mapping,
                **state_tensors,
                "rwkv_decode_batch_size": num_reqs,
                "rwkv_decode_rows": list(range(num_reqs)),
                "rwkv_decode_token_positions": list(range(num_reqs)),
                "rwkv_decode_query_start_loc": self.decode_query_start_loc[
                    : num_reqs + 1
                ],
                "slot_indices": None,
            }
        prefill_ranges = [
            (
                request,
                int(query_start_loc[request]),
                int(query_start_loc[request + 1]),
            )
            for request in range(num_reqs)
        ]
        prefill_rows = list(range(num_reqs))
        return {
            "query_start_loc": query_start_loc,
            "idx_mapping": idx_mapping,
            **state_tensors,
            "rwkv_prefill_token_ranges": prefill_ranges,
            "rwkv_prefill_rows": prefill_rows,
            **self._packed_prefill_inputs(prefill_ranges, prefill_rows),
        }

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        return {}
