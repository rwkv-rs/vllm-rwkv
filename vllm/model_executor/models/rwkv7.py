# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only RWKV7 model."""

import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

HEAD_SIZE = 64
DTYPE = torch.float16
CUDA_DEVICE = torch.device("cuda")
LOWRANK_SUFFIXES = (
    "att.w1",
    "att.w2",
    "att.a1",
    "att.a2",
    "att.g1",
    "att.g2",
    "att.v1",
    "att.v2",
)
LOWRANK_IN_ROWS_T = 7
LOWRANK_OUT_ROWS_T = 4
LOWRANK_FUSED_MIN_C = 1024
# Exact FP16-accumulation cuBLASLt winners measured on the project benchmark
# GPU (Blackwell sm_120, CUDA 13.0, PyTorch 2.11) against the current GemmEx
# baseline. The second value is a runtime heuristic-list index, not a stable
# algorithm ID, so production uses non-strict lookup and exact shape guards.
ATT_C2C_FP16_LT_4096 = {64: (32, 2)}
FFN_DOWN_FP16_LT_4096 = {64: (32, 3)}
# Exact runtime-layout low-rank candidates. Keep these separate from the
# dense tables: the rank is part of the dispatch contract and the aggregate
# benefit comes from repeated W/A/G/V projections at the same row count.
LOWRANK_IN_FP16_LT_4096 = {
    (8, 128): (128, 0),
    (8, 480): (128, 0),
    (8, 96): (128, 0),
    (16, 480): (32, 1),
    (16, 96): (32, 0),
}
LOWRANK_OUT_FP16_LT_4096 = {
    (8, 128): (0, 5),
    (8, 480): (0, 3),
    (8, 96): (0, 4),
    (16, 128): (0, 1),
    (16, 480): (32, 2),
}
CMIX_NOFC_ROW20_MAX_T = 5
CMIX_NOFC_MAX_ROWS = 19
CMIX_NOFC_T512_MIN_ROWS = 8
LN1_TMIX_FUSE = True
# The fused LN1/TMix owner is strongest for M=1 and for larger decode batches,
# but paired full-model measurements favor the separate kernels at these exact
# small T=1 shapes. Keep the exclusion table narrow: disabling the fusion also
# regresses rows >= 64 and the canonical B320 decode path.
LN1_TMIX_UNFUSED_BT = frozenset(((2, 1), (4, 1), (8, 1), (16, 1)))
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"
LNX_WARP_B1_T_4096 = frozenset((64, 96, 128, 160, 192, 240, 248, 264, 512))
MIX_3D_B1_T_4096 = frozenset((2, 4, 16, 64, 512))
# The direct-model harness clears this exact row set for paired A/B capture;
# production keeps M=1 grouped without a user-visible mode or fallback knob.
M1_RKV_GROUPED_ROWS = {1}
# The M=1 no-fc path prepares the sparse-down accumulator while reducing the
# FFN key split-K result, eliminating one zeroing launch per layer.
M1_CMIX_PREZERO_ROWS = {1}


@dataclass(frozen=True)
class RWKV7ExecutionProfile:
    wkv_mode: str
    wkv_state_dtype: torch.dtype
    allow_fp16_accumulation: bool
    gemm_accumulation_policy: str


def resolve_execution_profile(wkv_mode: str) -> RWKV7ExecutionProfile:
    if wkv_mode == "fp32io16":
        return RWKV7ExecutionProfile(
            wkv_mode=wkv_mode,
            wkv_state_dtype=torch.float32,
            allow_fp16_accumulation=False,
            gemm_accumulation_policy="fp32",
        )
    if wkv_mode == "fp16":
        return RWKV7ExecutionProfile(
            wkv_mode=wkv_mode,
            wkv_state_dtype=torch.float16,
            allow_fp16_accumulation=True,
            gemm_accumulation_policy="fp16",
        )
    raise ValueError(
        f"VLLM_RWKV7_WKV_MODE={wkv_mode!r} is invalid for RWKV7. "
        "Expected one of: fp16, fp32io16."
    )


@dataclass(frozen=True)
class PathConfig:
    rows: int
    cmix_mode: str


def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B * T
    use_nofc = rows <= CMIX_NOFC_MAX_ROWS or (rows == 20 and CMIX_NOFC_ROW20_MAX_T >= T)
    cmix_mode = (
        CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if use_nofc else CMIX_DENSE)
    )
    return PathConfig(rows=rows, cmix_mode=cmix_mode)


def use_tmix_kk_a_gate_2d(B: int, T: int, C: int, H: int) -> bool:
    """Use the exact 2D head grid validated for the 7.2B model shape."""
    return C == 4096 and H == 64 and 0 < B * T <= 65535


def use_tmix_lnx_warp(B: int, T: int, C: int, H: int) -> bool:
    """Use the one-warp LNX reduction only on validated batched shapes."""
    return (
        C == 4096
        and H == 64
        and B * T * H >= 4096
        and (B >= 2 or T in LNX_WARP_B1_T_4096)
    )


def use_tmix_mix6_3d(B: int, T: int, C: int) -> bool:
    """Use the validated B/T/channel launch grid for time mixing."""
    return (
        T != 1
        and C == 4096
        and 0 < B <= 65535
        and 0 < T <= 65535
        and (B >= 2 or T in MIX_3D_B1_T_4096)
    )


def use_cmix_mix_3d(B: int, T: int, C: int) -> bool:
    """Use the validated B/T/channel launch grid for channel mixing."""
    return use_tmix_mix6_3d(B, T, C)


def use_ln1_tmix_fusion(B: int, T: int) -> bool:
    """Choose the benchmark-backed LN1/TMix owner for a B/T shape."""
    return LN1_TMIX_FUSE and T == 1 and (B, T) not in LN1_TMIX_UNFUSED_BT


def is_lowrank_weight(key: str) -> bool:
    return key.endswith(LOWRANK_SUFFIXES)


def can_use_lowrank_fused(hidden_size: int, rows: int) -> bool:
    return hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_IN_ROWS_T


def can_use_lowrank_out_fused(hidden_size: int, rows: int) -> bool:
    return hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T


class RWKV7ForCausalLM(nn.Module):
    is_attention_free = True
    requires_uniform_decode_wave = True
    supports_pp = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Create the RWKV7 inference module; weights are loaded by vLLM."""
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.prefix = prefix
        self._validate_torch_compile_unsupported()

        hidden_size = int(getattr(self.config, "hidden_size", 0))
        vocab_size = int(getattr(self.config, "vocab_size", 0))
        head_size = int(getattr(self.config, "head_size", HEAD_SIZE))
        if hidden_size and head_size:
            num_attention_heads = hidden_size // head_size
        else:
            num_attention_heads = int(getattr(self.config, "num_attention_heads", 0))
            head_size = (
                hidden_size // num_attention_heads
                if hidden_size and num_attention_heads
                else HEAD_SIZE
            )
        num_hidden_layers = int(
            getattr(
                self.config,
                "num_hidden_layers",
                getattr(self.config, "n_layer", 0),
            )
        )

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.z: dict[str, torch.Tensor] = {}
        self.raw_weight_names: set[str] | None = None
        self.raw_weight_shapes: dict[str, tuple[int, ...]] | None = None
        self.total_num_layers = num_hidden_layers
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if num_attention_heads % self.tp_size != 0:
            raise ValueError(
                "RWKV7 requires num_attention_heads "
                f"({num_attention_heads}) to be divisible "
                f"by tensor_parallel_size ({self.tp_size})."
            )
        self.tp_num_heads = num_attention_heads // self.tp_size
        self.tp_hidden_size = self.tp_num_heads * head_size
        self.vocab_size_padded = self._get_padded_vocab_size(vocab_size)
        self.execution_profile = resolve_execution_profile(envs.VLLM_RWKV7_WKV_MODE)
        self.wkv_mode = self.execution_profile.wkv_mode
        self.wkv_state_dtype = self.execution_profile.wkv_state_dtype
        self.allow_fp16_accumulation = self.execution_profile.allow_fp16_accumulation
        self.logits_processor = LogitsProcessor(vocab_size, logits_as_input=True)
        self.register_buffer("_dummy_param", torch.empty(0), persistent=False)

    def _get_layer_range(self) -> tuple[int, int]:
        parallel_config = getattr(self.vllm_config, "parallel_config", None)
        if parallel_config is None:
            return 0, self.total_num_layers
        get_indices = getattr(self.model_config, "get_layers_start_end_indices", None)
        if get_indices is not None:
            return get_indices(parallel_config)
        return 0, self.total_num_layers

    def _is_pp_first_rank(self) -> bool:
        return get_pp_group().is_first_rank

    def _is_pp_last_rank(self) -> bool:
        return get_pp_group().is_last_rank

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        hidden_size = self.hidden_size
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, hidden_size), dtype=DTYPE, device=device
                ),
                "v_first": torch.zeros(
                    (batch_size, hidden_size), dtype=DTYPE, device=device
                ),
            }
        )

    def _get_padded_vocab_size(self, vocab_size: int) -> int:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size <= 1:
            return vocab_size
        return ((vocab_size + tp_size - 1) // tp_size) * tp_size

    def _tp_vocab_range(self, vocab_size: int | None = None) -> tuple[int, int, int]:
        vocab_size = self.vocab_size if vocab_size is None else vocab_size
        tp_size = getattr(self, "tp_size", 1)
        tp_rank = getattr(self, "tp_rank", 0)
        padded_vocab_size = self._get_padded_vocab_size(vocab_size)
        per_rank = padded_vocab_size // tp_size
        start = tp_rank * per_rank
        end = min(start + per_rank, vocab_size)
        return start, end, per_rank

    def _tp_slice(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        size = value.shape[dim]
        if size % tp_size != 0:
            raise ValueError(
                f"Cannot shard tensor dimension {dim} with size {size} "
                f"across tensor_parallel_size={tp_size}."
            )
        shard_size = size // tp_size
        start = getattr(self, "tp_rank", 0) * shard_size
        return value.narrow(dim, start, shard_size).contiguous()

    def _tp_hidden_slice(self, value: torch.Tensor, dim: int) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        shard_size = getattr(self, "tp_hidden_size", value.shape[dim])
        start = getattr(self, "tp_rank", 0) * shard_size
        return value.narrow(dim, start, shard_size).contiguous()

    def _tp_vocab_slice(self, value: torch.Tensor) -> torch.Tensor:
        tp_size = getattr(self, "tp_size", 1)
        if tp_size == 1:
            return value
        start, end, per_rank = self._tp_vocab_range(value.shape[0])
        out = value.new_zeros((per_rank, *value.shape[1:]))
        if end > start:
            out[: end - start].copy_(value[start:end])
        return out.contiguous()

    def _shard_weight_for_tp(self, key: str, value: torch.Tensor) -> torch.Tensor:
        if getattr(self, "tp_size", 1) == 1:
            return value
        if key in ("emb.weight", "head.weight"):
            return self._tp_vocab_slice(value)
        parts = key.split(".")
        if len(parts) < 4 or parts[0] != "blocks":
            return value
        submodule = parts[2]
        name = ".".join(parts[3:])
        if submodule == "att":
            if name == "r_k":
                return self._tp_slice(value, 0)
            if name in {
                "ln_x.weight",
                "ln_x.bias",
                "k_k",
                "k_a",
                "w0",
                "a0",
                "v0",
            }:
                return self._tp_hidden_slice(value, 0)
            if name in {"receptance.weight", "key.weight", "value.weight"}:
                return self._tp_hidden_slice(value, 0)
            if name == "output.weight":
                return self._tp_hidden_slice(value, 1)
            if name in {"w2", "a2", "g2", "v2"}:
                return self._tp_hidden_slice(value, 1)
        elif submodule == "ffn":
            if name == "key.weight":
                return self._tp_slice(value, 0)
            if name == "value.weight":
                return self._tp_slice(value, 1)
        return value

    def _tp_all_reduce(self, value: torch.Tensor) -> torch.Tensor:
        if getattr(self, "tp_size", 1) == 1:
            return value
        return tensor_model_parallel_all_reduce(value)

    def _is_weight_needed_on_rank(self, key: str) -> bool:
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", getattr(self, "total_num_layers", 0))
        total_layers = getattr(self, "total_num_layers", 0)
        if start_layer == 0 and end_layer >= total_layers:
            return True
        if key == "emb.weight" or key.startswith("blocks.0.ln0."):
            return start_layer == 0
        if key == "head.weight" or key.startswith("ln_out."):
            return end_layer >= total_layers
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "blocks":
            return start_layer <= int(parts[1]) < end_layer
        return True

    def _validate_torch_compile_unsupported(self) -> None:
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        if getattr(compilation_config, "mode", None) not in (
            None,
            CompilationMode.NONE,
        ):
            raise ValueError(
                "RWKV7 does not support torch.compile. Use non-compiled "
                "execution with CompilationMode.NONE."
            )

    @classmethod
    def get_model_state_cls(cls):
        from vllm.v1.worker.gpu.model_states.rwkv import RWKV7ModelState

        return RWKV7ModelState

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if hasattr(self, "_pending_weight_update"):
            pending = self._pending_weight_update
            loaded_names = set()
            for name, weight in weights:
                loaded_names.add(name)
                if self._weight_update_error is not None:
                    pending.setdefault(name, None)
                    continue
                try:
                    if (
                        self.raw_weight_names is None
                        or name not in self.raw_weight_names
                    ):
                        raise ValueError(
                            f"RWKV7 weight update received unexpected key {name!r}"
                        )
                    if name in pending:
                        raise ValueError(
                            f"RWKV7 weight update received duplicate key {name!r}"
                        )

                    detached = weight.detach()
                    expected_shape = (
                        getattr(self, "raw_weight_shapes", None) or {}
                    ).get(name)
                    if self._streaming_weight_update and expected_shape is None:
                        raise RuntimeError(
                            f"RWKV7 streaming update has no raw shape for {name!r}"
                        )
                    actual_shape = tuple(detached.shape)
                    if expected_shape is not None and actual_shape != expected_shape:
                        raise ValueError(
                            "RWKV7 raw weight shape mismatch for "
                            f"{name}: expected {expected_shape}, got {actual_shape}"
                        )

                    if self._streaming_weight_update:
                        if (
                            name in self._embedding_update_keys()
                            and self._is_weight_needed_on_rank(name)
                        ):
                            pending[name] = detached.to(device="cpu", copy=True)
                        else:
                            self._copy_raw_weight_to_runtime(name, detached)
                            pending[name] = None
                    else:
                        pending[name] = detached.to(device="cpu", copy=True)
                except Exception as exc:
                    self._record_weight_update_error(exc)
                    pending.setdefault(name, None)
            return loaded_names

        z = {name: weight.detach().cpu() for name, weight in weights}
        raw_weight_names = set(z.keys())
        self.raw_weight_names = raw_weight_names
        self.raw_weight_shapes = {
            name: tuple(weight.shape) for name, weight in z.items()
        }
        self._preprocess_weights(z)
        self._commit_preprocessed_weights(z)
        return raw_weight_names

    def start_weight_update(self) -> bool:
        """Handle checkpoint-format dense weight update chunks internally."""
        if hasattr(self, "_pending_weight_update"):
            raise RuntimeError("RWKV7 weight update is already active")
        streaming_weight_update = self._can_stream_weight_update()
        if (
            os.getenv("VLLM_RWKV7_STRICT_STREAMING_WEIGHT_UPDATE") == "1"
            and not streaming_weight_update
        ):
            raise RuntimeError(
                "strict RWKV7 online publication requires in-place streaming "
                "weight update capability; full-model staging is forbidden"
            )
        self._streaming_weight_update = streaming_weight_update
        self._pending_weight_update: dict[str, torch.Tensor | None] = {}
        self._weight_update_error: Exception | None = None
        return True

    def finish_weight_update(self) -> None:
        pending = getattr(self, "_pending_weight_update", None)
        if pending is None:
            raise RuntimeError("RWKV7 weight update is not active")
        try:
            if self.raw_weight_names is None:
                raise RuntimeError(
                    "RWKV7 online weight update requires a previous full "
                    "checkpoint load to establish raw weight names."
                )
            if self._weight_update_error is not None:
                raise self._weight_update_error
            received = set(pending.keys())
            missing = self.raw_weight_names - received
            unexpected = received - self.raw_weight_names
            if missing or unexpected:
                raise ValueError(
                    "RWKV7 weight update key mismatch: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            if self._streaming_weight_update:
                self._finish_streaming_weight_update(pending)
                torch.accelerator.synchronize()
                logger.info("RWKV7 streaming weight update is ready")
            else:
                z = {
                    name: value for name, value in pending.items() if value is not None
                }
                old_z = self.z
                try:
                    self._preprocess_weights(z)
                    self._commit_preprocessed_weights(
                        z,
                        reuse_existing_tensors=True,
                        existing_z=old_z,
                    )
                except Exception:
                    self.z = old_z
                    raise
        finally:
            self.abort_weight_update()

    def abort_weight_update(self) -> None:
        for name in (
            "_pending_weight_update",
            "_streaming_weight_update",
            "_weight_update_error",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _record_weight_update_error(self, error: Exception) -> None:
        if self._weight_update_error is None:
            self._weight_update_error = error

    @staticmethod
    def _embedding_update_keys() -> frozenset[str]:
        return frozenset(
            {
                "emb.weight",
                "blocks.0.ln0.weight",
                "blocks.0.ln0.bias",
            }
        )

    def _runtime_keys_for_raw_weight(self, key: str) -> tuple[str, ...]:
        if not self._is_weight_needed_on_rank(key):
            return ()
        if is_lowrank_weight(key):
            return key, key + ".t"
        return (key,)

    def _can_stream_weight_update(self) -> bool:
        """Return whether every local raw weight has a runtime destination."""
        if self.raw_weight_names is None or not self.z:
            return False
        raw_shapes = getattr(self, "raw_weight_shapes", None)
        if raw_shapes is None or set(raw_shapes) != self.raw_weight_names:
            return False
        return all(
            runtime_key in self.z
            for raw_key in self.raw_weight_names
            for runtime_key in self._runtime_keys_for_raw_weight(raw_key)
        )

    def _runtime_views_for_raw_weight(
        self, key: str, weight: torch.Tensor
    ) -> tuple[tuple[str, torch.Tensor], ...]:
        if not self._is_weight_needed_on_rank(key):
            return ()
        value = self._shard_weight_for_tp(key, weight.squeeze())
        lowrank = is_lowrank_weight(key)
        if not lowrank and (
            "key.weight" in key
            or "value.weight" in key
            or "receptance.weight" in key
            or "output.weight" in key
            or "head.weight" in key
        ):
            value = value.t()
        if key.endswith("att.r_k"):
            value = value.flatten()
        if lowrank:
            return ((key, value), (key + ".t", value.t()))
        return ((key, value),)

    def _copy_raw_weight_to_runtime(self, key: str, weight: torch.Tensor) -> None:
        with torch.no_grad():
            for runtime_key, value in self._runtime_views_for_raw_weight(key, weight):
                destination = self.z.get(runtime_key)
                if destination is None:
                    raise RuntimeError(
                        f"RWKV7 streaming update has no destination for {runtime_key!r}"
                    )
                if destination.shape != value.shape:
                    raise ValueError(
                        "RWKV7 streaming weight shape mismatch for "
                        f"{runtime_key}: expected {tuple(destination.shape)}, "
                        f"got {tuple(value.shape)}"
                    )
                destination.copy_(value)

    def _finish_streaming_weight_update(
        self, pending: dict[str, torch.Tensor | None]
    ) -> None:
        if not self._is_weight_needed_on_rank("emb.weight"):
            return
        embedding_inputs = self._embedding_update_keys()
        if not embedding_inputs.issubset(pending):
            missing = sorted(embedding_inputs - pending.keys())
            raise ValueError(
                f"RWKV7 streaming embedding update is incomplete: missing={missing}"
            )
        emb_src = pending["emb.weight"]
        ln0_w_src = pending["blocks.0.ln0.weight"]
        ln0_b_src = pending["blocks.0.ln0.bias"]
        assert emb_src is not None and ln0_w_src is not None and ln0_b_src is not None

        self._copy_raw_weight_to_runtime("blocks.0.ln0.weight", ln0_w_src)
        self._copy_raw_weight_to_runtime("blocks.0.ln0.bias", ln0_b_src)

        destination = self.z["emb.weight"]
        vocab_start, vocab_end, _ = self._tp_vocab_range(self.vocab_size)
        local_rows = vocab_end - vocab_start
        if (
            destination.shape[0] < local_rows
            or destination.shape[1] != self.hidden_size
        ):
            raise ValueError(
                "RWKV7 streaming embedding destination shape mismatch: "
                f"destination={tuple(destination.shape)}, rows={local_rows}, "
                f"hidden_size={self.hidden_size}"
            )
        device = destination.device
        emb_input = (
            emb_src.squeeze()[vocab_start:vocab_end]
            .to(device=device, non_blocking=False)
            .contiguous()
        )
        ln0_w = ln0_w_src.squeeze().to(device=device, non_blocking=False).contiguous()
        ln0_b = ln0_b_src.squeeze().to(device=device, non_blocking=False).contiguous()
        transformed = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
            emb_input, ln0_w, ln0_b
        )
        if transformed.shape != destination[:local_rows].shape:
            raise ValueError(
                "RWKV7 streaming embedding transform shape mismatch: "
                f"expected {tuple(destination[:local_rows].shape)}, "
                f"got {tuple(transformed.shape)}"
            )
        with torch.no_grad():
            destination.zero_()
            destination[:local_rows].copy_(transformed)

    def get_parameter(self, target: str) -> nn.Parameter:
        if target == "_dummy_param":
            return super().get_parameter(target)
        raise NotImplementedError(
            "RWKV7 does not support direct kernel-format weight updates; "
            "use checkpoint-format dense weight update instead."
        )

    def _commit_preprocessed_weights(
        self,
        z: dict[str, torch.Tensor],
        *,
        reuse_existing_tensors: bool = False,
        existing_z: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if reuse_existing_tensors and existing_z is not None:
            committed = existing_z
            for key in list(committed.keys()):
                if key not in z:
                    del committed[key]
            with torch.no_grad():
                for key, value in z.items():
                    old_value = committed.get(key)
                    if (
                        old_value is not None
                        and old_value.shape == value.shape
                        and old_value.dtype == value.dtype
                        and old_value.device == value.device
                    ):
                        old_value.copy_(value)
                    else:
                        committed[key] = value
            self.z = committed
        else:
            self.z = z
        torch.accelerator.synchronize()
        logger.info(
            "RWKV7 weights are ready L=%d C=%d H=%d N=%d V=%d",
            self.total_num_layers,
            self.hidden_size,
            self.num_attention_heads,
            self.head_size,
            self.vocab_size,
        )

    def _validate_raw_weight_shapes(self, z: dict[str, torch.Tensor]) -> None:
        r_k = z["blocks.0.att.r_k"].squeeze()
        emb = z["emb.weight"].squeeze()
        weight_heads, head_size = r_k.shape
        hidden_size = weight_heads * head_size
        vocab_size = emb.shape[0]
        max_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        num_hidden_layers = max_layer + 1

        checks = (
            ("hidden_size", hidden_size),
            ("vocab_size", vocab_size),
            ("head_size", head_size),
            ("num_hidden_layers", num_hidden_layers),
        )
        for name, actual in checks:
            expected = getattr(self.config, name, None)
            if expected is not None and int(expected) != actual:
                raise ValueError(
                    f"RWKV7 config {name}={expected} does not match raw "
                    f"checkpoint {name}={actual}."
                )

    def _preprocess_weights(self, z: dict[str, torch.Tensor]) -> None:
        """Apply the albatross faster3a weight layout preprocessing."""
        self._validate_raw_weight_shapes(z)
        num_attention_heads, head_size = z["blocks.0.att.r_k"].shape
        hidden_size = num_attention_heads * head_size
        vocab_size = z["emb.weight"].shape[0]
        assert head_size == HEAD_SIZE
        self.num_attention_heads = num_attention_heads
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        if num_attention_heads % getattr(self, "tp_size", 1) != 0:
            raise ValueError(
                "RWKV7 requires num_attention_heads "
                f"({num_attention_heads}) to be divisible "
                f"by tensor_parallel_size ({getattr(self, 'tp_size', 1)})."
            )
        max_layer = max(int(k.split(".")[1]) for k in z if k.startswith("blocks."))
        self.total_num_layers = max_layer + 1
        self.start_layer, self.end_layer = self._get_layer_range()
        self.tp_num_heads = num_attention_heads // getattr(self, "tp_size", 1)
        self.tp_hidden_size = self.tp_num_heads * head_size
        self.vocab_size_padded = self._get_padded_vocab_size(vocab_size)
        logger.info(
            "Detected RWKV7 model C=%d H=%d N=%d V=%d",
            hidden_size,
            num_attention_heads,
            head_size,
            vocab_size,
        )
        logger.info(
            "RWKV7 cmix no-fc path: rows<=%d row20_t<=%d",
            CMIX_NOFC_MAX_ROWS,
            CMIX_NOFC_ROW20_MAX_T,
        )

        emb_src = z["emb.weight"].squeeze()
        ln0_w_src = z["blocks.0.ln0.weight"].squeeze()
        ln0_b_src = z["blocks.0.ln0.bias"].squeeze()
        logger.info(
            "Preprocessing RWKV7 weights with profile=%s and GEMM accumulation=%s",
            self.execution_profile.wkv_mode,
            self.execution_profile.gemm_accumulation_policy,
        )
        for key in list(z.keys()):
            if not self._is_weight_needed_on_rank(key):
                del z[key]
                continue
            value = z[key].squeeze()
            value = self._shard_weight_for_tp(key, value)
            dev = CUDA_DEVICE
            is_lowrank = is_lowrank_weight(key)
            if not is_lowrank and (
                "key.weight" in key
                or "value.weight" in key
                or "receptance.weight" in key
                or "output.weight" in key
                or "head.weight" in key
            ):
                value = value.t()
            value = value.to(device=dev, dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if is_lowrank:
                z[key] = value
                z[key + ".t"] = value.t().contiguous()
            else:
                z[key] = value
        if self._is_weight_needed_on_rank("emb.weight"):
            emb_dev = CUDA_DEVICE
            ln0_w_bf16 = ln0_w_src.to(device=emb_dev).contiguous()
            ln0_b_bf16 = ln0_b_src.to(device=emb_dev).contiguous()
            vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(vocab_size)
            emb = torch.zeros(
                (vocab_per_rank, hidden_size), dtype=DTYPE, device=emb_dev
            )
            if vocab_end > vocab_start:
                local = torch.ops.rwkv7_v3a_ops.emb_ln0_bf16_to_f16(
                    emb_src[vocab_start:vocab_end].to(device=emb_dev).contiguous(),
                    ln0_w_bf16,
                    ln0_b_bf16,
                )
                emb[: vocab_end - vocab_start].copy_(local)
            z["emb.weight"] = emb

    def zero_state(self, B: int) -> list[torch.Tensor]:
        """Create RWKV recurrent state tensors for a batch."""
        local_heads = getattr(self, "tp_num_heads", self.num_attention_heads)
        return [
            torch.zeros(
                (self.total_num_layers, 2, B, self.hidden_size),
                dtype=DTYPE,
                device="cuda",
            ),
            torch.zeros(
                (
                    self.total_num_layers,
                    B,
                    local_heads,
                    self.head_size,
                    self.head_size,
                ),
                dtype=self.wkv_state_dtype,
                device="cuda",
            ),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | list[torch.Tensor] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        *,
        query_start_loc: torch.Tensor | None = None,
        idx_mapping: torch.Tensor | None = None,
        shift_state: torch.Tensor | None = None,
        wkv_state: torch.Tensor | None = None,
        elapsed: torch.Tensor | None = None,
        prefill_shift_state: torch.Tensor | None = None,
        prefill_wkv_state: torch.Tensor | None = None,
        prefill_elapsed: torch.Tensor | None = None,
        rwkv_decode_batch_size: int = 0,
        rwkv_decode_rows: list[int] | None = None,
        rwkv_decode_token_positions: torch.Tensor | list[int] | None = None,
        rwkv_decode_query_start_loc: torch.Tensor | None = None,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None = None,
        rwkv_prefill_rows: list[int] | None = None,
        rwkv_prefill_query_start_loc: torch.Tensor | None = None,
        rwkv_prefill_slot_indices: torch.Tensor | None = None,
        rwkv_prefill_token_positions: torch.Tensor | None = None,
        rwkv_prefill_req_id: torch.Tensor | None = None,
        slot_indices: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Run RWKV7 from Model Runner V2 request-indexed state tensors."""
        if query_start_loc is None:
            raise RuntimeError(
                "RWKV7 requires Model Runner V2 request-indexed state inputs."
            )

        assert query_start_loc is not None
        assert idx_mapping is not None
        assert shift_state is not None
        assert wkv_state is not None
        assert elapsed is not None

        is_first_pp_rank = getattr(self, "_is_pp_first_rank", lambda: True)()
        is_last_pp_rank = getattr(self, "_is_pp_last_rank", lambda: True)()
        if not (is_first_pp_rank and is_last_pp_rank):
            return self.forward_vllm_pp_stage(
                input_ids=input_ids,
                intermediate_tensors=intermediate_tensors,
                shift_state=shift_state,
                wkv_state=wkv_state,
                elapsed=elapsed,
                prefill_shift_state=prefill_shift_state,
                prefill_wkv_state=prefill_wkv_state,
                prefill_elapsed=prefill_elapsed,
                rwkv_decode_batch_size=rwkv_decode_batch_size,
                rwkv_decode_rows=rwkv_decode_rows,
                rwkv_decode_token_positions=rwkv_decode_token_positions,
                rwkv_decode_query_start_loc=rwkv_decode_query_start_loc,
                idx_mapping=idx_mapping,
                rwkv_prefill_token_ranges=rwkv_prefill_token_ranges,
                rwkv_prefill_rows=rwkv_prefill_rows,
                rwkv_prefill_query_start_loc=rwkv_prefill_query_start_loc,
                rwkv_prefill_slot_indices=rwkv_prefill_slot_indices,
                rwkv_prefill_token_positions=rwkv_prefill_token_positions,
                rwkv_prefill_req_id=rwkv_prefill_req_id,
                slot_indices=slot_indices,
                is_first_pp_rank=is_first_pp_rank,
                is_last_pp_rank=is_last_pp_rank,
            )

        assert input_ids is not None
        hidden_states = torch.empty(
            (input_ids.shape[0], self.hidden_size), dtype=DTYPE, device=CUDA_DEVICE
        )
        if rwkv_decode_rows is not None or rwkv_prefill_token_ranges is not None:
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions
            assert len(decode_rows) == self._decode_token_positions_length(
                decode_positions
            )
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                assert rwkv_decode_query_start_loc is not None
                decode_query_start_loc = rwkv_decode_query_start_loc[
                    : decode_batch_size + 1
                ]
                if slot_indices is not None:
                    decode_position_tensor = self._decode_token_positions_tensor(
                        decode_positions,
                        device=input_ids.device,
                    )
                    hidden_position_tensor = decode_position_tensor.to(
                        device=hidden_states.device
                    )
                    tokens = input_ids.index_select(0, decode_position_tensor).view(
                        decode_batch_size, 1
                    )
                    state = [shift_state, wkv_state, elapsed]
                    out = self.forward_tokens(
                        tokens,
                        state,
                        slot_indices=slot_indices[:decode_batch_size],
                        query_start_loc=decode_query_start_loc,
                        wkv_slot_indices=slot_indices[:decode_batch_size],
                    ).view(decode_batch_size, self.hidden_size)
                    hidden_states.index_copy_(0, hidden_position_tensor, out)
                else:
                    decode_position_list = self._decode_token_positions_list(
                        decode_positions
                    )
                    start, end = RWKV7ForCausalLM._contiguous_decode_token_range(
                        decode_batch_size, decode_rows, decode_position_list
                    )
                    tokens = input_ids[start:end].view(decode_batch_size, 1)
                    state = [
                        shift_state[:, :, :decode_batch_size, :],
                        wkv_state[:, :decode_batch_size, :, :, :],
                        elapsed[:decode_batch_size],
                    ]
                    out = self.forward_tokens(
                        tokens,
                        state,
                        query_start_loc=decode_query_start_loc,
                        wkv_slot_indices=idx_mapping[:decode_batch_size],
                    ).view(decode_batch_size, self.hidden_size)
                    hidden_states[start:end] = out.view(
                        decode_batch_size, self.hidden_size
                    )

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
            if (
                prefill_ranges
                and rwkv_prefill_query_start_loc is not None
                and rwkv_prefill_slot_indices is not None
                and rwkv_prefill_token_positions is not None
                and rwkv_prefill_req_id is not None
            ):
                input_position_tensor = rwkv_prefill_token_positions.to(
                    device=input_ids.device
                )
                hidden_position_tensor = rwkv_prefill_token_positions.to(
                    device=hidden_states.device
                )
                tokens = input_ids.index_select(0, input_position_tensor)
                state = [prefill_shift_state, prefill_wkv_state, prefill_elapsed]
                out = self.forward_varlen_hidden(
                    tokens,
                    state,
                    query_start_loc=rwkv_prefill_query_start_loc,
                    slot_indices=rwkv_prefill_slot_indices,
                    req_id=rwkv_prefill_req_id,
                )
                hidden_states.index_copy_(0, hidden_position_tensor, out)
                return hidden_states
            if prefill_ranges:
                raise RuntimeError(
                    "RWKV7 prefill requires canonical packed-varlen metadata."
                )
            return hidden_states

        raise RuntimeError("RWKV7 requires decode or packed-varlen prefill metadata.")

    def forward_vllm_pp_stage(
        self,
        *,
        input_ids: torch.Tensor | None,
        intermediate_tensors: IntermediateTensors | None,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed: torch.Tensor,
        prefill_shift_state: torch.Tensor | None,
        prefill_wkv_state: torch.Tensor | None,
        prefill_elapsed: torch.Tensor | None,
        rwkv_decode_batch_size: int,
        rwkv_decode_rows: list[int] | None,
        rwkv_decode_token_positions: torch.Tensor | list[int] | None,
        rwkv_decode_query_start_loc: torch.Tensor | None,
        idx_mapping: torch.Tensor,
        rwkv_prefill_token_ranges: list[tuple[int, int, int]] | None,
        rwkv_prefill_rows: list[int] | None,
        rwkv_prefill_query_start_loc: torch.Tensor | None,
        rwkv_prefill_slot_indices: torch.Tensor | None,
        rwkv_prefill_token_positions: torch.Tensor | None,
        rwkv_prefill_req_id: torch.Tensor | None,
        slot_indices: torch.Tensor | None,
        is_first_pp_rank: bool,
        is_last_pp_rank: bool,
    ) -> torch.Tensor | IntermediateTensors:
        if is_first_pp_rank:
            assert input_ids is not None
            total_tokens = input_ids.shape[0]
            incoming_hidden_states = None
            incoming_v_first = None
        else:
            assert intermediate_tensors is not None
            incoming_hidden_states = intermediate_tensors["hidden_states"].to(
                dtype=DTYPE
            )
            incoming_v_first = intermediate_tensors["v_first"].to(dtype=DTYPE)
            if incoming_v_first.shape[-1] == self.hidden_size:
                incoming_v_first = self._tp_hidden_slice(incoming_v_first, -1)
            total_tokens = incoming_hidden_states.shape[0]

        hidden_states = torch.empty(
            (total_tokens, self.hidden_size), dtype=DTYPE, device=CUDA_DEVICE
        )
        v_first_states = None
        if not is_last_pp_rank:
            v_first_states = torch.empty(
                (total_tokens, self.hidden_size),
                dtype=DTYPE,
                device=CUDA_DEVICE,
            )

        if rwkv_decode_rows is not None or rwkv_prefill_token_ranges is not None:
            decode_rows = rwkv_decode_rows or []
            decode_positions = rwkv_decode_token_positions
            assert len(decode_rows) == self._decode_token_positions_length(
                decode_positions
            )
            if decode_rows:
                decode_batch_size = rwkv_decode_batch_size
                assert decode_batch_size > 0
                assert rwkv_decode_query_start_loc is not None
                decode_query_start_loc = rwkv_decode_query_start_loc[
                    : decode_batch_size + 1
                ]
                if slot_indices is not None:
                    decode_position_tensor = self._decode_token_positions_tensor(
                        decode_positions,
                        device=hidden_states.device,
                    )
                    if is_first_pp_rank:
                        assert input_ids is not None
                        input_position_tensor = decode_position_tensor.to(
                            device=input_ids.device
                        )
                        tokens = input_ids.index_select(0, input_position_tensor).view(
                            decode_batch_size, 1
                        )
                        x = self.embed(tokens)
                        group_v_first = None
                    else:
                        assert incoming_hidden_states is not None
                        assert incoming_v_first is not None
                        x = incoming_hidden_states.index_select(
                            0, decode_position_tensor
                        ).view(decode_batch_size, 1, self.hidden_size)
                        group_v_first = incoming_v_first.index_select(
                            0, decode_position_tensor
                        ).view(
                            decode_batch_size,
                            1,
                            getattr(self, "tp_hidden_size", self.hidden_size),
                        )
                    state = [shift_state, wkv_state, elapsed]
                    decode_slot_indices = slot_indices[:decode_batch_size]
                    decode_wkv_slot_indices = decode_slot_indices
                else:
                    decode_position_list = self._decode_token_positions_list(
                        decode_positions
                    )
                    start, end = RWKV7ForCausalLM._contiguous_decode_token_range(
                        decode_batch_size, decode_rows, decode_position_list
                    )
                    if is_first_pp_rank:
                        assert input_ids is not None
                        tokens = input_ids[start:end].view(decode_batch_size, 1)
                        x = self.embed(tokens)
                        group_v_first = None
                    else:
                        assert incoming_hidden_states is not None
                        assert incoming_v_first is not None
                        x = incoming_hidden_states[start:end].view(
                            decode_batch_size, 1, self.hidden_size
                        )
                        group_v_first = incoming_v_first[start:end].view(
                            decode_batch_size,
                            1,
                            getattr(self, "tp_hidden_size", self.hidden_size),
                        )
                    state = [
                        shift_state[:, :, :decode_batch_size, :],
                        wkv_state[:, :decode_batch_size, :, :, :],
                        elapsed[:decode_batch_size],
                    ]
                    decode_slot_indices = None
                    decode_wkv_slot_indices = idx_mapping[:decode_batch_size]
                    decode_position_tensor = None
                path = select_path(decode_batch_size, 1)
                forward_kwargs = {}
                if decode_slot_indices is not None:
                    forward_kwargs["slot_indices"] = decode_slot_indices
                out, out_v_first = self.forward_layer_range(
                    x,
                    state,
                    path,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                    all_logits=True,
                    last_indices=None,
                    query_start_loc=decode_query_start_loc,
                    wkv_slot_indices=decode_wkv_slot_indices,
                    **forward_kwargs,
                )
                out = out.view(decode_batch_size, self.hidden_size)
                if decode_position_tensor is None:
                    hidden_states[start:end] = out.view(
                        decode_batch_size, self.hidden_size
                    )
                else:
                    hidden_states.index_copy_(0, decode_position_tensor, out)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    out_v_first = out_v_first.view(decode_batch_size, self.hidden_size)
                    if decode_position_tensor is None:
                        v_first_states[start:end] = out_v_first.view(
                            decode_batch_size, self.hidden_size
                        )
                    else:
                        v_first_states.index_copy_(
                            0, decode_position_tensor, out_v_first
                        )

            prefill_ranges = rwkv_prefill_token_ranges or []
            prefill_rows = rwkv_prefill_rows or []
            assert len(prefill_ranges) == len(prefill_rows)
            if prefill_shift_state is None:
                prefill_shift_state = shift_state
            if prefill_wkv_state is None:
                prefill_wkv_state = wkv_state
            if prefill_elapsed is None:
                prefill_elapsed = elapsed
            if (
                prefill_ranges
                and rwkv_prefill_query_start_loc is not None
                and rwkv_prefill_slot_indices is not None
                and rwkv_prefill_token_positions is not None
                and rwkv_prefill_req_id is not None
            ):
                hidden_position_tensor = rwkv_prefill_token_positions.to(
                    device=hidden_states.device
                )
                if is_first_pp_rank:
                    assert input_ids is not None
                    input_position_tensor = rwkv_prefill_token_positions.to(
                        device=input_ids.device
                    )
                    tokens = input_ids.index_select(0, input_position_tensor)
                    x = self.embed(tokens).view(tokens.numel(), self.hidden_size)
                    group_v_first = None
                else:
                    assert incoming_hidden_states is not None
                    assert incoming_v_first is not None
                    x = incoming_hidden_states.index_select(
                        0, hidden_position_tensor
                    ).view(-1, self.hidden_size)
                    group_v_first = incoming_v_first.index_select(
                        0, hidden_position_tensor
                    ).view(-1, getattr(self, "tp_hidden_size", self.hidden_size))
                state = [prefill_shift_state, prefill_wkv_state, prefill_elapsed]
                out, out_v_first = self.forward_varlen_layer_range(
                    x,
                    state,
                    query_start_loc=rwkv_prefill_query_start_loc,
                    slot_indices=rwkv_prefill_slot_indices,
                    req_id=rwkv_prefill_req_id,
                    v_first=group_v_first,
                    final=is_last_pp_rank,
                )
                hidden_states.index_copy_(0, hidden_position_tensor, out)
                if v_first_states is not None:
                    if out_v_first is None:
                        assert group_v_first is not None
                        out_v_first = group_v_first
                    if getattr(self, "tp_size", 1) > 1:
                        out_v_first = tensor_model_parallel_all_gather(out_v_first)
                    v_first_states.index_copy_(
                        0,
                        hidden_position_tensor,
                        out_v_first.view(-1, self.hidden_size),
                    )
                if not is_last_pp_rank:
                    assert v_first_states is not None
                    return IntermediateTensors(
                        {"hidden_states": hidden_states, "v_first": v_first_states}
                    )
                return hidden_states
            if prefill_ranges:
                raise RuntimeError(
                    "RWKV7 prefill requires canonical packed-varlen metadata."
                )

            if not is_last_pp_rank:
                assert v_first_states is not None
                return IntermediateTensors(
                    {"hidden_states": hidden_states, "v_first": v_first_states}
                )
            return hidden_states

        raise RuntimeError(
            "RWKV7 pipeline execution requires decode or packed-varlen "
            "prefill metadata."
        )

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        all_logits: bool = False,
        last_indices: torch.Tensor | None = None,
        slot_indices: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        wkv_slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        batch_size, token_count = tokens.shape
        x = self.embed(tokens)
        path = select_path(batch_size, token_count)
        if not all_logits and last_indices is None:
            return self.forward_from_x(
                x,
                state,
                path,
                slot_indices=slot_indices,
                query_start_loc=query_start_loc,
                wkv_slot_indices=wkv_slot_indices,
            )
        return self.forward_from_x(
            x,
            state,
            path,
            all_logits=all_logits,
            last_indices=last_indices,
            slot_indices=slot_indices,
            query_start_loc=query_start_loc,
            wkv_slot_indices=wkv_slot_indices,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    def project_logits_fp32(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the FP16 model output directly into FP32 logits."""
        weight = self.z["head.weight"]
        if (
            hidden_states.dim() >= 2
            and all(size == 1 for size in hidden_states.shape[:-1])
            and weight.size(1) % 64 == 0
        ):
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_fp32(
                hidden_states.contiguous(), weight
            )

        return torch.ops.rwkv7_v3a_ops.linear_f16_fp32_lt(
            hidden_states.contiguous(),
            weight,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.project_logits_fp32(hidden_states)
        logits = self.logits_processor(None, logits)
        if logits is not None and getattr(self, "tp_size", 1) > 1:
            logits = tensor_model_parallel_all_gather(logits)
            if logits is not None:
                logits = logits[..., : self.vocab_size]
        if logits is not None and logits.dtype != torch.float32:
            raise RuntimeError(f"RWKV7 logits must be FP32, but got {logits.dtype}.")
        return logits

    def compute_sampling_logits(
        self,
        hidden_states: torch.Tensor,
        logits_indices: torch.Tensor,
        input_batch: Any,
    ) -> torch.Tensor | None:
        num_reqs = getattr(input_batch, "num_reqs", None)
        num_draft_tokens = getattr(input_batch, "num_draft_tokens", None)
        num_scheduled_tokens = getattr(input_batch, "num_scheduled_tokens", None)
        if num_reqs is None or num_draft_tokens is None or num_scheduled_tokens is None:
            return None

        try:
            num_reqs = int(num_reqs)
            num_draft_tokens = int(num_draft_tokens)
        except (TypeError, ValueError):
            return None
        if num_reqs <= 0 or num_draft_tokens != 0:
            return None

        is_prefilling_np = getattr(input_batch, "is_prefilling_np", None)
        if is_prefilling_np is None:
            return None
        try:
            is_prefilling = is_prefilling_np[:num_reqs]
        except (TypeError, IndexError):
            return None
        if len(is_prefilling) != num_reqs:
            return None
        try:
            if bool(is_prefilling.any()):
                return None
        except (AttributeError, TypeError, RuntimeError):
            return None

        try:
            scheduled_tokens = num_scheduled_tokens[:num_reqs]
        except (TypeError, IndexError):
            return None
        if len(scheduled_tokens) != num_reqs:
            return None
        try:
            if bool((scheduled_tokens != 1).any()):
                return None
        except (AttributeError, TypeError, RuntimeError):
            return None

        if hidden_states.shape[0] < num_reqs:
            return None

        fast_path_metadata = getattr(
            input_batch,
            "rwkv_sampling_logits_contiguous",
            None,
        )
        if fast_path_metadata is None:
            if not isinstance(logits_indices, torch.Tensor):
                return None
            if (
                logits_indices.dim() != 1
                or logits_indices.numel() != num_reqs
                or logits_indices.dtype not in (torch.int32, torch.int64)
                or not logits_indices.is_contiguous()
            ):
                return None
            if logits_indices.is_cuda:
                return None
            if logits_indices.tolist() != list(range(num_reqs)):
                return None
        elif not bool(fast_path_metadata):
            return None

        return self.compute_logits(hidden_states[:num_reqs])

    @staticmethod
    def _decode_token_positions_length(
        decode_positions: torch.Tensor | list[int] | None,
    ) -> int:
        if decode_positions is None:
            return 0
        if isinstance(decode_positions, torch.Tensor):
            return int(decode_positions.numel())
        return len(decode_positions)

    @staticmethod
    def _decode_token_positions_tensor(
        decode_positions: torch.Tensor | list[int] | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if decode_positions is None:
            raise RuntimeError("RWKV7 decode token positions are missing")
        if isinstance(decode_positions, torch.Tensor):
            return decode_positions.to(device=device, dtype=torch.long)
        return torch.tensor(decode_positions, dtype=torch.long, device=device)

    @staticmethod
    def _decode_token_positions_list(
        decode_positions: torch.Tensor | list[int] | None,
    ) -> list[int]:
        if decode_positions is None:
            return []
        if isinstance(decode_positions, torch.Tensor):
            return decode_positions.tolist()
        return decode_positions

    @staticmethod
    def _contiguous_decode_token_range(
        decode_batch_size: int,
        decode_rows: list[int],
        decode_positions: list[int],
    ) -> tuple[int, int]:
        if decode_batch_size <= 0:
            raise RuntimeError("RWKV7 decode batch size must be positive")
        if len(decode_rows) != decode_batch_size:
            raise RuntimeError(
                "RWKV7 decode rows must match decode batch size "
                f"{decode_batch_size}; got {decode_rows}"
            )
        if len(decode_positions) != decode_batch_size:
            raise RuntimeError(
                "RWKV7 decode token positions must match decode batch size "
                f"{decode_batch_size}; got {decode_positions}"
            )
        for expected_row, row in enumerate(decode_rows):
            if row != expected_row:
                raise RuntimeError(
                    "RWKV7 decode rows must be contiguous prefix rows "
                    f"[0..{decode_batch_size - 1}]; got {decode_rows}"
                )
        start = decode_positions[0]
        end = start + decode_batch_size
        for offset, position in enumerate(decode_positions):
            if position != start + offset:
                raise RuntimeError(
                    "RWKV7 decode token positions must be a dense contiguous "
                    f"range; got {decode_positions}"
                )
        return start, end

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.device != self.z["emb.weight"].device:
            tokens = tokens.to(self.z["emb.weight"].device, non_blocking=True)
        if getattr(self, "tp_size", 1) == 1:
            return self.z["emb.weight"][tokens]
        vocab_start, vocab_end, vocab_per_rank = self._tp_vocab_range(self.vocab_size)
        mask = (tokens < vocab_start) | (tokens >= vocab_end)
        local = (tokens - vocab_start).clamp(min=0, max=vocab_per_rank - 1)
        out = self.z["emb.weight"][local]
        out.masked_fill_(mask.unsqueeze(-1), 0)
        return self._tp_all_reduce(out)

    def forward_from_x(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        all_logits: bool = False,
        last_indices=None,
        slot_indices: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        wkv_slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run RWKV7 from embedded input."""
        out, _ = self.forward_layer_range(
            x,
            state,
            path,
            v_first=None,
            final=True,
            all_logits=all_logits,
            last_indices=last_indices,
            slot_indices=slot_indices,
            query_start_loc=query_start_loc,
            wkv_slot_indices=wkv_slot_indices,
        )
        return out

    def forward_layer_range(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        path: PathConfig,
        *,
        v_first: torch.Tensor | None,
        final: bool,
        all_logits: bool,
        last_indices=None,
        slot_indices: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        wkv_slot_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        B, T, _ = x.shape
        if query_start_loc is None:
            query_start_loc = torch.arange(
                0,
                (B + 1) * T,
                T,
                dtype=torch.int32,
                device=x.device,
            )
        if wkv_slot_indices is None:
            wkv_slot_indices = (
                slot_indices
                if slot_indices is not None
                else torch.arange(B, dtype=torch.int32, device=x.device)
            )
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", self.total_num_layers)

        def advance_elapsed() -> None:
            if slot_indices is None:
                torch.ops.rwkv7_v3a_ops.advance_i32(state[2], T)
            else:
                torch.ops.rwkv7_v3a_ops.advance_i32_slots(state[2], slot_indices, T)

        if start_layer == 0 and v_first is None:
            v_first = x
        if start_layer >= end_layer:
            if final:
                x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
                return x, v_first
            return x, v_first

        xx = self.ln(
            x,
            z[f"blocks.{start_layer}.ln1.weight"],
            z[f"blocks.{start_layer}.ln1.bias"],
        )
        pre_mix = None

        for layer in range(start_layer, end_layer):
            local_layer = layer - start_layer
            p = f"blocks.{layer}."
            layer_v_first = x if layer == 0 else v_first
            assert layer_v_first is not None
            xx, v_first = self.tmix(
                layer,
                xx,
                state[0][local_layer],
                state[1][local_layer],
                state[2],
                layer_v_first,
                p + "att.",
                path,
                pre_mix,
                slot_indices=slot_indices,
                query_start_loc=query_start_loc,
                wkv_slot_indices=wkv_slot_indices,
            )
            pre_mix = None
            if T == 1:
                if slot_indices is None:
                    x, mixed = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16(
                        x.contiguous(),
                        xx.contiguous(),
                        state[0][local_layer][1],
                        z[p + "ln2.weight"],
                        z[p + "ln2.bias"],
                        z[p + "ffn.x_k"],
                    )
                    cmix_path = path
                else:
                    (
                        x,
                        mixed,
                    ) = torch.ops.rwkv7_v3a_ops.add_layer_norm_cmix_mix_f16_slots(
                        x.contiguous(),
                        xx.contiguous(),
                        state[0][local_layer][1],
                        z[p + "ln2.weight"],
                        z[p + "ln2.bias"],
                        z[p + "ffn.x_k"],
                        slot_indices,
                    )
                    cmix_path = path
                xx = self.cmix_from_mixed(mixed, p + "ffn.", cmix_path)
            else:
                x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
                xx = self.cmix(
                    xx,
                    state[0][local_layer],
                    p + "ffn.",
                    path,
                    slot_indices=slot_indices,
                )
            if layer + 1 < end_layer:
                p_next = f"blocks.{layer + 1}."
                if use_ln1_tmix_fusion(B, T):
                    if slot_indices is None:
                        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16(
                            x.contiguous(),
                            xx.contiguous(),
                            state[0][local_layer + 1][0],
                            z[p_next + "ln1.weight"],
                            z[p_next + "ln1.bias"],
                            z[p_next + "att.x_r"],
                            z[p_next + "att.x_w"],
                            z[p_next + "att.x_k"],
                            z[p_next + "att.x_v"],
                            z[p_next + "att.x_a"],
                            z[p_next + "att.x_g"],
                        )
                    else:
                        outs = (
                            torch.ops.rwkv7_v3a_ops.add_layer_norm_tmix_mix6_f16_slots(
                                x.contiguous(),
                                xx.contiguous(),
                                state[0][local_layer + 1][0],
                                z[p_next + "ln1.weight"],
                                z[p_next + "ln1.bias"],
                                z[p_next + "att.x_r"],
                                z[p_next + "att.x_w"],
                                z[p_next + "att.x_k"],
                                z[p_next + "att.x_v"],
                                z[p_next + "att.x_a"],
                                z[p_next + "att.x_g"],
                                slot_indices,
                            )
                        )
                    x, pre_mix = outs[0], outs[1:]
                    xx = x
                else:
                    x, xx = self.add_ln(
                        x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"]
                    )
            elif not final:
                x = self.add(x, xx)
                advance_elapsed()
                return x, v_first
            elif not all_logits:
                if last_indices is not None:
                    x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                    x = x[torch.arange(B, device=x.device), last_indices].contiguous()
                else:
                    x = self.add_last_ln(x, xx, z["ln_out.weight"], z["ln_out.bias"])
                advance_elapsed()
                return x, v_first
            else:
                x = self.add(x, xx)

        x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
        advance_elapsed()
        return x, v_first

    def ln(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.layer_norm_f16(x.contiguous(), weight, bias)

    def forward_all_logits(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        return self.compute_logits(self.forward_tokens(tokens, state, all_logits=True))

    def forward_all_hidden(
        self, tokens: torch.Tensor, state: list[torch.Tensor]
    ) -> torch.Tensor:
        """Return hidden states for every input position."""
        return self.forward_tokens(tokens, state, all_logits=True)

    def forward_varlen_hidden(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens.reshape(-1)
        x = self.embed(tokens).view(tokens.numel(), self.hidden_size)
        out, _ = self.forward_varlen_layer_range(
            x,
            state,
            query_start_loc=query_start_loc,
            slot_indices=slot_indices,
            req_id=req_id,
            v_first=None,
            final=True,
        )
        return out

    def forward_varlen_layer_range(
        self,
        x: torch.Tensor,
        state: list[torch.Tensor],
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
        v_first: torch.Tensor | None,
        final: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = self.z
        total_tokens = x.shape[0]
        path = PathConfig(total_tokens, CMIX_DENSE)
        start_layer = getattr(self, "start_layer", 0)
        end_layer = getattr(self, "end_layer", self.total_num_layers)

        def advance_elapsed() -> None:
            torch.ops.rwkv7_v3a_ops.advance_i32_varlen(
                state[2], query_start_loc, slot_indices
            )

        if start_layer == 0 and v_first is None:
            v_first = x
        if start_layer >= end_layer:
            if final:
                x = self.ln(x, z["ln_out.weight"], z["ln_out.bias"])
            return x, v_first

        xx = self.ln(
            x,
            z[f"blocks.{start_layer}.ln1.weight"],
            z[f"blocks.{start_layer}.ln1.bias"],
        )

        for layer in range(start_layer, end_layer):
            local_layer = layer - start_layer
            p = f"blocks.{layer}."
            layer_v_first = x if layer == 0 else v_first
            assert layer_v_first is not None
            xx, v_first = self.tmix_varlen(
                layer,
                xx,
                state[0][local_layer],
                state[1][local_layer],
                state[2],
                layer_v_first,
                p + "att.",
                path,
                query_start_loc=query_start_loc,
                slot_indices=slot_indices,
                req_id=req_id,
            )
            x, xx = self.add_ln(x, xx, z[p + "ln2.weight"], z[p + "ln2.bias"])
            xx = self.cmix_varlen(
                xx,
                state[0][local_layer],
                p + "ffn.",
                path,
                query_start_loc=query_start_loc,
                slot_indices=slot_indices,
                req_id=req_id,
            )
            if layer + 1 < end_layer:
                p_next = f"blocks.{layer + 1}."
                x, xx = self.add_ln(
                    x, xx, z[p_next + "ln1.weight"], z[p_next + "ln1.bias"]
                )
            elif not final:
                x = self.add(x, xx)
                advance_elapsed()
                return x, v_first
            else:
                x = self.ln(self.add(x, xx), z["ln_out.weight"], z["ln_out.bias"])
                advance_elapsed()
                return x, v_first

        raise AssertionError("unreachable RWKV7 varlen layer path")

    def forward_last_at(
        self,
        tokens: torch.Tensor,
        state: list[torch.Tensor],
        last_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_logits(
            self.forward_tokens(tokens, state, last_indices=last_indices)
        )

    def _project_tmix(
        self,
        layer: int,
        xr: torch.Tensor,
        xw: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        xa: torch.Tensor,
        xg: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        batch_size: int,
        time_steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Share the layout-independent time-mix projection pipeline."""
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        r, k, v = self.project_att_rkv(xr, xk, xv, p, path.rows)
        local_c = r.shape[-1]
        local_h = local_c // self.head_size
        use_lowrank_in = can_use_lowrank_fused(self.hidden_size, path.rows)
        use_lowrank_out = can_use_lowrank_out_fused(self.hidden_size, path.rows)

        v1 = None
        if use_lowrank_in and use_lowrank_out and layer != 0:
            w1, a1, g1, v1 = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                xv.contiguous(),
                z[p + "w1.t"],
                z[p + "a1.t"],
                z[p + "g1.t"],
                z[p + "v1.t"],
            )
        elif use_lowrank_in:
            w1, a1, g1 = torch.ops.rwkv7_v3a_ops.linear_wag_rank_in_f16(
                xw.contiguous(),
                xa.contiguous(),
                xg.contiguous(),
                z[p + "w1.t"],
                z[p + "a1.t"],
                z[p + "g1.t"],
            )
        else:
            w1 = self.linear_rank_in(xw, z.get(p + "w1"), z.get(p + "w1.t"), path.rows)
            a1 = self.linear_rank_in(xa, z.get(p + "a1"), z.get(p + "a1.t"), path.rows)
            g1 = self.linear_rank_in(xg, z.get(p + "g1"), z.get(p + "g1.t"), path.rows)

        v_done = False
        if use_lowrank_out and layer != 0 and v1 is not None:
            w, a, g, v = torch.ops.rwkv7_v3a_ops.linear_wagv_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                v1.contiguous(),
                z[p + "w2.t"],
                z[p + "a2.t"],
                z[p + "g2.t"],
                z[p + "v2.t"],
                v.contiguous(),
                v_first.contiguous(),
                z[p + "v0"],
            )
            v_done = True
        elif use_lowrank_out:
            w, a, g = torch.ops.rwkv7_v3a_ops.linear_wag_rank_out_f16(
                w1.contiguous(),
                a1.contiguous(),
                g1.contiguous(),
                z[p + "w2.t"],
                z[p + "a2.t"],
                z[p + "g2.t"],
            )
        else:
            w = self.linear_rank_out_act(
                w1, z.get(p + "w2"), z.get(p + "w2.t"), path.rows, 1
            )
            a = self.linear_rank_out(a1, z.get(p + "a2"), z.get(p + "a2.t"), path.rows)
            g = self.linear_rank_out_act(
                g1, z.get(p + "g2"), z.get(p + "g2.t"), path.rows, 2
            )

        value_shape = (batch_size, time_steps, local_c)
        kk_a_gate = (
            ops.tmix_kk_a_gate_2d
            if use_tmix_kk_a_gate_2d(batch_size, time_steps, local_c, local_h)
            else ops.tmix_kk_a_gate
        )
        k3, neg_kk3, kka3 = kk_a_gate(
            batch_size,
            time_steps,
            local_c,
            local_h,
            k.reshape(value_shape).contiguous(),
            z[p + "k_k"],
            z[p + "a0"],
            a.reshape(value_shape).contiguous(),
            z[p + "k_a"],
        )
        if k.dim() == 2:
            k = k3.reshape(-1, local_c)
            neg_kk = neg_kk3.reshape(-1, local_c)
            kka = kka3.reshape(-1, local_c)
        else:
            k, neg_kk, kka = k3, neg_kk3, kka3

        if layer == 0:
            v_first = v
        elif not v_done:
            if use_lowrank_out:
                if v1 is None:
                    v1 = self.linear_rank_in(
                        xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows
                    )
                v = torch.ops.rwkv7_v3a_ops.linear_t_vres_f16(
                    v1.contiguous(),
                    z[p + "v2.t"],
                    v.contiguous(),
                    v_first.contiguous(),
                    z[p + "v0"],
                )
            else:
                v12 = self.linear_rank_out(
                    self.linear_rank_in(
                        xv, z.get(p + "v1"), z.get(p + "v1.t"), path.rows
                    ),
                    z.get(p + "v2"),
                    z.get(p + "v2.t"),
                    path.rows,
                )
                v = ops.tmix_vres_gate(
                    batch_size,
                    time_steps,
                    local_c,
                    v.reshape(value_shape).contiguous(),
                    v_first.reshape(value_shape).contiguous(),
                    z[p + "v0"],
                    v12.reshape(value_shape).contiguous(),
                )
                if v.dim() == 3 and xv.dim() == 2:
                    v = v.reshape(-1, local_c)

        return r, w, k, v, neg_kk, kka, g, v_first

    def tmix_varlen(
        self,
        layer: int,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed_t: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B = int(slot_indices.numel())
        total_tokens = int(x.shape[0])
        xr, xw, xk, xv, xa, xg = ops.tmix_mix6_varlen(
            B,
            total_tokens,
            self.hidden_size,
            x.contiguous(),
            shift_state[0],
            slot_indices,
            z[p + "x_r"],
            z[p + "x_w"],
            z[p + "x_k"],
            z[p + "x_v"],
            z[p + "x_a"],
            z[p + "x_g"],
            query_start_loc,
            req_id,
        )

        r, w, k, v, neg_kk, kka, g, v_first = self._project_tmix(
            layer,
            xr,
            xw,
            xk,
            xv,
            xa,
            xg,
            v_first,
            p,
            path,
            batch_size=total_tokens,
            time_steps=1,
        )
        local_c = r.shape[-1]
        local_h = local_c // self.head_size

        y = torch.empty_like(r)
        self._run_wkv(
            query_start_loc,
            slot_indices,
            wkv_state,
            r,
            w,
            z[p + "w0"],
            k,
            v,
            neg_kk,
            kka,
            y,
            elapsed_t,
        )
        y = ops.tmix_lnx_rkvres_xg(
            total_tokens,
            1,
            local_c,
            local_h,
            y.view(total_tokens, 1, local_c).contiguous(),
            r.view(total_tokens, 1, local_c).contiguous(),
            k.view(total_tokens, 1, local_c).contiguous(),
            v.view(total_tokens, 1, local_c).contiguous(),
            z[p + "r_k"],
            z[p + "ln_x.weight"],
            z[p + "ln_x.bias"],
            g.view(total_tokens, 1, local_c).contiguous(),
        ).view(total_tokens, local_c)
        out = self.linear_att_c2c(y, z[p + "output.weight"], path.rows)
        return self._tp_all_reduce(out), v_first

    def cmix_varlen(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        req_id: torch.Tensor,
    ) -> torch.Tensor:
        ops = torch.ops.rwkv7_fast_ops_fp16
        total_tokens = int(x.shape[0])
        B = int(slot_indices.numel())
        mixed = ops.cmix_mix_varlen(
            B,
            total_tokens,
            self.hidden_size,
            x.contiguous(),
            shift_state[1],
            slot_indices,
            self.z[p + "x_k"],
            query_start_loc,
            req_id,
        )
        dense_path = PathConfig(path.rows, CMIX_DENSE)
        return self.cmix_from_mixed(
            mixed.view(total_tokens, 1, self.hidden_size), p, dense_path
        ).view(total_tokens, self.hidden_size)

    def _run_wkv(
        self,
        query_start_loc: torch.Tensor,
        slot_indices: torch.Tensor,
        state: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        w0: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neg_kk: torch.Tensor,
        kka: torch.Tensor,
        y: torch.Tensor,
        elapsed_t: torch.Tensor,
    ) -> None:
        """Run the canonical packed-varlen WKV operator for this precision."""
        hidden_size = r.shape[-1]
        r_rows = r.view(-1, hidden_size).contiguous()
        w_rows = w.view(-1, hidden_size).contiguous()
        k_rows = k.view(-1, hidden_size).contiguous()
        v_rows = v.view(-1, hidden_size).contiguous()
        neg_kk_rows = neg_kk.view(-1, hidden_size).contiguous()
        kka_rows = kka.view(-1, hidden_size).contiguous()
        y_rows = y.view(-1, hidden_size)
        if self.wkv_mode == "fp32io16":
            w_rows = torch.ops.rwkv7_fast_ops_fp16.add_vec(
                hidden_size,
                w_rows,
                w0,
            )
            torch.ops.rwkv7_wkv_fp32_v2.wkv(
                query_start_loc,
                slot_indices,
                state,
                r_rows,
                w_rows,
                k_rows,
                v_rows,
                neg_kk_rows,
                kka_rows,
                y_rows,
            )
            return
        torch.ops.rwkv7_wkv_fp16_v2.wkv(
            query_start_loc,
            slot_indices,
            state,
            r_rows,
            w_rows,
            w0,
            k_rows,
            v_rows,
            neg_kk_rows,
            kka_rows,
            y_rows,
            elapsed_t,
        )

    def tmix(
        self,
        layer: int,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
        elapsed_t: torch.Tensor,
        v_first: torch.Tensor,
        p: str,
        path: PathConfig,
        pre_mix=None,
        slot_indices: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        wkv_slot_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        if pre_mix is not None:
            xr, xw, xk, xv, xa, xg = pre_mix
        elif slot_indices is not None:
            xr, xw, xk, xv, xa, xg = ops.tmix_mix6_slot(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[0],
                slot_indices,
                z[p + "x_r"],
                z[p + "x_w"],
                z[p + "x_k"],
                z[p + "x_v"],
                z[p + "x_a"],
                z[p + "x_g"],
            )
        else:
            mix6 = (
                ops.tmix_mix6_3d
                if use_tmix_mix6_3d(B, T, self.hidden_size)
                else ops.tmix_mix6
            )
            xr, xw, xk, xv, xa, xg = mix6(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[0],
                z[p + "x_r"],
                z[p + "x_w"],
                z[p + "x_k"],
                z[p + "x_v"],
                z[p + "x_a"],
                z[p + "x_g"],
            )
        r, w, k, v, neg_kk, kka, g, v_first = self._project_tmix(
            layer,
            xr,
            xw,
            xk,
            xv,
            xa,
            xg,
            v_first,
            p,
            path,
            batch_size=B,
            time_steps=T,
        )
        local_c = r.shape[-1]
        local_h = local_c // self.head_size

        y = torch.empty_like(r)
        if query_start_loc is None or wkv_slot_indices is None:
            raise RuntimeError("RWKV7 WKV requires packed request metadata")
        self._run_wkv(
            query_start_loc,
            wkv_slot_indices,
            wkv_state,
            r,
            w,
            z[p + "w0"],
            k,
            v,
            neg_kk,
            kka,
            y,
            elapsed_t,
        )
        lnx = (
            ops.tmix_lnx_rkvres_xg_warp
            if use_tmix_lnx_warp(B, T, local_c, local_h)
            else ops.tmix_lnx_rkvres_xg
        )
        y = lnx(
            B,
            T,
            local_c,
            local_h,
            y.contiguous(),
            r.contiguous(),
            k.contiguous(),
            v.contiguous(),
            z[p + "r_k"],
            z[p + "ln_x.weight"],
            z[p + "ln_x.bias"],
            g.contiguous(),
        )
        out = self.linear_att_c2c(y, z[p + "output.weight"], path.rows)
        return self._tp_all_reduce(out), v_first

    def cmix(
        self,
        x: torch.Tensor,
        shift_state: torch.Tensor,
        p: str,
        path: PathConfig,
        *,
        slot_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if slot_indices is not None:
            mixed = ops.cmix_mix_slot(
                B,
                T,
                self.hidden_size,
                x.contiguous(),
                shift_state[1],
                slot_indices,
                z[p + "x_k"],
            )
            return self.cmix_from_mixed(mixed, p, path)

        mix = (
            ops.cmix_mix_3d if use_cmix_mix_3d(B, T, self.hidden_size) else ops.cmix_mix
        )
        mixed = mix(
            B, T, self.hidden_size, x.contiguous(), shift_state[1], z[p + "x_k"]
        )
        return self.cmix_from_mixed(mixed, p, path)

    def cmix_from_mixed(
        self, mixed: torch.Tensor, p: str, path: PathConfig
    ) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = mixed.shape
        if path.cmix_mode == CMIX_B1T1_NOFC:
            key_weight = z[p + "key.weight"]
            value_weight = z[p + "value.weight"]
            can_use_m1_splitk = (
                mixed.numel() == mixed.size(-1) and key_weight.size(1) % 64 == 0
            )
            if path.rows in M1_CMIX_PREZERO_ROWS and can_use_m1_splitk:
                hid, out = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk_prepare_zero(
                    mixed.contiguous(),
                    key_weight,
                    self.hidden_size,
                )
                ops.cmix_sparse_down_relu_one_out(
                    self.hidden_size,
                    value_weight.size(0),
                    hid.view(-1).contiguous(),
                    value_weight,
                    out,
                )
                return self._tp_all_reduce(out)
            if can_use_m1_splitk:
                hid = torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(
                    mixed.contiguous(),
                    key_weight,
                )
            else:
                hid = self.linear(mixed, key_weight)
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_one(
                    self.hidden_size,
                    value_weight.size(0),
                    hid.view(-1).contiguous(),
                    value_weight,
                )
            )
        hid = self.linear(mixed, z[p + "key.weight"])
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            F = z[p + "value.weight"].size(0)
            if (
                path.rows >= CMIX_NOFC_T512_MIN_ROWS
                and self.hidden_size % 512 == 0
                and F % 512 == 0
            ):
                return self._tp_all_reduce(
                    ops.cmix_sparse_down_relu_rows_t512(
                        B,
                        T,
                        self.hidden_size,
                        F,
                        hid.contiguous(),
                        z[p + "value.weight"],
                    )
                )
            return self._tp_all_reduce(
                ops.cmix_sparse_down_relu_rows(
                    B,
                    T,
                    self.hidden_size,
                    F,
                    hid.contiguous(),
                    z[p + "value.weight"],
                )
            )

        k = ops.relu_square(hid.contiguous())
        return self._tp_all_reduce(
            self.linear_ffn_down(k, z[p + "value.weight"], path.rows)
        )

    def linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if x.numel() == x.size(-1) and weight.size(1) % 64 == 0:
            return torch.ops.rwkv7_v3a_ops.linear_f16_m1_splitk(x.contiguous(), weight)
        return torch.ops.rwkv7_v3a_ops.linear_f16(
            x.contiguous(), weight, self.allow_fp16_accumulation
        )

    def linear_att_c2c(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        cfg = ATT_C2C_FP16_LT_4096.get(rows)
        if (
            self.allow_fp16_accumulation
            and self.hidden_size == 4096
            and tuple(weight.shape) == (4096, 4096)
            and cfg is not None
        ):
            workspace_mb, heuristic_index = cfg
            return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x.contiguous(),
                weight,
                workspace_mb,
                heuristic_index,
                False,
            )
        return self.linear(x, weight)

    def project_att_rkv(
        self,
        xr: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        p: str,
        rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project exact-M1 R/K/V with shared launches and separate weights."""
        weights = (
            self.z[p + "receptance.weight"],
            self.z[p + "key.weight"],
            self.z[p + "value.weight"],
        )
        inputs = (xr, xk, xv)
        weight_shape = tuple(weights[0].shape)
        if (
            rows in M1_RKV_GROUPED_ROWS
            and all(value.numel() == value.size(-1) for value in inputs)
            and all(tuple(weight.shape) == weight_shape for weight in weights)
            and all(
                value.size(-1) == weight.size(0)
                for value, weight in zip(inputs, weights)
            )
            and weights[0].size(1) % 64 == 0
        ):
            return torch.ops.rwkv7_v3a_ops.linear_rkv_f16_m1_splitk(
                *(value.contiguous() for value in inputs),
                *weights,
            )
        if rows == 1:
            # Keep the paired benchmark baseline pinned to the three independent
            # split-K projections even if row-1 Lt tuning is added later.
            return tuple(
                self.linear(value, weight) for value, weight in zip(inputs, weights)
            )
        return tuple(
            self.linear_att_c2c(value, weight, rows)
            for value, weight in zip(inputs, weights)
        )

    def linear_ffn_down(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        cfg = FFN_DOWN_FP16_LT_4096.get(rows)
        if (
            self.allow_fp16_accumulation
            and self.hidden_size == 4096
            and tuple(weight.shape) == (16384, 4096)
            and cfg is not None
        ):
            workspace_mb, heuristic_index = cfg
            return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x.contiguous(),
                weight,
                workspace_mb,
                heuristic_index,
                False,
            )
        return self.linear(x, weight)

    def linear_rank_in(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if rows <= LOWRANK_IN_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        cfg = LOWRANK_IN_FP16_LT_4096.get((rows, weight.size(1)))
        if (
            self.allow_fp16_accumulation
            and self.hidden_size == 4096
            and tuple(weight.shape) == (4096, weight.size(1))
            and cfg is not None
        ):
            workspace_mb, heuristic_index = cfg
            return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x.contiguous(),
                weight,
                workspace_mb,
                heuristic_index,
                False,
            )
        return self.linear(x, weight)

    def linear_rank_out(
        self, x: torch.Tensor, weight: torch.Tensor, weight_t: torch.Tensor, rows: int
    ) -> torch.Tensor:
        if self.hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_f16(x.contiguous(), weight_t)
        cfg = LOWRANK_OUT_FP16_LT_4096.get((rows, weight.size(0)))
        if (
            self.allow_fp16_accumulation
            and self.hidden_size == 4096
            and tuple(weight.shape) == (weight.size(0), 4096)
            and cfg is not None
        ):
            workspace_mb, heuristic_index = cfg
            return torch.ops.rwkv7_v3a_ops.linear_f16_lt_cfg(
                x.contiguous(),
                weight,
                workspace_mb,
                heuristic_index,
                False,
            )
        return self.linear(x, weight)

    def linear_rank_out_act(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_t: torch.Tensor,
        rows: int,
        act: int,
    ) -> torch.Tensor:
        if self.hidden_size >= LOWRANK_FUSED_MIN_C and rows <= LOWRANK_OUT_ROWS_T:
            return torch.ops.rwkv7_v3a_ops.linear_t_act_f16(
                x.contiguous(), weight_t, act
            )
        ops = torch.ops.rwkv7_fast_ops_fp16
        x = (
            ops.act_tanh(x.contiguous())
            if act == 1
            else ops.act_sigmoid(x.contiguous())
        )
        return self.linear_rank_out(x, weight, weight_t, rows)

    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_f16(x.contiguous(), y.contiguous())

    def add_ln(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outs = torch.ops.rwkv7_v3a_ops.add_layer_norm_f16(
            x.contiguous(), residual.contiguous(), weight, bias
        )
        return outs[0], outs[1]

    def add_last_ln(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.rwkv7_v3a_ops.add_last_layer_norm_f16(
            x.contiguous(), residual.contiguous(), weight, bias
        )
