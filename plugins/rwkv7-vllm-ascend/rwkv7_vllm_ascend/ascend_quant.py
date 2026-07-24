"""Fail-closed Ascend 910B3 RWKV-7 FFN weight-only quantization seam.

The serving adapters deliberately keep this module independent of vLLM and
SGLang.  Merely importing it does not import ``torch_npu`` or touch an NPU.
Quantization is disabled unless both environment variables documented in
``ASCEND_FFN_QUANT.md`` are set.  The only enabled scope is the explicitly
labelled raw-kernel-candidate scope; production admission remains closed until
an engine-level quality, HBM and latency artifact is committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

MANIFEST_FORMAT = "rwkv7-ascend-ffn-weight-only-v1"
MANIFEST_VERSION = 1
EXPECTED_DEVICE_NAME = "Ascend910B3"
EXPECTED_TORCH_VERSION = "2.9.0"
EXPECTED_TORCH_NPU_VERSION = "2.9.0"
EXPECTED_CANN_VERSION = "8.5.0"
EXPECTED_OPERATOR_SCHEMA_SHA256 = (
    "f99a37dcc4e7d07f803bb83ce3d4c93ccbd15c41af1267d5a07dcc0d62e7dff0"
)
VERIFIED_FFN_SHAPES = ((4096, 16384), (16384, 4096))  # (K, N)
RAW_CANDIDATE_ROWS = {4: (1, 8), 8: (17, 28)}
RAW_CANDIDATE_GROUP_SIZE = {4: 128, 8: 0}

ENABLE_ENV = "RWKV7_ASCEND_QUANT"
MANIFEST_ENV = "RWKV7_ASCEND_QUANT_MANIFEST"
RAW_ACK_ENV = "RWKV7_ASCEND_ALLOW_RAW_CANDIDATE"


class AscendQuantError(RuntimeError):
    """Base class for deterministic, fail-closed quantization errors."""


class AscendQuantConfigError(AscendQuantError):
    """The activation manifest is missing, malformed, or over-claims support."""


class AscendQuantRuntimeError(AscendQuantError):
    """The runtime is not the exact stack measured by the raw-op sweep."""


class AscendQuantLoadError(AscendQuantError):
    """The checkpoint does not match the explicit source/manifest contract."""


def _base_version(value: str) -> str:
    return str(value).split("+", 1)[0]


def _sha256_tensor(tensor: Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    # NumPy has no native bfloat16, but packed checkpoint tensors are restricted
    # to int8/int32/fp16, so the view is stable across supported hosts.
    return hashlib.sha256(memoryview(cpu.numpy()).cast("B")).hexdigest()


def _read_cann_version() -> str | None:
    candidates = (
        Path(
            "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info"
        ),
        Path(
            "/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info"
        ),
        Path("/usr/local/Ascend/cann/aarch64-linux/ascend_toolkit_install.info"),
    )
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except OSError:
            continue
        for line in text.splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() == "version":
                return value.strip().strip('"')
    return None


@dataclass(frozen=True)
class RuntimeFingerprint:
    device_name: str
    torch_version: str
    torch_npu_version: str
    cann_version: str
    operator_schema_sha256: str
    device: str = "npu:0"

    @classmethod
    def detect(cls) -> tuple[RuntimeFingerprint, Callable[..., Tensor]]:
        try:
            import torch_npu  # type: ignore
        except ImportError as exc:
            raise AscendQuantRuntimeError("torch_npu is required") from exc
        try:
            if not torch.npu.is_available():
                raise AscendQuantRuntimeError("torch.npu.is_available() is false")
            device_name = str(torch_npu.npu.get_device_name(0))
            op = torch_npu.npu_weight_quant_batchmatmul
            schema = str(torch.ops.npu.npu_weight_quant_batchmatmul.default._schema)
        except AscendQuantRuntimeError:
            raise
        except Exception as exc:
            raise AscendQuantRuntimeError(
                "unable to resolve the Ascend device or quant operator ABI"
            ) from exc
        return (
            cls(
                device_name=device_name,
                torch_version=_base_version(torch.__version__),
                torch_npu_version=_base_version(torch_npu.__version__),
                cann_version=str(_read_cann_version()),
                operator_schema_sha256=hashlib.sha256(
                    schema.encode("utf-8")
                ).hexdigest(),
            ),
            op,
        )

    def validate_exact(self) -> None:
        actual = {
            "device_name": self.device_name,
            "torch_version": _base_version(self.torch_version),
            "torch_npu_version": _base_version(self.torch_npu_version),
            "cann_version": self.cann_version,
            "operator_schema_sha256": self.operator_schema_sha256,
        }
        expected = {
            "device_name": EXPECTED_DEVICE_NAME,
            "torch_version": EXPECTED_TORCH_VERSION,
            "torch_npu_version": EXPECTED_TORCH_NPU_VERSION,
            "cann_version": EXPECTED_CANN_VERSION,
            "operator_schema_sha256": EXPECTED_OPERATOR_SCHEMA_SHA256,
        }
        if actual != expected:
            diff = {
                key: {"expected": expected[key], "actual": actual[key]}
                for key in expected
                if actual[key] != expected[key]
            }
            raise AscendQuantRuntimeError(
                "unverified Ascend quant runtime: " + json.dumps(diff, sort_keys=True)
            )


@dataclass(frozen=True)
class TensorRecord:
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    @classmethod
    def parse(cls, value: Any, name: str) -> TensorRecord:
        if not isinstance(value, Mapping):
            raise AscendQuantConfigError(f"tensors[{name!r}] must be an object")
        try:
            shape = tuple(int(item) for item in value["shape"])
            dtype = str(value["dtype"])
            digest = str(value["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AscendQuantConfigError(f"invalid tensor metadata for {name}") from exc
        if any(dim < 0 for dim in shape):
            raise AscendQuantConfigError(f"negative tensor dimension for {name}")
        if dtype not in {"int8", "int32", "float16"}:
            raise AscendQuantConfigError(
                f"unsupported packed dtype {dtype!r} for {name}"
            )
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise AscendQuantConfigError(f"invalid lowercase SHA256 for {name}")
        return cls(shape, dtype, digest)


@dataclass(frozen=True)
class AscendQuantManifest:
    backend: str
    bit: int
    group_size: int
    admitted_rows: tuple[int, ...]
    key_layers: tuple[int, ...]
    value_layers: tuple[int, ...]
    source: str
    tensor_records: Mapping[str, TensorRecord]
    manifest_path: str | None = None
    manifest_sha256: str | None = None

    @property
    def selected(self) -> frozenset[tuple[int, str]]:
        return frozenset(
            [(layer, "key") for layer in self.key_layers]
            + [(layer, "value") for layer in self.value_layers]
        )

    def module_path(self, layer: int, projection: str) -> str:
        return f"model.layers.{layer}.ffn.{projection}"

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        backend: str,
        manifest_path: str | None = None,
        manifest_sha256: str | None = None,
    ) -> AscendQuantManifest:
        required = {
            "format",
            "version",
            "backend",
            "bit",
            "group_size",
            "activation_dtype",
            "acceptance_scope",
            "production_accepted",
            "operator_schema_sha256",
            "verified_stack",
            "verified_ffn_shapes",
            "admitted_rows",
            "ffn",
            "source",
            "tensors",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise AscendQuantConfigError(f"manifest missing fields: {missing}")
        if raw["format"] != MANIFEST_FORMAT or int(raw["version"]) != MANIFEST_VERSION:
            raise AscendQuantConfigError("unsupported quant manifest format/version")
        if backend not in {"vllm", "sglang"} or raw["backend"] != backend:
            raise AscendQuantConfigError(
                f"manifest backend {raw['backend']!r} does not match {backend!r}"
            )
        if raw["activation_dtype"] != "float16":
            raise AscendQuantConfigError("only FP16 activations were measured")
        if raw["acceptance_scope"] != "raw-kernel-candidate-only":
            raise AscendQuantConfigError(
                "only raw-kernel-candidate-only scope exists; "
                "production is not admitted"
            )
        if raw["production_accepted"] is not False:
            raise AscendQuantConfigError(
                "production_accepted must remain false until a real "
                "engine E2E gate passes"
            )
        if raw["operator_schema_sha256"] != EXPECTED_OPERATOR_SCHEMA_SHA256:
            raise AscendQuantConfigError(
                "operator schema hash does not match the measured ABI"
            )
        expected_stack = {
            "device_name": EXPECTED_DEVICE_NAME,
            "torch_version": EXPECTED_TORCH_VERSION,
            "torch_npu_version": EXPECTED_TORCH_NPU_VERSION,
            "cann_version": EXPECTED_CANN_VERSION,
        }
        if raw["verified_stack"] != expected_stack:
            raise AscendQuantConfigError(
                "verified_stack must equal the exact measured stack"
            )
        try:
            shapes = tuple(
                tuple(int(x) for x in pair) for pair in raw["verified_ffn_shapes"]
            )
        except (TypeError, ValueError) as exc:
            raise AscendQuantConfigError(
                "verified_ffn_shapes must be integer [K,N] pairs"
            ) from exc
        if set(shapes) != set(VERIFIED_FFN_SHAPES) or len(shapes) != len(
            VERIFIED_FFN_SHAPES
        ):
            raise AscendQuantConfigError(
                "verified_ffn_shapes differs from measured RWKV-7 FFN shapes"
            )
        bit = int(raw["bit"])
        if bit not in (4, 8):
            raise AscendQuantConfigError("bit must be 4 or 8")
        group_size = int(raw["group_size"])
        if group_size != RAW_CANDIDATE_GROUP_SIZE[bit]:
            raise AscendQuantConfigError(
                f"bit={bit} requires measured group_size="
                f"{RAW_CANDIDATE_GROUP_SIZE[bit]}"
            )
        try:
            admitted_rows = tuple(int(row) for row in raw["admitted_rows"])
        except (TypeError, ValueError) as exc:
            raise AscendQuantConfigError("admitted_rows must be integers") from exc
        if not admitted_rows or len(set(admitted_rows)) != len(admitted_rows):
            raise AscendQuantConfigError("admitted_rows must be non-empty and unique")
        if not set(admitted_rows) <= set(RAW_CANDIDATE_ROWS[bit]):
            raise AscendQuantConfigError(
                f"unmeasured rows for W{bit}; candidates are {RAW_CANDIDATE_ROWS[bit]}"
            )
        ffn = raw["ffn"]
        if not isinstance(ffn, Mapping) or set(ffn) != {"key_layers", "value_layers"}:
            raise AscendQuantConfigError(
                "ffn must contain exactly key_layers and value_layers"
            )

        def parse_layers(name: str) -> tuple[int, ...]:
            try:
                values = tuple(int(item) for item in ffn[name])
            except (TypeError, ValueError) as exc:
                raise AscendQuantConfigError(
                    f"{name} must be integer layer ids"
                ) from exc
            if len(values) != len(set(values)) or any(item < 0 for item in values):
                raise AscendQuantConfigError(
                    f"{name} must contain unique non-negative ids"
                )
            return values

        key_layers = parse_layers("key_layers")
        value_layers = parse_layers("value_layers")
        if not key_layers and not value_layers:
            raise AscendQuantConfigError("at least one FFN projection must be selected")
        source = str(raw["source"])
        if source not in {"fp-checkpoint", "packed-checkpoint"}:
            raise AscendQuantConfigError(
                "source must be fp-checkpoint or packed-checkpoint"
            )
        tensor_raw = raw["tensors"]
        if not isinstance(tensor_raw, Mapping):
            raise AscendQuantConfigError("tensors must be an object")
        records = {
            str(name): TensorRecord.parse(value, str(name))
            for name, value in tensor_raw.items()
        }
        manifest = cls(
            backend=backend,
            bit=bit,
            group_size=group_size,
            admitted_rows=admitted_rows,
            key_layers=key_layers,
            value_layers=value_layers,
            source=source,
            tensor_records=records,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        manifest._validate_tensor_records()
        return manifest

    @classmethod
    def from_path(cls, path: str | Path, *, backend: str) -> AscendQuantManifest:
        manifest_path = Path(path)
        try:
            payload = manifest_path.read_bytes()
            raw = json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise AscendQuantConfigError(
                f"cannot read quant manifest {manifest_path}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise AscendQuantConfigError("quant manifest root must be an object")
        return cls.from_mapping(
            raw,
            backend=backend,
            manifest_path=str(manifest_path.resolve()),
            manifest_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def expected_component_shapes(
        self, layer: int, projection: str
    ) -> dict[str, tuple[tuple[int, ...], str]]:
        # Shape is selected from the measured pair when the module is constructed;
        # this helper is completed by AscendPackedLinear.expected_components().
        del layer, projection
        raise AssertionError("module dimensions are required")

    def _validate_tensor_records(self) -> None:
        if self.source == "fp-checkpoint":
            if self.tensor_records:
                raise AscendQuantConfigError(
                    "fp-checkpoint source requires an empty tensors object"
                )
            return
        expected_names = set()
        for layer, projection in self.selected:
            base = self.module_path(layer, projection)
            expected_names.update(
                {base + ".qweight", base + ".scales", base + ".offsets"}
            )
        if set(self.tensor_records) != expected_names:
            missing = sorted(expected_names - set(self.tensor_records))
            extra = sorted(set(self.tensor_records) - expected_names)
            raise AscendQuantConfigError(
                f"packed tensor manifest key mismatch; missing={missing}, extra={extra}"
            )


def _pack_int4_cpu(q_kn: Tensor) -> Tensor:
    """Pack signed int4 [K,N] into int32 [K,N/8], low nibble first."""
    if q_kn.ndim != 2 or q_kn.shape[1] % 8:
        raise ValueError("signed int4 matrix requires N divisible by 8")
    q = q_kn.detach().to(device="cpu", dtype=torch.int32).reshape(q_kn.shape[0], -1, 8)
    packed = torch.zeros(q.shape[0], q.shape[1], dtype=torch.int32)
    for nibble in range(8):
        packed.bitwise_or_((q[:, :, nibble] & 0xF) << (4 * nibble))
    return packed.contiguous()


class AscendPackedLinear(nn.Module):
    """Bias-free W8A16/W4A16 projection with no dense fallback weight.

    Persistent storage is exactly ``qweight``, ``scales`` and ``offsets``.
    ``offsets`` is a zero-length fp16 tensor for W8.  Runtime closures are bound
    once after loading and are not part of the state_dict.
    """

    def __init__(
        self, in_features: int, out_features: int, *, bit: int, group_size: int
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bit = int(bit)
        self.group_size = int(group_size)
        if self.bit not in (4, 8):
            raise ValueError("bit must be 4 or 8")
        if self.bit == 4 and (
            self.group_size != 128
            or self.in_features % self.group_size
            or self.out_features % 8
        ):
            raise ValueError("W4 requires group_size=128, K%128==0 and N%8==0")
        if self.bit == 8 and self.group_size != 0:
            raise ValueError("W8 group_size must be zero")
        self.register_buffer(
            "qweight",
            torch.empty(0, dtype=torch.int32 if self.bit == 4 else torch.int8),
        )
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("offsets", torch.empty(0, dtype=torch.float16))
        self._loaded_components: set[str] = set()
        self._kernels: dict[int, Callable[[Tensor], Tensor]] = {}
        self._expected_device_type: str | None = None

    def _apply(self, fn, recurse: bool = True):
        self._kernels.clear()
        self._expected_device_type = None
        return super()._apply(fn, recurse=recurse)

    def expected_components(self) -> dict[str, tuple[tuple[int, ...], str]]:
        if self.bit == 8:
            return {
                "qweight": ((self.in_features, self.out_features), "int8"),
                "scales": ((self.out_features,), "float16"),
                "offsets": ((0,), "float16"),
            }
        return {
            "qweight": ((self.in_features, self.out_features // 8), "int32"),
            "scales": (
                (self.in_features // self.group_size, self.out_features),
                "float16",
            ),
            "offsets": (
                (self.in_features // self.group_size, self.out_features),
                "float16",
            ),
        }

    @torch.no_grad()
    def load_fp_weight(self, weight: Tensor) -> None:
        if self._loaded_components:
            raise AscendQuantLoadError("projection was loaded more than once")
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise AscendQuantLoadError(
                f"source weight shape {tuple(weight.shape)} != "
                f"{(self.out_features, self.in_features)}"
            )
        if not weight.dtype.is_floating_point:
            raise AscendQuantLoadError(
                "source checkpoint weight must be floating point"
            )
        wf = weight.detach().to(device="cpu", dtype=torch.float32)
        if self.bit == 8:
            scales = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
            q_nk = torch.round(wf / scales[:, None]).clamp(-127, 127).to(torch.int8)
            self.qweight = q_nk.t().contiguous()
            self.scales = scales.to(torch.float16).contiguous()
            self.offsets = torch.empty(0, dtype=torch.float16)
        else:
            groups = self.in_features // self.group_size
            grouped = wf.reshape(self.out_features, groups, self.group_size)
            scales_ng = (grouped.abs().amax(dim=2) / 7.0).clamp_min(1e-8)
            q_nk = (
                torch.round(grouped / scales_ng[:, :, None]).clamp(-8, 7).to(torch.int8)
            )
            self.qweight = _pack_int4_cpu(
                q_nk.reshape(self.out_features, self.in_features).t().contiguous()
            )
            self.scales = scales_ng.t().to(torch.float16).contiguous()
            self.offsets = torch.zeros_like(self.scales)
        self._loaded_components = {"qweight", "scales", "offsets"}

    @torch.no_grad()
    def load_packed_component(self, component: str, tensor: Tensor) -> None:
        if component not in self.expected_components():
            raise AscendQuantLoadError(f"unexpected packed component {component!r}")
        if component in self._loaded_components:
            raise AscendQuantLoadError(f"duplicate packed component {component}")
        expected_shape, expected_dtype = self.expected_components()[component]
        if (
            tuple(tensor.shape) != expected_shape
            or str(tensor.dtype).removeprefix("torch.") != expected_dtype
        ):
            raise AscendQuantLoadError(
                f"{component} must be {expected_shape} {expected_dtype}, got "
                f"{tuple(tensor.shape)} {tensor.dtype}"
            )
        setattr(self, component, tensor.detach().contiguous().cpu())
        self._loaded_components.add(component)

    def validate_storage(self) -> None:
        expected = self.expected_components()
        if self._loaded_components != set(expected):
            raise AscendQuantLoadError(
                "missing packed components: "
                f"{sorted(set(expected) - self._loaded_components)}"
            )
        if list(self.named_parameters(recurse=False)):
            raise AscendQuantLoadError("quant projection must not retain parameters")
        if set(dict(self.named_buffers(recurse=False))) != {
            "qweight",
            "scales",
            "offsets",
        }:
            raise AscendQuantLoadError("quant projection buffer set changed")
        for name, (shape, dtype) in expected.items():
            tensor = getattr(self, name)
            if (
                tuple(tensor.shape) != shape
                or str(tensor.dtype).removeprefix("torch.") != dtype
            ):
                raise AscendQuantLoadError(f"invalid stored {name}")
        if hasattr(self, "weight"):
            raise AscendQuantLoadError("dense weight attribute is forbidden")

    def bind(
        self,
        op: Callable[..., Tensor],
        admitted_rows: tuple[int, ...],
        *,
        execution_device: str | torch.device,
    ) -> None:
        self.validate_storage()
        device = torch.device(execution_device)
        self.qweight = self.qweight.to(device=device, non_blocking=False).contiguous()
        self.scales = self.scales.to(device=device, non_blocking=False).contiguous()
        self.offsets = self.offsets.to(device=device, non_blocking=False).contiguous()
        qweight, scales, offsets = self.qweight, self.scales, self.offsets
        kernels: dict[int, Callable[[Tensor], Tensor]] = {}
        if self.bit == 8:
            for rows in admitted_rows:

                def kernel(x: Tensor, _op=op, _q=qweight, _s=scales):
                    return _op(x, _q, _s)

                kernels[int(rows)] = kernel
        else:
            group_size = self.group_size
            for rows in admitted_rows:

                def kernel(
                    x: Tensor,
                    _op=op,
                    _q=qweight,
                    _s=scales,
                    _o=offsets,
                    _g=group_size,
                ):
                    return _op(x, _q, _s, _o, None, None, None, _g, 1)

                kernels[int(rows)] = kernel
        self._kernels = kernels
        self._expected_device_type = device.type

    def forward(self, x: Tensor) -> Tensor:
        # These are tensor ABI guards, not runtime/acceptance-policy checks. The
        # expensive stack and shape admission policy was evaluated once at bind.
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise AscendQuantRuntimeError(
                f"quant projection requires [M,{self.in_features}], "
                f"got {tuple(x.shape)}"
            )
        if x.dtype is not torch.float16:
            raise AscendQuantRuntimeError(
                "quant projection accepts FP16 activations only"
            )
        if (
            self._expected_device_type is None
            or x.device.type != self._expected_device_type
        ):
            raise AscendQuantRuntimeError(
                "quant projection is unbound or on the wrong device"
            )
        kernel = self._kernels.get(int(x.shape[0]))
        if kernel is None:
            raise AscendQuantRuntimeError(
                f"M={x.shape[0]} has no pre-bound raw-candidate kernel; "
                f"admitted={tuple(self._kernels)}"
            )
        return kernel(x)

    def packed_weight_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.qweight, self.scales, self.offsets)
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit={self.bit}, group_size={self.group_size}, production_accepted=False"
        )


class AscendQuantActivation:
    """Construction, checkpoint mapping, and one-time kernel binding controller."""

    def __init__(
        self,
        manifest: AscendQuantManifest,
        runtime: RuntimeFingerprint,
        op: Callable[..., Tensor],
        *,
        execution_device: str | torch.device = "npu:0",
        allow_cpu_test: bool = False,
    ) -> None:
        runtime.validate_exact()
        device = torch.device(execution_device)
        if device.type != "npu" and not allow_cpu_test:
            raise AscendQuantRuntimeError(
                "production quant activation requires an NPU device"
            )
        self.manifest = manifest
        self.runtime = runtime
        self.op = op
        self.execution_device = device
        self.allow_cpu_test = bool(allow_cpu_test)
        self.modules: dict[str, AscendPackedLinear] = {}
        self._finished = False

    def make_ffn_linear(
        self,
        layer: int,
        projection: str,
        in_features: int,
        out_features: int,
    ) -> AscendPackedLinear | None:
        if (int(layer), str(projection)) not in self.manifest.selected:
            return None
        if (int(in_features), int(out_features)) not in VERIFIED_FFN_SHAPES:
            raise AscendQuantConfigError(
                f"unmeasured FFN shape K={in_features}, N={out_features}"
            )
        path = self.manifest.module_path(int(layer), str(projection))
        if path in self.modules:
            raise AscendQuantConfigError(f"duplicate selected projection {path}")
        module = AscendPackedLinear(
            int(in_features),
            int(out_features),
            bit=self.manifest.bit,
            group_size=self.manifest.group_size,
        )
        self.modules[path] = module
        return module

    def validate_construction(self, *, num_layers: int) -> None:
        invalid_layers = sorted(
            {layer for layer, _ in self.manifest.selected if layer >= int(num_layers)}
        )
        if invalid_layers:
            raise AscendQuantConfigError(
                f"manifest selects layers outside model: {invalid_layers}; "
                f"num_layers={num_layers}"
            )
        expected = {
            self.manifest.module_path(layer, projection)
            for layer, projection in self.manifest.selected
        }
        if set(self.modules) != expected:
            raise AscendQuantConfigError(
                "selected projection construction mismatch: "
                f"expected={sorted(expected)}, "
                f"actual={sorted(self.modules)}"
            )
        if self.manifest.source == "packed-checkpoint":
            for path, module in self.modules.items():
                for component, (shape, dtype) in module.expected_components().items():
                    record = self.manifest.tensor_records[path + "." + component]
                    if record.shape != shape or record.dtype != dtype:
                        raise AscendQuantConfigError(
                            f"packed metadata for {path}.{component} "
                            "does not match module ABI"
                        )

    def owns_selected_namespace(self, checkpoint_name: str) -> bool:
        return any(
            checkpoint_name == path + ".weight"
            or checkpoint_name.startswith(path + ".")
            for path in self.modules
        )

    def load_tensor(self, checkpoint_name: str, tensor: Tensor) -> bool:
        for path, module in self.modules.items():
            if checkpoint_name == path + ".weight":
                if self.manifest.source != "fp-checkpoint":
                    raise AscendQuantLoadError(
                        f"{checkpoint_name} supplied but manifest "
                        "requires packed-checkpoint"
                    )
                module.load_fp_weight(tensor)
                return True
            prefix = path + "."
            if checkpoint_name.startswith(prefix):
                component = checkpoint_name[len(prefix) :]
                if component not in {"qweight", "scales", "offsets"}:
                    raise AscendQuantLoadError(
                        f"unexpected selected projection tensor {checkpoint_name}"
                    )
                if self.manifest.source != "packed-checkpoint":
                    raise AscendQuantLoadError(
                        f"{checkpoint_name} supplied but manifest "
                        "requires fp-checkpoint"
                    )
                record = self.manifest.tensor_records[checkpoint_name]
                actual_dtype = str(tensor.dtype).removeprefix("torch.")
                if tuple(tensor.shape) != record.shape or actual_dtype != record.dtype:
                    raise AscendQuantLoadError(
                        f"packed tensor metadata mismatch for {checkpoint_name}"
                    )
                if _sha256_tensor(tensor) != record.sha256:
                    raise AscendQuantLoadError(
                        f"packed tensor SHA256 mismatch for {checkpoint_name}"
                    )
                module.load_packed_component(component, tensor)
                return True
        return False

    def finish_load(self) -> dict[str, Any]:
        if self._finished:
            raise AscendQuantLoadError(
                "quant checkpoint loader finalized more than once"
            )
        total_packed = 0
        total_fp = 0
        for path, module in sorted(self.modules.items()):
            try:
                module.bind(
                    self.op,
                    self.manifest.admitted_rows,
                    execution_device=self.execution_device,
                )
            except Exception as exc:
                if isinstance(exc, AscendQuantError):
                    raise
                raise AscendQuantLoadError(f"failed to bind {path}") from exc
            total_packed += module.packed_weight_bytes()
            total_fp += module.in_features * module.out_features * 2
        self._finished = True
        return {
            "enabled": True,
            "backend": self.manifest.backend,
            "bit": self.manifest.bit,
            "group_size": self.manifest.group_size,
            "source": self.manifest.source,
            "selected_modules": sorted(self.modules),
            "admitted_rows": list(self.manifest.admitted_rows),
            "packed_bytes": total_packed,
            "removed_fp16_weight_bytes": total_fp,
            "packed_storage_ratio": total_packed / total_fp,
            "runtime": {
                "device_name": self.runtime.device_name,
                "torch_version": self.runtime.torch_version,
                "torch_npu_version": self.runtime.torch_npu_version,
                "cann_version": self.runtime.cann_version,
                "operator_schema_sha256": self.runtime.operator_schema_sha256,
            },
            "manifest_path": self.manifest.manifest_path,
            "manifest_sha256": self.manifest.manifest_sha256,
            "acceptance_scope": "raw-kernel-candidate-only",
            "production_accepted": False,
        }


def activate_quant_from_env(
    *,
    backend: str,
    environ: Mapping[str, str] | None = None,
) -> AscendQuantActivation | None:
    """Return an active controller or the unchanged dense default.

    Any partially configured or over-claiming setup raises.  There is no
    production override: the explicit acknowledgement only admits measured raw
    operator candidates for further engine E2E experimentation.
    """
    env = os.environ if environ is None else environ
    enabled = env.get(ENABLE_ENV)
    path = env.get(MANIFEST_ENV)
    ack = env.get(RAW_ACK_ENV)
    if enabled in (None, "0") and path is None and ack is None:
        return None
    if enabled != "1" or not path:
        raise AscendQuantConfigError(
            f"quant activation requires {ENABLE_ENV}=1 and an explicit {MANIFEST_ENV}"
        )
    if ack != "1":
        raise AscendQuantConfigError(
            f"raw-candidate execution requires {RAW_ACK_ENV}=1; "
            "it is not production acceptance"
        )
    manifest = AscendQuantManifest.from_path(path, backend=backend)
    runtime, op = RuntimeFingerprint.detect()
    return AscendQuantActivation(manifest, runtime, op, execution_device=runtime.device)


__all__ = [
    "AscendPackedLinear",
    "AscendQuantActivation",
    "AscendQuantConfigError",
    "AscendQuantError",
    "AscendQuantLoadError",
    "AscendQuantManifest",
    "AscendQuantRuntimeError",
    "EXPECTED_CANN_VERSION",
    "EXPECTED_DEVICE_NAME",
    "EXPECTED_OPERATOR_SCHEMA_SHA256",
    "EXPECTED_TORCH_NPU_VERSION",
    "EXPECTED_TORCH_VERSION",
    "MANIFEST_FORMAT",
    "RAW_CANDIDATE_ROWS",
    "RuntimeFingerprint",
    "VERIFIED_FFN_SHAPES",
    "activate_quant_from_env",
]
