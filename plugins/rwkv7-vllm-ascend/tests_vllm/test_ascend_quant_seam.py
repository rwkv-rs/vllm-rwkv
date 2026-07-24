from __future__ import annotations

import hashlib

import pytest
import rwkv7_vllm_ascend.ascend_quant as q
import torch
import torch.nn.functional as F
from rwkv7_vllm_ascend.model import RWKV7FFN, RawLinear


def _digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    return hashlib.sha256(memoryview(cpu.numpy()).cast("B")).hexdigest()


def _runtime(device_name: str = q.EXPECTED_DEVICE_NAME) -> q.RuntimeFingerprint:
    return q.RuntimeFingerprint(
        device_name=device_name,
        torch_version=q.EXPECTED_TORCH_VERSION,
        torch_npu_version=q.EXPECTED_TORCH_NPU_VERSION,
        cann_version=q.EXPECTED_CANN_VERSION,
        operator_schema_sha256=q.EXPECTED_OPERATOR_SCHEMA_SHA256,
        device="cpu",
    )


def _mapping(*, bit: int, k: int, n: int, source: str, tensors=None):
    return {
        "format": q.MANIFEST_FORMAT,
        "version": 1,
        "backend": "vllm",
        "bit": bit,
        "group_size": 128 if bit == 4 else 0,
        "activation_dtype": "float16",
        "acceptance_scope": "raw-kernel-candidate-only",
        "production_accepted": False,
        "operator_schema_sha256": q.EXPECTED_OPERATOR_SCHEMA_SHA256,
        "verified_stack": {
            "device_name": q.EXPECTED_DEVICE_NAME,
            "torch_version": q.EXPECTED_TORCH_VERSION,
            "torch_npu_version": q.EXPECTED_TORCH_NPU_VERSION,
            "cann_version": q.EXPECTED_CANN_VERSION,
        },
        "verified_ffn_shapes": [[k, n]],
        "admitted_rows": [1] if bit == 4 else [17],
        "ffn": {"key_layers": [0], "value_layers": []},
        "source": source,
        "tensors": {} if tensors is None else tensors,
    }


def _activation(
    monkeypatch, *, bit: int, k: int, n: int, source="fp-checkpoint", tensors=None
):
    monkeypatch.setattr(q, "VERIFIED_FFN_SHAPES", ((k, n),))
    calls = []

    def fake_op(*args):
        calls.append(args)
        x, packed = args[:2]
        out_features = packed.shape[1] if bit == 8 else packed.shape[1] * 8
        return torch.zeros(x.shape[0], out_features, dtype=x.dtype, device=x.device)

    manifest = q.AscendQuantManifest.from_mapping(
        _mapping(bit=bit, k=k, n=n, source=source, tensors=tensors),
        backend="vllm",
    )
    activation = q.AscendQuantActivation(
        manifest,
        _runtime(),
        fake_op,
        execution_device="cpu",
        allow_cpu_test=True,
    )
    return activation, calls


def test_dense_default_and_dense_ffn_regression(monkeypatch):
    for name in (q.ENABLE_ENV, q.MANIFEST_ENV, q.RAW_ACK_ENV):
        monkeypatch.delenv(name, raising=False)
    assert q.activate_quant_from_env(backend="vllm") is None

    torch.manual_seed(9)
    ffn = RWKV7FFN(8, 16)
    assert isinstance(ffn.key, RawLinear)
    assert isinstance(ffn.value, RawLinear)
    assert set(ffn.state_dict()) == {"x_k", "key.weight", "value.weight"}
    torch.nn.init.normal_(ffn.key.weight, std=0.1)
    torch.nn.init.normal_(ffn.value.weight, std=0.1)
    x = torch.randn(3, 8)
    got = ffn.value(torch.relu(ffn.key(x)).square())
    expected = F.linear(
        torch.relu(F.linear(x, ffn.key.weight)).square(), ffn.value.weight
    )
    torch.testing.assert_close(got, expected)


@pytest.mark.parametrize("bit,k,n,rows", [(4, 128, 8, 1), (8, 8, 16, 17)])
def test_fp_checkpoint_mapping_packs_without_dense_copy(monkeypatch, bit, k, n, rows):
    activation, calls = _activation(monkeypatch, bit=bit, k=k, n=n)
    module = activation.make_ffn_linear(0, "key", k, n)
    assert isinstance(module, q.AscendPackedLinear)
    assert activation.make_ffn_linear(0, "value", n, k) is None
    activation.validate_construction(num_layers=1)
    assert activation.load_tensor("unrelated.weight", torch.empty(0)) is False
    assert activation.owns_selected_namespace("model.layers.0.ffn.key.weight")

    source = torch.linspace(-1, 1, n * k, dtype=torch.float32).reshape(n, k)
    assert activation.load_tensor("model.layers.0.ffn.key.weight", source)
    report = activation.finish_load()
    assert report["production_accepted"] is False
    assert report["acceptance_scope"] == "raw-kernel-candidate-only"
    assert report["packed_storage_ratio"] < 1.0
    assert dict(module.named_parameters()) == {}
    assert set(dict(module.named_buffers())) == {"qweight", "scales", "offsets"}
    assert not hasattr(module, "weight")
    assert module.scales.dtype is torch.float16
    assert module.offsets.dtype is torch.float16

    out = module(torch.ones(rows, k, dtype=torch.float16))
    assert out.shape == (rows, n)
    assert len(calls) == 1
    assert len(calls[0]) == (9 if bit == 4 else 3)
    with pytest.raises(q.AscendQuantRuntimeError, match="no pre-bound"):
        module(torch.ones(rows + 1, k, dtype=torch.float16))


def test_packed_manifest_hash_and_component_mapping(monkeypatch):
    k, n = 8, 16
    qweight = torch.arange(k * n, dtype=torch.int8).reshape(k, n)
    scales = torch.ones(n, dtype=torch.float16)
    offsets = torch.empty(0, dtype=torch.float16)
    base = "model.layers.0.ffn.key"
    tensors = {
        base + ".qweight": {
            "shape": [k, n],
            "dtype": "int8",
            "sha256": _digest(qweight),
        },
        base + ".scales": {"shape": [n], "dtype": "float16", "sha256": _digest(scales)},
        base + ".offsets": {
            "shape": [0],
            "dtype": "float16",
            "sha256": _digest(offsets),
        },
    }
    activation, calls = _activation(
        monkeypatch, bit=8, k=k, n=n, source="packed-checkpoint", tensors=tensors
    )
    module = activation.make_ffn_linear(0, "key", k, n)
    activation.validate_construction(num_layers=1)
    with pytest.raises(q.AscendQuantLoadError, match="SHA256"):
        activation.load_tensor(base + ".qweight", qweight.clone().fill_(0))

    # A rejected tensor does not mutate the module; authenticated tensors still load.
    activation.load_tensor(base + ".qweight", qweight)
    activation.load_tensor(base + ".scales", scales)
    activation.load_tensor(base + ".offsets", offsets)
    activation.finish_load()
    module(torch.ones(17, k, dtype=torch.float16))
    assert len(calls) == 1


def test_manifest_and_runtime_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(q, "VERIFIED_FFN_SHAPES", ((8, 16),))
    raw = _mapping(bit=8, k=8, n=16, source="fp-checkpoint")
    raw["production_accepted"] = True
    with pytest.raises(q.AscendQuantConfigError, match="production_accepted"):
        q.AscendQuantManifest.from_mapping(raw, backend="vllm")

    raw["production_accepted"] = False
    manifest = q.AscendQuantManifest.from_mapping(raw, backend="vllm")
    with pytest.raises(q.AscendQuantRuntimeError, match="unverified"):
        q.AscendQuantActivation(
            manifest,
            _runtime("Ascend910B4"),
            lambda *args: None,
            execution_device="cpu",
            allow_cpu_test=True,
        )

    with pytest.raises(q.AscendQuantConfigError, match="requires"):
        q.activate_quant_from_env(
            backend="vllm",
            environ={q.ENABLE_ENV: "1"},
        )
