# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.attention_layer_base import AttentionPostLoadModule
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
)
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def test_auto_weights_loader_does_not_eagerly_import_optional_loaders():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from vllm.model_executor.models.utils import AutoWeightsLoader; "
                "assert 'vllm.model_executor.model_loader.default_loader' "
                "not in sys.modules; "
                "assert 'vllm.model_executor.model_loader.reload.layerwise' "
                "not in sys.modules"
            ),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_model_loader_public_exports_remain_compatible():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import vllm.model_executor.model_loader as model_loader; "
                "assert 'DefaultModelLoader' in dir(model_loader); "
                "from vllm.model_executor.model_loader import DefaultModelLoader; "
                "from vllm.model_executor.model_loader.default_loader "
                "import DefaultModelLoader as DirectDefaultModelLoader; "
                "assert DefaultModelLoader is DirectDefaultModelLoader; "
                "assert 'vllm.model_executor.model_loader.reload.layerwise' "
                "not in sys.modules"
            ),
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_attention_post_load_marker_controls_dtype_finalization():
    class AttentionModule(torch.nn.Module, AttentionPostLoadModule):
        def __init__(self):
            super().__init__()
            self.processed_dtype = None

        def process_weights_after_loading(self, dtype):
            self.processed_dtype = dtype

    class NonAttentionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.called = False

        def process_weights_after_loading(self, dtype):
            self.called = True

    model = torch.nn.Module()
    model.attention = AttentionModule()
    model.non_attention = NonAttentionModule()
    model_config = SimpleNamespace(dtype=torch.float16, quantization=None)

    process_weights_after_loading(model, model_config, torch.device("cpu"))

    assert model.attention.processed_dtype is torch.float16
    assert model.non_attention.called is False


class ModuleWithBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2)

    def forward(self, x):
        return self.bn(x)


class ModuleWithNestedBatchNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nested_mod = ModuleWithBatchNorm()

    def forward(self, x):
        return self.nested_mod(x)


@pytest.mark.cpu_test
def test_module_with_batchnorm_can_load():
    """Ensure the auto weight loader can load batchnorm stats."""
    mod = ModuleWithBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithBatchNorm()

    assert not torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert not torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(new_mod.bn.running_mean == mod.bn.running_mean)
    assert torch.all(new_mod.bn.running_var == mod.bn.running_var)
    assert new_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_with_child_containing_batchnorm_can_autoload():
    """Ensure the auto weight loader can load nested modules batchnorm stats."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        yield from mod.state_dict().items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod)
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_skip_prefix():
    """Ensure the auto weight loader can skip prefix."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        # weights needed to be filtered out
        redundant_weights = {
            "prefix.bn.weight": torch.Tensor([1, 2]),
            "prefix.bn.bias": torch.Tensor([3, 4]),
        }
        yield from (mod.state_dict() | redundant_weights).items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod, skip_prefixes=["prefix."])
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


@pytest.mark.cpu_test
def test_module_skip_substr():
    """Ensure the auto weight loader can skip prefix."""
    mod = ModuleWithNestedBatchNorm()
    # Run some data through the module with batchnorm
    mod(torch.Tensor([[1, 2], [3, 4]]))

    # Try to load the weights to a new instance
    def weight_generator():
        # weights needed to be filtered out
        redundant_weights = {
            "nested_mod.0.substr.weight": torch.Tensor([1, 2]),
            "nested_mod.0.substr.bias": torch.Tensor([3, 4]),
            "nested_mod.substr.weight": torch.Tensor([1, 2]),
            "nested_mod.substr.bias": torch.Tensor([3, 4]),
        }
        yield from (mod.state_dict() | redundant_weights).items()

    new_mod = ModuleWithNestedBatchNorm()

    assert not torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert not torch.all(
        new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var
    )
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 0

    loader = AutoWeightsLoader(new_mod, skip_substrs=["substr."])
    loader.load_weights(weight_generator())

    # Ensure the stats are updated
    assert torch.all(
        new_mod.nested_mod.bn.running_mean == mod.nested_mod.bn.running_mean
    )
    assert torch.all(new_mod.nested_mod.bn.running_var == mod.nested_mod.bn.running_var)
    assert new_mod.nested_mod.bn.num_batches_tracked.item() == 1


class raise_if_cuda_sync:
    def __enter__(self):
        self.previous_debug_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")

    def __exit__(self, exception_type, exception_value, traceback):
        torch.cuda.set_sync_debug_mode(self.previous_debug_mode)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Skip if not cuda")
def test_merge_multimodal_embeddings_no_sync():
    inputs_embeds = torch.zeros(
        [5, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0"
    )
    multimodal_embeddings = [
        torch.ones([3, 10], dtype=torch.bfloat16, device=f"{DEVICE_TYPE}:0")
    ]
    is_multimodal = torch.tensor([True, False, True, True, False], device="cpu")
    with raise_if_cuda_sync():
        _merge_multimodal_embeddings(
            inputs_embeds, multimodal_embeddings, is_multimodal
        )
