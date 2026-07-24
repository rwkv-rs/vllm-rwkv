import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]


def test_no_fake_serving_import_in_plugin():
    for path in (ROOT / "rwkv7_vllm_ascend").glob("*.py"):
        tree = ast.parse(path.read_text())
        imports = {
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
        }
        assert not (imports & {"fastapi", "uvicorn", "starlette"})


def test_registers_real_vllm_architecture():
    pytest.importorskip("vllm")
    from rwkv7_vllm_ascend.plugin import register

    from vllm import ModelRegistry

    register()
    assert "NativeRWKV7ForCausalLM" in ModelRegistry.models
    assert ModelRegistry.models["NativeRWKV7ForCausalLM"] is not None


def test_cache_shape_keeps_wkv_fp32():
    pytest.importorskip("vllm")
    from rwkv7_vllm_ascend.model import RWKV7ForCausalLM

    class HF:
        hidden_size = 128
        n_layer = 2
        head_size = 64

    class MC:
        hf_config = HF()
        dtype = torch.float16

    class VC:
        model_config = MC()

    assert RWKV7ForCausalLM.get_mamba_state_shape_from_config(VC()) == (
        (2, 64, 64),
        (128,),
        (128,),
    )
    assert RWKV7ForCausalLM.get_mamba_state_dtype_from_config(VC())[0] is torch.float32


def test_plugin_registers_mamba_config_handler():
    from rwkv7_vllm_ascend.plugin import register

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    register()
    for architecture in (
        "NativeRWKV7ForCausalLM",
        "RWKV7ForCausalLM",
        "Rwkv7ForCausalLM",
    ):
        assert issubclass(MODELS_CONFIG_MAP[architecture], MambaModelConfig)
        assert MODELS_CONFIG_MAP[architecture] is not MambaModelConfig
