"""vLLM general plugin entry point."""

from __future__ import annotations

import importlib.metadata

SUPPORTED_VLLM = {"0.18.0"}
ARCHITECTURES = ("NativeRWKV7ForCausalLM", "RWKV7ForCausalLM", "Rwkv7ForCausalLM")


def register() -> None:
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    class RWKV7MambaConfig(MambaModelConfig):
        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            if vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "RWKV-7 prefix caching is not validated; "
                    "pass --disable-prefix-caching"
                )
            if vllm_config.speculative_config is not None:
                raise NotImplementedError(
                    "RWKV-7 speculative decoding is not validated"
                )
            MambaModelConfig.verify_and_update_config(vllm_config)
            if vllm_config.cache_config.mamba_cache_mode != "none":
                raise NotImplementedError("RWKV-7 only supports mamba_cache_mode=none")

    version = importlib.metadata.version("vllm").split("+")[0]
    if version not in SUPPORTED_VLLM:
        raise RuntimeError(
            f"rwkv7-vllm-ascend supports vLLM {sorted(SUPPORTED_VLLM)}, found {version}"
        )
    target = "rwkv7_vllm_ascend.model:RWKV7ForCausalLM"
    for architecture in ARCHITECTURES:
        ModelRegistry.register_model(architecture, target)
        MODELS_CONFIG_MAP[architecture] = RWKV7MambaConfig
