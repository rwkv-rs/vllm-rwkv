# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Literal

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from torch import nn

    from vllm.model_executor.model_loader.base_loader import BaseModelLoader

logger = init_logger(__name__)

# Reminder: Please update docstring in `LoadConfig`
# if a new load format is added here
LoadFormats = Literal[
    "auto",
    "hf",
    "bitsandbytes",
    "dummy",
    "fastsafetensors",
    "instanttensor",
    "mistral",
    "modelexpress",
    "npcache",
    "pt",
    "runai_streamer",
    "runai_streamer_sharded",
    "rwkv_pth",
    "safetensors",
    "sharded_state",
    "tensorizer",
]
_BUILTIN_LOADERS = {
    "auto": ("default_loader", "DefaultModelLoader"),
    "hf": ("default_loader", "DefaultModelLoader"),
    "bitsandbytes": ("bitsandbytes_loader", "BitsAndBytesModelLoader"),
    "dummy": ("dummy_loader", "DummyModelLoader"),
    "fastsafetensors": ("default_loader", "DefaultModelLoader"),
    "instanttensor": ("default_loader", "DefaultModelLoader"),
    "mistral": ("default_loader", "DefaultModelLoader"),
    "modelexpress": ("modelexpress_loader", "ModelExpressModelLoader"),
    "npcache": ("default_loader", "DefaultModelLoader"),
    "pt": ("default_loader", "DefaultModelLoader"),
    "runai_streamer": ("runai_streamer_loader", "RunaiModelStreamerLoader"),
    "runai_streamer_sharded": ("sharded_state_loader", "ShardedStateLoader"),
    "rwkv_pth": ("rwkv_pth_loader", "RWKV7PthModelLoader"),
    "safetensors": ("default_loader", "DefaultModelLoader"),
    "sharded_state": ("sharded_state_loader", "ShardedStateLoader"),
    "tensorizer": ("tensorizer_loader", "TensorizerLoader"),
}
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {}

_PUBLIC_EXPORTS = {
    "BaseModelLoader": ("base_loader", "BaseModelLoader"),
    "BitsAndBytesModelLoader": ("bitsandbytes_loader", "BitsAndBytesModelLoader"),
    "DefaultModelLoader": ("default_loader", "DefaultModelLoader"),
    "DummyModelLoader": ("dummy_loader", "DummyModelLoader"),
    "ModelExpressModelLoader": ("modelexpress_loader", "ModelExpressModelLoader"),
    "RunaiModelStreamerLoader": (
        "runai_streamer_loader",
        "RunaiModelStreamerLoader",
    ),
    "RWKV7PthModelLoader": ("rwkv_pth_loader", "RWKV7PthModelLoader"),
    "ShardedStateLoader": ("sharded_state_loader", "ShardedStateLoader"),
    "TensorizerLoader": ("tensorizer_loader", "TensorizerLoader"),
    "get_architecture_class_name": ("utils", "get_architecture_class_name"),
    "get_model_architecture": ("utils", "get_model_architecture"),
    "get_model_cls": ("utils", "get_model_cls"),
}


def __getattr__(name: str):
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _PUBLIC_EXPORTS.keys())


def register_model_loader(load_format: str):
    """Register a customized vllm model loader.

    When a load format is not supported by vllm, you can register a customized
    model loader to support it.

    Args:
        load_format (str): The model loader format name.

    Examples:
        >>> from vllm.config.load import LoadConfig
        >>> from vllm.model_executor.model_loader import (
        ...     get_model_loader,
        ...     register_model_loader,
        ... )
        >>> from vllm.model_executor.model_loader.base_loader import BaseModelLoader
        >>>
        >>> @register_model_loader("my_loader")
        ... class MyModelLoader(BaseModelLoader):
        ...     def download_model(self):
        ...         pass
        ...
        ...     def load_weights(self):
        ...         pass
        >>>
        >>> load_config = LoadConfig(load_format="my_loader")
        >>> type(get_model_loader(load_config))
        <class 'MyModelLoader'>
    """  # noqa: E501

    def _wrapper(model_loader_cls):
        if (
            load_format in _BUILTIN_LOADERS
            or load_format in _LOAD_FORMAT_TO_MODEL_LOADER
        ):
            logger.warning(
                "Load format `%s` is already registered, and will be "
                "overwritten by the new loader class `%s`.",
                load_format,
                model_loader_cls,
            )
        from vllm.model_executor.model_loader.base_loader import BaseModelLoader

        if not issubclass(model_loader_cls, BaseModelLoader):
            raise ValueError(
                "The model loader must be a subclass of `BaseModelLoader`."
            )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = model_loader_cls
        logger.info(
            "Registered model loader `%s` with load format `%s`",
            model_loader_cls,
            load_format,
        )
        return model_loader_cls

    return _wrapper


def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    """Get a model loader based on the load format."""
    load_format = load_config.load_format
    loader_cls = _LOAD_FORMAT_TO_MODEL_LOADER.get(load_format)
    if loader_cls is None and load_format in _BUILTIN_LOADERS:
        module_name, class_name = _BUILTIN_LOADERS[load_format]
        loader_cls = getattr(
            import_module(f"{__name__}.{module_name}"),
            class_name,
        )
        _LOAD_FORMAT_TO_MODEL_LOADER[load_format] = loader_cls
    if loader_cls is None:
        raise ValueError(f"Load format `{load_format}` is not supported")
    return loader_cls(load_config)


def get_model(
    *,
    vllm_config: VllmConfig,
    model_config: ModelConfig | None = None,
    prefix: str = "",
    load_config: LoadConfig | None = None,
) -> nn.Module:
    loader = get_model_loader(load_config or vllm_config.load_config)
    if model_config is None:
        model_config = vllm_config.model_config
    return loader.load_model(
        vllm_config=vllm_config, model_config=model_config, prefix=prefix
    )


__all__ = [
    "get_model",
    "get_model_loader",
    "get_architecture_class_name",
    "get_model_architecture",
    "get_model_cls",
    "register_model_loader",
    "BaseModelLoader",
    "BitsAndBytesModelLoader",
    "ModelExpressModelLoader",
    "DefaultModelLoader",
    "DummyModelLoader",
    "RunaiModelStreamerLoader",
    "RWKV7PthModelLoader",
    "ShardedStateLoader",
    "TensorizerLoader",
]
