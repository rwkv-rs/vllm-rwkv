# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.transformers_utils.configs.rwkv7 import (
    download_rwkv7_pth_source,
    try_parse_rwkv7_pth_source,
)


class RWKV7PthModelLoader(DefaultModelLoader):
    """Load a raw RWKV-7 ``.pth`` checkpoint."""

    def _prepare_weights(
        self,
        model_name_or_path: str,
        subfolder: str | None,
        revision: str | None,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        del fall_back_to_pt, allow_patterns_overrides
        if subfolder is not None:
            raise ValueError("RWKV7 raw .pth checkpoints do not support subfolders.")

        source = try_parse_rwkv7_pth_source(model_name_or_path)
        if source is None:
            raise ValueError(
                "load_format='rwkv_pth' requires a supported RWKV-7 "
                f"raw .pth source, but got {model_name_or_path!r}."
            )

        checkpoint = download_rwkv7_pth_source(
            source,
            cache_dir=self.load_config.download_dir,
            revision=revision,
        )
        return str(checkpoint.parent), [str(checkpoint)], False
