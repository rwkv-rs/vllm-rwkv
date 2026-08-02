# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build-profile selection shared by setup and its focused tests."""

from __future__ import annotations

import os
from collections.abc import Iterable

BUILD_PROFILES = ("full", "rwkv", "rwkv-nvfp4")
RWKV_EXTENSION_NAMES = (
    "vllm._rapid_sampling",
    "vllm.cumem_allocator",
    "vllm.rwkv7_ops",
)
RWKV_NVFP4_EXTENSION_NAMES = (
    *RWKV_EXTENSION_NAMES,
    "vllm._C_stable_libtorch",
)


def _validate_profile(profile: str) -> None:
    if profile not in BUILD_PROFILES:
        accepted = ", ".join(BUILD_PROFILES)
        raise ValueError(
            f"Invalid build profile {profile!r}; accepted values: {accepted}"
        )


def resolve_build_profile() -> str:
    profile = os.getenv("VLLM_BUILD_PROFILE", "full")
    if profile not in BUILD_PROFILES:
        accepted = ", ".join(BUILD_PROFILES)
        raise ValueError(
            f"Invalid VLLM_BUILD_PROFILE={profile!r}; accepted values: {accepted}"
        )
    return profile


def profile_build_temp(build_temp: str, profile: str) -> str:
    """Return a profile-specific CMake reuse directory."""
    _validate_profile(profile)
    return f"{build_temp}-{profile}" if profile != "full" else build_temp


def select_extension_names(names: Iterable[str], profile: str) -> list[str]:
    _validate_profile(profile)
    names = list(names)
    if profile == "full":
        return names
    required = (
        RWKV_NVFP4_EXTENSION_NAMES if profile == "rwkv-nvfp4" else RWKV_EXTENSION_NAMES
    )
    available = set(names)
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(
            "RWKV build profile is missing required extension(s): " + ", ".join(missing)
        )
    return [name for name in required if name in available]
