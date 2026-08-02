# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared RWKV7 source provenance contract."""

from urllib.parse import urlsplit

import regex as re

FLA_RWKV_REPOSITORY = "https://github.com/rwkv-rs/fla-rwkv.git"
FLA_RWKV_REVISION = "f1888c9a81cb24bfa46d84dc9ab38c4abd746225"
FLASH_RWKV_REPOSITORY = "https://github.com/rwkv-rs/FlashRWKV.git"
FLASH_RWKV_REVISION = "9fe104c8c748771ba981058a6efcd95c150e453d"

_GITHUB_REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def canonicalize_github_repository_url(repository: object) -> str:
    """Return a strict ASCII canonical URL for a public GitHub repository."""
    if not isinstance(repository, str) or not repository.isascii():
        raise ValueError("GitHub repository URL must be an ASCII string")
    if any(
        ord(character) <= 0x20 or ord(character) == 0x7F for character in repository
    ):
        raise ValueError(
            "GitHub repository URL must not contain ASCII controls, spaces, or DEL"
        )

    value = repository.removeprefix("git+")
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("GitHub repository URL has an invalid port") from error
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.hostname.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("GitHub repository URL has forbidden authority or metadata")

    path = parsed.path
    if "%" in path or "//" in path:
        raise ValueError("GitHub repository URL path is not canonical ASCII")
    if path.endswith("/"):
        path = path[:-1]
    components = path.removeprefix("/").split("/")
    if len(components) != 2 or any(
        _GITHUB_REPOSITORY_COMPONENT.fullmatch(component) is None
        for component in components
    ):
        raise ValueError("GitHub repository URL must identify owner/repository")

    owner, name = components
    if name.lower().endswith(".git"):
        name = name[:-4]
    if not name or name.lower().endswith(".git"):
        raise ValueError("GitHub repository URL permits at most one .git suffix")
    return f"https://github.com/{owner.lower()}/{name.lower()}.git"


__all__ = [
    "FLASH_RWKV_REPOSITORY",
    "FLASH_RWKV_REVISION",
    "FLA_RWKV_REPOSITORY",
    "FLA_RWKV_REVISION",
    "canonicalize_github_repository_url",
]
