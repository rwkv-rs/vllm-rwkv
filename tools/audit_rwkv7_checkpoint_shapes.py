# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit local RWKV-7 checkpoints against the checked-in shape provenance."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

DEFAULT_FIXTURE = (
    Path(__file__).parents[1]
    / "tests"
    / "transformers_utils"
    / "fixtures"
    / "rwkv7_g1_checkpoint_shapes.json"
)
SHAPE_KEYS = (
    "emb.weight",
    "blocks.0.att.w1",
    "blocks.0.att.a1",
    "blocks.0.att.v1",
    "blocks.0.att.g1",
    "blocks.0.ffn.key.weight",
)
BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Read every checkpoint byte and verify its recorded LFS SHA-256.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        while chunk := checkpoint.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_shapes(path: Path) -> dict[str, object]:
    state_dict = torch.load(
        path,
        map_location="meta",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(state_dict, dict):
        raise ValueError(f"{path}: checkpoint is not a state_dict")

    block_indices = {
        int(match.group(1))
        for key in state_dict
        if (match := BLOCK_KEY_RE.match(key)) is not None
    }
    if not block_indices or block_indices != set(range(max(block_indices) + 1)):
        raise ValueError(f"{path}: checkpoint block indices are not contiguous")

    missing = [key for key in SHAPE_KEYS if key not in state_dict]
    if missing:
        raise ValueError(f"{path}: missing audited tensors: {', '.join(missing)}")

    return {
        "num_hidden_layers": len(block_indices),
        **{key: list(state_dict[key].shape) for key in SHAPE_KEYS},
    }


def main() -> None:
    args = parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    expected_by_name = {
        checkpoint["filename"]: checkpoint for checkpoint in fixture["checkpoints"]
    }

    failures: list[str] = []
    for path in args.checkpoints:
        expected = expected_by_name.get(path.name)
        if expected is None:
            failures.append(f"{path}: filename is not present in the fixture")
            continue
        if not path.is_file():
            failures.append(f"{path}: checkpoint does not exist")
            continue
        if path.stat().st_size != expected["size"]:
            failures.append(
                f"{path}: size={path.stat().st_size}, expected={expected['size']}"
            )
            continue

        actual = checkpoint_shapes(path)
        for key in ("num_hidden_layers", *SHAPE_KEYS):
            if actual[key] != expected[key]:
                failures.append(
                    f"{path}: {key}={actual[key]}, expected={expected[key]}"
                )

        if args.verify_sha256:
            actual_sha256 = file_sha256(path)
            if actual_sha256 != expected["sha256"]:
                failures.append(
                    f"{path}: sha256={actual_sha256}, expected={expected['sha256']}"
                )

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(args.checkpoints)} RWKV-7 checkpoint(s)")


if __name__ == "__main__":
    main()
