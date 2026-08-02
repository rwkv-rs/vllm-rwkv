# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from packaging.requirements import Requirement

from vllm.model_executor.models.rwkv7_wkv_backend import (
    FLA_RWKV_REVISION,
    FLASH_RWKV_REVISION,
)
from vllm.transformers_utils.rwkv7_provenance import (
    TRANSFORMERS_RWKV_REPOSITORY,
    TRANSFORMERS_RWKV_REQUIREMENT,
    TRANSFORMERS_RWKV_REVISION,
)


def _rwkv_requirements() -> dict[str, Requirement]:
    path = Path(__file__).parents[2] / "requirements" / "rwkv.txt"
    return {
        requirement.name: requirement
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
        for requirement in (Requirement(line),)
    }


def test_rwkv_profile_pins_transformers_and_operator_forks() -> None:
    requirements = _rwkv_requirements()

    assert requirements["transformers"] == Requirement(TRANSFORMERS_RWKV_REQUIREMENT)
    assert requirements["transformers"].url == (
        f"git+{TRANSFORMERS_RWKV_REPOSITORY}@{TRANSFORMERS_RWKV_REVISION}"
    )
    assert requirements["flash-linear-attention"].extras == {"flash-rwkv"}
    assert requirements["flash-linear-attention"].url == (
        f"git+https://github.com/rwkv-rs/fla-rwkv.git@{FLA_RWKV_REVISION}"
    )
    assert requirements["flash-rwkv"].url == (
        f"git+https://github.com/rwkv-rs/FlashRWKV.git@{FLASH_RWKV_REVISION}"
    )


def test_fresh_process_accepts_pinned_rwkv7_or_fails_closed() -> None:
    code = textwrap.dedent(
        """
        import json

        from vllm.transformers_utils.rwkv7_provenance import (
            validate_transformers_rwkv7_runtime_provenance,
        )

        try:
            provenance = validate_transformers_rwkv7_runtime_provenance()
        except RuntimeError as error:
            print(json.dumps({"status": "rejected", "error": str(error)}))
        else:
            print(json.dumps({"status": "accepted", "provenance": provenance}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    result = json.loads(completed.stdout.splitlines()[-1])

    if result["status"] == "rejected":
        assert TRANSFORMERS_RWKV_REQUIREMENT in result["error"] or (
            "flash-linear-attention[flash-rwkv]" in result["error"]
        )
        return

    provenance = result["provenance"]
    assert provenance["repository"] == TRANSFORMERS_RWKV_REPOSITORY
    assert provenance["revision"] == TRANSFORMERS_RWKV_REVISION
    assert provenance["config_module"] == (
        "transformers.models.rwkv7.configuration_rwkv7"
    )
    assert provenance["causal_lm_module"] == (
        "transformers.models.rwkv7.modeling_rwkv7"
    )
    assert provenance["operator_runtime"]["revision"] == FLA_RWKV_REVISION
    assert provenance["operator_runtime"]["flash_rwkv_revision"] == FLASH_RWKV_REVISION
