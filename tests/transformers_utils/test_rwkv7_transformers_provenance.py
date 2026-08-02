# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import subprocess
import sys
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


def _write_fake_transformers_distribution(
    root: Path,
    *,
    requested_revision: str = TRANSFORMERS_RWKV_REVISION,
    commit_id: str = TRANSFORMERS_RWKV_REVISION,
) -> None:
    package = root / "transformers"
    rwkv7 = package / "models" / "rwkv7"
    rwkv7.mkdir(parents=True)
    (package / "models" / "__init__.py").write_text("", encoding="utf-8")
    (rwkv7 / "configuration_rwkv7.py").write_text(
        "class Rwkv7Config:\n    pass\n",
        encoding="utf-8",
    )
    (rwkv7 / "modeling_rwkv7.py").write_text(
        "\n".join(
            (
                "class Rwkv7ForCausalLM:",
                "    pass",
                "",
                "def validate_rwkv7_runtime_provenance():",
                "    return {",
                f"        'revision': {FLA_RWKV_REVISION!r},",
                f"        'flash_rwkv_revision': {FLASH_RWKV_REVISION!r},",
                "    }",
                "",
            )
        ),
        encoding="utf-8",
    )
    (rwkv7 / "__init__.py").write_text(
        "from .configuration_rwkv7 import Rwkv7Config\n"
        "from .modeling_rwkv7 import Rwkv7ForCausalLM\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "from .models.rwkv7 import Rwkv7Config, Rwkv7ForCausalLM\n"
        "class AutoConfig:\n"
        "    @staticmethod\n"
        "    def for_model(name):\n"
        "        assert name == 'rwkv7'\n"
        "        return Rwkv7Config()\n"
        "class AutoModelForCausalLM:\n"
        "    _model_mapping = {Rwkv7Config: Rwkv7ForCausalLM}\n",
        encoding="utf-8",
    )

    dist_info = root / "transformers-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: transformers\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": TRANSFORMERS_RWKV_REPOSITORY,
                "vcs_info": {
                    "vcs": "git",
                    "requested_revision": requested_revision,
                    "commit_id": commit_id,
                },
            }
        ),
        encoding="utf-8",
    )


def _run_fresh_provenance(root: Path) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).parents[2]
    code = """
import json
from vllm.transformers_utils.rwkv7_provenance import (
    validate_transformers_rwkv7_runtime_provenance,
)
print(json.dumps(validate_transformers_rwkv7_runtime_provenance()))
"""
    python_path = os.pathsep.join(
        [str(root), str(repo), os.environ.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": python_path,
        },
    )


def test_fresh_process_accepts_exact_pinned_rwkv7_provenance(tmp_path: Path) -> None:
    _write_fake_transformers_distribution(tmp_path)
    completed = _run_fresh_provenance(tmp_path)

    assert completed.returncode == 0, completed.stderr
    provenance = json.loads(completed.stdout.splitlines()[-1])
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


def test_fresh_process_rejects_wrong_transformers_revision(tmp_path: Path) -> None:
    wrong_revision = "0" * 40
    _write_fake_transformers_distribution(
        tmp_path,
        requested_revision=wrong_revision,
        commit_id=wrong_revision,
    )
    completed = _run_fresh_provenance(tmp_path)

    assert completed.returncode != 0
    assert "Transformers provenance mismatch" in completed.stderr
    assert TRANSFORMERS_RWKV_REQUIREMENT in completed.stderr
