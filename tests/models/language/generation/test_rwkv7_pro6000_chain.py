# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused real-provider RWKV7 generation on the controlled PRO6000 chain."""

from __future__ import annotations

import contextlib
import functools
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from safetensors import safe_open

FLASH_RWKV_TEST_REVISION = "8b3d08a9a9430df23fb9da9b35fb0aa625faa1fb"
FLA_RWKV_TEST_REVISION = "606752b7dff79eb326eeebf2d046102027da5306"
TRANSFORMERS_RWKV_TEST_REVISION = "07dcea5f38c14177439374aa5ea6e1af2bca7266"

EXPECTED_GPU_NAME = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"
BOS_TOKEN_ID = 0
PROMPT_TOKEN_IDS = [1, 7, 11, 3]
NORMALIZED_PROMPT_TOKEN_IDS = [BOS_TOKEN_ID, *PROMPT_TOKEN_IDS]
RECURRENT_DECODE_STEPS = 2
GENERATED_TOKENS = RECURRENT_DECODE_STEPS + 1

requires_pro6000_chain = pytest.mark.skipif(
    os.getenv("RWKV7_RUN_PRO6000_CHAIN") != "1",
    reason="set RWKV7_RUN_PRO6000_CHAIN=1 in the focused PRO6000 job",
)


def _direct_url_revision(distribution_name: str) -> tuple[str, str, str]:
    distribution = importlib_metadata.distribution(distribution_name)
    direct_url_text = distribution.read_text("direct_url.json")
    assert direct_url_text is not None, f"{distribution_name} is not a VCS install"
    direct_url = json.loads(direct_url_text)
    vcs_info = direct_url["vcs_info"]
    assert vcs_info["vcs"] == "git"
    return (
        str(direct_url["url"]),
        str(vcs_info["requested_revision"]).lower(),
        str(vcs_info["commit_id"]).lower(),
    )


def _assert_controlled_test_dependencies() -> dict[str, dict[str, str]]:
    expected = {
        "flash-linear-attention": FLA_RWKV_TEST_REVISION,
        "flash-rwkv": FLASH_RWKV_TEST_REVISION,
        "transformers": TRANSFORMERS_RWKV_TEST_REVISION,
    }
    observed = {}
    for distribution_name, revision in expected.items():
        repository, requested, resolved = _direct_url_revision(distribution_name)
        assert requested == revision
        assert resolved == revision
        observed[distribution_name] = {
            "repository": repository,
            "revision": resolved,
        }
    return observed


def _install_test_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hf_rwkv7: Any,
    vllm_provenance: Any,
) -> dict[str, Any]:
    """Point validators at installed test heads without changing product pins."""
    monkeypatch.setattr(
        hf_rwkv7,
        "RWKV7_FLA_REVISION",
        FLA_RWKV_TEST_REVISION,
    )
    monkeypatch.setattr(
        hf_rwkv7,
        "RWKV7_FLA_REQUIREMENT",
        "flash-linear-attention[flash-rwkv] @ "
        "git+https://github.com/rwkv-rs/fla-rwkv.git@"
        f"{FLA_RWKV_TEST_REVISION}",
    )
    monkeypatch.setattr(
        hf_rwkv7,
        "RWKV7_FLASH_RWKV_REVISION",
        FLASH_RWKV_TEST_REVISION,
    )
    hf_rwkv7._load_fla_rwkv7_contract.cache_clear()

    monkeypatch.setattr(
        vllm_provenance,
        "TRANSFORMERS_RWKV_REVISION",
        TRANSFORMERS_RWKV_TEST_REVISION,
    )
    monkeypatch.setattr(
        vllm_provenance,
        "TRANSFORMERS_RWKV_REQUIREMENT",
        "transformers @ git+https://github.com/rwkv-rs/transformers-rwkv.git@"
        f"{TRANSFORMERS_RWKV_TEST_REVISION}",
    )
    monkeypatch.setattr(
        vllm_provenance,
        "FLA_RWKV_REVISION",
        FLA_RWKV_TEST_REVISION,
    )
    monkeypatch.setattr(
        vllm_provenance,
        "FLASH_RWKV_REVISION",
        FLASH_RWKV_TEST_REVISION,
    )
    vllm_provenance.validate_transformers_rwkv7_runtime_provenance.cache_clear()
    return vllm_provenance.validate_transformers_rwkv7_runtime_provenance()


@contextlib.contextmanager
def _test_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
    *,
    hf_rwkv7: Any,
    vllm_provenance: Any,
) -> Iterator[dict[str, Any]]:
    try:
        yield _install_test_only_provenance(
            monkeypatch,
            hf_rwkv7=hf_rwkv7,
            vllm_provenance=vllm_provenance,
        )
    finally:
        hf_rwkv7._load_fla_rwkv7_contract.cache_clear()
        vllm_provenance.validate_transformers_rwkv7_runtime_provenance.cache_clear()


@pytest.fixture
def test_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, Any]]:
    import transformers.models.rwkv7.modeling_rwkv7 as hf_rwkv7

    import vllm.transformers_utils.rwkv7_provenance as vllm_provenance

    with _test_only_provenance(
        monkeypatch,
        hf_rwkv7=hf_rwkv7,
        vllm_provenance=vllm_provenance,
    ) as provenance:
        yield provenance


def test_test_only_provenance_does_not_leak_into_later_validation() -> None:
    @functools.cache
    def fake_hf_contract() -> str:
        return fake_hf_rwkv7.RWKV7_FLA_REVISION

    fake_hf_rwkv7 = SimpleNamespace(
        RWKV7_FLA_REVISION="product-fla",
        RWKV7_FLA_REQUIREMENT="product-requirement",
        RWKV7_FLASH_RWKV_REVISION="product-flash",
        _load_fla_rwkv7_contract=fake_hf_contract,
    )

    @functools.cache
    def fake_validator() -> dict[str, str]:
        return {
            "transformers": fake_vllm_provenance.TRANSFORMERS_RWKV_REVISION,
            "fla": fake_vllm_provenance.FLA_RWKV_REVISION,
            "flash": fake_vllm_provenance.FLASH_RWKV_REVISION,
            "hf_fla": fake_hf_contract(),
        }

    fake_vllm_provenance = SimpleNamespace(
        TRANSFORMERS_RWKV_REVISION="product-transformers",
        TRANSFORMERS_RWKV_REQUIREMENT="product-requirement",
        FLA_RWKV_REVISION="product-fla",
        FLASH_RWKV_REVISION="product-flash",
        validate_transformers_rwkv7_runtime_provenance=fake_validator,
    )
    product_provenance = fake_validator()
    fake_validator.cache_clear()
    fake_hf_contract.cache_clear()

    for _ in range(2):
        with pytest.MonkeyPatch.context() as provenance_patch:
            with _test_only_provenance(
                provenance_patch,
                hf_rwkv7=fake_hf_rwkv7,
                vllm_provenance=fake_vllm_provenance,
            ) as synthetic_provenance:
                assert synthetic_provenance != product_provenance
                assert fake_validator.cache_info().currsize == 1
                assert fake_hf_contract.cache_info().currsize == 1

            assert fake_validator.cache_info().currsize == 0
            assert fake_hf_contract.cache_info().currsize == 0

        assert (
            product_provenance["transformers"]
            == fake_vllm_provenance.TRANSFORMERS_RWKV_REVISION
        )
        assert product_provenance["fla"] == fake_vllm_provenance.FLA_RWKV_REVISION
        assert product_provenance["flash"] == fake_vllm_provenance.FLASH_RWKV_REVISION
        assert product_provenance["hf_fla"] == fake_hf_rwkv7.RWKV7_FLA_REVISION
        assert fake_hf_contract() == product_provenance["hf_fla"]
        assert fake_validator() == product_provenance


def _write_tiny_standard_hf_artifact(artifact: Path) -> None:
    from transformers.models.rwkv7 import Rwkv7Config, Rwkv7ForCausalLM

    torch.manual_seed(17)
    config = Rwkv7Config(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        head_size=64,
        context_length=16,
        bos_token_id=BOS_TOKEN_ID,
        eos_token_id=BOS_TOKEN_ID,
    )
    model = Rwkv7ForCausalLM(config).eval().to(dtype=torch.bfloat16)
    model.save_pretrained(artifact, safe_serialization=True)

    saved_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert saved_config["model_type"] == "rwkv7"
    assert saved_config["architectures"] == ["Rwkv7ForCausalLM"]
    assert saved_config["dtype"] == "bfloat16"
    safetensors_path = artifact / "model.safetensors"
    assert safetensors_path.is_file()
    with safe_open(safetensors_path, framework="pt") as weights:
        weight_names = weights.keys()
        assert {weights.get_tensor(key).dtype for key in weight_names} == {
            torch.bfloat16
        }
    assert not list(artifact.glob("*.pth"))


@requires_pro6000_chain
def test_controlled_chain_matches_final_product_provenance() -> None:
    """The focused GPU chain must exercise the exact product revisions."""
    from vllm.transformers_utils.rwkv7_provenance import (
        TRANSFORMERS_RWKV_REVISION,
        validate_transformers_rwkv7_runtime_provenance,
    )

    dependencies = _assert_controlled_test_dependencies()
    assert TRANSFORMERS_RWKV_REVISION == TRANSFORMERS_RWKV_TEST_REVISION
    validate_transformers_rwkv7_runtime_provenance.cache_clear()
    provenance = validate_transformers_rwkv7_runtime_provenance()
    assert provenance["revision"] == TRANSFORMERS_RWKV_REVISION
    assert provenance["operator_runtime"]["revision"] == FLA_RWKV_TEST_REVISION
    assert (
        provenance["operator_runtime"]["flash_rwkv_revision"]
        == FLASH_RWKV_TEST_REVISION
    )
    assert dependencies["transformers"]["revision"] == TRANSFORMERS_RWKV_REVISION


@requires_pro6000_chain
def test_tiny_hf_artifact_generates_through_public_recurrent_flash_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    test_only_provenance: dict[str, Any],
) -> None:
    assert torch.cuda.is_available()
    assert torch.accelerator.device_count() == 1
    assert torch.cuda.get_device_name(0) == EXPECTED_GPU_NAME

    dependencies = _assert_controlled_test_dependencies()
    provenance = test_only_provenance
    assert provenance["config_module"] == (
        "transformers.models.rwkv7.configuration_rwkv7"
    )
    assert provenance["causal_lm_module"] == (
        "transformers.models.rwkv7.modeling_rwkv7"
    )
    assert provenance["operator_runtime"]["revision"] == FLA_RWKV_TEST_REVISION
    assert provenance["operator_runtime"]["flash_rwkv_revision"] == (
        FLASH_RWKV_TEST_REVISION
    )

    import flash_rwkv
    import flash_rwkv.ops as flash_rwkv_ops
    import flash_rwkv.reference as flash_rwkv_reference
    from fla.ops.rwkv7 import get_last_rwkv7_provider
    from fla.ops.rwkv7.backends.flash_rwkv import (
        FLASH_RWKV_SOURCE_REVISION,
        validate_flash_rwkv_installation,
    )

    import vllm.model_executor.models.rwkv7 as vllm_rwkv7
    from vllm import LLM, SamplingParams

    assert FLASH_RWKV_SOURCE_REVISION == FLASH_RWKV_TEST_REVISION
    flash_provenance = validate_flash_rwkv_installation()
    assert flash_provenance.revision == FLASH_RWKV_TEST_REVISION

    forbidden_calls: list[str] = []

    def forbid(name: str):
        def forbidden(*_args: Any, **_kwargs: Any):
            forbidden_calls.append(name)
            raise AssertionError(f"default vLLM RWKV7 path called forbidden {name}")

        return forbidden

    # The canonical package keeps the independent oracle in the reference
    # module. Patch both implementation and public aliases so an accidental
    # fallback cannot hide behind either import path.
    monkeypatch.setattr(
        flash_rwkv_reference,
        "rwkv7_reference",
        forbid("reference"),
    )
    monkeypatch.setattr(
        flash_rwkv_ops,
        "rwkv7_reference",
        forbid("reference"),
    )
    monkeypatch.setattr(
        flash_rwkv,
        "infer_chunk_bf16_forward",
        forbid("chunk"),
        raising=False,
    )
    monkeypatch.setattr(
        flash_rwkv,
        "infer_chunk_bf16_forward_varlen",
        forbid("chunk-varlen"),
        raising=False,
    )
    monkeypatch.setattr(
        flash_rwkv_ops,
        "infer_chunk_bf16_forward",
        forbid("chunk"),
    )
    monkeypatch.setattr(
        flash_rwkv_ops,
        "infer_chunk_bf16_forward_varlen",
        forbid("chunk-varlen"),
    )

    real_recurrent = vllm_rwkv7.run_fla_rwkv7_recurrent_from_decay_logits
    calls: list[dict[str, Any]] = []

    def traced_recurrent(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        state_pool = kwargs["state_pool"]
        cu_seqlens = kwargs["cu_seqlens"]
        state_indices = kwargs["state_indices"]
        selected = state_pool.index_select(0, state_indices.to(dtype=torch.long))
        before = float(selected.float().square().sum().item())
        output = real_recurrent(*args, **kwargs)
        selected_after = state_pool.index_select(
            0,
            state_indices.to(dtype=torch.long),
        )
        after = float(selected_after.float().square().sum().item())
        calls.append(
            {
                "state_pointer": state_pool.data_ptr(),
                "offsets": cu_seqlens.tolist(),
                "slots": state_indices.tolist(),
                "before": before,
                "after": after,
                "mode": kwargs["mode"],
            }
        )
        return output

    monkeypatch.setattr(
        vllm_rwkv7,
        "run_fla_rwkv7_recurrent_from_decay_logits",
        traced_recurrent,
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_USE_RAPID_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")

    artifact = tmp_path / "tiny-rwkv7-hf"
    _write_tiny_standard_hf_artifact(artifact)

    llm = None
    request: Any = [{"prompt_token_ids": PROMPT_TOKEN_IDS}]
    try:
        llm = LLM(
            model=str(artifact),
            skip_tokenizer_init=True,
            enforce_eager=True,
            dtype="float16",
            max_model_len=16,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            gpu_memory_utilization=0.15,
            disable_log_stats=True,
        )
        outputs = llm.generate(
            request,
            SamplingParams(
                temperature=1.0,
                top_k=1,
                max_tokens=GENERATED_TOKENS,
                ignore_eos=True,
            ),
            use_tqdm=False,
        )
    finally:
        if llm is not None:
            with contextlib.suppress(Exception):
                llm.shutdown()  # type: ignore[attr-defined]

    assert len(outputs) == 1
    assert outputs[0].prompt_token_ids == NORMALIZED_PROMPT_TOKEN_IDS
    assert len(outputs[0].outputs) == 1
    generated_token_ids = list(outputs[0].outputs[0].token_ids)
    assert len(generated_token_ids) == GENERATED_TOKENS
    assert get_last_rwkv7_provider() == "flash_rwkv"
    assert forbidden_calls == []

    calls_by_state = defaultdict(list)
    for call in calls:
        assert call["slots"] == list(dict.fromkeys(call["slots"]))
        calls_by_state[call["state_pointer"]].append(call)
    assert len(calls_by_state) == 2
    for layer_calls in calls_by_state.values():
        assert [call["offsets"] for call in layer_calls] == [
            [0, len(NORMALIZED_PROMPT_TOKEN_IDS)],
            *([[0, 1]] * RECURRENT_DECODE_STEPS),
        ]
        assert len({tuple(call["slots"]) for call in layer_calls}) == 1
        assert layer_calls[0]["before"] == 0.0
        assert layer_calls[0]["after"] > 0.0
        assert layer_calls[1]["before"] == layer_calls[0]["after"]
        assert layer_calls[2]["before"] == layer_calls[1]["after"]
        assert {call["mode"] for call in layer_calls} == {"fp32io16"}

    evidence = {
        "device": torch.cuda.get_device_name(0),
        "dependencies": dependencies,
        "provider": get_last_rwkv7_provider(),
        "transformers_config_module": provenance["config_module"],
        "transformers_model_module": provenance["causal_lm_module"],
        "requested_prompt_token_ids": PROMPT_TOKEN_IDS,
        "normalized_prompt_token_ids": NORMALIZED_PROMPT_TOKEN_IDS,
        "generated_token_ids": generated_token_ids,
        "wkv_calls": calls,
    }
    print("RWKV7_PRO6000_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
