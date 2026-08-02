# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from itertools import combinations
from types import SimpleNamespace

import pytest

from benchmarks.rwkv7 import benchmark_direct_model as bench


class _FakeRecurrentModel:
    vocab_size = 31

    def __init__(self) -> None:
        self.embed_calls = 0
        self.forward_initial_states = []

    def zero_state(self, batch_size):
        torch = pytest.importorskip("torch")
        return [
            torch.zeros((batch_size, 1)),
            torch.zeros((batch_size, 1)),
            torch.zeros((batch_size,), dtype=torch.int32),
        ]

    def embed(self, input_ids):
        self.embed_calls += 1
        return input_ids.to(dtype=input_ids.dtype).float().unsqueeze(-1)

    def forward_from_x(
        self,
        embedded,
        state,
        path,
        *,
        query_start_loc,
        wkv_slot_indices,
    ):
        self.forward_initial_states.append(bench._clone_state(state))
        hidden = state[0].unsqueeze(1) + embedded.cumsum(dim=1)
        state[0].copy_(hidden[:, -1])
        state[1].copy_(hidden[:, -1] * 2)
        state[2].add_(embedded.shape[1])
        return hidden

    @staticmethod
    def project_logits_fp32(hidden):
        return hidden.float()


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("all", bench.FP16_LT_FAMILIES),
        ("attention-c2c", ("attention-c2c",)),
        ("ffn-down", ("ffn-down",)),
        ("lowrank", ("lowrank-in", "lowrank-out")),
        ("lowrank-in", ("lowrank-in",)),
        ("lowrank-out", ("lowrank-out",)),
    ],
)
def test_disabled_fp16_lt_families_are_explicit(family, expected) -> None:
    assert bench._disabled_fp16_lt_families(family) == expected


def test_disabled_fp16_lt_families_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unknown FP16 Lt tuning family"):
        bench._disabled_fp16_lt_families("ffn-key")


@pytest.mark.parametrize(
    ("variant", "batch_size", "token_count", "production_enabled", "expected"),
    [
        ("baseline", 16, 1, True, True),
        ("tuned", 1, 1, True, True),
        ("tuned", 2, 1, True, False),
        ("tuned", 1, 2, True, False),
        ("baseline", 1, 1, False, False),
        ("tuned", 1, 1, False, False),
    ],
)
def test_ln1_tmix_fuse_variant_matches_albatross_policy(
    variant,
    batch_size,
    token_count,
    production_enabled,
    expected,
) -> None:
    assert (
        bench._ln1_tmix_fuse_for_variant(
            variant,
            batch_size,
            token_count,
            production_enabled,
        )
        is expected
    )


def test_ln1_tmix_fuse_variant_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="unknown model comparison variant"):
        bench._ln1_tmix_fuse_for_variant("unknown", 1, 1, True)


def test_measurement_result_reports_shape_dtype_and_throughput() -> None:
    torch = pytest.importorskip("torch")
    output = torch.empty((2, 3), dtype=torch.float32)

    result = bench._measurement_result(
        [2.0, 1.0, 3.0],
        output,
        batch_size=2,
        token_count=4,
        provider="paired",
    )

    assert result["p10_ms"] == pytest.approx(1.2)
    assert result["p50_ms"] == 2.0
    assert result["p90_ms"] == pytest.approx(2.8)
    assert result["output_shape"] == [2, 3]
    assert result["output_dtype"] == "torch.float32"
    assert result["tokens_per_s_p50"] == 4000.0


def test_direct_forward_times_embedding_and_decode_uses_real_prefix_state() -> None:
    torch = pytest.importorskip("torch")
    model = _FakeRecurrentModel()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    decode_token = torch.tensor([[7], [8]])

    _, prefix_state = bench._direct_forward(model, prompt)
    expected_prefix = bench._clone_state(prefix_state)
    _, decode_state = bench._direct_forward(
        model,
        decode_token,
        bench._clone_state(prefix_state),
    )

    assert model.embed_calls == 2
    assert torch.equal(model.forward_initial_states[-1][0], expected_prefix[0])
    assert not torch.equal(
        model.forward_initial_states[-1][0], torch.zeros_like(prefix_state[0])
    )
    assert torch.equal(decode_state[2], torch.tensor([4, 4], dtype=torch.int32))


def test_correctness_gate_matches_chunked_logits_and_final_state() -> None:
    torch = pytest.importorskip("torch")
    model = _FakeRecurrentModel()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    result, prefix_state = bench._correctness_gate(
        model,
        input_ids,
        prompt_tokens=3,
    )

    assert result["passed"] is True
    assert result["compared_tokens"] == 2
    assert result["max_abs_logit_error"] == 0.0
    assert result["max_abs_state_error"] == 0.0
    assert prefix_state[0].item() == 6.0
    assert prefix_state[2].item() == 3


def test_artifact_provenance_hashes_config_weights_and_source(tmp_path) -> None:
    source_sha256 = "ab" * 32
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "rwkv7"}),
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"second")

    result = bench._artifact_provenance(
        str(tmp_path),
        SimpleNamespace(rwkv_source_sha256=source_sha256, _commit_hash=None),
    )

    assert result["kind"] == "local_path"
    assert result["source_checkpoint_sha256"] == source_sha256
    assert len(result["config_sha256"]) == 64
    assert len(result["weights_sha256"]) == 64
    assert [weight["name"] for weight in result["weight_files"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert all(len(weight["sha256"]) == 64 for weight in result["weight_files"])


def test_canonical_parser_matches_transformers_workload_defaults() -> None:
    args = bench._build_parser().parse_args(["--model", "model-hf"])

    assert args.batch_size == 1
    assert args.prompt_tokens == 128
    assert args.decode_tokens == 32
    assert args.seed == 0
    assert args.warmup == 3
    assert args.iterations == 20
    assert args.expected_wkv_mode == "fp16"
    assert args.expected_gemm_accumulation == "fp16"
    assert args.diagnostic_cases is False


def test_runtime_provenance_resolves_wkv_gemm_and_stage_kernel_contract() -> None:
    torch = pytest.importorskip("torch")
    model = SimpleNamespace(
        hidden_size=4096,
        execution_profile=SimpleNamespace(
            wkv_mode="fp16",
            wkv_state_dtype=torch.float16,
            allow_fp16_accumulation=True,
            gemm_accumulation_policy="fp16",
        ),
    )

    runtime = bench._runtime_provenance(model)
    prefill = bench._stage_kernel_provenance(model, 1, 128)
    decode = bench._stage_kernel_provenance(model, 1, 1)

    assert runtime["wkv"]["operator"] == "torch.ops.rwkv7_wkv_fp16_v2.wkv"
    assert runtime["gemm"]["accumulation_policy"] == "fp16"
    assert prefill["rows"] == 128
    assert decode["rows"] == 1
    assert decode["m1_rkv_grouped"] is True


def test_model_comparison_modes_are_mutually_exclusive() -> None:
    parser = bench._build_parser()

    modes = (
        "--compare-fp16-lt-tuning",
        "--compare-m1-rkv",
        "--compare-m1-cmix-prezero",
        "--compare-gemm-accumulation",
        "--compare-ln1-tmix-fusion",
    )
    for first, second in combinations(modes, 2):
        with pytest.raises(SystemExit):
            parser.parse_args(["--model", "model.pth", first, second])
