# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import PretrainedConfig

STANDARD_RWKV7_ARCHITECTURE = "Rwkv7ForCausalLM"
RWKV7_LOW_RANK_PROJECTIONS = frozenset(("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"))


class RWKV7Config(PretrainedConfig):
    model_type = "rwkv7"
    architectures = [STANDARD_RWKV7_ARCHITECTURE]
    attribute_map = {"max_position_embeddings": "context_length"}

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 2048,
        head_size: int = 64,
        num_hidden_layers: int = 24,
        context_length: int | None = None,
        max_position_embeddings: int | None = None,
        intermediate_size: int | None = None,
        embedding_layer_norm_fused: bool = False,
        **kwargs,
    ):
        if context_length is None:
            context_length = max_position_embeddings or 4096
        elif (
            max_position_embeddings is not None
            and context_length != max_position_embeddings
        ):
            raise ValueError(
                "RWKV7 config context_length and max_position_embeddings must match."
            )
        kwargs.setdefault("architectures", self.architectures)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = hidden_size // head_size
        self.context_length = context_length
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.embedding_layer_norm_fused = embedding_layer_norm_fused


def _rwkv7_config_value(config: object, name: str) -> object:
    if isinstance(config, dict):
        if name == "max_position_embeddings":
            return config.get(name, config.get("context_length"))
        return config.get(name)
    if name == "max_position_embeddings":
        return getattr(
            config,
            name,
            getattr(config, "context_length", None),
        )
    return getattr(config, name, None)


def validate_rwkv7_hf_artifact_config(config: object) -> None:
    required_values = (
        "vocab_size",
        "hidden_size",
        "head_size",
        "num_hidden_layers",
        "max_position_embeddings",
    )
    architectures = _rwkv7_config_value(config, "architectures")
    allowed_architectures = [[STANDARD_RWKV7_ARCHITECTURE]]
    values = {name: _rwkv7_config_value(config, name) for name in required_values}
    valid_values = all(
        not isinstance(value, bool) and isinstance(value, int) and value > 0
        for value in values.values()
    )
    divisible_head_size = valid_values and (
        int(values["hidden_size"]) % int(values["head_size"]) == 0
    )
    if (
        _rwkv7_config_value(config, "model_type") != "rwkv7"
        or architectures not in allowed_architectures
        or not valid_values
        or not divisible_head_size
    ):
        raise ValueError(
            "Invalid RWKV7 Hugging Face artifact config: expected model_type="
            f"'rwkv7', architectures={allowed_architectures}, positive integer "
            "vocab_size/hidden_size/head_size/num_hidden_layers/"
            "max_position_embeddings, and hidden_size divisible by head_size."
        )


def rwkv7_hf_to_internal_weight_name(name: str) -> str | None:
    if name == "model.embeddings.weight":
        return "emb.weight"
    if name == "head.weight":
        return name
    if name.startswith("model.blocks."):
        internal_name = name.removeprefix("model.")
        parts = internal_name.split(".")
        if len(parts) >= 5 and parts[-2] in RWKV7_LOW_RANK_PROJECTIONS:
            return internal_name.removesuffix(".weight")
        if parts[-1] in RWKV7_LOW_RANK_PROJECTIONS:
            return None
        return internal_name
    if name.startswith("model.ln_out."):
        return name.removeprefix("model.")
    return None


def rwkv7_internal_to_hf_weight_name(name: str) -> str | None:
    if name == "emb.weight":
        return "model.embeddings.weight"
    if name == "head.weight":
        return name
    if rwkv7_is_legacy_low_rank_weight_name(name):
        return f"model.{name}.weight"
    if name.startswith("blocks.") or name.startswith("ln_out."):
        return f"model.{name}"
    return None


def rwkv7_is_legacy_low_rank_weight_name(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) == 4
        and parts[0] == "blocks"
        and parts[1].isdigit()
        and parts[2] == "att"
        and parts[3] in RWKV7_LOW_RANK_PROJECTIONS
    )


def _projection_ranks(hidden_size: int) -> tuple[int, int, int]:
    return (
        max(32, round(2.5 * hidden_size**0.5 / 32) * 32),
        max(32, round(1.7 * hidden_size**0.5 / 32) * 32),
        max(32, round(5.0 * hidden_size**0.5 / 32) * 32),
    )


def rwkv7_legacy_checkpoint_weight_shapes(
    config: object,
) -> dict[str, tuple[int, ...]]:
    validate_rwkv7_hf_artifact_config(config)
    if bool(_rwkv7_config_value(config, "embedding_layer_norm_fused")):
        raise ValueError(
            "RWKV7 vLLM loading requires an unfused standard artifact; "
            "convert without embedding_layer_norm_fused."
        )
    hidden_size = int(_rwkv7_config_value(config, "hidden_size"))
    vocab_size = int(_rwkv7_config_value(config, "vocab_size"))
    num_layers = int(_rwkv7_config_value(config, "num_hidden_layers"))
    head_size = int(_rwkv7_config_value(config, "head_size"))
    num_heads = hidden_size // head_size
    intermediate_size_value = _rwkv7_config_value(config, "intermediate_size")
    intermediate_size = int(intermediate_size_value or 4 * hidden_size)
    decay_rank, value_rank, gate_rank = _projection_ranks(hidden_size)
    shapes = {
        "emb.weight": (vocab_size, hidden_size),
        "head.weight": (vocab_size, hidden_size),
        "ln_out.weight": (hidden_size,),
        "ln_out.bias": (hidden_size,),
    }
    vectors = ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "k_k", "k_a")
    for layer_id in range(num_layers):
        block = f"blocks.{layer_id}"
        if layer_id == 0:
            shapes[f"{block}.ln0.weight"] = (hidden_size,)
            shapes[f"{block}.ln0.bias"] = (hidden_size,)
        for norm in ("ln1", "ln2"):
            shapes[f"{block}.{norm}.weight"] = (hidden_size,)
            shapes[f"{block}.{norm}.bias"] = (hidden_size,)
        attention = f"{block}.att"
        for name in vectors:
            shapes[f"{attention}.{name}"] = (1, 1, hidden_size)
        if layer_id > 0:
            shapes[f"{attention}.v0"] = (1, 1, hidden_size)
            shapes[f"{attention}.v1"] = (hidden_size, value_rank)
            shapes[f"{attention}.v2"] = (value_rank, hidden_size)
        shapes.update(
            {
                f"{attention}.w1": (hidden_size, decay_rank),
                f"{attention}.w2": (decay_rank, hidden_size),
                f"{attention}.a1": (hidden_size, decay_rank),
                f"{attention}.a2": (decay_rank, hidden_size),
                f"{attention}.g1": (hidden_size, gate_rank),
                f"{attention}.g2": (gate_rank, hidden_size),
                f"{attention}.r_k": (num_heads, head_size),
                f"{attention}.ln_x.weight": (hidden_size,),
                f"{attention}.ln_x.bias": (hidden_size,),
            }
        )
        for projection in ("receptance", "key", "value", "output"):
            shapes[f"{attention}.{projection}.weight"] = (
                hidden_size,
                hidden_size,
            )
        feed_forward = f"{block}.ffn"
        shapes[f"{feed_forward}.x_k"] = (1, 1, hidden_size)
        shapes[f"{feed_forward}.key.weight"] = (
            intermediate_size,
            hidden_size,
        )
        shapes[f"{feed_forward}.value.weight"] = (
            hidden_size,
            intermediate_size,
        )
    return shapes


def rwkv7_checkpoint_weight_shapes(config: object) -> dict[str, tuple[int, ...]]:
    standard_shapes = {}
    for internal_name, legacy_shape in rwkv7_legacy_checkpoint_weight_shapes(
        config
    ).items():
        hf_name = rwkv7_internal_to_hf_weight_name(internal_name)
        if hf_name is None:
            raise ValueError(
                f"RWKV7 checkpoint key has no HF mapping: {internal_name!r}"
            )
        standard_shapes[hf_name] = (
            tuple(reversed(legacy_shape))
            if rwkv7_is_legacy_low_rank_weight_name(internal_name)
            else legacy_shape
        )
    return standard_shapes
