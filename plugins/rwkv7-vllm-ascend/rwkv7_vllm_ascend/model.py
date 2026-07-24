"""RWKV-7 vLLM V1 model using the native recurrent MambaSpec cache path.

The implementation intentionally uses ordinary PyTorch/torch_npu operations as
its correctness backend. AscendC fusion can replace the token recurrence
without changing vLLM's scheduler or cache lifecycle.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import HasInnerState, IsAttentionFree
from vllm.sequence import IntermediateTensors

from .ascend_quant import (
    AscendQuantActivation,
    AscendQuantConfigError,
    activate_quant_from_env,
)

LOG = logging.getLogger(__name__)
EXP_HALF = 0.6065306597126334


def _attr(config, *names, default=None, required=False):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    if required:
        raise ValueError(f"RWKV-7 config requires one of {names}")
    return default


def _rank(config, kind: str, default: int) -> int:
    aliases = {
        "w": ("decay_low_rank_dim", "decay_lora_dim", "time_decay_rank", "w_lora_rank"),
        "a": ("a_low_rank_dim", "aaa_lora_dim", "time_aaa_rank", "a_lora_rank"),
        "g": ("gate_low_rank_dim", "gate_lora_dim", "time_gate_rank", "g_lora_rank"),
        "v": ("v_low_rank_dim", "value_lora_dim", "time_value_rank", "v_lora_rank"),
    }[kind]
    return int(_attr(config, *aliases, default=default))


class RawLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x):
        return F.linear(x, self.weight)


class _LoRA(nn.Module):
    def __init__(self, input_size: int, rank: int, output_size: int, bias: bool):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(input_size, rank, bias=False),
            nn.Identity(),
            nn.Linear(rank, output_size, bias=bias),
        )


class RWKV7Attention(nn.Module):
    def __init__(self, config, layer_idx: int, hidden: int, heads: int, head_dim: int):
        super().__init__()
        attention_hidden = heads * head_dim
        self.hidden_size, self.num_heads, self.head_dim = hidden, heads, head_dim
        self.attention_hidden_size = attention_hidden
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            self.register_parameter(name, nn.Parameter(torch.empty(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.empty(attention_hidden))
        self.k_a = nn.Parameter(torch.empty(attention_hidden))
        self.r_k = nn.Parameter(torch.empty(heads, head_dim))
        self.r_proj = RawLinear(attention_hidden, hidden)
        self.k_proj = RawLinear(attention_hidden, hidden)
        self.v_proj = RawLinear(attention_hidden, hidden)
        self.o_proj = RawLinear(hidden, attention_hidden)
        self.w_lora = _LoRA(hidden, _rank(config, "w", 64), attention_hidden, True)
        self.a_lora = _LoRA(hidden, _rank(config, "a", 64), attention_hidden, True)
        self.g_lora = _LoRA(hidden, _rank(config, "g", 128), attention_hidden, False)
        self.v_lora = (
            _LoRA(hidden, _rank(config, "v", 32), attention_hidden, True)
            if layer_idx > 0
            else None
        )
        self.g_norm = nn.GroupNorm(heads, attention_hidden, eps=head_dim * 1e-5)


class RWKV7FFN(nn.Module):
    def __init__(
        self,
        hidden: int,
        intermediate: int,
        *,
        layer_idx: int = 0,
        ascend_quant: AscendQuantActivation | None = None,
    ):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(hidden))
        quant_key = (
            ascend_quant.make_ffn_linear(layer_idx, "key", hidden, intermediate)
            if ascend_quant is not None
            else None
        )
        quant_value = (
            ascend_quant.make_ffn_linear(layer_idx, "value", intermediate, hidden)
            if ascend_quant is not None
            else None
        )
        self.key = (
            quant_key if quant_key is not None else RawLinear(intermediate, hidden)
        )
        self.value = (
            quant_value if quant_value is not None else RawLinear(hidden, intermediate)
        )


class RWKV7Block(MambaBase, nn.Module):
    def __init__(
        self,
        config,
        model_config,
        cache_config,
        layer_idx: int,
        prefix: str,
        ascend_quant: AscendQuantActivation | None = None,
    ):
        super().__init__()
        hidden = int(_attr(config, "hidden_size", "n_embd", required=True))
        head_size = int(
            _attr(config, "head_dim", "head_size", "head_size_a", default=64)
        )
        value_dims = getattr(config, "value_dim", None)
        attention_hidden = int(
            _attr(
                config,
                "attention_hidden_size",
                default=(
                    value_dims[layer_idx]
                    if isinstance(value_dims, (list, tuple))
                    else hidden
                ),
            )
        )
        if attention_hidden % head_size:
            raise ValueError("attention_hidden_size must be divisible by head_dim")
        heads = attention_hidden // head_size
        intermediate = int(
            _attr(config, "intermediate_size", "dim_ffn", default=hidden * 4)
        )
        self.hidden, self.heads, self.head_size, self.attention_hidden = (
            hidden,
            heads,
            head_size,
            attention_hidden,
        )
        self.layer_idx, self.prefix = layer_idx, prefix
        self.model_config, self.cache_config = model_config, cache_config
        self.pre_norm = nn.LayerNorm(hidden) if layer_idx == 0 else nn.Identity()
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.attn = RWKV7Attention(config, layer_idx, hidden, heads, head_size)
        self.ffn = RWKV7FFN(
            hidden, intermediate, layer_idx=layer_idx, ascend_quant=ascend_quant
        )
        self.kv_cache = (torch.tensor([]), torch.tensor([]), torch.tensor([]))
        try:
            from vllm.config import get_current_vllm_config

            static = get_current_vllm_config().compilation_config.static_forward_context
            if prefix in static:
                raise ValueError(f"duplicate RWKV recurrent layer: {prefix}")
            static[prefix] = self
        except (RuntimeError, AssertionError):
            pass

    @property
    def mamba_type(self) -> str:
        return "mamba1"

    def get_state_shape(self):
        return (
            (self.heads, self.head_size, self.head_size),
            (self.hidden,),
            (self.hidden,),
        )

    def get_state_dtype(self):
        return (torch.float32, self.model_config.dtype, self.model_config.dtype)

    def _mix(self, x, prev, weight):
        return x + (prev - x) * weight

    def _token_recurrence(self, x, state, att_prev, ffn_prev, v_first):
        a = self.attn
        residual = self.pre_norm(x) if self.layer_idx == 0 else x
        xx = self.attn_norm(residual)
        xr = self._mix(xx, att_prev, a.x_r.view(-1))
        xw = self._mix(xx, att_prev, a.x_w.view(-1))
        xk = self._mix(xx, att_prev, a.x_k.view(-1))
        xv = self._mix(xx, att_prev, a.x_v.view(-1))
        xa = self._mix(xx, att_prev, a.x_a.view(-1))
        xg = self._mix(xx, att_prev, a.x_g.view(-1))
        r = a.r_proj(xr)
        k = a.k_proj(xk)
        v = a.v_proj(xv)
        w_raw = a.w_lora.lora[2](torch.tanh(a.w_lora.lora[0](xw)))
        aa = torch.sigmoid(a.a_lora.lora[2](a.a_lora.lora[0](xa)))
        g = a.g_lora.lora[2](torch.sigmoid(a.g_lora.lora[0](xg)))
        if self.layer_idx == 0:
            next_v_first = v
        else:
            if v_first is None:
                raise RuntimeError("v_first missing for nonzero RWKV layer")
            vm = torch.sigmoid(a.v_lora.lora[2](a.v_lora.lora[0](xv)))
            v = v + (v_first - v) * vm
            next_v_first = v_first
        w = torch.exp(-EXP_HALF * torch.sigmoid(w_raw)).view(
            self.heads, 1, self.head_size
        )
        kk = (k * a.k_k).view(self.heads, self.head_size)
        kk = F.normalize(kk.float(), dim=-1, eps=1e-8).to(k.dtype)
        k = k * (1 + (aa - 1) * a.k_a)
        kh, vh, rh, ah = (z.view(self.heads, self.head_size) for z in (k, v, r, aa))
        ab = (-kk).unsqueeze(-1) @ (kk * ah).unsqueeze(-2)
        vk = vh.unsqueeze(-1) @ kh.unsqueeze(-2)
        state = state * w.float() + state @ ab.float() + vk.float()
        y = (
            (state.to(r.dtype) @ rh.unsqueeze(-1))
            .squeeze(-1)
            .reshape(self.attention_hidden)
        )
        y = F.group_norm(
            y.view(1, self.attention_hidden, 1),
            self.heads,
            a.g_norm.weight,
            a.g_norm.bias,
            a.g_norm.eps,
        ).view(self.attention_hidden)
        sk = (rh * kh * a.r_k).sum(-1, keepdim=True)
        y = (y.view(self.heads, self.head_size) + sk * vh).reshape(
            self.attention_hidden
        ) * g
        x = residual + a.o_proj(y)
        fx = self.ffn_norm(x)
        mixed = self._mix(fx, ffn_prev, self.ffn.x_k)
        ff = torch.relu(self.ffn.key(mixed)).square()
        x = x + self.ffn.value(ff)
        return x, state, xx, fx, next_v_first

    def _segments(self, metadata, token_count: int):
        # vLLM V1 separates decode/prefill state slots. Decode tokens precede
        # prefill tokens in the flattened token buffer.
        segments = []
        nd_tokens = int(metadata.num_decode_tokens)
        nd = int(metadata.num_decodes)
        if nd_tokens != nd:
            raise NotImplementedError(
                "RWKV-7 speculative decode is not supported; "
                "expected one token per decode request"
            )
        decode_ids = getattr(metadata, "state_indices_tensor_d", None)
        prefill_ids = getattr(metadata, "state_indices_tensor_p", None)
        combined_ids = getattr(metadata, "state_indices_tensor", None)
        if decode_ids is None and combined_ids is not None:
            decode_ids = combined_ids[:nd]
        for i in range(min(nd, token_count)):
            slot_value = decode_ids[i]
            if getattr(slot_value, "ndim", 0):
                slot_value = slot_value.reshape(-1)[0]
            slot = int(slot_value.item())
            # A newly admitted one-token prompt is classified as decode by
            # vLLM Mamba metadata. Its recycled slot must be zeroed explicitly.
            seq_lens = getattr(metadata, "seq_lens", None)
            fresh = seq_lens is not None and int(seq_lens[i].item()) == 1
            if slot >= 0:
                segments.append((i, i + 1, slot, fresh))

        npref = int(metadata.num_prefills)
        trace_path = os.getenv("RWKV7_TRACE_PATH")
        if trace_path and getattr(self, "layer_idx", -1) == 0:
            trace = {
                "num_decodes": nd,
                "num_decode_tokens": nd_tokens,
                "num_prefills": npref,
                "num_prefill_tokens": int(metadata.num_prefill_tokens),
                "decode_state_slots": (
                    decode_ids.detach().cpu().tolist()
                    if decode_ids is not None
                    else None
                ),
                "prefill_state_slots": (
                    prefill_ids.detach().cpu().tolist()
                    if prefill_ids is not None
                    else None
                ),
                "query_start_loc_p": (
                    metadata.query_start_loc_p.detach().cpu().tolist()
                    if metadata.query_start_loc_p is not None
                    else None
                ),
                "has_initial_states_p": (
                    metadata.has_initial_states_p.detach().cpu().tolist()
                    if metadata.has_initial_states_p is not None
                    else None
                ),
            }
            with open(trace_path, "a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(trace, sort_keys=True) + "\n")
        if npref:
            starts = metadata.query_start_loc_p.detach().cpu().tolist()
            if prefill_ids is None and combined_ids is not None:
                prefill_ids = combined_ids[nd : nd + npref]
            initial = getattr(metadata, "has_initial_states_p", None)
            for i in range(npref):
                begin = nd_tokens + int(starts[i])
                end = nd_tokens + int(starts[i + 1])
                slot_value = prefill_ids[i]
                if getattr(slot_value, "ndim", 0):
                    slot_value = slot_value.reshape(-1)[0]
                slot = int(slot_value.item())
                # False means a newly assigned/reused slot; zero stale state.
                fresh = initial is not None and not bool(initial[i].item())
                if slot >= 0 and begin < token_count:
                    segments.append((begin, min(end, token_count), slot, fresh))
        return segments

    def forward(self, hidden_states: torch.Tensor, v_first: torch.Tensor | None):
        context = get_forward_context()
        all_metadata = context.attn_metadata
        output = hidden_states.clone()
        vf_output = (
            hidden_states.new_zeros((hidden_states.shape[0], self.attention_hidden))
            if self.layer_idx == 0
            else v_first
        )
        if all_metadata is None:
            state = torch.zeros(
                self.heads,
                self.head_size,
                self.head_size,
                dtype=torch.float32,
                device=hidden_states.device,
            )
            ap = torch.zeros(
                self.hidden, dtype=hidden_states.dtype, device=hidden_states.device
            )
            fp = torch.zeros_like(ap)
            for pos in range(hidden_states.shape[0]):
                vt = None if v_first is None else v_first[pos]
                output[pos], state, ap, fp, vo = self._token_recurrence(
                    hidden_states[pos], state, ap, fp, vt
                )
                if self.layer_idx == 0:
                    vf_output[pos] = vo
            return output, vf_output
        if not isinstance(all_metadata, dict) or self.prefix not in all_metadata:
            raise RuntimeError(f"missing vLLM recurrent metadata for {self.prefix}")
        if self.cache_config.enable_prefix_caching:
            raise NotImplementedError(
                "RWKV-7 prefix caching is disabled until "
                "block-boundary state parity is validated"
            )
        metadata = all_metadata[self.prefix]
        try:
            cache = self.kv_cache[context.virtual_engine]
        except (AttributeError, TypeError):
            cache = self.kv_cache[0]
        if len(cache) != 3:
            raise RuntimeError("RWKV-7 cache must contain (wkv, att_x, ffn_x)")
        wkv_cache, att_cache, ffn_cache = cache
        for begin, end, slot, fresh in self._segments(metadata, hidden_states.shape[0]):
            if fresh:
                stale_nonzero = None
                if (
                    trace_path := os.getenv("RWKV7_TRACE_PATH")
                ) and self.layer_idx == 0:
                    stale_nonzero = bool(
                        torch.count_nonzero(wkv_cache[slot]).item()
                        or torch.count_nonzero(att_cache[slot]).item()
                        or torch.count_nonzero(ffn_cache[slot]).item()
                    )
                state = torch.zeros_like(wkv_cache[slot])
                ap = torch.zeros_like(att_cache[slot])
                fp = torch.zeros_like(ffn_cache[slot])
                if stale_nonzero is not None:
                    zero_event = {
                        "event": "fresh_state_zero",
                        "slot": slot,
                        "pre_zero_had_nonzero": stale_nonzero,
                        "post_zero_nonzero": False,
                    }
                    with open(trace_path, "a", encoding="utf-8") as trace_file:
                        trace_file.write(json.dumps(zero_event, sort_keys=True) + "\n")
            else:
                state = wkv_cache[slot]
                ap = att_cache[slot]
                fp = ffn_cache[slot]
            for pos in range(begin, end):
                vt = None if v_first is None else v_first[pos]
                output[pos], state, ap, fp, vo = self._token_recurrence(
                    hidden_states[pos], state, ap, fp, vt
                )
                if self.layer_idx == 0:
                    vf_output[pos] = vo
            wkv_cache[slot].copy_(state)
            att_cache[slot].copy_(ap)
            ffn_cache[slot].copy_(fp)
        return output, vf_output


class RWKV7Model(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        ascend_quant: AscendQuantActivation | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        hidden = int(_attr(config, "hidden_size", "n_embd", required=True))
        layers = int(_attr(config, "num_hidden_layers", "n_layer", required=True))
        self.embeddings = VocabParallelEmbedding(
            int(config.vocab_size),
            hidden,
            params_dtype=vllm_config.model_config.dtype,
            prefix=f"{prefix}.embeddings" if prefix else "embeddings",
        )
        self.layers = nn.ModuleList(
            [
                RWKV7Block(
                    config,
                    vllm_config.model_config,
                    vllm_config.cache_config,
                    i,
                    f"{prefix}.layers.{i}.linear_attn"
                    if prefix
                    else f"layers.{i}.linear_attn",
                    ascend_quant=ascend_quant,
                )
                for i in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)

    def embed_input_ids(self, input_ids):
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds=None,
    ):
        if intermediate_tensors is not None:
            raise NotImplementedError(
                "pipeline parallelism is not supported by the RWKV-7 plugin"
            )
        x = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        v_first = None
        for block in self.layers:
            x, v_first = block(x, v_first)
        return self.norm(x)


class RWKV7ForCausalLM(nn.Module, HasInnerState, IsAttentionFree):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if os.getenv("RWKV7_ALLOW_UNVALIDATED_ASCEND") != "1":
            try:
                device_name = torch.npu.get_device_name(0)
            except Exception as exc:
                raise RuntimeError(
                    "RWKV-7 vLLM plugin requires the validated Ascend 910B3 device"
                ) from exc
            normalized_device_name = "".join(
                char for char in device_name.lower() if char.isalnum()
            )
            if normalized_device_name != "ascend910b3":
                raise RuntimeError(
                    f"unvalidated Ascend device {device_name!r}; only Ascend 910B3 "
                    "is enabled by default"
                )
        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise NotImplementedError(
                "RWKV-7 Ascend plugin currently requires TP=1 and PP=1"
            )
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self._ascend_quant = activate_quant_from_env(backend="vllm")
        if self._ascend_quant is not None:
            if vllm_config.model_config.dtype is not torch.float16:
                raise AscendQuantConfigError(
                    "Ascend weight-only quantization requires vLLM --dtype half"
                )
            if prefix:
                raise AscendQuantConfigError(
                    "Ascend quant manifests use canonical unprefixed model paths"
                )
        self.model = RWKV7Model(
            vllm_config,
            prefix=(f"{prefix}.model" if prefix else "model"),
            ascend_quant=self._ascend_quant,
        )
        if self._ascend_quant is not None:
            layers = int(_attr(config, "num_hidden_layers", "n_layer", required=True))
            self._ascend_quant.validate_construction(num_layers=layers)
        hidden = int(_attr(config, "hidden_size", "n_embd", required=True))
        self.lm_head = ParallelLMHead(
            int(config.vocab_size),
            hidden,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )
        self.logits_processor = LogitsProcessor(int(config.vocab_size))
        self.make_empty_intermediate_tensors = lambda **_: None

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        c = vllm_config.model_config.hf_config
        hidden = int(_attr(c, "hidden_size", "n_embd", required=True))
        n = int(_attr(c, "head_dim", "head_size", "head_size_a", default=64))
        value_dims = getattr(c, "value_dim", None)
        attention_hidden = int(
            _attr(
                c,
                "attention_hidden_size",
                default=(
                    value_dims[0] if isinstance(value_dims, (list, tuple)) else hidden
                ),
            )
        )
        if attention_hidden % n:
            raise ValueError("attention_hidden_size must be divisible by head_dim")
        return ((attention_hidden // n, n, n), (hidden,), (hidden,))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        d = vllm_config.model_config.dtype
        return (torch.float32, d, d)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        prefixes = ("_orig_mod.", "backbone.", "rwkv.")
        aliases = {
            "head.weight": "lm_head.weight",
            "emb.weight": "model.embeddings.weight",
            "ln_out.weight": "model.norm.weight",
            "ln_out.bias": "model.norm.bias",
        }
        for original_name, tensor in weights:
            name = original_name
            for p in prefixes:
                if name.startswith(p):
                    name = name[len(p) :]
            name = aliases.get(name, name)
            if name.startswith("blocks."):
                name = "model." + name
            if self._ascend_quant is not None:
                if self._ascend_quant.load_tensor(name, tensor):
                    loaded.add(name)
                    continue
                if self._ascend_quant.owns_selected_namespace(name):
                    raise ValueError(
                        f"checkpoint tensor violates Ascend quant manifest: {name}"
                    )
            if name not in params:
                LOG.debug(
                    "ignoring checkpoint tensor not owned by vLLM model: %s",
                    original_name,
                )
                continue
            param = params[name]
            value = tensor
            if value.shape != param.shape:
                if value.ndim == 2 and value.T.shape == param.shape:
                    value = value.T
                elif value.numel() == param.numel():
                    value = value.reshape(param.shape)
            if value.shape != param.shape:
                raise ValueError(
                    f"shape mismatch for {original_name} -> {name}: "
                    f"checkpoint {tuple(value.shape)}, model {tuple(param.shape)}"
                )
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, value)
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(
                f"RWKV-7 checkpoint is missing {len(missing)} parameters; "
                f"first: {missing[:8]}"
            )
        if self._ascend_quant is not None:
            self.ascend_quant_status = self._ascend_quant.finish_load()
            LOG.warning(
                "RWKV-7 Ascend W%d raw-kernel-candidate seam active; "
                "production_accepted=false",
                self._ascend_quant.manifest.bit,
            )
        return loaded
