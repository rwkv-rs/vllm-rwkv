"""Pure-PyTorch fallbacks for the Albatross faster3a `rwkv7_*` custom CUDA ops.

Goal: let `vllm-rwkv`'s `vllm/model_executor/models/rwkv7.py` (and the standalone
`rwkv7_fast_v3a.py`) run **unchanged** on CPU (or any non-CUDA device) by
providing Python implementations of every `torch.ops.rwkv7_v3a_ops` /
`rwkv7_fast_ops_fp16` / `rwkv7_wkv_fp16_v2` / `rwkv7_wkv_fp32_v2` op.

The CUDA kernels are *only* registered for the CUDA dispatch key. Without CUDA the
libraries are never loaded, so the op namespaces don't even exist — we therefore
*define* the ops (if absent) and register generic (Composite) impls that any
backend without a dedicated kernel falls through to.

Math provenance:
  * layout conventions + call sites : BlinkDL/Albatross faster3a_2605/rwkv7_fast_v3a.py
  * WKV recurrence (decay + dithering): faster3a_2605/cuda/rwkv7_wkv_fp16_v2.cu
  * per-token TMix/CMix equations   : rwkv7_hf/native.py (verified cos=1.0 vs official rwkv)

The Albatross fp16 decay `exp2(A/(1+exp2(B*w)))` and native.py's
`exp(-EXP_HALF*sigmoid(w))` are the SAME function rewritten (verified equal at
w=0 -> 0.7385, w=1 -> 0.6418); we use the exact Albatross form + rotator
dithering so the output tracks the V100 CUDA ground truth as tightly as
fp16-accumulation order allows.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- WKV decay / dithering constants (rwkv7_wkv_fp16_v2.cu) -----------------
_A = -0.8750387749145276          # NEXP_HALF_LOG2_E
_B = -1.4426950408889634          # NLOG2_E
_TWO_NEG_41 = 4.547473508864641e-13
_ROT1 = 2654435769                # Knuth multiplicative dither
HEAD = 64


# ============================================================================
# helpers
# ============================================================================
def _ln_eps(eps=1e-5):
    return float(eps) if eps is not None else 1e-5


def _decay(w_raw: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Albatross w_delta decay multiplier on the state (= w_delta + 1).

    w_raw, phase broadcast to the same shape. Returns float32 decay.
    """
    base = torch.exp2(_A / (1.0 + torch.exp2(_B * w_raw.float())))
    rot = _rotator(phase)
    return base + rot


def _rotator(phase: torch.Tensor) -> torch.Tensor:
    """float((ROT1 * phase) wrapped to int32) * TWO_NEG_41, matching the C++ int
    overflow -> float conversion in the kernel."""
    p = (_ROT1 * phase) & 0xFFFFFFFF            # uint32 range
    p = p.to(torch.int32).to(torch.float32)      # 2's-complement wrap -> signed float
    return _TWO_NEG_41 * p


def _shifted(x: torch.Tensor, shift_state: torch.Tensor) -> torch.Tensor:
    """sx[t] = (t==0) ? shift_state : x[t-1], shape [B,T,C]."""
    sx = torch.empty_like(x)
    sx[:, 0, :] = shift_state
    if x.shape[1] > 1:
        sx[:, 1:, :] = x[:, :-1, :]
    return sx


# ============================================================================
# rwkv7_v3a_ops
# ============================================================================
def _op_layer_norm_f16(x, weight, bias, eps=1e-5):
    return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), _ln_eps(eps)).to(x.dtype)


def _op_emb_ln0_bf16_to_f16(emb, weight, bias, eps=1e-5):
    # emb/weight/bias are bf16; apply ln0 (the embedding pre-norm), return fp16
    out = F.layer_norm(emb.float(), (emb.shape[-1],), weight.float(), bias.float(), _ln_eps(eps))
    return out.to(torch.float16)


def _op_linear_f16(x, weight, *a):
    # weight is [K,N] (transpose layout) -> x @ weight  (extra cfg/tile int args ignored)
    return (x.float() @ weight.float()).to(x.dtype)


def _op_linear_orig(x, weight_orig, *a):
    # weight_orig is [N,K] (orig layout) -> x @ weight_orig.T == F.linear (extra args ignored)
    return F.linear(x, weight_orig)


def _op_linear_t(x, weight_t):
    # weight_t is [N,K] -> F.linear(x, weight_t)
    return F.linear(x, weight_t)


def _op_linear_t_act(x, weight_t, act):
    fn = torch.tanh if act == 1 else torch.sigmoid
    return F.linear(fn(x), weight_t)


def _op_linear_t_vres(x, weight_t, v, v_first, v0):
    v12 = F.linear(x, weight_t)
    return v + (v_first - v) * torch.sigmoid(v12 + v0)


def _op_linear_wag_rank_in(xw, xa, xg, w1_t, a1_t, g1_t):
    return [F.linear(xw, w1_t), F.linear(xa, a1_t), F.linear(xg, g1_t)]


def _op_linear_wagv_rank_in(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t):
    return [F.linear(xw, w1_t), F.linear(xa, a1_t), F.linear(xg, g1_t), F.linear(xv, v1_t)]


def _op_linear_wag_rank_out(w1, a1, g1, w2_t, a2_t, g2_t):
    return [F.linear(torch.tanh(w1), w2_t), F.linear(a1, a2_t), F.linear(torch.sigmoid(g1), g2_t)]


def _op_linear_wagv_rank_out(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0):
    w = F.linear(torch.tanh(w1), w2_t)
    a = F.linear(a1, a2_t)
    g = F.linear(torch.sigmoid(g1), g2_t)
    v12 = F.linear(v1, v2_t)
    v_new = v + (v_first - v) * torch.sigmoid(v12 + v0)
    return [w, a, g, v_new]


def _op_add_f16(x, y):
    return x + y


def _op_add_layer_norm_f16(x, residual, weight, bias, eps=1e-5):
    s = (x + residual)
    ln = F.layer_norm(s.float(), (s.shape[-1],), weight.float(), bias.float(), _ln_eps(eps)).to(x.dtype)
    return [s.to(x.dtype), ln]


def _op_add_last_layer_norm_f16(x, residual, weight, bias, eps=1e-5):
    # x: [B,T,C] -> take last token of (x+residual)
    s = (x + residual)[:, -1, :]
    return F.layer_norm(s.float(), (s.shape[-1],), weight.float(), bias.float(), _ln_eps(eps)).to(x.dtype)


def _op_advance_i32(x, amount):
    x += int(amount)
    return None


# ============================================================================
# rwkv7_fast_ops_fp16
# ============================================================================
def _op_tmix_mix6(B, T, C, x, shift_state, x_r, x_w, x_k, x_v, x_a, x_g):
    sx = _shifted(x, shift_state)
    xx = sx - x
    out = []
    for vec in (x_r, x_w, x_k, x_v, x_a, x_g):
        out.append(x + xx * vec.reshape(1, 1, C))
    shift_state.copy_(x[:, -1, :])          # in-place: store last token as new prev
    return out


def _op_tmix_kk_a_gate(B, T, C, H, k, k_k, a0, a12, k_a):
    a = torch.sigmoid(a12 + a0.reshape(1, 1, C))
    kk = F.normalize((k * k_k.reshape(1, 1, C)).view(B, T, H, HEAD), dim=-1, p=2).view(B, T, C)
    k_new = k * (1 + (a - 1) * k_a.reshape(1, 1, C))
    neg_kk = -kk
    kka = kk * a
    return [k_new, neg_kk, kka]


def _op_tmix_lnx_rkvres_xg(B, T, C, H, y, r, k, v, r_k, weight, bias, g):
    # group-norm (RWKV-7 ln_x == g_norm) over heads, then sk*v residual, then *g
    y2 = y.reshape(B * T, C)
    out = F.group_norm(y2.float(), H, weight.float(), bias.float(), eps=HEAD * 1e-5).to(y.dtype)
    out = out.reshape(B, T, C)
    rh = r.view(B, T, H, HEAD)
    kh = k.view(B, T, H, HEAD)
    sk = (rh * kh * r_k.reshape(1, 1, H, HEAD)).sum(dim=-1, keepdim=True)        # [B,T,H,1]
    out = out + (sk * v.view(B, T, H, HEAD)).reshape(B, T, C)
    return out * g


def _op_tmix_vres_gate(B, T, C, v, v_first, v0, v12):
    return v + (v_first - v) * torch.sigmoid(v12 + v0.reshape(1, 1, C))


def _op_cmix_mix(B, T, C, x, shift_state, x_k):
    sx = _shifted(x, shift_state)
    xx = sx - x
    mixed = x + xx * x_k.reshape(1, 1, C)
    shift_state.copy_(x[:, -1, :])
    return mixed


def _op_relu_square(x):
    return torch.relu(x).pow(2)


def _op_act_tanh(x):
    return torch.tanh(x)


def _op_act_sigmoid(x):
    return torch.sigmoid(x)


def _op_add_vec(C, x, vec):
    return x + vec.reshape(*([1] * (x.dim() - 1)), C)


# cmix sparse fast-paths -> dense equivalent (used only when CMIX_SPARSE != "off")
def _op_cmix_sparse_one(C, F_, x, shift_state, x_k, key_fc, value_fc):
    mixed = _op_cmix_mix(1, 1, C, x, shift_state, x_k)
    hid = F.linear(mixed.view(1, 1, C), key_fc).view(-1)          # [F]
    return F.linear(torch.relu(hid).pow(2).view(1, -1), value_fc.t()).view(1, 1, C)


def _op_cmix_sparse_rows(B, T, C, F_, x, shift_state, x_k, key_fc, value_fc):
    mixed = _op_cmix_mix(B, T, C, x, shift_state, x_k)
    hid = F.linear(mixed, key_fc)                                  # [B,T,F]
    act = torch.relu(hid).pow(2)
    return F.linear(act, value_fc.t())


def _op_cmix_sparse_down_relu_rows(B, T, C, F_, preact, value_fc):
    act = torch.relu(preact).pow(2)                                # preact: [B,T,F]
    return F.linear(act, value_fc.t())


# ============================================================================
# rwkv7_wkv_fp16_v2 / rwkv7_wkv_fp32_v2  (the recurrence — THE critical op)
# ============================================================================
def _wkv_run(state, r, w_in, k, v, a, b, y, elapsed_t, w0=None, *, fp32_state=False):
    """In-place WKV7 recurrence. state:[B,H,64,64]; r,w,k,v,a,b:[B,T,C]; y:[B,T,C].
    a = neg_kk, b = kka. elapsed_t:[B] int32 (position counter, drives dithering)."""
    B, T, C = r.shape
    H = state.shape[1]
    N = HEAD
    dev = r.device
    s32 = state.float()                                               # fp32 accumulation
    arange_c = torch.arange(C, device=dev, dtype=torch.int64)
    for t in range(T):
        rt = r[:, t].float().view(B, H, N)
        kt = k[:, t].float().view(B, H, N)
        vt = v[:, t].float().view(B, H, N)
        at = a[:, t].float().view(B, H, N)                                    # neg_kk
        bt = b[:, t].float().view(B, H, N)                                    # kka
        w_raw = w_in[:, t].clone()
        if w0 is not None:
            w_raw = w_raw + w0
        phase = (elapsed_t.to(torch.int64).view(B, 1)
                 + arange_c.view(1, C) + t)                          # [B,C]
        decay = _decay(w_raw, phase).view(B, H, N)                   # over key dim (per channel)
        ab = at.view(B, H, N, 1) * bt.view(B, H, 1, N)              # [B,H,N,N]; a is already neg_kk
        vk = vt.view(B, H, N, 1) * kt.view(B, H, 1, N)
        s32 = s32 * decay.view(B, H, 1, N) + s32 @ ab + vk
        yt = (s32 * rt.view(B, H, 1, N)).sum(dim=3)                  # [B,H,N] contract key dim j
        y[:, t] = yt.reshape(B, C).to(y.dtype)
    state.copy_(s32.to(state.dtype))
    return None


def _op_wkv_seq(B, T, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    return _wkv_run(state, r, w, k, v, a, b, y, elapsed_t)


def _op_wkv_seq_w0(B, T, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    return _wkv_run(state, r, w, k, v, a, b, y, elapsed_t, w0=w0)


def _op_wkv_one(B, C, H, state, r, w, k, v, a, b, y, elapsed_t):
    return _wkv_run(state, r.view(B, 1, C), w.view(B, 1, C), k.view(B, 1, C),
                    v.view(B, 1, C), a.view(B, 1, C), b.view(B, 1, C), y.view(B, 1, C), elapsed_t)


def _op_wkv_one_w0(B, C, H, state, r, w, w0, k, v, a, b, y, elapsed_t):
    return _wkv_run(state, r.view(B, 1, C), w.view(B, 1, C), k.view(B, 1, C),
                    v.view(B, 1, C), a.view(B, 1, C), b.view(B, 1, C), y.view(B, 1, C), elapsed_t, w0=w0)


# ============================================================================
# op schemas + registration
# ============================================================================
_V3A_DEFS = [
    "layer_norm_f16(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor",
    "layer_norm_f16_small(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor",
    "layer_norm_f16_small512(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor",
    "emb_ln0_bf16_to_f16(Tensor emb, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor",
    "linear_f16(Tensor x, Tensor weight) -> Tensor",
    "linear_f16_orig(Tensor x, Tensor weight_orig) -> Tensor",
    "linear_orig_rows_f16(Tensor x, Tensor weight_orig, int row_tile, int out_tile) -> Tensor",
    "linear_orig_rows_cfg_f16(Tensor x, Tensor weight_orig, int threads, int row_tile, int out_tile) -> Tensor",
    "linear_orig_rows_exact_f16(Tensor x, Tensor weight_orig, int threads, int out_tile, bool use4) -> Tensor",
    "linear_orig_wmma16_f16(Tensor x, Tensor weight_orig) -> Tensor",
    "linear_f16_orig_lt(Tensor x, Tensor weight_orig) -> Tensor",
    "linear_f16_orig_lt_cfg(Tensor x, Tensor weight_orig, int workspace_mb, int algo_index) -> Tensor",
    "linear_f16_lt(Tensor x, Tensor weight) -> Tensor",
    "linear_f16_m1_splitk(Tensor x, Tensor weight) -> Tensor",
    "linear_f16_m1_splitk_cfg(Tensor x, Tensor weight, int chunk_k) -> Tensor",
    "linear_f16_m1_splitk_tile(Tensor x, Tensor weight, int chunk_k, int tile_cols) -> Tensor",
    "linear_f16_m1_splitk_warpred_tile(Tensor x, Tensor weight, int chunk_k, int tile_cols) -> Tensor",
    "linear_f16_rows_splitk(Tensor x, Tensor weight, int chunk_k) -> Tensor",
    "linear_t_f16(Tensor x, Tensor weight_t) -> Tensor",
    "linear_t_act_f16(Tensor x, Tensor weight_t, int act) -> Tensor",
    "linear_t_vres_f16(Tensor x, Tensor weight_t, Tensor v, Tensor v_first, Tensor v0) -> Tensor",
    "linear_wag_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor w1_t, Tensor a1_t, Tensor g1_t) -> Tensor[]",
    "linear_wagv_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor xv, Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]",
    "linear_wag_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor w2_t, Tensor a2_t, Tensor g2_t) -> Tensor[]",
    "linear_wagv_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor v1, Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, Tensor v, Tensor v_first, Tensor v0) -> Tensor[]",
    "add_f16(Tensor x, Tensor y) -> Tensor",
    "add_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor[]",
    "add_last_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor",
    "add_layer_norm_cmix_mix_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps=1e-5) -> Tensor[]",
    "add_layer_norm_cmix_mix_f16_cfg(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps, int threads) -> Tensor[]",
    "add_layer_norm_cmix_mix_f16_scalar_stats(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps=1e-5) -> Tensor[]",
    "add_layer_norm_tmix_mix6_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps=1e-5) -> Tensor[]",
    "add_layer_norm_tmix_mix6_f16_cfg(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps, int threads) -> Tensor[]",
    "add_layer_norm_tmix_mix6_f16_scalar_stats(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps=1e-5) -> Tensor[]",
    "advance_i32(Tensor(a!) x, int amount) -> ()",
]
_V3A_IMPLS = {
    "layer_norm_f16": _op_layer_norm_f16,
    "layer_norm_f16_small": _op_layer_norm_f16,
    "layer_norm_f16_small512": _op_layer_norm_f16,
    "emb_ln0_bf16_to_f16": _op_emb_ln0_bf16_to_f16,
    "linear_f16": _op_linear_f16,
    "linear_f16_lt": _op_linear_f16,
    "linear_f16_m1_splitk": _op_linear_f16,
    "linear_f16_m1_splitk_cfg": _op_linear_f16,
    "linear_f16_m1_splitk_tile": _op_linear_f16,
    "linear_f16_m1_splitk_warpred_tile": _op_linear_f16,
    "linear_f16_rows_splitk": _op_linear_f16,
    "linear_f16_orig": _op_linear_orig,
    "linear_orig_rows_f16": _op_linear_orig,
    "linear_orig_rows_cfg_f16": _op_linear_orig,
    "linear_orig_rows_exact_f16": _op_linear_orig,
    "linear_orig_wmma16_f16": _op_linear_orig,
    "linear_f16_orig_lt": _op_linear_orig,
    "linear_f16_orig_lt_cfg": _op_linear_orig,
    "linear_t_f16": _op_linear_t,
    "linear_t_act_f16": _op_linear_t_act,
    "linear_t_vres_f16": _op_linear_t_vres,
    "linear_wag_rank_in_f16": _op_linear_wag_rank_in,
    "linear_wagv_rank_in_f16": _op_linear_wagv_rank_in,
    "linear_wag_rank_out_f16": _op_linear_wag_rank_out,
    "linear_wagv_rank_out_f16": _op_linear_wagv_rank_out,
    "add_f16": _op_add_f16,
    "add_layer_norm_f16": _op_add_layer_norm_f16,
    "add_last_layer_norm_f16": _op_add_last_layer_norm_f16,
    "advance_i32": _op_advance_i32,
}

# add_layer_norm_*_cmix_mix / tmix_mix6 fused ops
def _op_add_layer_norm_cmix_mix(x, residual, shift_state, weight, bias, x_k, eps=1e-5, threads=None):
    s = (x + residual)
    ln = F.layer_norm(s.float(), (s.shape[-1],), weight.float(), bias.float(), _ln_eps(eps)).to(x.dtype)
    sx = _shifted(ln, shift_state)
    mixed = ln + (sx - ln) * x_k.reshape(1, 1, x_k.shape[-1])
    shift_state.copy_(ln[:, -1, :])
    return [s.to(x.dtype), mixed]


def _op_add_layer_norm_tmix_mix6(x, residual, shift_state, weight, bias,
                                 x_r, x_w, x_k, x_v, x_a, x_g, eps=1e-5, threads=None):
    s = (x + residual)
    ln = F.layer_norm(s.float(), (s.shape[-1],), weight.float(), bias.float(), _ln_eps(eps)).to(x.dtype)
    sx = _shifted(ln, shift_state)
    xx = sx - ln
    C = ln.shape[-1]
    out = [s.to(x.dtype)]
    for vec in (x_r, x_w, x_k, x_v, x_a, x_g):
        out.append(ln + xx * vec.reshape(1, 1, C))
    shift_state.copy_(ln[:, -1, :])
    return out


_V3A_IMPLS["add_layer_norm_cmix_mix_f16"] = lambda x, r, ss, w, b, xk, eps=1e-5: _op_add_layer_norm_cmix_mix(x, r, ss, w, b, xk, eps)
_V3A_IMPLS["add_layer_norm_cmix_mix_f16_cfg"] = lambda x, r, ss, w, b, xk, eps, threads: _op_add_layer_norm_cmix_mix(x, r, ss, w, b, xk, eps, threads)
_V3A_IMPLS["add_layer_norm_cmix_mix_f16_scalar_stats"] = lambda x, r, ss, w, b, xk, eps=1e-5: _op_add_layer_norm_cmix_mix(x, r, ss, w, b, xk, eps)
_V3A_IMPLS["add_layer_norm_tmix_mix6_f16"] = lambda x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps=1e-5: _op_add_layer_norm_tmix_mix6(x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps)
_V3A_IMPLS["add_layer_norm_tmix_mix6_f16_cfg"] = lambda x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps, threads: _op_add_layer_norm_tmix_mix6(x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps, threads)
_V3A_IMPLS["add_layer_norm_tmix_mix6_f16_scalar_stats"] = lambda x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps=1e-5: _op_add_layer_norm_tmix_mix6(x, r, ss, w, b, xr, xw, xk, xv, xa, xg, eps)

_FAST_DEFS = [
    "tmix_mix6(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g) -> Tensor[]",
    "tmix_mix6_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads) -> Tensor[]",
    "tmix_mix6_t1_c4096(int B, Tensor x, Tensor(a!) shift_state, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, int threads, int vec, bool half_math=False) -> Tensor[]",
    "tmix_kk_a_gate(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a) -> Tensor[]",
    "tmix_kk_a_gate_update_shift(int B, int T, int C, int H, Tensor k, Tensor k_k, Tensor a0, Tensor a12, Tensor k_a, Tensor x, Tensor(a!) shift_state) -> Tensor[]",
    "tmix_lnx_rkvres_xg(int B, int T, int C, int H, Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor",
    "tmix_vres_gate(int B, int T, int C, Tensor v, Tensor v_first, Tensor v0, Tensor v12) -> Tensor",
    "cmix_sparse_one(int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor",
    "cmix_sparse_rows(int B, int T, int C, int F, Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor key_fc, Tensor value_fc) -> Tensor",
    "cmix_sparse_down_one(int C, int F, Tensor act, Tensor value_fc) -> Tensor",
    "cmix_sparse_down_rows(int B, int T, int C, int F, Tensor act, Tensor value_fc) -> Tensor",
    "cmix_sparse_down_relu_one(int C, int F, Tensor preact, Tensor value_fc) -> Tensor",
    "cmix_sparse_down_relu_rows(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor",
    "cmix_sparse_down_relu_rows_t512(int B, int T, int C, int F, Tensor preact, Tensor value_fc) -> Tensor",
    "cmix_mix(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k) -> Tensor",
    "cmix_mix_cfg(int B, int T, int C, Tensor x, Tensor(a!) shift_state, Tensor x_k, int threads) -> Tensor",
    "relu_square(Tensor x) -> Tensor",
    "act_tanh(Tensor x) -> Tensor",
    "act_sigmoid(Tensor x) -> Tensor",
    "add_vec(int C, Tensor x, Tensor vec) -> Tensor",
]
_FAST_IMPLS = {
    "tmix_mix6": lambda B, T, C, x, ss, xr, xw, xk, xv, xa, xg: _op_tmix_mix6(B, T, C, x, ss, xr, xw, xk, xv, xa, xg),
    "tmix_mix6_cfg": lambda B, T, C, x, ss, xr, xw, xk, xv, xa, xg, threads: _op_tmix_mix6(B, T, C, x, ss, xr, xw, xk, xv, xa, xg),
    "tmix_mix6_t1_c4096": lambda B, x, ss, xr, xw, xk, xv, xa, xg, threads, vec, half_math=False: _op_tmix_mix6(B, 1, 4096, x, ss, xr, xw, xk, xv, xa, xg),
    "tmix_kk_a_gate": _op_tmix_kk_a_gate,
    "tmix_kk_a_gate_update_shift": lambda B, T, C, H, k, kk, a0, a12, ka, x, ss: _op_tmix_kk_a_gate(B, T, C, H, k, kk, a0, a12, ka),
    "tmix_lnx_rkvres_xg": _op_tmix_lnx_rkvres_xg,
    "tmix_vres_gate": _op_tmix_vres_gate,
    "cmix_sparse_one": _op_cmix_sparse_one,
    "cmix_sparse_rows": _op_cmix_sparse_rows,
    "cmix_sparse_down_one": lambda C, F_, act, vfc: F.linear(torch.relu(act).pow(2).view(1, -1), vfc.t()).view(-1),
    "cmix_sparse_down_rows": lambda B, T, C, F_, act, vfc: F.linear(torch.relu(act).pow(2), vfc.t()),
    "cmix_sparse_down_relu_one": lambda C, F_, preact, vfc: F.linear(torch.relu(preact).pow(2).view(1, -1), vfc.t()).view(-1),
    "cmix_sparse_down_relu_rows": _op_cmix_sparse_down_relu_rows,
    "cmix_sparse_down_relu_rows_t512": _op_cmix_sparse_down_relu_rows,
    "cmix_mix": _op_cmix_mix,
    "cmix_mix_cfg": lambda B, T, C, x, ss, xk, threads: _op_cmix_mix(B, T, C, x, ss, xk),
    "relu_square": _op_relu_square,
    "act_tanh": _op_act_tanh,
    "act_sigmoid": _op_act_sigmoid,
    "add_vec": _op_add_vec,
}

_WKV16_DEFS = [
    "wkv_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()",
    "wkv_seq_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()",
    "wkv_one(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()",
    "wkv_one_w0(int B, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y, Tensor elapsed_t) -> ()",
]
_WKV16_IMPLS = {
    "wkv_seq": _op_wkv_seq, "wkv_seq_w0": _op_wkv_seq_w0,
    "wkv_one": _op_wkv_one, "wkv_one_w0": _op_wkv_one_w0,
}
_WKV32_DEFS = [
    "forward(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()",
    "forward_seq(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()",
    "forward_small(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()",
    "forward_block(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()",
]
# fp32 variant: no elapsed_t arg; dithering disabled (fp32 mode). Same recurrence.
def _op_wkv32(B, T, C, H, state, r, w, k, v, a, b, y):
    elapsed = torch.zeros(B, dtype=torch.int32, device=r.device)
    return _wkv_run(state, r, w, k, v, a, b, y, elapsed, fp32_state=True)
_WKV32_IMPLS = {n: _op_wkv32 for n in ("forward", "forward_seq", "forward_small", "forward_block")}


_LIBS = []  # keep Library objects alive — definitions are dropped when GC'd


def _register_namespace(name: str, defs: list[str], impls: dict):
    try:
        lib = torch.library.Library(name, "DEF")
        for d in defs:
            lib.define(d)
    except RuntimeError:
        # namespace already exists (e.g. CUDA .so loaded) — attach to existing
        lib = torch.library.Library(name, "EXISTING")
    for op, fn in impls.items():
        lib.impl(op, fn)
    _LIBS.append(lib)


def install():
    """Define (if absent) and register all rwkv7_* op fallbacks. Import this then
    `rwkv7_pytorch_fallback.install()` before constructing/running the model without CUDA."""
    _register_namespace("rwkv7_v3a_ops", _V3A_DEFS, _V3A_IMPLS)
    _register_namespace("rwkv7_fast_ops_fp16", _FAST_DEFS, _FAST_IMPLS)
    _register_namespace("rwkv7_wkv_fp16_v2", _WKV16_DEFS, _WKV16_IMPLS)
    _register_namespace("rwkv7_wkv_fp32_v2", _WKV32_DEFS, _WKV32_IMPLS)


if __name__ == "__main__":
    install()
    print("registered rwkv7_* op fallbacks on device:", torch.accelerator.current_accelerator())
