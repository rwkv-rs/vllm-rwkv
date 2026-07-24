import sys
from types import SimpleNamespace as NS

import torch

sys.path.insert(0, "/data/work/hf-adapter")
from rwkv7_hf.native import attn_step_batched, ffn_step_batched
from rwkv7_vllm_ascend.model import RWKV7Block


def config():
    return NS(
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        head_dim=4,
        attention_hidden_size=8,
        value_dim=[8, 8],
        decay_low_rank_dim=3,
        a_low_rank_dim=3,
        gate_low_rank_dim=4,
        v_low_rank_dim=2,
    )


def test_one_token_block_matches_canonical_hf_oracle():
    torch.manual_seed(123)
    c = config()
    mc = NS(dtype=torch.float32)
    cc = NS(enable_prefix_caching=False)
    block = RWKV7Block(c, mc, cc, 1, "model.layers.1")
    for p in block.parameters():
        p.data.normal_(0, 0.05)
    x = torch.randn(8)
    att_prev = torch.randn(8)
    ffn_prev = torch.randn(8)
    v_first = torch.randn(8)
    state = torch.randn(2, 4, 4, dtype=torch.float32)

    got = block._token_recurrence(
        x.clone(), state.clone(), att_prev.clone(), ffn_prev.clone(), v_first.clone()
    )

    residual = x
    h = block.attn_norm(residual)
    att, new_att, new_state, new_v = attn_step_batched(
        block.attn, 1, h[None], att_prev[None], v_first[None], state[None]
    )
    after_att = residual + att[0]
    h2 = block.ffn_norm(after_att)
    ff, new_ffn = ffn_step_batched(block.ffn, h2[None], ffn_prev[None])
    expected = after_att + ff[0]

    torch.testing.assert_close(got[0], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got[1], new_state[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got[2], new_att[0])
    torch.testing.assert_close(got[3], new_ffn[0])
    torch.testing.assert_close(got[4], new_v[0])
