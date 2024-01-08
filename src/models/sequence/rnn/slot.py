import math
import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange

from src.models.sequence.rnn.slot_triton.slot_triton import flash_rnn_onc_no_materialize_v5
from src.models.sequence.rnn.slot_triton.slot_triton_v0 import stickbreaking_att_mix_block_parallel_triton


def resize_stb_score(stb_score, num_head, num_chunk):
    if len(stb_score.shape)  == 3:
        B, T, S = stb_score.size()
        stb_score = stb_score.unsqueeze(1).expand(B, num_head, T, S)
    else:
        B, T, _H, S = stb_score.size()
        stb_score = stb_score.transpose(1, 2) # (B, H, T, S)

    # truncate such beta <= 0.9999
    # stb_score = stb_score.clamp(max=math.log(0.9999 / (1 - 0.9999) ))
    stb_score = rearrange(stb_score, 'b h (n c) s -> b h n c s', n=num_chunk).contiguous()
    return stb_score

def stickbreaking_att_mix_block_parallel_flash(
    x: torch.Tensor,
    stb_score: torch.Tensor,
    q_proj, k_proj, v_proj,
    num_head: int,
    num_chunk: int,
    init_hidden: torch.Tensor = None,
    rotary_emb = None,
    group_norm = None,
) -> torch.Tensor:
    B, T, _ = x.size()
    chunk_size = T // num_chunk

    query, key, value = q_proj(x), k_proj(x), v_proj(x)
    qk_head_size = query.shape[-1] // num_head
    v_head_size = value.shape[-1] // num_head

    query = (rearrange(query, 'b (n c) (h d) -> b h n c d', h=num_head, n=num_chunk)).contiguous() 
    key = rearrange(key, 'b (n c) (h d) -> b h n c d', h=num_head, n=num_chunk).contiguous()
    value = rearrange(value, 'b (n c) (h d) -> b h n c d', h=num_head, n=num_chunk).contiguous()

    if rotary_emb is not None:
        query = rotary_emb.rotate_queries_or_keys(query)
        key = rotary_emb.rotate_queries_or_keys(key)

    stb_score = resize_stb_score(stb_score, num_head, num_chunk)
    S = stb_score.shape[-1]

    if init_hidden is None:
        prev_hk = torch.zeros(B, num_head, qk_head_size, S)
        prev_hv = torch.zeros(B, num_head, S, v_head_size)
    else:
        prev_hk, prev_hv = init_hidden
    prev_hk, prev_hv = prev_hk.to(x.device), prev_hv.to(x.device)

    y, next_hk, next_hv = flash_rnn_onc_no_materialize_v5(query, key, value, stb_score, prev_hk, prev_hv)

    y = y.reshape(B, T, num_head, v_head_size)
    if group_norm is not None:
        y = group_norm(y)
    y = y.reshape(B, T, -1)
    return y, (next_hk, next_hv)


if __name__ == "__main__":
    B, T, H, D = 2, 32, 8, 256
    S = 8
    C = 16
    num_chunk = T // C
    head_dim = D // H
    device = "cuda:0"
    torch.set_default_dtype(torch.float32)


    x = torch.randn(B, T, D).to(device)
    # stb_score = torch.randn(B, T, H, S).to(device)
    stb_score = torch.randn(B, T, S).to(device) / 10
    q_proj = nn.Linear(D, D, bias=False).to(device)
    k_proj = nn.Linear(D, D, bias=False).to(device)
    v_proj = nn.Linear(D, D, bias=False).to(device)

    y1, _ = stickbreaking_att_mix_block_parallel_triton(x, stb_score, q_proj, k_proj, v_proj, num_head=H, num_chunk=num_chunk)
    y2, _ = stickbreaking_att_mix_block_parallel_flash(x, stb_score, q_proj, k_proj, v_proj, num_head=H, num_chunk=num_chunk)
    print(torch.max(torch.abs(y1 - y2)))