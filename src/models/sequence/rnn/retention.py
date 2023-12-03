# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError

class MultiScaleRetention(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads=4,
                 layer_idx=None,
                 device=None,
                 dtype=None):

        super().__init__()
        self.factor = 2
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.head_dim = self.embed_dim * self.factor // self.num_heads
        self.key_dim = self.embed_dim // self.num_heads
        self.scaling = self.key_dim ** -0.5
        self.layer_idx = layer_idx

        self.gate_fn = get_activation_fn(activation="swish")

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True).to(device=device, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True).to(device=device, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim * self.factor, bias=True).to(device=device, dtype=dtype)
        self.g_proj = nn.Linear(self.embed_dim, self.embed_dim * self.factor, bias=True).to(device=device, dtype=dtype)
        self.out_proj = nn.Linear(self.embed_dim * self.factor, self.embed_dim, bias=True).to(device=device, dtype=dtype)

        # 1e-5 is used in the official implementation
        self.group_norm = LayerNorm(self.head_dim, eps=1e-5, elementwise_affine=False).to(device=device, dtype=dtype)

        self.xpos = RetNetRelPos(self.embed_dim, self.num_heads).to(device=device, dtype=dtype)

    def post_init(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (1 - 1 / scale).view(self.num_heads, 1, 1) + kv / scale.view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask
    ):
        mask, cross_decay, inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim

        # reduce kv in one chunk
        kv = kr_t @ (v * mask[:, -1, :, None])
        kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, self.head_dim).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        cross_output = (qr * inner_decay) @ kv_recurrent
        output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output

    def forward(
        self,
        x,
        incremental_state=None,
        chunkwise_recurrent=False,
        return_attention=False,
    ):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = self.xpos(tgt_len)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)
        if return_attention:
            return output, None
        return output

class RetNetRelPos(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, n_embd // n_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(n_head, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = None

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            retention_rel_pos = ((sin, cos), (mask, cross_decay, inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
