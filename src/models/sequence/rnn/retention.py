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
    def __init__(self, d_model, n_heads=4, layer_idx=None, device=None, dtype=None, pt_residual=False, ih_residual=False):
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

        self.pt_residual = pt_residual # prev-token residual
        self.ih_residual = ih_residual # induciton-head residual
        assert (pt_residual and ih_residual) is False, "only one residual connection is allowed"

        # reset the residual connection if it is not the second-to-last layer
        if self.layer_idx != 2:
            self.pt_residual = False
            self.ih_residual = False

        if self.pt_residual:
            self.token_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.t0 = torch.nn.Parameter(torch.zeros(self.embed_dim))
            self.t1 = torch.nn.Parameter(torch.ones(self.embed_dim))
        elif self.ih_residual:
            self.t0 = torch.nn.Parameter(torch.zeros(self.embed_dim))
            self.t1 = torch.nn.Parameter(torch.ones(self.embed_dim))

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
    
        

    def forward(self, x, return_attention=False, input_ids=None):
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

        output = self.parallel_forward(qr, kr, v, inner_mask)

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        output = self.gate_fn(g) * output
        output = self.out_proj(output)

        if self.pt_residual:
            h0 = self.token_shift(output)
            h1 = output
            output = self.t0 * h0 + self.t1 * h1
        elif self.ih_residual:
            h0 = induction_head(input_ids, output)
            h1 = output
            output = self.t0 * h0 + self.t1 * h1

        # attention output is not supported from now
        if return_attention:
            return output, None
        else:
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

def induction_head(x, hidden_state):
    """
    Args:
        x: bsz x input_len
        hidden_state: bsz x input_len x d_model
    Output:
        bsz x input_len x d_model
    """
    bsz, seq_len = x.shape

    # import pbd; pdb.set_trace()
    same_mask = x[:, :, None] == x[:, None, :]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=-1)
    ih_mask = torch.logical_and(same_mask, causal_mask).float()
    ih_mask_norm = ih_mask / ih_mask.sum(dim=2, keepdim=True)
    ih_mask_norm = torch.nan_to_num(ih_mask_norm, 0)
    output = torch.einsum("bmn,bnz->bmz", ih_mask_norm, hidden_state)
    return output


if __name__ == "__main__":
    x = torch.LongTensor([[1, 2, 1, 3, 1], [1, 3, 2, 3, 4]])
    y = torch.randn((2, 5, 32))
    output = induction_head(x, y)
    print(output)
