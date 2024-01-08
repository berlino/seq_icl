import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange

import torch

import triton
import triton.language as tl

import numpy as np
import math

def cum_ij(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    x = x.to(torch.float64)
    c = torch.cumsum(x, dim=-1)
    c = F.pad(c, (1, 0), value=0)
    c_ij = c.unsqueeze(-1) - c.unsqueeze(-2)  # cij = c_i - c_j
    return c_ij.to(orig_dtype), c.to(orig_dtype)

def stickbreaking_with_residual(stb_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask: created with triu(diag=1)
    """
    orig_dtype = stb_scores.dtype
    stb_scores = stb_scores.to(torch.float32)

    logz = F.logsigmoid(stb_scores) # (B, nc, ns, cs)
    log_beta = F.logsigmoid(-stb_scores) # (B, nc, ns, cs)
    cij_log_beta, cum_log_beta = cum_ij(log_beta) # (B, nc, ns, cs + 1, cs + 1), (B, nc, ns, cs + 1)
    cij_log_beta = cij_log_beta.masked_fill(mask, -torch.inf)
    cij_log_beta = cij_log_beta[:, :, :, :, 1:] # (B, nc, ns, cs + 1, cs)

    logp = logz[:, :, :, None, :] + cij_log_beta # (B, nc, ns, cs + 1, cs)

    p_all = logp.exp() # (B, nc, ns, cs + 1, cs)
    p = p_all[:,:,:,:-1,:].to(orig_dtype) # (B, nc, ns, cs, cs)
    p_residual = p_all[:,:,:, -1, :].to(orig_dtype) # (B, nc, ns, cs)

    p_hat_all = cum_log_beta.exp() # (B, nc, ns, cs + 1)
    p_hat = p_hat_all[:,:,:,:-1].to(orig_dtype) # (B, nc, ns, cs)
    p_hat_residual = p_hat_all[:, :, :, -1].to(orig_dtype) # (B, nc, ns)

    return p, p_residual, p_hat, p_hat_residual


@triton.jit
def fwd_sequential_scan(x, gate, init_hidden, final_hidden, y, B, ns, nc, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Parallelize over B, ns, C
    Args:
        x: (B, ns, nc, C)
        gate: (B, ns, nc, num_C_blocks)
        init: (B, ns, C)
    
    BLOCK_N <-> num_C_blocks, num_C_blocks x 256 = C
    """
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    
    offset_ns = tl.program_id(1)
    if offset_ns >= ns:
        return

    offset_C = tl.program_id(2)
    x_ptr = tl.arange(0, BLOCK_M) + offset_b * ns * nc * C + offset_ns * nc * C + offset_C * BLOCK_M        
    gate_ptr = offset_b * ns * nc * BLOCK_N + offset_ns * nc * BLOCK_N + offset_C
    h_ptr = tl.arange(0, BLOCK_M) + offset_b * ns * C + offset_ns * C + offset_C * BLOCK_M
    y_tm1 = tl.load(init_hidden + h_ptr)  # y_{t-1}

    for _ in range(nc):        
        tl.store(y + x_ptr, y_tm1.to(y.dtype.element_ty) )
        x_t = tl.load(x + x_ptr).to(tl.float32)
        g_t = tl.load(gate + gate_ptr).to(tl.float32)            
        y_tm1 = y_tm1 * g_t + x_t

        x_ptr += C
        gate_ptr += BLOCK_N
    
    tl.store(final_hidden + h_ptr, y_tm1.to(final_hidden.dtype.element_ty))


@triton.jit
def bwd_sequential_scan(grad_y, x, gate, y, B, ns, nc, C, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """
    Args:
        x: (B, ns, nc, C)
        gate: (B, ns, nc, num_C_blocks)
    
    BLOCK_N <-> num_C_blocks, num_C_blocks x 256 = C
    """
    offset_b = tl.program_id(0)
    if offset_b >= B:
        return
    
    offset_ns = tl.program_id(1)
    if offset_ns >= ns:
        return

    offset_C = tl.program_id(2)
    x_ptr = tl.arange(0, BLOCK_M) + offset_b * ns * nc * C + offset_ns * nc * C + offset_C * BLOCK_M  + (nc - 1) * C
    gate_ptr = offset_b * ns * nc * BLOCK_N + offset_ns * nc * BLOCK_N + (nc - 1) * BLOCK_N + offset_C
    dy_ta1 = tl.zeros([BLOCK_M,], dtype=tl.float32)  # dy_{t+1}

    for _ in range(nc - 1, -1, -1):
        # dx_t = dy_{t+1}
        tl.store(x + x_ptr, dy_ta1.to(x.dtype.element_ty))

        # dy_t += dy_{t+1} * g_t
        # dg_t = dy_{t+1} * y_t
        g_t = tl.load(gate + gate_ptr).to(tl.float32)
        y_t = tl.load(y + x_ptr).to(tl.float32)
        dy_t = tl.load(grad_y + x_ptr).to(tl.float32)

        dg_t = tl.sum(dy_ta1 * y_t, axis=0).to(tl.float32)
        tl.store(gate + gate_ptr, dg_t.to(gate.dtype.element_ty))

        dy_ta1 = dy_ta1 * g_t + dy_t

        x_ptr -= C
        gate_ptr -= BLOCK_N



class StbChunkRNN(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, gate, init_hidden, final_hidden):
        B, ns, nc, C = x.shape
        num_warps = 8
        assert C % 256 == 0
        num_C_blocks = int(C/256)

        x = x.contiguous()
        gate_expanded = gate.unsqueeze(-1).expand(B, ns, nc, num_C_blocks).contiguous()
        y =  torch.zeros_like(x).contiguous()
                                    
        fwd_sequential_scan[(B, ns, num_C_blocks)](x, gate_expanded, init_hidden, final_hidden, y, B, ns, nc, C, BLOCK_M=256, BLOCK_N=num_C_blocks, num_warps=num_warps)
        ctx.save_for_backward(x, gate_expanded, y)
        return y
            
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_y):
        x, gate_expanded, y = ctx.saved_tensors 
        B, ns, nc, C = x.shape
        num_C_blocks = int(C/256)
        
        num_warps = 8
        bwd_sequential_scan[(B, ns, num_C_blocks)](grad_y, x, gate_expanded, y, B, ns, nc, C, BLOCK_M=256, BLOCK_N=num_C_blocks, num_warps=num_warps)
        gate = gate_expanded.sum(dim=-1)
        # None for init_hidden and final_hidden
        return x, gate, None, None
        
stb_chunk_rnn_triton = StbChunkRNN.apply

def stickbreaking_att_mix_block_parallel_triton(
    x: torch.Tensor,
    stb_score: torch.Tensor,
    q_proj, k_proj, v_proj,
    num_head: int,
    num_chunk: int,
    init_hidden: torch.Tensor = None,
    rotary_emb = None,
    group_norm = None,
) -> torch.Tensor:
    assert rotary_emb is None, "rotary_emb is not supported yet"
    B, T, C = x.size()
    S = stb_score.shape[-1]
    chunk_size = T // num_chunk
    head_size = C // num_head

    # chunk parallel
    x = x.view(B, num_chunk, chunk_size, C) # (B, nc, cs, C)

    q = q_proj(x).view(B, num_chunk, chunk_size, num_head, -1) # normal query and weak qk, (B, nc, cs, nh, qk_d)
    k = k_proj(x).view(B, num_chunk, chunk_size, num_head, -1) # (B, nc, cs, nh, qk_d)
    qk_head_size = q.size(-1)

    v = v_proj(x).view(B, num_chunk, chunk_size, num_head, head_size) # (B, nc, cs, nh, d)
    gamma = torch.einsum('bcihd,bcjhd->bchij', q, k) # / math.sqrt(qk_head_size) # (B, nc, nh, cs, cs)

    # in chunk stickbreaking
    stb_score = stb_score.reshape(B, num_chunk, chunk_size, S) # (B, nc, cs, ns)
    stb_score = stb_score.transpose(2, 3) # (B, nc, ns, cs)

    mask = torch.triu(torch.ones(chunk_size+1, chunk_size+1, dtype=torch.bool), diagonal=1).to(stb_score.device)
    p, p_last, p_hat, p_hat_last = stickbreaking_with_residual(stb_score, mask) # (B, nc, ns, cs, cs), (B, nc, ns, cs), (B, nc, ns, cs), (B, nc, ns)

    # this is the triton version.
    h_t = torch.einsum('bcsi,bcid->bscd', p_last, x) # (B, ns, nc, C)
    p_hat_last = p_hat_last.transpose(1, 2) # (B, ns, nc)

    if init_hidden is None:
        init_hidden = torch.zeros_like(h_t[:, :, 0, :])
    final_hidden = torch.zeros_like(init_hidden)
    h = stb_chunk_rnn_triton(h_t, p_hat_last, init_hidden, final_hidden) # (B, ns, nc, C)

    # chunk parallel
    # inter chunk value
    k_h = k_proj(h).view(B, S, num_chunk, num_head, qk_head_size) # (B, ns, nc, nh, d)
    v_h = v_proj(h).view(B, S, num_chunk, num_head, head_size) # (B, ns, nc, nh, d)
    gamma0 = torch.einsum('bcihd,bschd->bchsi', q, k_h) # / math.sqrt(k.size(-1)) # (B, nc, nh, ns, cs)

    g_logit = torch.einsum('bcsij,bchij->bchsi', p, gamma) + torch.einsum('bcsi,bchsi->bchsi', p_hat, gamma0) # (B, nc, nh, ns, cs)
    g = torch.softmax(g_logit, dim=3) # (B, nc, nh, ns, cs)
    att = torch.einsum('bchsi,bcsij->bchij', g, p) # (B, nc, nh, cs, cs)

    y_0 = torch.einsum('bchsi,bcsi,bschd->bcihd', g, p_hat, v_h) # (B, nc, cs, nh, d)
    y = torch.einsum('bchij,bcjhd->bcihd', att, v) + y_0 # (B, nc, cs, nh, d)
    y = y.reshape(B, T, num_head, head_size)

    if group_norm is not None:
        y = group_norm(y)
    y = y.reshape(B, T, C)
    return y, final_hidden