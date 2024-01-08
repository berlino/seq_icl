import time
import math
from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
import torch


import triton
import triton.language as tl

import numpy as np
import math

@triton.jit
def _fwd_recurrence_qs(
    S, p, O, prev_memory,
    next_memory, 
    NUM_BLOCK, 
    D_MODEL, NUM_SLOT,
    BLOCK_S: tl.constexpr, BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    
    
    S = S + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]

    O = O + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :] 

    p = p + offset_bh * NUM_BLOCK * NUM_SLOT + tl.arange(0, BLOCK_S) + offset_s * BLOCK_S  

    prev_memory = prev_memory + offset_bh * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :] 

    next_memory = next_memory + offset_bh * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :] 


    acc = tl.zeros([BLOCK_MODEL, BLOCK_S], dtype=tl.float32)
    acc += tl.load(prev_memory) 
    
    tl.store(O, acc.to(O.dtype.element_ty))
    O += NUM_SLOT * D_MODEL

    for i in range(NUM_BLOCK-1):
        p_i = tl.load(p)
        S_i = tl.load(S) 
        acc = acc * p_i[None, :] + S_i

        tl.store(O, acc.to(O.dtype.element_ty))

        p +=  NUM_SLOT
        S +=  NUM_SLOT * D_MODEL
        O +=  NUM_SLOT * D_MODEL        

       
    p_i = tl.load(p)
    S_i = tl.load(S) 
    acc = acc * p_i[None, :] + S_i
    tl.store(next_memory, acc.to(next_memory.dtype.element_ty))
 

# compared to the functtion above, input's shape is changed from B*H*N*D*S ->  B*H*N*S*D, S first. (To avoid some big matrix transpotations..) 
@triton.jit
def _fwd_recurrence_output(
    S, p, O, prev_memory,
    next_memory,
    NUM_BLOCK, 
    D_MODEL, NUM_SLOT,
    BLOCK_S: tl.constexpr, BLOCK_MODEL: tl.constexpr
  ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    
    
    S = S + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[None, :]  +  (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]) * D_MODEL

    O = O + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[None, :]  +  (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]) * D_MODEL 

    p = p + offset_bh * NUM_BLOCK * NUM_SLOT + (tl.arange(0, BLOCK_S) + offset_s * BLOCK_S ) 

    prev_memory = prev_memory + offset_bh * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[None, :]  + (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None] ) * D_MODEL

    next_memory = next_memory + offset_bh * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL +  tl.arange(0, BLOCK_MODEL)[None, :]  + (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None] ) * D_MODEL

    acc = tl.zeros([BLOCK_S, BLOCK_MODEL], dtype=tl.float32)
    acc += tl.load(prev_memory)    
    
    tl.store(O, acc.to(O.dtype.element_ty))

    O +=  D_MODEL * NUM_SLOT

    for i in range(NUM_BLOCK-1):
        p_i = tl.load(p)
        S_i = tl.load(S) 
        acc = acc * p_i[:, None] + S_i

        tl.store(O, acc.to(O.dtype.element_ty))

        p +=  NUM_SLOT 
        S +=  D_MODEL * NUM_SLOT
        O +=  D_MODEL * NUM_SLOT
    
    p_i = tl.load(p)
    S_i = tl.load(S) 
    acc = acc * p_i[:, None] + S_i
    tl.store(next_memory, acc.to(next_memory.dtype.element_ty))




@triton.jit
def _bwd_recurrence_qs(
    S, p, 
    DS, Dp,
    
    NUM_BLOCK, 
    D_MODEL, NUM_SLOT,
    BLOCK_S: tl.constexpr, BLOCK_MODEL: tl.constexpr
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    S = S + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :]  + (NUM_BLOCK - 2) * NUM_SLOT * D_MODEL

    DS = DS + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :] + (NUM_BLOCK - 1) * NUM_SLOT * D_MODEL

    p = p + offset_bh * NUM_BLOCK * NUM_SLOT + tl.arange(0, BLOCK_S) + offset_s * BLOCK_S + (NUM_BLOCK - 1) * NUM_SLOT 

    Dp = Dp + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d * NUM_SLOT * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[:, None] * NUM_SLOT + offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[None, :] + (NUM_BLOCK - 1) * NUM_SLOT * D_MODEL

    Dacc = tl.zeros([BLOCK_MODEL, BLOCK_S], dtype=tl.float32)

    for i in range(NUM_BLOCK - 2, -1, -1):
        S_i = tl.load(S)

        DS_i = tl.load(DS)

        dp_i = DS_i * S_i
        tl.store(Dp, dp_i)

        p_i = tl.load(p)

        Dacc = Dacc * p_i + DS_i 
                
        tl.store(S, Dacc.to(S.dtype.element_ty))
        
        
        S -= NUM_SLOT * D_MODEL
        DS -= NUM_SLOT * D_MODEL
        p -= NUM_SLOT
        Dp -= NUM_SLOT * D_MODEL 




@triton.jit
def _bwd_recurrence_output(
    S, p, 
    DS, Dp,
    NUM_BLOCK, 
    D_MODEL, NUM_SLOT,
    BLOCK_S: tl.constexpr, BLOCK_MODEL: tl.constexpr
 ):
    offset_bh = tl.program_id(0)
    offset_d = tl.program_id(1)
    offset_s = tl.program_id(2)    

    S = S + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[None, :]  + (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]) * D_MODEL   + (NUM_BLOCK - 2) * NUM_SLOT * D_MODEL

    DS = DS + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[None, :]  + (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]) * D_MODEL   + (NUM_BLOCK - 1) * NUM_SLOT * D_MODEL

    p = p + offset_bh * NUM_BLOCK * NUM_SLOT + tl.arange(0, BLOCK_S) + offset_s * BLOCK_S +  (NUM_BLOCK - 1) * NUM_SLOT 

    Dp = Dp + offset_bh * NUM_BLOCK * D_MODEL * NUM_SLOT + offset_d  * BLOCK_MODEL  +  tl.arange(0, BLOCK_MODEL)[None, :]  + (offset_s * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]) * D_MODEL   + (NUM_BLOCK - 1) * NUM_SLOT * D_MODEL
    
    Dacc = tl.zeros([BLOCK_S, BLOCK_MODEL], dtype=tl.float32)

    for i in range(NUM_BLOCK - 2, -1, -1):
        S_i = tl.load(S)

        DS_i = tl.load(DS)

        dp_i = DS_i * S_i
        tl.store(Dp, dp_i)

        p_i = tl.load(p)

        Dacc = Dacc * p_i[:, None] + DS_i 

        tl.store(S, Dacc.to(S.dtype.element_ty))
        
        S -= NUM_SLOT * D_MODEL
        DS -= NUM_SLOT * D_MODEL
        p -= NUM_SLOT
        Dp -= NUM_SLOT * D_MODEL 
        


def stickbreaking_with_residual(logits: torch.Tensor) -> torch.Tensor:
    orig_dtype = logits.dtype
    logits = logits.to(torch.float32)

    logz = F.logsigmoid(logits) 
    log_beta = F.logsigmoid(-logits)  
    
    cum_log_beta = torch.cumsum(log_beta, dim=-2)  

    cum_exp = cum_log_beta.exp()
    # logz_exp = logz.exp()
    log_beta_exp = log_beta.exp()

    p_hat_residual = cum_exp[..., -2, :]
    
    p_hat = torch.cat([ torch.zeros_like(cum_log_beta[..., 0:1, :]), cum_log_beta[..., :-1, :]], dim=-2)

    qf = p_hat
    kf = (cum_log_beta - logz)


    p_residual = (qf[..., -1, None, :] + log_beta[..., -1, None, :] - kf).exp()


    return qf.float(), kf.float(), p_residual.to(orig_dtype), p_hat.exp().to(orig_dtype), p_hat_residual.to(orig_dtype)


def compute_qs_intra_materialize(qk, qf, kf):
    return ((qf.unsqueeze(-2) + kf.unsqueeze(-3)).exp().to(qk) * qk.unsqueeze(-1)).sum(-2)

def compute_output_intra_materialize(g, qf, kf):
    return ((qf.unsqueeze(-2) + kf.unsqueeze(-3)).exp().to(g) * g.unsqueeze(-2)).sum(-1)


def flash_rnn_onc_no_materialize_v5(q, k, v, stb_score, h_k_prev=None, h_v_prev=None):
    # h_k_prev: (B, H, D, S)
    # h_v_prev: (B, H, S, D)

    num_chunk, chunk_size = q.shape[2], q.shape[3]
    num_slot = stb_score.shape[-1]
    
    qf, kf, p_residual, p_hat, p_hat_residual = stickbreaking_with_residual(stb_score)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, device=q.device, dtype=torch.bool), diagonal=0)
    qk = (q @ k.transpose(-1, -2)).masked_fill_(mask, 0)
    qs_intra = compute_qs_intra_materialize(qk, qf, kf) 
        
    to_add_1 = k.transpose(-1, -2) @ p_residual
    to_add_2 = p_residual.transpose(-1, -2) @ v 
    
    if h_k_prev is None:
        h_k_prev = torch.zeros_like(to_add_1[:, :, 0])
        h_v_prev = torch.zeros_like(to_add_2[:, :, 0])

    h_k, h_k_next = TorchFlashRNN_Stage1.apply(p_hat_residual, to_add_1, h_k_prev)
    qs = compute_qs(q, h_k, p_hat, qs_intra)

    h_v, h_v_next = TorchFlashRNN_Stage2.apply(p_hat_residual, to_add_2, h_v_prev)
    output_inter = compute_output_inter(qs, p_hat, h_v)
    
    output_intra = compute_output_intra_materialize(qs, qf, kf)
    output_intra = output_intra.masked_fill(mask, 0) @ v

    output = output_inter  + output_intra
    return rearrange(output, 'b h n c d -> b (n c) (h d)'), h_k_next.detach(), h_v_next.detach()


def torch_rnn_stage1(p_hat_residual, to_add_1, prev_memory):
    p_hat_residual = p_hat_residual.contiguous()
    to_add_1 = to_add_1.contiguous()
    prev_memory = prev_memory.contiguous()

    B, H, D, S = to_add_1.shape[0], to_add_1.shape[1],  to_add_1.shape[-2], to_add_1.shape[-1]
    num_block = to_add_1.shape[2]
    p_hat_residual = p_hat_residual.contiguous()
    to_add_1 = to_add_1.contiguous()
    output = torch.empty_like(to_add_1).contiguous()
    output[:, :, 0] = prev_memory

    acc1 = prev_memory

    for i in range(1, num_block + 1):
        p2 = p_hat_residual[:, :, i-1]
        acc1 = acc1 * p2.unsqueeze(-2) + to_add_1[:, :, i-1]
        if i != num_block:
            output[:, :, i] = acc1.clone()            
    return output, acc1


def torch_rnn_stage2(p_hat_residual, to_add_2, prev_memory):
    p_hat_residual = p_hat_residual.contiguous()
    to_add_2 = to_add_2.contiguous()
    prev_memory = prev_memory.contiguous()

    B, H, S, D = to_add_2.shape[0], to_add_2.shape[1],  to_add_2.shape[-2], to_add_2.shape[-1]
    acc2 = prev_memory 
    num_block = to_add_2.shape[2]
    
    output = torch.empty_like(to_add_2).contiguous()
    output[:, :, 0] = prev_memory

    for i in range(1, num_block+1):
        p2 = p_hat_residual[:, :, i-1, ]
        acc2 = acc2 * p2.unsqueeze(-1) + to_add_2[:, :, i-1]
        if i != num_block:
            output[:, :, i] = acc2.clone()
        
    return output, acc2

class TorchFlashRNN_Stage1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_hat_residual, to_add_1, prev_memory):        
        #prev_memory does not require grad
        p_hat_residual = p_hat_residual.contiguous()
        to_add_1 = to_add_1.contiguous()
        prev_memory = prev_memory.contiguous()

        B, H, D, S = to_add_1.shape[0], to_add_1.shape[1],  to_add_1.shape[-2], to_add_1.shape[-1]
        num_block = to_add_1.shape[2]
        p_hat_residual = p_hat_residual.contiguous()
        to_add_1 = to_add_1.contiguous()
        output = torch.empty_like(to_add_1).contiguous()
        output[:, :, 0] = prev_memory

        BLOCK_MODEL = 16
        BLOCK_S = 32

        assert to_add_1.is_contiguous()
        assert p_hat_residual.is_contiguous()
        grid = (to_add_1.shape[0] * to_add_1.shape[1], D//BLOCK_MODEL, S//BLOCK_S)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL
        ctx.BLOCK_S = BLOCK_S

        next_memory = torch.empty_like(prev_memory)

        _fwd_recurrence_qs[(to_add_1.shape[0] * to_add_1.shape[1], D//BLOCK_MODEL, S//BLOCK_S)](
            to_add_1,  
            p_hat_residual,
            output,
            prev_memory,
            next_memory,
            D_MODEL=D, NUM_BLOCK=num_block, NUM_SLOT=S, 
            BLOCK_MODEL=BLOCK_MODEL, BLOCK_S=BLOCK_S
        )
        
        ctx.save_for_backward(output, p_hat_residual)        
        return output, next_memory

    @staticmethod
    def backward(ctx, DO, Dnext_memory=None):
        DO = DO.contiguous()
        
        to_add_1, p_hat_residual = ctx.saved_tensors

        B, H, D, S = to_add_1.shape[0], to_add_1.shape[1], to_add_1.shape[-2], to_add_1.shape[-1]

        Dacc1 = torch.zeros(B, H, D, S).to(to_add_1)

        num_block = to_add_1.shape[2]

        grid = ctx.grid 
        BLOCK_S = ctx.BLOCK_S 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        D_p = torch.empty_like(DO)

        _bwd_recurrence_qs[grid](
            to_add_1, p_hat_residual,
            DO, D_p, 
            NUM_BLOCK = num_block, 
            D_MODEL = D, 
            NUM_SLOT = S,
            BLOCK_S = BLOCK_S,
            BLOCK_MODEL = BLOCK_MODEL
        )
        to_add_1[:, :, -1] = 0
        D_p[:, :, 0] = 0

        return D_p.sum(-2), to_add_1, None, None 


    
class TorchFlashRNN_Stage2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p_hat_residual, to_add_2, prev_memory):
        p_hat_residual = p_hat_residual.contiguous()
        to_add_2 = to_add_2.contiguous()
        prev_memory = prev_memory.contiguous()

        B, H, S, D = to_add_2.shape[0], to_add_2.shape[1],  to_add_2.shape[-2], to_add_2.shape[-1]
        acc2 = prev_memory 
        num_block = to_add_2.shape[2]
        
        output = torch.empty_like(to_add_2).contiguous()
        output[:, :, 0] = prev_memory

        BLOCK_MODEL = 16
        BLOCK_S = 32

        assert to_add_2.is_contiguous()
        assert p_hat_residual.is_contiguous()
        grid = (to_add_2.shape[0] * to_add_2.shape[1], D//BLOCK_MODEL, S//BLOCK_S)
        ctx.grid = grid 
        ctx.BLOCK_MODEL = BLOCK_MODEL
        ctx.BLOCK_S = BLOCK_S

        next_memory = torch.empty_like(prev_memory)

        _fwd_recurrence_output[(to_add_2.shape[0] * to_add_2.shape[1], D//BLOCK_MODEL, S//BLOCK_S)](
            to_add_2.contiguous(),  
            p_hat_residual.contiguous(),
            output,
            prev_memory,
            next_memory,
            D_MODEL=D, NUM_BLOCK=num_block, NUM_SLOT=S, 
            BLOCK_MODEL=BLOCK_MODEL, BLOCK_S=BLOCK_S
        )

        ctx.save_for_backward(output, p_hat_residual)        

        return output, next_memory


    @staticmethod
    def backward(ctx, DO, Dnext_memory=None):
        DO = DO.contiguous()

        to_add_2, p_hat_residual = ctx.saved_tensors

        B, H, D, S = to_add_2.shape[0], to_add_2.shape[1],  to_add_2.shape[-1], to_add_2.shape[-2]

        Dacc2 = torch.zeros(B, H, S, D).to(to_add_2)
        num_block = to_add_2.shape[2]

        grid = ctx.grid 
        BLOCK_S = ctx.BLOCK_S 
        BLOCK_MODEL = ctx.BLOCK_MODEL 
        D_p = torch.zeros_like(DO)

        _bwd_recurrence_output[grid](
            to_add_2, p_hat_residual,
            DO, D_p, 
            NUM_BLOCK = num_block, 
            D_MODEL = D, 
            NUM_SLOT = S,
            BLOCK_S = BLOCK_S,
            BLOCK_MODEL = BLOCK_MODEL
        )
        to_add_2[:, :, -1] = 0
        D_p[:, :, 0] = 0

        return D_p.sum(-1), to_add_2, None, None 

                       
@torch.jit.script
def compute_qs(q, memory_inter, p_hat, qs_intra):
    # return ((q @ memory_inter) * p_hat + qs_intra)  # linear attention
    return ((q @ memory_inter) * p_hat + qs_intra).softmax(-1)

@torch.jit.script
def compute_output_inter(qs, p_hat, memory_inter):
    return (qs * p_hat) @ memory_inter
