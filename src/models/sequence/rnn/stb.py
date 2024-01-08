import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def stickbreaking(key_logit: torch.Tensor, mask: torch.Tensor, ignore_current: bool = True) -> torch.Tensor:
    """
    Tensor shape:
        key_logit: (B, L, H)
        mask: (L, L), created with diagnoal=0 if ignore_current=True; diagnoal=1 if ignore_current=False (using torch.triu)
    """
    key_logit = key_logit.transpose(1, 2)  # B x H x L
    log_beta = F.logsigmoid(key_logit)
    log_neg_beta = F.logsigmoid(-key_logit)
    log_b = torch.cumsum(log_neg_beta, dim=-1)

    if ignore_current:
        # log_b - log_neg_beta is prefix sum
        log_beta_b = (log_b - log_neg_beta)[:, :, :, None] - log_b[:, :, None, :]
    else:
        log_beta_b = log_b[:, :, :, None] - log_b[:, :, None, :]

    mask = mask[None, None, :, :].expand_as(log_beta_b)
    log_beta_b = log_beta_b.masked_fill(mask, -torch.inf)
    log_att = log_beta[:, :, None, :] + log_beta_b.to(log_beta.dtype)
    return log_att


def stickbreaking_att(stb_score: torch.Tensor, value: torch.Tensor, ignore_current: bool = True) -> torch.Tensor:
    """
    Original version of stickbreaking attention.
    """
    B, L, H = stb_score.shape
    if ignore_current:
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=0).to(stb_score.device)
    else:
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1).to(stb_score.device)

    B, L, H, D = value.shape
    log_att = stickbreaking(stb_score, mask, ignore_current)
    y = torch.einsum('bhij,bjhd->bihd', log_att.exp(), value).reshape(B, L, H * D)
    return y

@torch.jit.script
def block_scan(stb_score: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    M: num_chunks, C: chunk_size
    Tensor shape:
        stb_score: (B, H, C, M)
        value: (B, H, C, M, D)
    """
    log_beta = F.logsigmoid(stb_score)
    log_neg_beta = F.logsigmoid(-stb_score)
    log_b = torch.cumsum(log_neg_beta, dim=-1)

    log_beta_b = log_beta - log_b
    normalizer = stb_score.shape[-1]
    log_beta_b = log_beta_b - normalizer

    # HACKY! convert to float64 to avoid overflow
    log_b = log_b.to(torch.float64)
    log_beta_b = log_beta_b.to(torch.float64)

    beta_b = log_beta_b.exp()
    beta_b_val_t = torch.einsum('bhmi,bhmid->bhmid', beta_b, value)
    cum_beta_b_val_t = torch.cumsum(beta_b_val_t, dim=-2)
    b = (log_b + normalizer).exp()
    y = torch.einsum('bhmi,bhmid->bhmid', b, cum_beta_b_val_t)

    y = y.to(stb_score.dtype)
    log_b = log_b.to(stb_score.dtype)
    
    return y, log_b

# @torch.jit.script
def stickbreaking_att_blockwise(stb_score: torch.Tensor, value: torch.Tensor, chunk_size: int, ignore_current=True) -> torch.Tensor:
    """
    Args:
        key_logit: (B, L, H)
        value: (B, L, H, D) or (B, L, H)
        chunk_size: the size of the chunk
        ignore_current: if True, ignore the current position when computing the attention

    Note that the mask arguments is not needed here.
    """
    if len(value.shape) == 3:
        # add head dim if missing
        value = value.unsqueeze(-1)

    B_, L_, H_ = stb_score.shape
    B, L, H, D = value.shape
    stb_score = stb_score.transpose(1, 2)  # B x H x L
    value = value.transpose(1, 2)  # B x H x L x D
    assert B_ == B and L_ == L and H_ == H

    M = chunk_size
    C = L // chunk_size
    assert L % C == 0

    # truncate such beta <= 0.99
    stb_score = stb_score.clamp(max=math.log( 0.99 / (1 - 0.99) ))

    block_key_logit = stb_score.reshape(B, H, C, M)
    block_val_t = value.reshape(B, H, C, M, D)

    block_v, block_gates = block_scan(block_key_logit, block_val_t)

    block_updates = []
    for nc in range(C):
        if nc == 0:
            # B x H x C x D
            block_update = torch.zeros_like(block_v[:, :, 0, :, :])
        else:
            last_block_update = block_updates[-1]
            # B x H x 1 x D
            last_block_v = block_v[:, :, nc - 1, -1:, :] + last_block_update[:, :, -1:, :]
            # B x H x C x 1
            cur_block_gates = (block_gates[:, :, nc, :].unsqueeze(-1)).exp()
            # B x H x C x D
            block_update = (cur_block_gates * last_block_v)
        block_updates.append(block_update)
    block_state_v = torch.stack(block_updates, dim=2)
    final_block_v = block_v + block_state_v
    final_block_v = final_block_v.reshape(B, H, L, D)

    # shift right by one if ignore_current
    y = final_block_v.transpose(1, 2).reshape(B, L, H * D)
    if ignore_current:
        y = F.pad(y, (0, 0, 1, -1), mode='constant', value=0)
    return y


class STB(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.att_hidden % config.n_head == 0
        # assert config.att_hidden == config.n_embd

        self.stb_gate = nn.Linear(config.n_embd, config.n_head)
        self.value_proj = nn.Linear(config.n_embd, config.att_hidden)

        self.out_gate = nn.Linear(config.n_embd, config.att_hidden)
        self.out_proj = nn.Linear(config.att_hidden, config.n_embd)

        # regularization
        self.att_dropout = nn.Dropout(config.attn_pdrop)
        self.att_chunk_size = config.att_chunk_size

        assert config.block_size % config.att_chunk_size == 0
        # self.register_buffer("mask_diag1", torch.triu(torch.ones(config.att_chunk_size, config.att_chunk_size, dtype=torch.bool), diagonal=1))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.att_hidden = config.att_hidden
        self.head_size = config.att_hidden // config.n_head

        self.att_func = config.att_func


    def forward(self, x):
        B, L, C = x.size()

        stb_score = self.stb_gate(x) # / (self.head_size ** 0.5) # (B, T, n_head)
        stb_score = stb_score.transpose(1, 2) # (B, n_head, T)
        value = self.value_proj(x)
        value = value.view(B, L, self.n_head, self.head_size) # (B, L, nh, hs)
        rnn_output = stickbreaking_att_blockwise(stb_score, value, self.att_chunk_size)

        y1 = rnn_output * torch.sigmoid(self.out_gate(x))
        y2 = self.out_proj(y1)
        return y2

if __name__ == "__main__":
    B, L, H, D = 2, 128, 16, 256
    ignore_current = False
    stb_score = torch.randn(B, L, H)
    value = torch.randn(B, L, H, D)

    y1 = stickbreaking_att(stb_score, value, ignore_current=ignore_current)
    y2 = stickbreaking_att_blockwise(stb_score, value, chunk_size=16, ignore_current=ignore_current)
    print(torch.max(torch.abs(y1 - y2)))