
import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange

from src.models.sequence.rope import RotaryEmbedding

def flash_attention(query, key, value, mask=None, dropout=None):
    """
    Args:
        either mask or is_causal must be provided
    """
    # assert mask is not None
    # return F.scaled_dot_product_attention(query, key, value, mask, dropout.p,)
    assert mask is not None
    return F.scaled_dot_product_attention(query, key, value, None, dropout.p, is_causal=True)


class MultiHeadAttentionWithRope(SequenceModule):
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.att_qkv = nn.Linear(d_model, d_model * 3)

        self.n_head = n_heads 
        self.head_size = d_model // n_heads

        self.rope = RotaryEmbedding(self.head_size)
        freqs = self.rope.freqs.data
        del self.rope.freqs
        self.rope.register_buffer('freqs', freqs)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2), dtype=torch.bool, device=src.device), diagonal=1)

        B, L, _ = x.size()
        query, key, value = self.att_qkv(x).split(self.att_hidden, dim=2)

        # B * H * n_head * head_size
        query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)

        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        x = flash_attention(query, key, value, mask=attn_mask)

        x = x.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_size)
        return self.out_proj(x), None 