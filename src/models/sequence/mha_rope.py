
import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange

from src.models.sequence.rope import RotaryEmbedding


class MultiHeadAttentionWithRope(SequenceModule):
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.att_qkv = nn.Linear(d_model, d_model * 3)

        self.d_model = d_model
        self.n_head = n_heads
        self.head_size = d_model // n_heads
        self.causal = causal

        self.rope = RotaryEmbedding(self.head_size)
        freqs = self.rope.freqs.data
        del self.rope.freqs
        self.rope.register_buffer('freqs', freqs)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, src, attn_mask=None, return_attention=False):
        assert self.causal and attn_mask is None

        x = src
        B, L, _ = x.size()
        query, key, value = self.att_qkv(x).split(self.d_model, dim=2)

        # B * H * n_head * head_size
        query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)

        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        x = F.scaled_dot_product_attention(query, key, value, is_causal=True)

        x = x.transpose(1, 2).contiguous().view(B, L, self.n_head * self.head_size)

        if return_attention:
            return self.out_proj(x), None
        return self.out_proj(x)