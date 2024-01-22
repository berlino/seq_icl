import torch
import torch.nn.functional as F
from torch import nn

def induction_head(x, hidden_state, shift_step=1, ngram=1):
    """
    Args:
        x: bsz x input_len
        hidden_state: bsz x input_len x d_model
        shift_right: use the second token from the bigram
    Output:
        bsz x input_len x d_model
    """
    bsz, seq_len = x.shape

    # bsz x L x L
    # match ngrams in the input sequence
    mask_0 = x[:, None, :] == x[:, :, None]

    causal_mask = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=-1
    )
    mask_0 = torch.logical_and(mask_0, causal_mask)

    masks = [mask_0.long()]
    for _ in range(1, ngram):
        mask_0 = F.pad(mask_0, (1, -1, 1, -1), "constant", False)
        masks.append(mask_0.long())

    ih_mask = torch.stack(masks, dim=-1).sum(dim=-1) >= ngram

    if shift_step > 0:
        ih_mask = F.pad(ih_mask, (shift_step, -shift_step), "constant", False)


    ih_mask = torch.logical_and(ih_mask, causal_mask)


    ih_mask_norm = ih_mask / ih_mask.sum(dim=2, keepdim=True)
    ih_mask_norm = torch.nan_to_num(ih_mask_norm, 0)
    output = torch.einsum("bmn,bnz->bmz", ih_mask_norm, hidden_state)
    return output


class Ngram(nn.Module):
    def __init__(self, d_model, ngram=1, layer_idx=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.ngram = ngram

        # self.t0 = torch.nn.Parameter(torch.zeros(self.d_model))
        # self.t1 = torch.nn.Parameter(torch.ones(self.d_model))

        self.t0 = nn.Linear(self.d_model, self.d_model)
        self.t1 = nn.Linear(self.d_model, self.d_model)
        # self.out_proj = nn.Linear(self.d_model, self.d_model)
        # self.out_gate = nn.Linear(self.d_model, self.d_model)


    def forward(self, x, return_attention=False, input_ids=None):
        bsz, seq_len, _ = x.shape
        h0 = induction_head(input_ids, x, ngram=self.ngram)
        h1 = x
        y = self.t0(h0) + self.t1(h1)
        # y_gate = self.out_gate(y)
        # y = self.out_proj(y) * F.swish(y_gate)
        return y