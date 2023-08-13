import numpy as np
import torch
import torch.nn as nn
from src.models.sequence.rnn.scan_triton import real_scan_tie_input_gate, real_scan_tie_input_gate_fused

class GILRLayer(nn.Module):
    def __init__(
            self,
            d_model,
            factor=1,
            dropout=0.2,
            fuse_forget_gate=True,
            layer_idx=None,
            device=None,
            dtype=None,
        ):
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()
        self.d_model = d_model
        self.fuse_forget_gate = fuse_forget_gate
        
        self.in_proj = nn.Linear(self.d_model, self.d_model*3*factor, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(factor * self.d_model, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_model * factor, self.d_model, **factory_kwargs)
        self.swish =  nn.SiLU()

        self.layer_idx = layer_idx
    
    def forward(self, x):
        u = self.in_proj(x)
        v, o, f = u.chunk(3,dim=-1)
        
        if not self.fuse_forget_gate:
            f = f.sigmoid()            
            v = real_scan_tie_input_gate(v.contiguous(), f.contiguous())
        else:
            v = real_scan_tie_input_gate_fused(v.contiguous(), f.contiguous())

        return self.out_proj( 
            self.layer_norm(
                self.dropout(v * self.swish(o))
            )
        )
