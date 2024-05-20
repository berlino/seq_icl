from torch import nn
from mamba_ssm import Mamba as MambaSSM

class MambaLayer(nn.Module):
    def __init__(self, d_model, layer_idx=None, device=None, dtype=None):
        super().__init__()
        self.inner_mamba = MambaSSM(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            use_fast_path=False,
        )

    def forward(self, x, return_attention=False, input_ids=None):
        assert return_attention is False
        output = self.inner_mamba(x)
        return output