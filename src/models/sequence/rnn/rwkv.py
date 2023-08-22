import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.sequence.rnn.scan_triton import wkv_triton_vanilla, wkv_triton_log_space

class RWKVLayer(nn.Module):
    init_x: Tensor
    init_state: Tensor

    def __init__(
            self,
            d_model,
            dropout=None,
            layer_idx=None,
            device=None,
            dtype=None,
        ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.time_decay = nn.Parameter(torch.randn(d_model))
        self.time_first = nn.Parameter(torch.randn(d_model))

        self.time_mix_k = nn.Parameter(torch.randn(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.randn(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.randn(1, 1, d_model))

        self.key = nn.Linear(d_model, d_model, False, **factory_kwargs)
        self.value = nn.Linear(d_model, d_model, False, **factory_kwargs)
        self.receptance = nn.Linear(d_model, d_model, False, **factory_kwargs)
        self.output = nn.Linear(d_model, d_model, False, **factory_kwargs)

        # init_state = torch.zeros(1, 2, 1, d_model, **factory_kwargs)
        # self.wkv_fn = wkv_triton_vanilla

        init_state = torch.full((1, 3, 1, d_model), float("-inf"), **factory_kwargs)
        self.wkv_fn = wkv_triton_log_space

        self.register_buffer("init_x", torch.zeros(1, 1, d_model), persistent=False)
        self.register_buffer("init_state", init_state, persistent=False)

        self.layer_idx = layer_idx

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor) -> Tensor:
        bsz, _, _ = x.shape

        last_x = self.init_x.repeat_interleave(bsz, dim=0)
        last_state = self.init_state.repeat_interleave(bsz, dim=0)
        last_x = self.time_shift(last_x, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        w = torch.exp(w)
        wkv, next_state = self.wkv_fn(w, u, k, v, last_state)
        rwkv = wkv * sr

        return self.output(rwkv)