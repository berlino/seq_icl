import numpy as np
import torch
import torch.nn as nn

class LSTMLayer(nn.Module):
    def __init__(
            self,
            d_model,
            n_layer=1,
            dropout=0.1,
            layer_idx=None,
            device=None,
            dtype=None,
            reinit=True,
        ):
        """
        Args:
            layer_idx (int): index of the layer to extract, not used for now
        """
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layer, batch_first=True, dropout=dropout, bidirectional=False).to(device)
        
        self.layer_idx = layer_idx

        if reinit:
            self._reinitialize()
    
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)
    
    def forward(self, x):
        output, h = self.lstm(x)
        return output