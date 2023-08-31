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
        ):
        """
        Args:
            layer_idx (int): index of the layer to extract, not used for now
        """
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layer, batch_first=True, dropout=dropout, bidirectional=False).to(device)
        
        self.layer_idx = layer_idx
    
    def forward(self, x):
        output, h = self.lstm(x)
        return output