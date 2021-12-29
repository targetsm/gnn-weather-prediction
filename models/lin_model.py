import torch
from torch.nn import Transformer
import numpy as np

class LinearModel(torch.nn.Module):
    def __init__(self, device):
        super(LinearModel, self).__init__()
        self.input_dim = 2048
        self.output_dim = 2048
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.device = device

    def forward(self, x, y):
        x = x[:,-1].to(self.device)
        x = x.reshape(x.shape[0], self.input_dim)
        if not self.training:
            with torch.no_grad():
                out = self.linear(x)
                return out.reshape(y.shape)
        out = self.linear(x)
        return out.reshape(y.shape)
