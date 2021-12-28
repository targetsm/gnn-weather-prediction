import torch
from torch.nn import Transformer
import numpy as np

class CustomModule(torch.nn.Module):
    def __init__(self, time_step, lead_time):
        super(CustomModule, self).__init__()
        self.time_step = time_step
        self.lead_time = lead_time
        self.input_dim = 2048
        self.output_dim = 2048
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, y):
        x = x[:,-1]
        x = x.reshape(x.shape[0], self.input_dim)
        if not self.training:
            with torch.no_grad():
                out = self.linear(x)
                return out.reshape(y.shape)
        out = self.linear(x)
        return out.reshape(y.shape)
