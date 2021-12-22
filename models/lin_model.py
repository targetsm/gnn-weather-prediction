import torch
from torch.nn import Transformer
import numpy as np

class CustomModule(torch.nn.Module):
    def __init__(self, sample_time, lead_time):
        super(CustomModule, self).__init__()
        self.sample_time = sample_time
        self.lead_time = lead_time
        self.input_dim = sample_time*2048
        self.output_dim = lead_time*2048
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        print(x.shape)
        x_shape = x.shape
        x = x.reshape(x.shape[0], self.input_dim)
        print(x.shape)
        print(self.linear)
        out = self.linear(x)
        return out.reshape([x_shape[0], self.lead_time, x_shape[2], x_shape[3], x_shape[4]])