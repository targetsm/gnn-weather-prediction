import torch
from models.graph_model import GraphModel


class LinearModel(torch.nn.Module):
    def __init__(self, device):
        super(LinearModel, self).__init__()
        self.input_dim = 2048
        self.output_dim = 2048
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = torch.nn.Linear(self.input_dim, self.output_dim)
        self.linear3 = torch.nn.Linear(self.input_dim, self.output_dim)
        self.non_linear = torch.nn.Sigmoid()
        self.device = device
        self.graph_conv = GraphModel(device)

    def forward(self, x, y):
        x = self.graph_conv(x,y)
        x = x.reshape(x.shape[0], self.input_dim)
        if not self.training:
            with torch.no_grad():
                tmp = self.non_linear(self.linear(x))
                out = self.linear(tmp)
                return out.reshape(y.shape)
        tmp = self.non_linear(self.linear(x))
        out = self.linear(tmp)

        return out.reshape(y.shape)
