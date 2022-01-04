import torch
from models.graph_model import GraphModel
from torch.nn import Conv2d


class SimpleModel2(torch.nn.Module):
    def __init__(self, device):
        super(SimpleModel2, self).__init__()
        self.input_dim = 2048
        self.output_dim = 2048
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)
        self.non_linear = torch.nn.Sigmoid()
        self.device = device
        self.conv = Conv2d(12, 1, kernel_size=1, device=device)
        self.graph_conv = GraphModel(device)

    def forward(self, x, y):
        x = self.conv(x.squeeze(dim=4).permute(0,3,1,2))
        x = x.permute(0,3,1,2).unsqzueeze(4)
        out = self.graph_conv(x,y)
        return out
