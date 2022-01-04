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
        self.conv = Conv2d(120, 1, kernel_size=1, device=device)
        self.conv2 = Conv2d(1, 1, kernel_size=1, device=device)
        self.graph_conv = GraphModel(device)

    def forward(self, x, y):
        #print(x.squeeze(4).shape)
        x = self.conv(x.squeeze(dim=4))
        x = x.unsqueeze(4)
        tmp = x.reshape(x.shape[0], self.input_dim)
        out = self.linear(tmp)
        out = out.reshape(x.shape)
        out2 = self.graph_conv(out, y)
        out = self.conv2(out2.squeeze(dim=4)).unsqueeze(4)
        return out
