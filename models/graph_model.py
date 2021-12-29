import torch
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, TransformerConv
from torch_geometric.data import Data
from torch_geometric.utils import grid

class GraphModel(torch.nn.Module):
    ''' use just the last image and run graph nn over it'''
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.conv = GCNConv(-1, 1)

    def forward(self, batch, labels):
        output_shape = labels.shape[1:]
        output = []
        for x in batch:
            edge_index, pos = grid(x.shape[1], x.shape[2], device=self.device)
            x = x.moveaxis(0, 2)
            x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
            out = self.conv(x, edge_index)
            output.append(out.reshape(output_shape))
        return torch.stack(output)
