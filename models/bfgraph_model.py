import torch
from torch import nn
from torch.nn import Conv3d, Sigmoid, Tanh, ReLU, BatchNorm3d
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import grid


class BFGraph(nn.Module):
    def __init__(self, input_size, num_filters, num_residual_blocks, device):
        super().__init__()
        self.device = device
        self.num_filters = num_filters
        self.input_size = input_size
        self.num_residual_blocks = num_residual_blocks

        self.l_stack_graph = GCNConv(in_channels=-1, out_channels=num_filters, improved=True).to(device)
        # self.l_stack_graph = GATv2Conv(in_channels=-1, out_channels=num_filters, heads=4).to(device)

        """
        self.l_stack_conv3d = Conv3d(in_channels=input_size[-1], out_channels=num_filters, kernel_size=kernel_size,
                                     padding='same').to(device)
        """

        self.layer_li = nn.ModuleList()

        for i in range(self.num_residual_blocks):
            wave_net_res_layer = GCNConv(in_channels=-1, out_channels=num_filters, improved=True).to(device)
            self.layer_li.append(wave_net_res_layer)

        self.l4_graph = GCNConv(in_channels=-1, out_channels=1, improved=True).to(device)

    def forward(self, l_input, y):

        B, T, H, W, F = l_input.shape
        edge_index, pos = grid(H, W, device=self.device)
        l_input = l_input.permute(0, 2, 3, 4, 1)
        l_input = l_input.reshape(B, H * W, T * F)

        l_skip_connections = [self.l_stack_graph(l_input, edge_index)]

        for i in range(self.num_residual_blocks):
            l_skip_connection_output = self.layer_li[i](ReLU()(l_skip_connections[-1]), edge_index)
            l_skip_connections.append(l_skip_connection_output)

        l_sum = torch.stack(l_skip_connections, dim=0).sum(dim=0)
        relu = ReLU()(l_sum)

        l4_conv3d_output = self.l4_graph(relu, edge_index)

        l4_conv3d_output = l4_conv3d_output.reshape(B, H, W, 1, 1)
        l4_conv3d_output = l4_conv3d_output.permute(0, 3, 1, 2, 4)

        return l4_conv3d_output
