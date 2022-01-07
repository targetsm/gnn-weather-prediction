import torch
from torch import nn
from torch.nn import Conv3d, Sigmoid, Tanh, ReLU, BatchNorm3d
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.utils import grid


class GraphWaveNetResLayer(nn.Module):
    def __init__(self, num_filters, time_step, kernel_size, dilation_rate, device):
        super().__init__()
        self.device = device
        # Gated activation.

        self.graph_layer = GCNConv(in_channels=-1, out_channels=num_filters * time_step, improved=True).to(device)
        # self.graph_layer = GATv2Conv(in_channels=-1, out_channels=num_filters * time_step, heads=4).to(device)

        self.l_sigmoid_conv3d = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                       dilation=dilation_rate, padding='same').to(device)
        self.l_tanh_conv3d = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                    dilation=dilation_rate, padding='same').to(device)

        self.BN = BatchNorm3d(num_filters)

        self.l_skip_connection = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=1,
                                        padding='same').to(device)

    def forward(self, l_input, edge_index, B, F, H, W, T):
        graph_output = self.graph_layer(l_input, edge_index)

        graph_output = graph_output.reshape(B, H, W, T, F)
        graph_output = graph_output.permute(0, 4, 1, 2, 3)

        l_sigmoid_conv3d_output = Sigmoid()(self.l_sigmoid_conv3d(graph_output))
        l_tanh_conv3d_output = Tanh()(self.l_tanh_conv3d(graph_output))
        l_mul = l_sigmoid_conv3d_output * l_tanh_conv3d_output
        l_skip_connection_output = self.l_skip_connection(l_mul)

        l_skip_connection_output = self.BN(l_skip_connection_output)

        l_skip_connection_output = l_skip_connection_output.permute(0, 2, 3, 4, 1)
        l_skip_connection_output = l_skip_connection_output.reshape(B, H * W, T * F)
        l_residual = l_input + l_skip_connection_output

        return l_residual, l_skip_connection_output


class GraphWavenetModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, num_residual_blocks, device):
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

        self.wave_net_res_layer_li = nn.ModuleList()

        for i in range(self.num_residual_blocks):
            wave_net_res_layer = GraphWaveNetResLayer(num_filters // input_size[0], input_size[0], kernel_size,
                                                      2 ** (i + 1), device)
            self.wave_net_res_layer_li.append(wave_net_res_layer)

        """
        self.l1_conv3d = Conv3d(in_channels=num_filters, out_channels=input_size[-1], kernel_size=(3, 3, 3),
                                padding='same').to(device)

        self.l2_conv3d = Conv3d(in_channels=input_size[-1], out_channels=input_size[-1], kernel_size=(1, 1, 3),
                                padding='same').to(device)

        self.l3_conv3d = Conv3d(in_channels=input_size[-1], out_channels=input_size[-1], kernel_size=(1, 1, 3),
                                padding='same').to(device)

        self.l4_conv3d = Conv3d(in_channels=input_size[-1], out_channels=1,
                                kernel_size=(1, 1, input_size[0])).to(device)
        """

        self.l1_graph = GCNConv(in_channels=-1, out_channels=num_filters, improved=True).to(device)
        # self.l1_graph = GATv2Conv(in_channels=-1, out_channels=num_filters, heads=4).to(device)

        self.l2_graph = GCNConv(in_channels=-1, out_channels=num_filters // 2, improved=True).to(device)
        # self.l2_graph = GATv2Conv(in_channels=-1, out_channels=num_filters, heads=4).to(device)

        self.l3_graph = GCNConv(in_channels=-1, out_channels=num_filters // 4, improved=True).to(device)
        # self.l3_graph = GATv2Conv(in_channels=-1, out_channels=num_filters, heads=4).to(device)

        self.l4_graph = GCNConv(in_channels=-1, out_channels=1, improved=True).to(device)
        # self.l4_graph = GATv2Conv(in_channels=-1, out_channels=1, heads=4).to(device)

    def forward(self, l_input, y):

        B, T, H, W, F = l_input.shape
        edge_index, pos = grid(H, W, device=self.device)
        l_input = l_input.permute(0, 2, 3, 4, 1)
        l_input = l_input.reshape(B, H * W, T * F)

        l_stack_conv3d_outputs = [self.l_stack_graph(l_input, edge_index)]

        l_skip_connections = []
        for i in range(self.num_residual_blocks):
            l_stack_conv3d_output, l_skip_connection_output = self.wave_net_res_layer_li[i](l_stack_conv3d_outputs[-1],
                                                                                            edge_index, B,
                                                                                            self.num_filters // T, H, W,
                                                                                            T)
            l_skip_connections.append(l_skip_connection_output)

            l_stack_conv3d_outputs.append(l_stack_conv3d_output)

        l_sum = torch.stack(l_skip_connections, dim=0).sum(dim=0)
        relu = ReLU()(l_sum)

        l1_conv3d_output = ReLU()(self.l1_graph(relu, edge_index))

        l2_conv3d_output = ReLU()(self.l2_graph(l1_conv3d_output, edge_index))

        l3_conv3d_output = ReLU()(self.l3_graph(l2_conv3d_output, edge_index))

        l4_conv3d_output = self.l4_graph(l3_conv3d_output, edge_index)

        l4_conv3d_output = l4_conv3d_output.reshape(B, H, W, 1, 1)
        l4_conv3d_output = l4_conv3d_output.permute(0, 3, 1, 2, 4)

        return l4_conv3d_output
