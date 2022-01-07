import torch
from torch import nn
from torch.nn import Conv3d, Sigmoid, Tanh, ReLU


class WaveNetResLayer(nn.Module):
    def __init__(self, num_filters, kernel_size, dilation_rate, device):
        super().__init__()
        # Gated activation.
        self.l_sigmoid_conv3d = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                       dilation=dilation_rate, padding='same').to(device)
        self.l_tanh_conv3d = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size,
                                    dilation=dilation_rate, padding='same').to(device)

        self.l_skip_connection = Conv3d(in_channels=num_filters, out_channels=num_filters, kernel_size=1,
                                        padding='same').to(device)

    def forward(self, l_input):
        l_sigmoid_conv3d_output = Sigmoid()(self.l_sigmoid_conv3d(l_input))
        l_tanh_conv3d_output = Tanh()(self.l_tanh_conv3d(l_input))
        l_mul = l_sigmoid_conv3d_output * l_tanh_conv3d_output
        l_skip_connection_output = self.l_skip_connection(l_mul)
        l_residual = l_input + l_skip_connection_output

        return l_residual, l_skip_connection_output


class WavenetModel(nn.Module):
    def __init__(self, input_size, num_filters, kernel_size, num_residual_blocks, lead_time, device):
        super().__init__()
        self.lead_time = lead_time
        self.input_size = input_size
        self.num_residual_blocks = num_residual_blocks

        self.l_stack_conv3d = Conv3d(in_channels=input_size[-1], out_channels=num_filters, kernel_size=kernel_size,
                                     padding='same').to(device)

        self.wave_net_res_layer_li = nn.ModuleList()

        for i in range(self.num_residual_blocks):
            wave_net_res_layer = WaveNetResLayer(num_filters, kernel_size, 2 ** (i + 1), device)
            self.wave_net_res_layer_li.append(wave_net_res_layer)

        self.l1_conv3d = Conv3d(in_channels=num_filters, out_channels=input_size[-1], kernel_size=(3, 3, 3),
                                padding='same').to(device)

        self.l2_conv3d = Conv3d(in_channels=input_size[-1], out_channels=input_size[-1], kernel_size=(1, 1, 3),
                                padding='same').to(device)

        self.l3_conv3d = Conv3d(in_channels=input_size[-1], out_channels=input_size[-1], kernel_size=(1, 1, 3),
                                padding='same').to(device)

        self.l4_conv3d = Conv3d(in_channels=input_size[-1], out_channels=1,
                                kernel_size=(1, 1, input_size[0])).to(device)

    def forward(self, l_input, y):

        l_input = l_input.permute(0, 4, 2, 3, 1)

        l_stack_conv3d_outputs = [self.l_stack_conv3d(l_input)]

        l_skip_connections = []
        for i in range(self.num_residual_blocks):
            l_stack_conv3d_output, l_skip_connection_output = self.wave_net_res_layer_li[i](l_stack_conv3d_outputs[-1])
            l_skip_connections.append(l_skip_connection_output)
            l_stack_conv3d_outputs.append(l_stack_conv3d_output)

        l_sum = torch.stack(l_skip_connections, dim=0).sum(dim=0)
        relu = ReLU()(l_sum)

        l1_conv3d_output = ReLU()(self.l1_conv3d(relu))

        l2_conv3d_output = ReLU()(self.l2_conv3d(l1_conv3d_output))

        l3_conv3d_output = ReLU()(self.l3_conv3d(l2_conv3d_output))

        l4_conv3d_output = self.l4_conv3d(l3_conv3d_output)

        return l4_conv3d_output.permute(0, 4, 2, 3, 1)
