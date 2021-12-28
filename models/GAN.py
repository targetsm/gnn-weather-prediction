import torch
from torch.nn import Transformer
import numpy as np

from torch import nn
from torch_geometric.nn import GATv2Conv, BatchNorm, global_mean_pool, GATConv


def create_nn(layer_dims, flag):
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        layers.append(nn.ReLU())
    if flag == 1:
        layers = layers[:-1]
    return torch.nn.Sequential(*layers)


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_vertices, batch_size):
        super().__init__()
        self.num_features = num_features
        self.num_vertices = num_vertices
        self.batch_size = batch_size

        self.output_shape = [self.batch_size, 1, 32, 64, self.num_features] # we preict one instance of num_features on a 32x64 latlon grid
        self.hidden = 2
        self.in_head = 2
        self.out_head = 1

        self.cov_dims = [self.num_features, 32, 32] # 32 hidden states

        self.gat1 = GATv2Conv(self.cov_dims[0], self.cov_dims[1], heads=self.in_head)
        self.norm1 = BatchNorm(self.cov_dims[1] * self.in_head)
        self.elu1 = nn.ELU()

        self.gat2 = GATv2Conv(self.cov_dims[1] * self.in_head + self.num_features, self.cov_dims[2], heads=self.out_head)
        self.norm2 = BatchNorm(self.cov_dims[2] * self.out_head)
        self.elu2 = nn.ELU()

        self.ra_ffnn_dims = [self.num_vertices * self.cov_dims[2] * self.out_head, 128, self.num_features*(64*32)]
        self.ra_ffnn = create_nn(self.ra_ffnn_dims, 0)


    def forward(self, input, edge_index):
        x = input.reshape(-1, self.num_features) # num_vertices*batch_size x num_features
        edge_index = edge_index

        y = self.gat1(x, edge_index)
        y = self.norm1(y)
        y = self.elu1(y)

        y = torch.cat((y, x), dim=1) # add residual connection before GATv2Conv layer
        y = self.gat2(y, edge_index)
        y = self.norm2(y)
        y = self.elu2(y)
        
        y = y.reshape(self.batch_size, -1)
        output = self.ra_ffnn(y).reshape(self.output_shape)

        return output


class CustomModule(torch.nn.Module):

    def __init__(self):
        super(CustomModule, self).__init__()
        self.model = Transformer(d_model=2048)

    def forward(self, x, y):
        x = x[:,:,:,:,[0]]
        self.lead_time = y.shape[1]
        x_shape = x.shape
        x = x.reshape(x_shape[0], x_shape[1], x_shape[2]*x_shape[3]*x_shape[4]).transpose(0,1)
        y_shape = y.shape
        y = y.reshape(y_shape[0], y_shape[1], y_shape[2]*y_shape[3]*y_shape[4]).transpose(0,1)
        #print(self.lead_time)
        if not self.training:
            src_mask = Transformer.generate_square_subsequent_mask(self.lead_time + 1)
            output = torch.from_numpy(np.full([y.shape[0]+1, y.shape[1], y.shape[2]], x[-1])[:self.lead_time + 1])
            #print(output.shape)
            with torch.no_grad():
                for i in range(1, self.lead_time + 1):
                    print(i)
                    output[i] = self.model(x, output[:i], tgt_mask=src_mask[:i, :i])[
                        i - 1]
            return output[:-1].reshape(y_shape)
        #print(x.shape, y.shape)
        output = self.model(x, y)
        return output.reshape(y_shape)
