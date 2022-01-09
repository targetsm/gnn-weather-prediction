from torch.nn import Module
from torch import nn
import torch
from models.GCN import GCN
import models.util as util
import numpy as np


class AttModel(Module):

    def __init__(self, device):
        super(AttModel, self).__init__()
        in_features = 256
        kernel_size = 10  # M
        num_stage = 2
        dct_n = 10
        d_model = 128
        self.device = device
        self.kernel_size = kernel_size
        self.d_model = d_model
        self.dct_n = dct_n

        self.convQ = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.convK = nn.Sequential(nn.Conv1d(in_channels=in_features, out_channels=d_model, kernel_size=6,
                                             bias=False),
                                   nn.ReLU(),
                                   nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=5,
                                             bias=False),
                                   nn.ReLU())

        self.gcn = GCN(input_feature=(dct_n) * 2, hidden_feature=512, p_dropout=0.3,
                           num_stage=num_stage,
                           node_n=in_features)

        self.multihead_attn = nn.MultiheadAttention(self.d_model, 16)

        self.ff_nn = nn.Sequential(
            nn.Linear(in_features=(in_features*dct_n), out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=d_model),
            nn.ReLU(),
        )

        self.rev_ff_nn = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=(in_features*dct_n)),
            nn.ReLU(),
        )

        self.encode = nn.Linear(2048, in_features)
        self.decode = nn.Linear(in_features, 2048)

    def forward(self, batch, lables):
        """
        The forward pass.
        :param batch: Current batch of data.
        :return: Each forward pass must return a dictionary with keys {'seed', 'predictions'}.
        """
        batch_size = batch.shape[0]
        src = batch
        output_n = 72  # number of output frames T
        input_n = batch.shape[1] # number of input frames N

        dct_n = self.dct_n
        src = src.reshape((batch_size, input_n, 2048))  # [bs,in_n,dim]
        src = self.encode(src)
        src_tmp = src.clone()
        bs = src.shape[0]
        src_key_tmp = src_tmp.transpose(1, 2)[:, :, :(input_n - output_n)].clone()
        src_query_tmp = src_tmp.transpose(1, 2)[:, :, -self.kernel_size:].clone()

        dct_m, idct_m = util.get_dct_matrix(self.kernel_size + output_n)
        dct_m = torch.from_numpy(dct_m).float().to(self.device)
        idct_m = torch.from_numpy(idct_m).float().to(self.device)

        vn = input_n - self.kernel_size - output_n + 1
        vl = self.kernel_size + output_n
        idx = np.expand_dims(np.arange(vl), axis=0) + \
              np.expand_dims(np.arange(vn), axis=1)
        src_value_tmp = src_tmp[:, idx].clone().reshape(
            [bs * vn, vl, -1])
        src_value_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), src_value_tmp).reshape(
            [bs, vn, dct_n, -1]).transpose(2, 3).reshape(
            [bs, vn, -1])  #

        idx = list(range(-self.kernel_size, 0, 1)) + [-1] * output_n
        outputs = []

        key_tmp = self.convK(src_key_tmp / 1000.0).transpose(0, 1).transpose(0, 2)

        query_tmp = self.convQ(src_query_tmp / 1000.0).transpose(0, 1).transpose(0, 2)
        value_tmp = self.ff_nn(src_value_tmp).transpose(1, 2).transpose(0, 1).transpose(0, 2)

        mh_attn_out, weights = self.multihead_attn(query_tmp, key_tmp, value_tmp)

        af_ff_out = self.rev_ff_nn(mh_attn_out).transpose(0, 1).reshape(batch_size, -1, dct_n)

        input_gcn = src_tmp[:, idx]  # shape:[16, 34, 135]
        dct_in_tmp = torch.matmul(dct_m[:dct_n].unsqueeze(dim=0), input_gcn).transpose(1, 2)

        dct_in_tmp = torch.cat([dct_in_tmp, af_ff_out], dim=-1)

        dct_out_tmp = self.gcn(dct_in_tmp)
        out_gcn = torch.matmul(idct_m[:, :dct_n].unsqueeze(dim=0),
                               dct_out_tmp[:, :, :dct_n].transpose(1, 2))
        outputs.append(out_gcn.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)

        out_sq = outputs.squeeze()
        out_sq = out_sq[:, -1:, :]
        out_sq = self.decode(out_sq)

        return out_sq.reshape(lables.shape)
