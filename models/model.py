import torch
from torch.nn import Transformer
import numpy as np

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
