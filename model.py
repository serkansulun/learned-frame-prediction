import torch.nn as nn
import constants as c


def conv_same(in_channels, out_channels, kernel_size, bias=True):
    # Convolution which does not change image size
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class GenModel(nn.Module):
    def __init__(self, input_ch=c.HIST_LEN):
        super(GenModel, self).__init__()
        self.n_res_blocks = 32
        self.res_ch = 256
        self.res_k = 3
        self.res_activation = 'ReLU'
        self.res_scale = 0.1
        self.input_ch = input_ch

        act = eval('nn.{}(True)'.format(self.res_activation))

        head = [conv_same(self.input_ch, self.res_ch, self.res_k)]

        body = [ResBlock(conv_same, self.res_ch, self.res_k, act=act, res_scale=self.res_scale)
                for _ in range(self.n_res_blocks)]
        body.append(conv_same(self.res_ch, self.res_ch, self.res_k))

        tail = [conv_same(self.res_ch, c.PRED_LEN, self.res_k), nn.Tanh()]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x
