# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F



class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()

    def forward(self, x):
        pass


class ColorModel(nn.Module):
    def __init__(self, name='UNet'):
        super(ColorModel, self).__init__()
        if name == 'UNet':
            self.model = UNet()
        else:
            self.model = FCN()

    def forward(self, x):

        x = self.model(x)
        return x