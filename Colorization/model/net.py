# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import unet
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
            self.model = unet.UNet()
        else:
            self.model = FCN()

    def forward(self, x):

        x = self.model(x)
        return x


class OverallLoss(nn.Module):

    def __init__(self):
        super(OverallLoss, self).__init__()
        self.color_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()

    def forward(self, gen, target, gray_gen, gray_target, gamma=0.5):

        color = self.color_loss(gen, target)
        content = self.content_loss(gray_gen, gray_target)

        loss = (1/(gen.size()[2] * gen.size()[3]))*(color + gamma*content)

        return loss


