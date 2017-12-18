# -*- coding: utf-8 -*-

from __future__ import  division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_chans)
        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class DownConv(nn.Module):

    def __init__(self, in_chans, out_chans):
        super(DownConv, self).__init__()
        self.conv = ConvBlock(in_chans, out_chans)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_chans, out_chans, bilinear=False):
        super(UpConv, self).__init__()
        self.convblock = ConvBlock(2*out_chans, out_chans)
        if bilinear:
            self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upconv = nn.ConvTranspose2d(in_chans, out_chans, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        if diff_x > 0:
            x2 = F.pad(x2, (diff_x // 2, int((diff_x+1) / 2), diff_y // 2, int((diff_y+1) / 2)))
        else:
            diff_x = -diff_x
            diff_y = -diff_y
            x1 = F.pad(x1, (diff_x // 2, int((diff_x+1) / 2), diff_y // 2, int((diff_y+1) / 2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.convblock(x)
        return x


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.in_convs = ConvBlock(1, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)
        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        self.out_convs = nn.Sequential(
            nn.Conv2d(64, 3, 1, padding=0)
        )

    def forward(self, x):

        x0 = self.in_convs(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x0)

        x = self.out_convs(x8)

        # gray_images = x[:,0]+x[:,1]+x[:2]

        return x


def __test_unet__():

    unet = UNet()
    import numpy as np

    image = np.random.normal(size=[3, 100, 100])
    image = Variable(torch.FloatTensor(image))
    image = image.unsqueeze(1)
    x = unet(image)

    #print(x.size())


