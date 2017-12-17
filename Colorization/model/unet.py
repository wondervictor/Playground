# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F


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
        x = self.conv(x)
        x = self.pool(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_chans, out_chans, bilinear=True):
        super(UpConv, self).__init__()
        self.convblock = ConvBlock(in_chans, out_chans)
        if bilinear:
            self.upconv = nn.UpsamplingBilinear2d(scale_factor=2)
        else:
            self.upconv = nn.ConvTranspose2d(in_chans, out_chans, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        diff_x = x1.size()[2] - x2.size()[2]
        diff_y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diff_x // 2, int(diff_x / 2), diff_y // 2, int(diff_y / 2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.convblock(x)
        return x


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.convs = nn.Sequential(
            DownConv(1, 64),
            DownConv(64, 128),
            DownConv(128, 256),
            DownConv(256, 512),
            UpConv(512, 1024),
            UpConv(1024, 512),
            UpConv(512, 256),
            UpConv(256, 128)
        )

        self.out_convs = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 3, 1, padding=0)
        )

    def forward(self, x):

        x = self.convs(x)
        x = self.out_convs(x)

        return x
