# -*- coding: utf-8 -*-

"""
Colorization GAN network

"""
import torch
from torch import nn
from torch.autograd import Variable
from torch import functional as F

from unet import UNet


class ColorGenerator(nn.Module):

    def __init__(self):
        super(ColorGenerator, self).__init__()
        self.genrator = UNet()

    def forward(self, x):
        return self.genrator(x)


class Disciminator(nn.Module):
    """
    PatchDiscriminator
    """
    def __init__(self, in_chan, ndf=64, num_layers=3, use_sigmoid=False):
        super(Disciminator, self).__init__()

        sequence = [
            nn.Conv2d(in_channels=in_chan, out_channels=ndf, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        prev_chan = 1
        for n in xrange(1, num_layers):
            cur_chan = min(2**n, 8)
            sequence += [
                nn.Conv2d(in_channels=ndf*prev_chan, out_channels=ndf*cur_chan, kernel_size=4, padding=1, stride=1),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            prev_chan = cur_chan

        cur_chan = min(2**num_layers, 8)
        sequence += [
            nn.Conv2d(in_channels=prev_chan*ndf, out_channels=ndf*cur_chan, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf*cur_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=ndf*cur_chan, out_channels=1, kernel_size=4, stride=1, padding=1)
        ]

        if use_sigmoid:
            sequence += [ nn.Sigmoid() ]

        self.layers = nn.Sequential(*sequence)

    def forward(self, x):

        return self.layers(x)

