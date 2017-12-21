# -*- coding: utf-8 -*-

"""
Colorization GAN network

"""
import torch
from torch import nn
from torch.autograd import Variable
from torch import functional as F
import torch.optim as optimizer

from unet import UNet, GrayLayer


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):

        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


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
            nn.Conv2d(
                in_channels=in_chan,
                out_channels=ndf,
                kernel_size=4,
                padding=1,
                stride=2
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        prev_chan = 1
        for n in xrange(1, num_layers):
            cur_chan = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    in_channels=ndf*prev_chan,
                    out_channels=ndf*cur_chan,
                    kernel_size=4,
                    padding=1,
                    stride=1
                ),
                nn.BatchNorm2d(ndf*cur_chan),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            prev_chan = cur_chan

        cur_chan = min(2**num_layers, 8)
        sequence += [
            nn.Conv2d(
                in_channels=prev_chan*ndf,
                out_channels=ndf*cur_chan,
                kernel_size=4,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(ndf*cur_chan),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=ndf*cur_chan,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.layers = nn.Sequential(*sequence)

    def forward(self, x):

        return self.layers(x)


class ColorGAN(object):

    def __init__(self, opt):

        self.is_train = opt.train
        self.use_sigmoid = opt.use_sigmoid
        self.use_gpu = opt.gpu

        self.generator = ColorGenerator()
        self.gray_layer = GrayLayer(self.use_gpu)

        if self.is_train:
            self.discriminator = Disciminator(in_chan=4, use_sigmoid=self.use_sigmoid)

        if self.use_gpu:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.gray_layer = self.gray_layer.cuda()

        lr = opt.lr
        self.generator_optimizer = optimizer.Adam(lr=lr, params=self.generator.parameters())
        self.discriminator_optimizer = optimizer.Adam(lr=lr, params=self.discriminator.parameters())

        self.epoches = opt.epoches
        self.batch_size = opt.batch_size
        self.gan_criterion = GANLoss()
        self.l1_criterion = nn.L1Loss()
        self.current_loss = dict()

        print("init network")

    def train_step(self, real_y):

        #real_y = Variable(torch.FloatTensor(real_y))
        real_x = self.gray_layer(real_y)

        if self.use_gpu:
            real_x = real_x.cuda()
            real_y = real_y.cuda()

        fake_y = self.generator(real_x)

        # Update Discriminator
        discrim_fake = self.discriminator(torch.cat([real_x, fake_y.detach()], dim=1))
        discrim_real = self.discriminator(torch.cat([real_x, real_y], dim=1))

        discrim_fake_loss = self.gan_criterion(discrim_fake, False)
        discrim_real_loss = self.gan_criterion(discrim_real, True)

        discrim_loss = 0.5 * (discrim_fake_loss + discrim_real_loss)

        self.discriminator_optimizer.zero_grad()
        discrim_loss.backward()
        self.discriminator_optimizer.step()

        # Update Generator
        l1_color_loss = self.l1_criterion(fake_y, real_y)
        fake_x = self.gray_layer(fake_y)
        l1_content_loss = self.l1_criterion(fake_x, real_x)

        gen_discrim_fake = self.discriminator(torch.cat([real_x, fake_y],dim=1))
        gen_gan_loss = self.gan_criterion(gen_discrim_fake, True)
        gen_loss = l1_color_loss + l1_content_loss + gen_gan_loss

        self.generator_optimizer.zero_grad()
        gen_loss.backward()
        self.generator_optimizer.step()

        self.current_loss = {
            "gen_content_loss": l1_content_loss.cpu().data[0],
            "gen_color_loss": l1_color_loss.cpu().data[0],
            "gen_gan_loss": gen_gan_loss.cpu().data[0],
            "discrim_fake_loss": discrim_fake_loss.cpu().data[0],
            "discrim_real_loss": discrim_real_loss.cpu().data[0],
        }

        return self.current_loss

    def generate(self, model_path):

        pass

    def save(self, index, directory):

        if self.use_gpu:
            torch.save(self.generator.cpu().state_dict(), directory + "generator_cpu_params_%s.pth" % index)
            torch.save(self.discriminator.cpu().state_dict(), directory + "generator_cpu_params_%s.pth" % index)
            torch.save(self.generator.state_dict(), directory + "generator_gpu_params_%s.pth" % index)
            torch.save(self.discriminator.state_dict(), directory + "generator_gpu_params_%s.pth" % index)
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
        else:
            torch.save(self.generator.cpu().state_dict(), directory + "generator_cpu_params_%s.pth" % index)
            torch.save(self.discriminator.cpu().state_dict(), directory + "generator_cpu_params_%s.pth" % index)


def __test__gan():
    from PIL import Image
    from torchvision import transforms
    import collections

    Option = collections.namedtuple('Option', 'train, gpu, lr, batch_size, epoches,use_sigmoid')
    crop = transforms.RandomCrop([256, 256])
    to_tensor = transforms.ToTensor()
    path = '../test_images/2012.jpg'
    image = Image.open(path)
    image = crop(image)
    image = to_tensor(image)

    image = Variable(image)
    image = image.unsqueeze(0)

    opt = Option(
        train=True,
        gpu=0,
        lr=0.001,
        batch_size=1,
        epoches=1,
        use_sigmoid=False
    )
    gan = ColorGAN(opt)
    p = gan.train_step(image)
    print(p)

#__test__gan()





