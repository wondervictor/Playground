# -*- coding: utf-8 -*-


import argparse
import torch
import numpy as np
import torch.optim as optimizer
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import unet
from model.net import OverallLoss
from data_provider import get_data


def gray(image):
    r = image[0]
    g = image[1]
    b = image[2]
    tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return tensor


transform = transforms.Compose([
    transforms.ToPILImage(),
])

transform_totensor = transforms.Compose([
    transforms.ToTensor()
])


def batch_gray(images):
    """
    Grayscale for Batch Images
    :param images: [B, C, H, W]
    :return: [B, H, W]
    """
    batch_size = images.size()[0]
    h = images.size()[2]
    w = images.size()[3]
    result = Variable(torch.zeros([batch_size, h, w]))
    for i in xrange(batch_size):
        result[i] = gray(images[i])

    return result


def train(opt):

    """

    Y'=F(X)
    Loss1 = |Gray(Y')-X|
    Loss2 = |Y'-Y|
    :param opt: options
    :return:
    """

    epoches = opt.epoches
    batch_size = opt.batch_size
    model = opt.model
    # 'UNet'

    generator = unet.UNet()
    content_criterion = nn.MSELoss()
    color_criterion = nn.MSELoss()
    #criterion = OverallLoss()
    gamma = 0.5

    if opt.gpu == 1:

        generator = generator.cuda()
        content_criterion = content_criterion.cuda()
        color_criterion = color_criterion.cuda()

    train_optimizer = optimizer.Adam(lr=opt.lr, params=generator.parameters())

    print("Load Data ...")
    data = get_data(opt.path, opt.dataset)
    num_samples = len(data)

    for epoch in xrange(epoches):

        loss = 0
        for batch in xrange(num_samples/batch_size):
            images = np.array(data[batch:batch+batch_size])
            images = images.transpose((0, 3, 1, 2))
            images = Variable(torch.FloatTensor(images))
            gray_images = batch_gray(images)

            if opt.gpu == 1:
                images = images.cuda()
                gray_images = gray_images.cuda()
            gray_images = gray_images.unsqueeze(1)
            gen_images = generator(gray_images)
            gen_gray = batch_gray(gen_images)
            if opt.gpu == 1:
                gen_gray = gen_gray.cuda()

            color_loss = color_criterion(gen_images, images)
            current_loss = color_loss
            #content_loss = content_criterion(gen_gray, ge)
            #current_loss = criterion(gen_images, images, gen_gray, gray_images)
            loss += current_loss.cpu().data.numpy()[0]
            train_optimizer.zero_grad()
            current_loss.backward()
            train_optimizer.step()

            if batch % 5 == 0:
                print("Epoch: %s Batch: %d Loss: %s" %
                      (epoch, batch, loss/(batch+1)))

        torch.save(generator.state_dict(), 'model_params/epoch_%s_params.model' % epoch)


def inference(opt):

    # if type(model) == str:
    #     model = unet.UNet()
    #     model.load_state_dict(torch.load(model))
    #
    # if type(x) != list:
    #     x = Variable(torch.FloatTensor(x))
    #     x = x.unsqueeze(0)

    assert opt.model_params is None, "Model Param Dir cannot be None"
    assert opt.image_name is None, "Image cannot be None"

    image = Image.open(opt.image_name)

    x = transform_totensor(image)
    x = Variable(torch.FloatTensor(x))

    model = unet.UNet()
    model.load_state_dict(torch.load(opt.model_params))

    if opt.gpu == 1:
        model = model.cuda()
        x = x.cuda()

    gen_images = model(x).cpu()

    batch_nums = gen_images.size()[0]

    for i in xrange(batch_nums):

        img = transform(gen_images[i].data)
        img.save('out_%s.jpg' % i, 'JPEG')

    print("Inference Ended")


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--infer', type=int, default=0, help='Inference with trained model')
    args.add_argument('--model_params', type=str, default=None, help='trained model dir')
    args.add_argument('--epoches', type=int, default=30, help='epoch')
    args.add_argument('--gpu', type=int, default=0, help='use gpu')
    args.add_argument('--model', type=str, default="unet", help='network model')
    args.add_argument('--path', type=str, default="./", help='dataset dir')
    args.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    args.add_argument('--dataset', type=str, default="cifar", help='dataset')
    args.add_argument('--image_name', type=str, default=None, help='dataset')

    params = args.parse_args()

    if params.infer == 1:
        inference(params)
    else:
        train(opt=params)




