# -*- coding:utf-8 -*-

import os
import threading
import Queue
import random
import argparse

import torch
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from skimage import color, io


from model.color_gan import ColorGAN

resize = transforms.Scale(256)
crop = transforms.RandomCrop([256, 256])
to_tensor = transforms.ToTensor()

image_path_list = 'pascal_voc_image_list.txt'


def rgb2lab(img):
    return color.rgb2lab(img)


def lab2rgb(img):
    return color.lab2rgb(img)


def crop_image_lab(img):
    """
    Random crop images to 256 * 256
    :param img:
    :return:
    """
    if img.size[0] < 256 or img.size[1] < 256:
        img = resize(img)

    img = crop(img)
    img = np.array(img)
    lab_image = rgb2lab(img)
    return torch.FloatTensor([lab_image[:, :, 0]]), \
        torch.FloatTensor([lab_image[:, :, 0], lab_image[:, :, 1],lab_image[:, :, 2]])


def crop_image(img):
    """
    Random crop images to 256 * 256
    :param img:
    :return:
    """
    if img.size[0] < 256 or img.size[1] < 256:
        img = resize(img)

    img = crop(img)
    return to_tensor(img)


def load_image_path():

    with open(image_path_list, 'r') as f:
        paths = f.readlines()
        paths = [x.rstrip('\n\r') for x in paths]

    return paths


def init_data(image_queue, directory, paths, init_size, use_lab):

    for i in xrange(init_size):
        path = directory + paths[i]
        img = Image.open(path)
        if use_lab:
            img = crop_image_lab(img)
        else:
            img = crop_image(img)
        image_queue.put(img)


def load_data_concurrently(image_queue, paths, directory, max_size, use_lab):

    def create_image(q, paths, max_size, dir_, use_lab):

        while True:
            while q.qsize() < max_size:

                idx = random.randint(0, len(paths)-1)
                path = dir_ + paths[idx]
                img = Image.open(path)
                if not use_lab:
                    x = crop_image(img)
                else:
                    x = crop_image_lab(img)
                q.put(x)

    image_thread = threading.Thread(target=create_image, args=(image_queue, paths, max_size, directory, use_lab))
    image_thread.daemon = True
    image_thread.start()


def load_batch_data(batch_size, image_queue, use_lab):

    image_height = 256
    if use_lab:
        input_tensor = torch.zeros((batch_size, 1, image_height, image_height))
        result_tensor = torch.zeros((batch_size, 3, image_height, image_height))

        for x in xrange(batch_size):
            m, n = image_queue.get()
            result_tensor[x] = n
            input_tensor[x] = m
        print("Get data from queue/%s" % image_queue.qsize())
        return result_tensor, input_tensor
    else:
        result_tensor = torch.zeros((batch_size, 3, image_height, image_height))

        for x in xrange(batch_size):
            n = image_queue.get()
            result_tensor[x] = n
        print("Get data from queue/%s" % image_queue.qsize())
        return result_tensor


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epoches', type=int, default=300, help='Training epoches')
    parser.add_argument('--gpu', type=int, default=0, help='Training with GPU')
    parser.add_argument('--train', type=int, default=1, help='Training?')
    parser.add_argument('--use_sigmoid', type=bool, default=False, help='Use Sigmoid?')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/', help='dataset directory')
    parser.add_argument('--params_dir', type=str, default='model_params/', help='dataset directory')
    parser.add_argument('--use_lab', type=int, default=0, help='use CIE Lab Color Space')

    params = parser.parse_args()

    use_lab = params.use_lab
    if use_lab == 1:
        use_lab = True
    else:
        use_lab = False

    color_gan = ColorGAN(params)

    data_queue = Queue.Queue()

    image_paths = load_image_path()
    init_data(data_queue, params.data_dir, image_paths, 3000, use_lab)
    load_data_concurrently(data_queue, image_paths, params.data_dir, 5000, use_lab)

    for epoch in xrange(params.epoches):

        if params.use_lab:
            output_images, input_images = load_batch_data(params.batch_size, data_queue, use_lab)
            batch_input = Variable(torch.FloatTensor(input_images))
            batch_result = Variable(torch.FloatTensor(output_images))

            if params.gpu:
                batch_input = batch_input.cuda()
                batch_result = batch_result.cuda()
            losses = color_gan.train_step(batch_result, batch_input)
        else:
            output_images = load_batch_data(params.batch_size, data_queue, use_lab)
            batch_result = Variable(torch.FloatTensor(output_images))

            if params.gpu:
                batch_result = batch_result.cuda()
            losses = color_gan.train_step(batch_result)

        """
            "gen_content_loss": l1_content_loss.cpu().data[0],
            "gen_color_loss": l1_color_loss.cpu().data[0],
            "gen_gan_loss": gen_gan_loss.cpu().data[0],
            "discrim_fake_loss": discrim_fake_loss.cpu().data[0],
            "discrim_real_loss": discrim_real_loss.cpu().data[0],
        """
        print("Iteration: %s" % epoch)
        print("Generator Loss:\nGAN: %s Content: %s Color: %s"
              % (losses['gen_gan_loss'], losses['gen_content_loss'], losses['gen_color_loss']))
        print("Discriminator Loss\nFake Loss: %s Real Loss: %s\n"
              % (losses['discrim_fake_loss'], losses['discrim_real_loss']))

        if (epoch+1) % 100 == 0:

            color_gan.save(epoch, params.params_dir)


if __name__ == '__main__':
    main()




