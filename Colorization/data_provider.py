# -*- coding: utf-8 -*-

import pickle
import numpy as np
from torchvision import transforms
from PIL import Image


def get_data(path, name='cifar'):

    data = []

    if name == 'cifar':

        with open(path, 'r') as f:
            datadict = pickle.load(f)
            x = datadict['data']
            data = x.reshape(50000, 32, 32, 3).astype("float")
            return data
    elif name == 'VOC':
        toPIL = transforms.ToPILImage()
        random_crop = transforms.RandomCrop([256, 256])

        return data

    else:
        return data






