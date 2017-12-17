# -*- coding: utf-8 -*-

import pickle
import numpy as np


def get_data(path, name='cifar'):

    data = []

    if name == 'cifar':

        with open(path, 'r') as f:
            datadict = pickle.load(f)
            x = datadict['data']
            data = x.reshape(50000, 3072).astype("float")
            return data

    else:
        return data



