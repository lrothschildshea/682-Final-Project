#file to import data, relabel data, etc.

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

#this method is currently not used - possibly remove in the future if not needed
def getData():
    data = get_CIFAR10_data()
    for k,v in list(data.items()):
        print(('%s:' % k, v.shape))
    return data

def relabelData(data, score):
    temp_y_train = data['y_train'].copy()
    temp_y_val = data['y_val'].copy()
    temp_y_test = data['y_test'].copy()
    for i in range(0,len(temp_y_train)):
        if temp_y_train[i] == score:
            temp_y_train[i] = 1
        else:
            temp_y_train[i] = 0
    for j in range(0, len(temp_y_val)):
        if temp_y_val[j] == score:
            temp_y_val[j] = 1
        else:
            temp_y_val[j] = 0
    for k in range(0, len(temp_y_test)):
        if temp_y_test[k] == score:
            temp_y_test[k] = 1
        else:
            temp_y_test[k] = 0
    data['y_train'] = temp_y_train
    data['y_val'] = temp_y_val
    data['y_test'] = temp_y_test
    return data

def getDataPyTorch():
    NUM_TRAIN = 49000
    transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True,
                             transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64, 
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=True,
                           transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, 
                            transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

    return (loader_train, loader_val, loader_test)

def relabelDataPyTorch(data, score):
    loader_train, loader_val, loader_test = data
    temp_loader_train = loader_train
    temp_loader_val = loader_val
    temp_loader_test = loader_test

    for batch in temp_loader_train:
        batch_labels = batch[1]
        for i in range(0,list(batch_labels.size())[0]):
            if(batch_labels[i] == torch.LongTensor([score])):
                batch_labels[i] = torch.LongTensor([1])
            else:
                batch_labels[i] = torch.LongTensor([0])
        batch[1] = batch_labels

    for batch in temp_loader_train:
        print(batch)

relabelDataPyTorch(getDataPyTorch(), 2)