from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

hold = []

def getDataPyTorch():
    NUM_TRAIN = 49000
    transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    cifar10_train = dset.CIFAR10('./dataset', train=True, download=True, transform=transform)
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=True, transform=transform)
    loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=64)

    return (loader_train, loader_val, loader_test)

def relabelDataPyTorch(data, score, device):
    loader_train, loader_val, loader_test = data
    list_loader_train = []
    list_loader_val = []
    list_loader_test = []
    og_loader_test = torch.tensor([], dtype=torch.long)
    og_loader_test = og_loader_test.to(device=device, dtype=torch.long)
    for batch in loader_train:
        batch_labels = batch[1]
        for i in range(0,list(batch_labels.size())[0]):
            if(batch_labels[i] == torch.LongTensor([score])):
                batch_labels[i] = torch.LongTensor([1])
            else:
                batch_labels[i] = torch.LongTensor([0])
        batch[1] = batch_labels
        list_loader_train.append(batch)
    
    for batch in loader_val:
        batch_labels = batch[1]
        for j in range(0,list(batch_labels.size())[0]):
            if(batch_labels[j] == torch.LongTensor([score])):
                batch_labels[j] = torch.LongTensor([1])
            else:
                batch_labels[j] = torch.LongTensor([0])
        batch[1] = batch_labels
        list_loader_val.append(batch)
    
    for batch in loader_test:
        og_loader_test = torch.cat((og_loader_test, batch[1].to(device=device, dtype=torch.long)), 0)
        b = copy.deepcopy(batch)
        batch_labels = b[1]
        for k in range(0,list(batch_labels.size())[0]):
            if(batch_labels[k] == torch.LongTensor([score])):
                batch_labels[k] = torch.LongTensor([1])
            else:
                batch_labels[k] = torch.LongTensor([0])
        b[1] = batch_labels
        list_loader_test.append(b)

    return (list_loader_train, list_loader_val, list_loader_test, og_loader_test)
