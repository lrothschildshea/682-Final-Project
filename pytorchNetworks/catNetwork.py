import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

def catNetwork(learning_rate, channels):
    model = nn.Sequential()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return model, optimizer
