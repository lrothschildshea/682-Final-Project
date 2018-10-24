import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

from utils import *

def airplaneNetwork(learning_rate, channels):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(channels[0]*32*32, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return model, optimizer
