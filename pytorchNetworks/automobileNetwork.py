import torch.nn as nn
import torch.optim as optim
from .utils import *

def automobileNetwork(learning_rate, channels):
    model = nn.Sequential(
        nn.Conv2d(channels[0], channels[1], (5, 5), padding=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(channels[2]*32*32, 2),
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return model, optimizer
