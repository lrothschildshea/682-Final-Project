import torch.nn as nn
import torch.optim as optim
from .utils import *

def horseNetwork(learning_rate, channels):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(channels[0]*32*32, 2),
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return model, optimizer
