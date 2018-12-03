import torch.nn as nn
import torch.optim as optim
from .utils import *

def deerNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 32, (5, 5), padding=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*32*32, 2),
    )
    optimizer = optim.SGD(model.parameters(), lr=.01)

    return model, optimizer
