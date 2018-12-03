import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight)

def catNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 16, (3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, (5,5), padding=2),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        nn.Conv2d(32, 24, (7,7), padding=3),
        nn.ReLU(),
        nn.Conv2d(24, 64, (5,5), padding=2),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        nn.Conv2d(64, 48, (3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(48, 32, (3,3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        Flatten(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    )
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    return model, optimizer
