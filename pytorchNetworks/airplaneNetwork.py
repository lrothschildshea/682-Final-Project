import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def airplaneNetwork():
    model = nn.Sequential(
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3, 64, (5, 5), padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, (3, 3), padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 24, (5, 5), padding=2),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(24, 64, (7, 7), padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 32, (3, 3), padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 24, (5, 5), padding=2),
        nn.BatchNorm2d(24),
        nn.ReLU(),
        nn.Conv2d(24, 16, (5, 5), padding=2),
        Flatten(),
        nn.Linear(16384, 2),
    )

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=.0009)

    return model, optimizer
