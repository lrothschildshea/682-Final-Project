import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)

def allLabelsNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 24, (5, 5), padding=2),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(24, 32, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 64, (7, 7), padding=3),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(64, 32, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 24, (7, 7), padding=3),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(24, 32, (5, 5), padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 24, (5, 5), padding=2),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(24, 32, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(32, 64, (7, 7), padding=3),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(64, 32, (3, 3), padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 24, (7, 7), padding=3),
        nn.ReLU(),
        nn.Dropout(p=.1),
        nn.Conv2d(24, 32, (5, 5), padding=2),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d((2, 2)),
        Flatten(),
        nn.Linear(512, 10),
    )

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=.0009)

    return model, optimizer
