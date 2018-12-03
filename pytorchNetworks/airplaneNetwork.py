import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def airplaneNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1),
        nn.ELU(alpha = 1.0),
        nn.Conv2d(16, 32, 5, 1, 2),
        nn.ELU(alpha = 1.0),
        nn.BatchNorm2d(32),
        nn.Dropout(0.2),
        nn.MaxPool2d(2,2),
        nn.Conv2d(32, 24, 5, 1, 2),
        nn.ELU(alpha = 1.0),
        nn.Conv2d(24, 64, 5, 1, 2),
        nn.ELU(alpha = 1.0),
        nn.BatchNorm2d(64),
        nn.Dropout(0.2),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 64, 7, 1, 3),
        nn.ELU(alpha = 1.0),
        nn.Conv2d(64, 128, 9, 1, 4),
        nn.ELU(alpha = 1.0),
        nn.BatchNorm2d(128),
        nn.Dropout(0.2),
        nn.MaxPool2d(2,2),
        Flatten(),
        nn.Linear(2048, 64),
        nn.BatchNorm1d(64),
        nn.ELU(alpha = 1.0),
        nn.Linear(64, 2)
    )

    model.apply(init_weights)
    optimizer = optim.RMSprop(model.parameters(), lr=0.002)

    return model, optimizer
