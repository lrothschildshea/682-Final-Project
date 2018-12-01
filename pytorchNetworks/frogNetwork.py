import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def frogNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 64, (3, 3), padding=2),
        nn.LeakyReLU(.001),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(64, 48, (5, 5), padding=3),
        nn.LeakyReLU(.001),
        nn.Dropout(p=.3),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(48, 32, (3, 3), padding=2),
        nn.LeakyReLU(.001),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(32, 64, (3, 3), padding=2),
        nn.LeakyReLU(.001),
        nn.Dropout(p=.2),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(64, 32, (3, 3), padding=2),
        nn.LeakyReLU(.001),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(32, 24, (5, 5), padding=3),
        nn.LeakyReLU(.001),
        nn.Dropout(p=.1),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Conv2d(24, 48, (5, 5), padding=3),
        Flatten(),
        nn.Linear(768, 2),
    )

    model.apply(init_weights)
    optimizer = optim.RMSprop(model.parameters(), lr=.0007)

    return model, optimizer
