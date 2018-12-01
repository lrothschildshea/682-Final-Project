import torch.nn as nn
import torch.optim as optim
from .utils import *

def catNetwork(learning_rate, channels):
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.BatchNorm2d(32),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose2d(32, 64, 5, 1, 2),
        nn.BatchNorm2d(64),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2,2),
        nn.Conv2d(64, 64, 5, 1, 2),
        nn.BatchNorm2d(64),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.01),
        nn.ConvTranspose2d(64, 128, 7, 1, 3),
        nn.BatchNorm2d(128),
        nn.Dropout(0.2),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2,2),
        nn.Tanh(),
        Flatten(),
        nn.Linear(8192, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 64)
    )
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    return model, optimizer
