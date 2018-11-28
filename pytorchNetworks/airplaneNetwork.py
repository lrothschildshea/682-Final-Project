import torch.nn as nn
import torch.optim as optim
from .utils import *

def airplaneNetwork(learning_rate, channels):
    model = nn.Sequential(
        nn.Conv2d(3, 32, (5, 5), padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 32, (3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        nn.Conv2d(32, 16, (7, 7), padding=3),
        nn.ReLU(),
        nn.Conv2d(16, 24, (5, 5), padding=2),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        nn.Conv2d(24, 64, (3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 32, (7, 7), padding=3),
        nn.ReLU(),
        nn.MaxPool2d((2,2), stride=2),
        Flatten(),
        nn.Linear(512, 2),
    )
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return model, optimizer
