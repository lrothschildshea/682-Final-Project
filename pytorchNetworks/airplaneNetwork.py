import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

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
        nn.Conv2d(32, 64, (9, 9), padding=4),
        nn.ReLU(),
        nn.Conv2d(64, 128, (7, 7), padding=3),
        nn.ReLU(),
        nn.MaxPool2d((4,4), stride=2),
        Flatten(),
        nn.Linear(128, 2),
    )

    model.apply(init_weights)
    optimizer = optim.RMSprop(model.parameters(), lr=.0005)

    return model, optimizer
