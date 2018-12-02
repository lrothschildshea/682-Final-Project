import torch.nn as nn
import torch.optim as optim
from .utils import *

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

def allLabelsNetwork():
    model = nn.Sequential(
        nn.Conv2d(3, 32, (5, 5), padding=2),
        nn.ReLU(),
        nn.Conv2d(32, 16, (3, 3), padding=1),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), stride=2),
        nn.Dropout(p=.05),
        nn.Conv2d(16, 12, (5, 5), padding=2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(3072, 10),
    )

    model.apply(init_weights)
    optimizer = optim.RMSprop(model.parameters(), lr=.00011)

    return model, optimizer
