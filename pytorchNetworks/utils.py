import torch.nn as nn
import numpy as np

def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
