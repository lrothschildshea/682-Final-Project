#file to import data, relabel data, etc.

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data

data = get_CIFAR10_data()
for k,v in list(data.items()):
  print(('%s:' % k, v.shape))