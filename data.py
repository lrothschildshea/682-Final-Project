#file to import data, relabel data, etc.

from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from data_utils import get_CIFAR10_data

#this method is currently not used - possibly remove in the future if not needed
def getData():
    data = get_CIFAR10_data()
    for k,v in list(data.items()):
        print(('%s:' % k, v.shape))
    return data