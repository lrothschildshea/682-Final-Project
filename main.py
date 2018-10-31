from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
from time import time

from pytorchNetworks.airplaneNetwork import *
from pytorchNetworks.automobileNetwork import *
from pytorchNetworks.birdNetwork import *
from pytorchNetworks.catNetwork import *
from pytorchNetworks.deerNetwork import *
from pytorchNetworks.dogNetwork import *
from pytorchNetworks.frogNetwork import *
from pytorchNetworks.horseNetwork import *
from pytorchNetworks.shipNetwork import *
from pytorchNetworks.truckNetwork import *
from main_utils import *
from data import relabelDataPyTorch, getDataPyTorch

start = time()

loader_train, loader_val, loader_test = getDataPyTorch()

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

NUM_LABELS = 10

models = [None]*10
optimizers = [None]*10
lltst = [None]*10
out = [None]*10

models[0], optimizers[0] = airplaneNetwork(.01, [3, 32, 32])
models[1], optimizers[1] = automobileNetwork(.01, [3, 32, 32])
models[2], optimizers[2] = birdNetwork(.01, [3, 32, 32])
models[3], optimizers[3] = catNetwork(.01, [3, 32, 32])
models[4], optimizers[4] = deerNetwork(.01, [3, 32, 32])
models[5], optimizers[5] = dogNetwork(.01, [3, 32, 32])
models[6], optimizers[6] = frogNetwork(.01, [3, 32, 32])
models[7], optimizers[7] = horseNetwork(.01, [3, 32, 32])
models[8], optimizers[8] = shipNetwork(.01, [3, 32, 32])
models[9], optimizers[9] = truckNetwork(.01, [3, 32, 32])

for i in range(NUM_LABELS):
    print('Training Model #' + str(i+1))
    llt, llv, lltst[i], lbltst = relabelDataPyTorch((loader_train, loader_val, loader_test), i, device)
    train_model(models[i], optimizers[i], device, llt, llv, i)

for i in range(NUM_LABELS):
    print('Checking Accuracy for Model #' + str(i+1))
    out[i] = check_accuracy(lltst[i], models[i], device, i, False)

labels = torch.zeros(10000, dtype=torch.long)
labels = labels.to(device=device, dtype=torch.long)
print('Combining Labels')
for i in range(NUM_LABELS):
    for j in range(10000):
        if out[i][j] == 1:
            labels[j] = i

correct = (labels == lbltst).sum()

print('Got %d / 10000 correct (%.2f)' % (correct, (float(correct)/100.0)))

end = time()
print()
print('Runtime: %d Minutes and %f Seconds' % (((end-start)//60), ((end-start)%60)))
