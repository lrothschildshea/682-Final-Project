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
from data import relabelDataPyTorch

start = time()

NUM_TRAIN = 49000

transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

cifar10_train = dset.CIFAR10('./dataset', train=True, download=True, transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./dataset', train=True, download=True, transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./dataset', train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

models = [None]*10
optimizers = [None]*10

models[0], optimizers[0] = airplaneNetwork(.01, [3, 32, 32])
models[1], optimizers[1] = automobileNetwork(.1, [3])
models[2], optimizers[2] = birdNetwork(.1, [3])
models[3], optimizers[3] = catNetwork(.1, [3])
models[4], optimizers[4] = deerNetwork(.1, [3])
models[5], optimizers[5] = dogNetwork(.1, [3])
models[6], optimizers[6] = frogNetwork(.1, [3])
models[7], optimizers[7] = horseNetwork(.1, [3])
models[8], optimizers[8] = shipNetwork(.1, [3])
models[9], optimizers[9] = truckNetwork(.1, [3])

lt = [None]*10
llt = [None]*10
llv = [None]*10
lltst = [None]*10

for i in range(10):
    print('Training Model #' + str(i+1))
    lt[i], llt[i], llv[i], lltst[i] = relabelDataPyTorch((loader_train, loader_val, loader_test), i)
    train_model(models[i], optimizers[i], device, dtype, llt[i], llv[i], i) #perhaps redo the parameters once we figure out how the data will be imported

for i in range(10):
    print('Checking Accuracy for Model #' + str(i+1))
    check_accuracy(lltst[i], models[i], device, dtype, i)

end = time()
print()
print('Runtime: %d Minutes and %f Seconds' % (((end-start)//60), ((end-start)%60)))
