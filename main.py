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
from pytorchNetworks.allLabelsNetwork import *
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
NUM_EPOCHS = 20

models = [None]*10
optimizers = [None]*10
lltst = [None]*10
out = [None]*10
all_scores = [None]*10

models[0], optimizers[0] = airplaneNetwork()        #airplane
models[1], optimizers[1] = automobileNetwork()      #automobile
models[2], optimizers[2] = birdNetwork()         #bird
models[3], optimizers[3] = catNetwork()        #cat
models[4], optimizers[4] = deerNetwork()         #deer
models[5], optimizers[5] = dogNetwork()        #dog
models[6], optimizers[6] = frogNetwork()         #frog
models[7], optimizers[7] = horseNetwork()         #horse
models[8], optimizers[8] = shipNetwork()        #ship
models[9], optimizers[9] = truckNetwork()        #truck
m, o = allLabelsNetwork()

for i in [2]:
    print('Training Model #' + str(i+1))
    llt, llv, lltst[i], lbltst = relabelDataPyTorch((loader_train, loader_val, loader_test), i, device)
    train_model(models[i], optimizers[i], device, llt, llv, epochs=NUM_EPOCHS)

'''
print('Training 10 label network')
train_model(m, o, device, loader_train, loader_val, epochs=NUM_EPOCHS)

print('Checking Accuracy for All Labels Model')
check_accuracy(loader_test, m, device, False)
'''
for i in [2]:
    print('Checking Accuracy for Model #' + str(i+1))
    out[i],all_scores[i] = check_accuracy(lltst[i], models[i], device, False)

labels = combine_labels(out, all_scores, NUM_LABELS, device)
correct = (labels == lbltst).sum()
print('Got %d / 10000 correct (%.2f)' % (correct, (float(correct)/100.0)))
count_collisions(out, NUM_LABELS, device)

end = time()
print()
print('Runtime: %d Minutes and %f Seconds' % (((end-start)//60), ((end-start)%60)))

# Makes noise to indicate completion
print('\a')
