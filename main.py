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
import numpy as np
from visuals import imageGrid, imageStrip
import copy

start = time()

loader_train, loader_val, loader_test = getDataPyTorch()

loader_test = copy.deepcopy(loader_test)

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('Device:', device)

NUM_LABELS = 10
NUM_EPOCHS = 10
NUM_TRAINING = 1

if NUM_EPOCHS < 1:
    NUM_EPOCHS = 1
if NUM_TRAINING < 1:
    NUM_TRAINING = 1

models = [None]*(10*NUM_TRAINING)
optimizers = [None]*(10*NUM_TRAINING)
lltst = [None]*10
out = [None]*10
all_scores = [None]*10
best_models = [None]*10

print()
print('Creating Models')
for i in range(NUM_TRAINING):
    models[i], optimizers[i] = airplaneNetwork()
    models[NUM_TRAINING + i], optimizers[NUM_TRAINING + i] = automobileNetwork()
    models[2*NUM_TRAINING + i], optimizers[2*NUM_TRAINING + i] = birdNetwork()
    models[3*NUM_TRAINING + i], optimizers[3*NUM_TRAINING + i] = catNetwork()
    models[4*NUM_TRAINING + i], optimizers[4*NUM_TRAINING + i] = deerNetwork() 
    models[5*NUM_TRAINING + i], optimizers[5*NUM_TRAINING + i] = dogNetwork()
    models[6*NUM_TRAINING + i], optimizers[6*NUM_TRAINING + i] = frogNetwork()
    models[7*NUM_TRAINING + i], optimizers[7*NUM_TRAINING + i] = horseNetwork()
    models[8*NUM_TRAINING + i], optimizers[8*NUM_TRAINING + i] = shipNetwork()
    models[9*NUM_TRAINING + i], optimizers[9*NUM_TRAINING + i] = truckNetwork()
m, o = allLabelsNetwork()

for i in range(NUM_LABELS):
    llt, llv, lltst[i], lbltst = relabelDataPyTorch((loader_train, loader_val, loader_test), i, device)
    best_acc = -1
    for j in range(NUM_TRAINING):
        print('Training Model #' + str(i+1) + '-' + str(j+1))
        idx = i*NUM_TRAINING + j
        train_model(models[idx], optimizers[idx], device, llt, llv, epochs=NUM_EPOCHS)
        _, _, _, _,acc = check_accuracy(llv, models[idx], device, False)
        if acc.item() > best_acc:
            best_acc = acc.item()
            best_models[i] = models[idx]
'''
print('Training 10 label network')
train_model(m, o, device, loader_train, loader_val, epochs=NUM_EPOCHS)

print()
print()
print()
print()
print('Checking Accuracy for All Labels Model')
check_accuracy(loader_test, m, device, False)
'''
data_fp = [None]*10
data_fn = [None]*10
for i in range(NUM_LABELS):
    print('Checking Accuracy for Model #' + str(i+1))
    out[i],all_scores[i], data_fp[i], data_fn[i], _ = check_accuracy(lltst[i], models[i], device, False, c = i)



labels = combine_labels(out, all_scores, NUM_LABELS, device)
correct = (labels == lbltst).sum()
print(' %d / 10000 correct (%.2f)' % (correct, (float(correct)/100.0)))
iterator = count_collisions(out, NUM_LABELS, device, loader_test)

##UNCOMMENT NEXT LINES FOR VISUALS
imageGrid(data_fp, 5)
imageGrid(data_fn, 5)
imageStrip(iterator)

end = time()
print()
print('Runtime: %d Minutes and %f Seconds' % (((end-start)//60), ((end-start)%60)))

# Makes noise to indicate completion
print('\a')
