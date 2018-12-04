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
NUM_EPOCHS = 1

models = [None]*50
optimizers = [None]*50
lltst = [None]*10
out = [None]*10
all_scores = [None]*10
best_models = [None]*10

for i in range(5):
    models[i], optimizers[i] = airplaneNetwork()

for i in range(5,10):
    models[i], optimizers[i] = automobileNetwork()

for i in range(10,15):
    models[i], optimizers[i] = birdNetwork()

for i in range(15, 20):
    models[i], optimizers[i] = catNetwork()

for i in range(20, 25):
    models[i], optimizers[i] = deerNetwork() 

for i in range(25, 30):
    models[i], optimizers[i] = dogNetwork()

for i in range(30, 35):
    models[i], optimizers[i] = frogNetwork()

for i in range(35, 40):
    models[i], optimizers[i] = horseNetwork()

for i in range(40, 45):
    models[i], optimizers[i] = shipNetwork()

for i in range(45, 50):
    models[i], optimizers[i] = truckNetwork()

m, o = allLabelsNetwork()

for i in range(NUM_LABELS):
    llt, llv, lltst[i], lbltst = relabelDataPyTorch((loader_train, loader_val, loader_test), i, device)
    best_acc = -1
    for j in range(5):
        print('Training Model #' + str(i+1) + '-' + str(j+1))
        idx = i*5 + j
        train_model(models[idx], optimizers[idx], device, llt, llv, epochs=NUM_EPOCHS)
        _, _, acc = check_accuracy(llv, models[idx], device, False)
        if acc.item() > best_acc:
            best_acc = acc.item()
            best_models[i] = models[idx]



print('Training 10 label network')
train_model(m, o, device, loader_train, loader_val, epochs=NUM_EPOCHS)

print('Checking Accuracy for All Labels Model')
check_accuracy(loader_test, m, device, False)

for i in range(NUM_LABELS):
    print('Checking Accuracy for Model #' + str(i+1))
    out[i], all_scores[i], _ = check_accuracy(lltst[i], best_models[i], device, False)

labels = combine_labels(out, all_scores, NUM_LABELS, device)
correct = (labels == lbltst).sum()
print(' %d / 10000 correct (%.2f)' % (correct, (float(correct)/100.0)))
count_collisions(out, NUM_LABELS, device)

end = time()
print()
print('Runtime: %d Minutes and %f Seconds' % (((end-start)//60), ((end-start)%60)))

# Makes noise to indicate completion
print('\a')
