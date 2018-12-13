import torch
import torch.nn.functional as F

import numpy as np
from visuals import imageGrid, produceSingleImage, imageStrip

def check_accuracy(loader, model, device, train, c = None, check_val = False):  
    data_fp = None
    data_fn = None
    count_fp = None
    count_fn = None
    if train:
        print('        Checking accuracy on validation set')
    elif check_val:
        data_fp = np.full(5,None)
        data_fn = np.full(5,None)
        count_fp = 0
        count_fn = 0
        print('    Checking accuracy on validation set')
    else:
        data_fp = np.full(5,None)
        data_fn = np.full(5,None)
        count_fp = 0
        count_fn = 0
        print('    Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()

    all_scores = torch.tensor([])
    out = torch.tensor([])
    all_scores = all_scores.to(device = device, dtype = torch.float)
    out = out.to(device=device, dtype=torch.long)
    with torch.no_grad():

        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            _, preds = scores.max(1)
            if not train:
                out = torch.cat((out, preds), 0)
                all_scores = torch.cat((all_scores, scores), 0)
                if (count_fp < 5 or count_fn < 5) and c != None:
                    temp_x = x.cpu().numpy()
                    temp_y = y.cpu().numpy()
                    temp_preds = preds.cpu().numpy()
                    for t in range(temp_y.size):
                        #false positive
                        if count_fp < 5 and temp_y[t] == 0 and temp_preds[t] == 1:
                            data_fp[count_fp] = temp_x[t]
                            count_fp += 1
                        #false negative
                        if count_fn < 5 and temp_y[t] == 1 and temp_preds[t] == 0:
                            data_fn[count_fn] = temp_x[t]
                            count_fn += 1
                        if count_fp == 5 and count_fn == 5:
                            break

            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples

        if train:
            print('         %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        else:
            print('     %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            print()
            return out, all_scores, data_fp, data_fn, num_correct

def train_model(model, optimizer, device, loader_train, loader_val, epochs=1):
    model = model.to(device=device)
    for e in range(epochs):
        print('    epoch #' + str(e + 1))
        print()
        for itn, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if itn % 175 == 0:
                print('        Iteration %d, loss = %.4f' % (itn, loss.item()))
                check_accuracy(loader_val, model, device, True)
                print()

def combine_labels(labelset, scoreset, num_labels, device):
    labels = torch.ones(10000, dtype=torch.long) * 10
    labels = labels.to(device=device, dtype=torch.long)
    print('Combining Labels')
    for i in range(num_labels):
        for j in range(10000):
            if labelset[i][j] == 1:
                if labels[j] == 10:
                    labels[j] = i
                elif  torch.abs(scoreset[labels[j]][j][0] - scoreset[labels[j]][j][1]) < torch.abs(scoreset[i][j][0] - scoreset[i][j][1]):
                    labels[j] = i
    return labels

def combine_labels_2(labelset, num_labels, device):
    labels = torch.ones(10000, dtype=torch.long) * 10
    labels = labels.to(device=device, dtype=torch.long)
    print('Combining Labels')
    for i in range(num_labels):
        for j in range(10000):
            if labelset[i][j] == 1:
                labels[j] = i

    return labels

def combine_labels_3(labelset, scoreset, num_labels, device):
    labels = torch.ones(10000, dtype=torch.long) * 10
    labels = labels.to(device=device, dtype=torch.long)
    print('Combining Labels')
    for i in range(num_labels):
        for j in range(10000):
            if labelset[i][j] == 1:
                if labels[j] == 10:
                    labels[j] = i
                elif  np.exp(scoreset[labels[j]][j][1])/(np.exp(scoreset[labels[j]][j][1]) + np.exp(scoreset[labels[j]][j][0])) < np.exp(scoreset[i][j][1])/(np.exp(scoreset[i][j][0]) + np.exp(scoreset[i][j][1])):
                    labels[j] = i
    return labels

def count_collisions(labelset, num_labels, device, loader_test):
    print('Counting label collisions and unlabled images')
    labels = torch.zeros(10000, dtype=torch.long)
    labels = labels.to(device=device, dtype=torch.long)

    collision = 0
    unlabeled = 0
    iterator = []
    classes = np.zeros(10)
    num = np.ones(10)

    for i in range(num_labels):
        labels += labelset[i]
    for i in range(10000):
        if labels[i] > 1:
            collision += labels[i]-1
        if labels[i] == 0:
            unlabeled += 1
            for t,(x, y) in enumerate(loader_test):
                temp_x = x.cpu().numpy()
                temp_y = y.cpu().numpy()
                for s,c in enumerate(temp_y):
                    img = temp_x[s]
                    if t*64 + s == i:
                        classes[c] += 1
                        if len(iterator) < 10:
                            if num[c] == 1:
                                num[c] -= 1
                                iterator.append((img, c))
    
    print("%d Label Collsions" % collision)
    print("%d unlabeled" % unlabeled)
    titles = ['Airplanes', 'Automobiles', 'Birds', 'Cats', 'Deer', 'Dogs', 'Frogs', 'Horses', 'Ships', 'Trucks']
    for t,title in enumerate(titles):
        print("%i/%i = %3.2f of unlabeled images were %s" % (classes[t], unlabeled, classes[t]/unlabeled, title))
    
    return iterator
