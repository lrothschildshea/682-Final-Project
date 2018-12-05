import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from data import *

def visualizeAnImage(label, loader_train):
    found = False
    for i in loader_train:
        c = i[1].numpy()
        arr = i[0].numpy()
        for t,j in enumerate(arr):
            if c[t] == label:
                produceSingleImage(j, c[t])
                found = True
                break
        if found == True:
            break

def produceSingleImage(inp, label):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)

    img = compressImage(inp)

    print("shape: ", img.shape)
    print("class: ", classes[label])

    plt.imshow(img, interpolation = 'bicubic')
    plt.axis('off')
    plt.show()

def compressImage(img):
    R = img[0]
    G = img[1]
    B = img[2]

    return np.dstack((R,G,B))

def imageGrid(loader, sample_per_class):
    #data = collectImages(loader, sample_per_class)
    data = loader
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    for s,c in enumerate(classes):
        class_array = data[s]
        for t,i in enumerate(class_array):
            plt_class = s * len(class_array) + t + 1
            plt.subplot(num_classes, sample_per_class, plt_class)
            comp = compressImage(i)
            plt.imshow(comp)
            plt.axis('off')
    plt.show()

def collectImages(loader, sample_per_class):
    all_data = np.full(10,None)
    for this_class in range(10):
        class_array = np.full(sample_per_class, None)
        temp_sample_per_class = sample_per_class
        for i in loader_train:
            c = i[1].numpy()
            for t,j in enumerate(i[0].numpy()):
                if c[t] == this_class:
                    temp_sample_per_class -= 1
                    class_array[temp_sample_per_class] = j
                    if temp_sample_per_class == 0:
                        break
            if temp_sample_per_class == 0:
                        break
        all_data[this_class] = class_array
        if temp_sample_per_class == 0:
            continue
    return all_data

#LABEL = -1
#while(LABEL < 0 or LABEL > 9):
    #try:
        #LABEL = int(input("Choose a Class: "))
        #if LABEL < 0 or LABEL > 9:
            #print("Please enter an Integer from 0-9.")
    #except ValueError:
        #print("Please enter an Integer from 0-9.")
#loader_train, loader_val, loader_test = getDataPyTorch()
#imageGrid(loader_train, 5)
#visualizeAnImage(LABEL, loader_train)
