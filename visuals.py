import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from data import *

def visualizeAnImage(label):
    loader_train, loader_val, loader_test = getDataPyTorch()
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    found = False
    for i in loader_train:
        c = i[1].numpy()
        arr = i[0].numpy()
        for t,j in enumerate(arr):
            if c[t] == label:
                R = j[0]
                G = j[1]
                B = j[2]
                
                img = np.dstack((R,G,B))

                print("shape: ", img.shape)
                print("class: ", classes[c[t]])

                plt.imshow(img, interpolation = 'bicubic')
                plt.axis('off')

                found = True
                break
        if found == True:
            break
    plt.show()

LABEL = -1
while(LABEL < 0 or LABEL > 9):
    try:
        LABEL = int(input("Choose a Class: "))
        if LABEL < 0 or LABEL > 9:
            print("Please enter an Integer from 0-9.")
    except ValueError:
        print("Please enter an Integer from 0-9.")
visualizeAnImage(LABEL)
