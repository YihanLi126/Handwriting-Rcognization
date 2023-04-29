import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import dataset

dataiter = iter(dataset.trainloader)
images, labels = dataiter.next()

print("Images Shape: ",images.shape)
print("Labels Shape: ",labels.shape)

# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
    plt.savefig(
    "plots/dataset_vis.png",
    format="png",
    dpi=1000,
    pad_inches=0,
    bbox_inches="tight",
)

plt.show()