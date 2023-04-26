import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import dataset
import joblib

# Build the network
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

# Define the Loss
criterion = nn.NLLLoss()
images, labels = next(iter(dataset.trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

# Training
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 18
iter=np.arange(0,epochs)
losses=np.zeros(epochs)
for e in range(epochs):
    running_loss = 0
    for images, labels in dataset.trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        # Training pass
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        #This is where the model learns by backpropagating
        loss.backward()
        #And optimizes its weights here
        optimizer.step()
        running_loss += loss.item()
    losses[e]=running_loss
    print("Epoch {} - Training loss: {}".format(e, running_loss/len(dataset.trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

joblib.dump(model,"model/rec_model.dat")
print('Model saved : {}'.format("model/rec_model.dat"))

plt.scatter(iter,losses)
plt.xlabel("Iteration")
plt.ylabel("Loss*1000")
plt.xlim([0,epochs])
plt.savefig(
    "plots/loss_curvs.png",
    format="png",
    dpi=1000,
    pad_inches=0,
    bbox_inches="tight",
)
plt.show()


