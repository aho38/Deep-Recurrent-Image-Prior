from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from matplotlib.pyplot import imshow, pause

"""
By Alex Ho AND ALEX HO ONLY
ASSISTED BY NO ONE
"""

def getData(datasets,batch_size=1):

    if datasets == 'MNIST':
        train_loader = DataLoader(MNIST(download=True, root=".", transform=ToTensor(), train=True),
                batch_size=batch_size, shuffle=True)

        dataiter = iter(train_loader)
        images, labels = dataiter.next()
    return images

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    return img_noisy_np

def getPlot(img_np):
    grid = torchvision.utils.make_grid(img_np,nrow=2,padding=0)
    imshow(grid[1,:,:],cmap=plt.get_cmap('gray'))


def optim(num_epochs, input, true, net , train, n_steps, optimizer, total_step = 1):
    """Optimization of our funciton

    Args:
        num_epochs: number of epochs
        input: our input to our network
        true: true label that is expected
        net: our network
        train: training algorithm that spit out loss value and our output image
    """
    for epoch in range(num_epochs):

        sumloss = 0

        for i in range(total_step):
            out = net(input,n_steps)
            out.size()
            loss1, output = train(out,true)
            sumloss += loss1

        loss = sumloss/total_step

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss))

        getPlot(torch.tensor(out))

class RNN(nn.Module):
    """docstring for RNN."""

    def __init__(self, n_neurons, gamma=0.001, epsilon=0.01):
        super(RNN, self).__init__()
        self.W = nn.Parameter(torch.randn(n_neurons, n_neurons))
        self.b = nn.Parameter(torch.randn(1,n_neurons))
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_neurons = n_neurons

    def forward(self, X, n_steps=30):
        output = []
        self.ht = torch.zeros(X.shape[0], self.n_neurons)
        states = []
        states.append(self.ht)

        for i in range(n_steps):
            alpha = torch.mm(states[i],self.W)+self.b
            self.ht = states[i] + self.epsilon*torch.tanh(alpha)
            states.append(self.ht)
            out = self.ht
        return out
