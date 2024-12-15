# Example loader for a small fully-connected neural network on MNIST

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.datasets as datasets

from src_lla.loss_landscapes.metrics.metric import Metric
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper

# original MLPSmall implementation by marcellodebernardi distributed under MIT license
# https://github.com/marcellodebernardi/loss-landscapes/

# training hyperparameters
IN_DIM = 28 * 28
OUT_DIM = 10
BATCH_SIZE = 2048

class MLPSmall(torch.nn.Module):
    """ Fully connected feed-forward neural network with one hidden layer. """
    def __init__(self, x_dim, y_dim):
        super().__init__()
        self.linear_1 = torch.nn.Linear(x_dim, 32)
        self.linear_2 = torch.nn.Linear(32, y_dim)

    def forward(self, x):
        h = F.relu(self.linear_1(x))
        return F.softmax(self.linear_2(h), dim=1)



class Flatten(object):
    """ Transforms a PIL image to a flat numpy array. """
    def __call__(self, sample):
        return np.array(sample, dtype=np.float32).flatten()              



def ModelInit(N_CLASSES=None, weight_path=None, device='cpu'):
    
    # init model
    model = MLPSmall(IN_DIM, OUT_DIM)
    model.to(device)
    
    # load model weights
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path,map_location=device))
        
    #model.eval()
    
    return model
    


def CustomLoader(data_path = None, batch_size = BATCH_SIZE, wrap=False, shuffle=False):
    
    mnist_train = datasets.MNIST(root='../data', train=True, download=True, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=shuffle,drop_last=True)
    
    if wrap:
        train_loader = LoaderWrapper(train_loader)
    
    return train_loader



class LoaderWrapper():
    def __init__(self, train_loader: DataLoader):

        self.loader = train_loader
        self.iterator = iter(self.loader)
    
    def __iter__(self):
        
        return self
    
    def __next__(self):
        
        output = self.iterator.__next__()
        
        # modify to get real x,y from your custom output!
        x, y = output[0] 
        
        return x, y   
    


class CustomLoss(Metric):
    def __init__(self, inputs: torch.Tensor, target: torch.Tensor, device):
        super().__init__()
        
        # original part
        self.inputs = inputs
        self.target = target
        self.device = device
        
        # custom part
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        

    def __call__(self, model_wrapper: ModelWrapper, model=None, batch=None, use_wrapper=True, return_pred=False):

        # modify if inputs and/or targets are more complex
        if batch is None:
            inputs = self.inputs
            target = self.target
        else:
            inputs = batch[0]
            target = batch[1]

        if use_wrapper:
            pred = model_wrapper.forward(inputs.to(self.device))
        else:
            # assumes that model is provided
            pred = model(inputs.to(self.device))
        
        targets = target.type(torch.LongTensor)
        loss = self.loss_fn(pred, targets.to(self.device))
        
        if not return_pred: 
            return loss.item()
        else:
            return loss, pred
