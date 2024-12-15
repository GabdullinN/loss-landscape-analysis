
# Example loader for LeNet on MNIST

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

import os

import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

from src_lla.loss_landscapes.metrics.metric import Metric
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper

# Lenet implementation by lychengrex under MIT License
# original repo: https://github.com/lychengrex/LeNet-5-Implementation-Using-Pytorch/tree/master

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)        


def ModelInit(N_CLASSES=None, weight_path=None, device='cpu'):
    
    # init model
    model = LeNet()
    
    # load model weights
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path,map_location=device))
    #model.eval()
    model.to(device)
    
    return model
    

def CustomLoader(data_path = None, batch_size = 2048, wrap=False, shuffle=False,drop_last=True):
    
    mnist_train = datasets.MNIST(root='../data', train=True, download=True,transform=ToTensor()) #, transform=Flatten())
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=2048, shuffle=shuffle,drop_last=drop_last)
    
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
