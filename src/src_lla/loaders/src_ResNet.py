
# Example loader for ResNet on ImageNet
# ImageNet dataset is NOT provided with this library

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

import os

import torchvision
from torchvision import transforms

from src_lla.loss_landscapes.metrics.metric import Metric
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper


def ModelInit(N_CLASSES=None, weight_path=None, device='cpu'):
    
    # init model
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    # load model weights
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path,map_location=device))
    #model.eval()
    model.to(device)
    
    return model
    


def CustomLoader(data_path = 'datasets/ImageNet/', batch_size = 64, wrap=False, shuffle=False):

    transform = transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]) 

    dataset = torchvision.datasets.ImageFolder(root=data_path, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
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