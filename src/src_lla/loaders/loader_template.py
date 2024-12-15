# Loader template to be imported into lla_train.py or lla_eval.py
# You can account for any specifics of you data_loader, model, and loss evaluation via model inference here

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import os
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from src_lla.loss_landscapes.metrics.metric import Metric
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper


def ModelInit(N_CLASSES=10, weight_path=None, device='cpu'):
    
    '''
    initinitalize your model with this function
    must return torch model onject with device specified
    add extra init params if necessary, but the ones listed below must be included
    
    :N_CLASSES - number of classes for classifier
    :weight_path - path to weights, random init if None
    :deivce - 'cpu' or 'cuda'
    '''

    # modify to load your data and do necessary preprocessing
    model = _init_your_model(N_CLASSES)

    # do not modify below this line
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path,map_location=device))

    model.to(device)
    return model
    

def CustomLoader(data_path='path_to_data', batch_size=32, wrap=False, shuffle=False):
    
    '''
    load your data here, do preprocessing if necessary
    returns torch DataLoader object which in turn MUST return inputs,labels pair for loss eval
    !!! if DataLoader has different outputs because of data structure, wrap it with LoaderWrapper !!!

    :data_path - path to data directory
    :batch_size - batch_size for train_loader
    :wrap - whether to use LoaderWrapper or not
    '''
    
    # modify to load your data and do necessary preprocessing
    train_data = _load_your_data(data_path)

    # do not modify below this line
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    if wrap:
        train_loader = LoaderWrapper(train_loader)
    return train_loader



class LoaderWrapper():
    def __init__(self, train_loader: DataLoader):

        '''
        wraps train_loader to convert its output to inputs, labels format
        '''
        # do not modify
        self.loader = train_loader
        self.iterator = iter(self.loader)
        self.batch_size = self.loader.batch_size
    
    def __iter__(self):
        # do not modify
        return self
    
    def __len__(self):
        # do not modify
        return len(self.loader)
    
    def __next__(self):
        # do not modify
        output = self.iterator.__next__()
        
        #modify to get inputs,labels from your train_loader output!
        inputs, labels = output[0] 

        # do not modify
        return inputs, labels



class CustomLoss(Metric):
    """ 
    Specify your model inference and loss computation. 
    Models of any complexity can be described here as long as output format is maintained
    
    inits automatically with inputs, labels data from train_loader, requires the device
    specify loss function in __init__ and everything necesary for its calculation in __call__
    write model inference in __call__. model_wrapper.forward supports flags of original model.forward(x, flags)

    returns either a single float loss.item() for landscapes or torch objects loss, preds for training and other purposes
    """
    
    def __init__(self, inputs, target, device='cpu'):
        super().__init__()

        '''
        :x - input tensors
        :y - labels
        :device - 'cpu' or 'cuda'
        '''
        
        # do not modify
        self.inputs = inputs 
        self.target = target
        self.device = device
        
        # insert your loss function here (optional)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
               

    def __call__(self, model_wrapper: ModelWrapper, model=None, batch=None, use_wrapper=True, return_pred=False):

        '''
        write your model inference and loss evaluation procedure here
        
        :model_wrapper - wrapped model object used for landscape evaluation
        :model - original model object used for training  
        :batch - batch of data for evaluation; if provided overwrites x,y specified during init
        :use_wrapper - wheather to use model_wrapper (see above); if False the model will be used
        :return_pred - if True returns torch objects for training, otherwise returns only loss value for landscape evaluation

        To use CustomLoss for training:
        use_wrapper=False and model != None allows to pass grads to model so CustomLoss can be used during training
        when bacth is provided it will be used instead of self.x,self.y which is needed for training.
        return_pred = True returns loss as torch obect connected to computational graph so loss.backward() can work; also returns preds
        '''

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
