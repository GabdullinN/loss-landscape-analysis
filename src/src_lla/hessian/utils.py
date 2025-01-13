# Vector operations used in hessian decomposition
# These functions are needed because torch stores model parameters as a list (layers) of tensors (weights)

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024


import torch
import numpy as np


def list_prod(v1, v2):
    """
    takes two lists of tensors as input and returns their inner product
    
    :v1 - list of tensors
    :v2 - list of tensors
    returns a single float
    """
    return sum([torch.sum(x1 * x2) for (x1, x2) in zip(v1, v2)]) 


def list_mul(v1,v2):
    """
    takes two lists of tensors as input and returns element-wise mult for all tensors
    
    :v1 - list of tensors
    :v2 - list of tensors
    returns list of tensors
    """
    
    return [torch.mul(x1,x2) for (x1, x2) in zip(v1, v2)]


def update_vect(vect, update, alpha=1):
    """
    updates a vect as vect = vect + update*alpha

    :vect - list of tensors
    :update - list of data used for update
    returns updated vect 
    """
    for i, _ in enumerate(vect):
        vect[i].data.add_(update[i] * alpha)
    return vect


def list_norm(v):
    """
    normalization of a list of tensors

    :v - list of tensors
    returns normalized v
    """
    s = torch.sqrt(list_prod(v, v))
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grads(model):
    """
    requires model
    returns parameters (weights) and gradients of the model
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def ortho_vect(vect, update):
    """
    creates a vector that is orthogonal to each vector in vect and normalizes it
    
    :vect - list of tensors
    returns another vect (list of tensors)
    """
    for param in update:
        vect = update_vect(vect, param, alpha=-list_prod(vect, param))
    return list_norm(vect)


def gen_mask_tens(mask_idx,params):
    """
    generate tensors of masks of shape params using mask index list
    
    :mask_idx - list of indexes
    :params - list of tensors
    returns list of tensors
    """
    
    mt = []
    for i,p in enumerate(params):
        el = torch.ones(p.shape).to(p.device) if i in mask_idx else torch.zeros(p.shape).to(p.device)
        mt.append(el)
        
    return mt
        

def hes_prod(grads,params,v,mask=None):
    """
    calcuates hessian vector product

    :grads - gradients of model parameters
    :params - model parameters
    :v - list of tensors
    :mask - list of tensors of zeros and ones 
    returns a product of hessian H with v
    """

    H = torch.autograd.grad(grads, params, grad_outputs=v, retain_graph=True)
    
    # zero all tensors but torch.ones in mask
    if mask is not None:
        H = list_mul(H,mask)
                             
    return H
