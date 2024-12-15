# Functions to extract vectors from optimizer and hessian; the main analysis and visualization function viz_lla

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import torch
from copy import deepcopy
import warnings
from src_lla.hessian import hessian_calc
from src_lla.hessian.viz import eval_save_esd
from src_lla.loss_landscapes.model_interface.model_parameters import ModelParameters
from src_lla.loss_landscapes.viz import eval_save_landscape

# TODO: take this from config or allow user input with flag
default_viz_dir = 'viz_results'
default_res_dir = 'analysis_results'


def vec_from_optim(optimizer, lr = 1):

    """
    extracts m1 and m2 vectors from adam optimizer and wraps them for landscape plotting

    :optimizer - adam optimizer object
    returns wrapped direction vectors a1 and a2
    """
    
    # assumes all model parameters are in optim parameters

    m1 = []
    m2 = []
    for i in range(len(optimizer.state_dict()['state'])):
        m1.append(optimizer.state_dict()['state'][i]['exp_avg'] * lr)
        m2.append(optimizer.state_dict()['state'][i]['exp_avg_sq'] * lr)
    
    a1 = ModelParameters(m1)
    a2 = ModelParameters(m2)
    
    return a1, a2


def vec_model_optim(optimizer, model, lr = 1):

    """
    extracts m1 and m2 vectors from adam optimizer and wraps them for landscape plotting
    allows for missing parameters in optimizer (running av of batchnorm etc)
    
    :optimizer - adam optimizer object
    returns wrapped direction vectors a1 and a2
    """

    m1 = []
    m2 = []
    for module, param in model.named_parameters():
        m1.append(torch.zeros(size=param.size(), dtype=param.dtype).to(param.device))
        m2.append(torch.zeros(size=param.size(), dtype=param.dtype).to(param.device))

    for i, key in enumerate(optimizer.state_dict()['state'].keys()):
        m1[key] = optimizer.state_dict()['state'][key]['exp_avg'] * lr
        m2[key] = optimizer.state_dict()['state'][key]['exp_avg_sq'] * lr
    
    a1 = ModelParameters(m1)
    a2 = ModelParameters(m2)
    
    return a1, a2


def vec_H_eigenvects(hessian):

    """
    extracts top two eigenvectors from hessian and wraps them for landscape plotting

    :hessian - hessian decomposition object
    returns wrapped direction vectors a1 and a2
    """
    
    top_eigenvalues, top_eigenvectors = hessian.eigs_calc(top_n=2)
    
    a1 = ModelParameters(top_eigenvectors[0])
    a2 = ModelParameters(top_eigenvectors[1])
    
    return a1, a2


def viz_lla(model, metric, device, dist=1, steps=40, num_plots=4, num_per_plot=2, axes='random', normalization=None, order=2, cur_name='', mode='add', 
            b_sqrt=True, viz_dev=False, cap_loss=None, raa=None, viz_dir=None, eval_hessian = False, 
            optimizer=None, to_viz=False,to_save=True, return_loss=False,res_dir=default_res_dir,calc_crit=False,n_kh=0.5):

    """
    set-up function for hessian and loss landscape analysis with concequent visualization
    calculates necessary parameters for eval_save_landscape to evaluate and plot loss landscape

    :model - torch model object
    :metric - custom loss metric object
    :device - 'cpu' or 'cuda'
    :dist - distance in weight space
    :steps - size of landscape map
    :num_plots - number of plots for random axes
    :num_per_plot - number of plots in one horizontal line
    :axes - type of axes used to plot loss landscape
    :normalization - type of normalization for direction vectors
    :order - L1/L2 normalization type
    :cur_name - tag of experiment used in names of output files
    :mode - type of equation used for weight update
    :b_sqrt - whether to use square root for b in adam weight update equation
    :viz_dev - alias  for all_modes, whether to use a predefined set of settings
    :cap_loss - value at which loss will be clipped
    :viz_dir - path to directory for output files
    :eval_hessian - whether to calculate hessian spectral decomposition
    :optimizer - torch optimizer object (only Adam is supported)
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :return_loss - whether to return loss array used to generate the landscape plot (only supported for random axes)
    :raa - aliase of freeze, number of layer to freeze the weights
    """

    allowed_axes = ['random', 'adam', 'hessian']
    allowed_normalization = ['None', 'weight', 'model', 'layer', 'filter']
    allowed_order = [1,2]

    if order not in allowed_order:
        warnings.warn('Argument {} for order is not supported, allowed {}!'.format(order,allowed_order))
        warnings.warn('Setting order to 2')
        order = 2
    
    if cap_loss is not None and cap_loss <= 0:
        warnings.warn('Loss cap value cannot be negative or zero, ignorring this setting')
        cap_loss = None      
        
    if normalization not in allowed_normalization and normalization is not None:       
        raise AttributeError('Argument {} for norm is not supported, allowed {}!'.format(normalization,allowed_normalization))

    if axes not in allowed_axes:
        raise AttributeError('Argument {} for axes is not supported, allowed {}!'.format(axes,allowed_axes))
        
    if not b_sqrt and mode != 'adameq':
        warnings.warn('Warning! nobsqrt flag will be ignored since mode is not adameq')
    
    if (axes == 'random' or axes == 'hessian') and (mode == 'moment' or mode == 'adameq'):
        raise AttributeError('Using Adam update equation for random or hessian axes is prohibited')

    if calc_crit and not eval_hessian:
        raise AttributeError('Contradictory flags: Hessian criteria calculation is requested, but eval_hessian is False!')
    
    if to_save and viz_dir is None:
        viz_dir = default_viz_dir 
    if to_save and res_dir is None:
        res_dir = default_res_dir 

    # get hessian decomposition if it is needed
    if eval_hessian or axes == 'hessian': #or viz_dev: 
        hessian = hessian_calc(model, metric)

    # calc and save hessian eigenvalue spectral density
    if eval_hessian:
        eval_save_esd(hessian,to_save=to_save,to_viz=to_viz,exp_name=cur_name,viz_dir=viz_dir,calc_crit=calc_crit,n_kh=n_kh,res_dir=res_dir)

    # loss landscape settings and eval->save the plot
    if axes != 'random':
        # use adam moment axes
        if axes == 'adam':
            if optimizer is None:
                raise AttributeError('Adam moment axes evaluation is requested but optimizer is not specified!')
            else:
                try:
                    dir_one, dir_two = vec_from_optim(optimizer, 1) 
                except KeyError as _: # model too complex so the simple methods failed
                    dir_one, dir_two = vec_model_optim(optimizer, model, 1)
        # use hessian first and second eigenvectors as axes
        elif axes == 'hessian':
            dir_one, dir_two = vec_H_eigenvects(hessian)
            print('taking dirs from Hessian...')
    # use random axes
    else: #axes == 'random':
        dir_one = None
        dir_two = None
    loss_vals = eval_save_landscape(model, metric, STEP=dist, STEPS=steps, num_plots = num_plots, num_per_plot=num_per_plot, a1 = dir_one, a2 = dir_two, 
                        normalization = normalization, order=order, exp_name=cur_name, mode=mode, b_sqrt=b_sqrt, 
                        viz_dev=viz_dev,cap_loss=cap_loss,raa=raa,viz_dir=viz_dir,axes=axes,to_viz=to_viz,to_save=to_save,return_loss=return_loss)

    if eval_hessian or axes == 'hessian':
        hessian.reset()

    if return_loss:
        return loss_vals
    else:
        return None
