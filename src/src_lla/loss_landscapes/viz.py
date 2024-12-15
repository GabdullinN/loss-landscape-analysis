# Visualization functions

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import src_lla.loss_landscapes
import src_lla.loss_landscapes.metrics
from src_lla.loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from src_lla.loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to

import copy

# TODO: take this from config or allow user input with flag
default_viz_dir = 'viz_results'


# unused functions for debugging
def freeze_params_list(model):
    saved_params = []
    for layer, param in model.named_parameters():
        saved_params.append(param.cpu().detach())
    return saved_params

        
def eval_save_landscape(model, metric, STEP = 1, STEPS=40, num_plots=4, num_per_plot=2, a1 = None, a2 = None, to_save = True, to_viz = False,
                        normalization='filter', order=2, exp_name = '', mode='add', raa=None, b_sqrt=True, viz_dev=False, 
                        cap_loss=None, dcopy=True, return_loss=False, viz_dir=None, axes='random'):
    
    """
    evaluates and saves loss landscape for given model using metric
    uses src_lla.loss_landscapes.random_plane as the main evaluation 'engine'

    :model - torch model object
    :metric - custom loss metric object
    :step - step in weight space
    :steps - size of landscape map
    :num_plots - number of plots for random axes
    :num_per_plot - number of plots in one horizontal line
    :a1 - first direction vector
    :a2 - second direction vector
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :normalization - type of normalization for direction vectors
    :order - L1/L2 normalization type
    :exp_name - tag of experiment used in names of output files
    :mode - type of equation used for weight update
    :raa - alias of freeze, number of layer to freeze the weights
    :b_sqrt - whether to use square root for b in adam weight update equation
    :viz_dev - alias for all_modes, whether to use a predefined set of settings
    :cap_loss - value at which loss will be clipped
    :dcopy - whether some functions are allowed to deepcopy the model
    :return_loss - whether to return loss array used to generate the landscape plot (only supported for random axes)
    :viz_dir - path to directory for output files
    :axes - type of axes used to plot loss landscape
    """
    
    if to_viz and viz_dir is None:
        viz_dir = default_viz_dir
    
    # original eval_plot_landscape mode
    if not viz_dev and a1 is None and a2 is None and num_plots > 1:
        
        n_h = num_plots // num_per_plot
        n_v = num_plots // n_h

        figure, axis = plt.subplots(n_h, n_v,figsize=(int(15), int(15*(num_plots/4)/(num_per_plot/2))),subplot_kw=dict(projection='3d')) 

        loss_all = {}

        for h in range(n_h):
            for v in range(n_v):

                loss_data = src_lla.loss_landscapes.random_plane(model, metric, distance=STEP, steps=STEPS, normalization=normalization, order=order, deepcopy_model=True, a1=a1, a2=a2, mode=mode, raa=raa, b_sqrt=b_sqrt)

                if cap_loss:
                    loss_data = loss_data.clip(0,cap_loss)

                tag = str(h)+str(v)
                loss_all[tag] = loss_data
                
                if num_plots == 2:
                    ax = axis[v]
                else:
                    ax = axis[h][v]
                #ax = plt.axes(projection='3d')
                X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
                Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
                ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
                
        if to_save:
            figure.savefig(os.path.join(viz_dir, exp_name + '.png'))

        if to_viz:
            plt.show() 
        else:
            plt.close()

        if return_loss:
            return loss_all
        else:
            return None

    # original eval_save_landscape modes
    elif not viz_dev:
        loss_data = src_lla.loss_landscapes.random_plane(model, metric, distance=STEP, steps=STEPS, normalization=normalization, order=order, deepcopy_model=dcopy, a1=a1, a2=a2, mode=mode, raa=raa, b_sqrt=b_sqrt)
        if cap_loss:
            loss_data = loss_data.clip(0,cap_loss)

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
        Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
        ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Loss Landscape for {}'.format(exp_name))
        
        if to_save:
            fig.savefig(os.path.join(viz_dir, exp_name + '.png'))  
        if to_viz:
            plt.show() 
        else:
            plt.close()

        if return_loss:
            return loss_data
        else:
            return None
    
    else:
        # viz dev - fixed parameter settings

        n_h = 2
        if axes == 'random':
            n_v = 2
        elif axes == 'hessian':
            n_v = 4
        elif axes == 'adam':
            n_v = 5 # 6 for dev version

        figure, axis = plt.subplots(n_v, n_h,figsize=(15,20),subplot_kw=dict(projection='3d')) 
        #print('making {} by {} grit'.format(n_v,n_h))
        
        # rand vectors for typical add mode
        # calc rand axes first so they are the same for all rand axes plots below
        model_start_wrapper = wrap_model(copy.deepcopy(model) if dcopy else model)
        start_point = model_start_wrapper.get_module_parameters()
        dir_one = rand_u_like(start_point)
        dir_two = orthogonal_to(dir_one)
        
        res = {}
        titles = {}
        
        # random axes part that is always shown
        # mode=add with dist=1 and norm=None
        res['00'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization=None, deepcopy_model=dcopy, a1=dir_one, a2=dir_two, mode='add', raa=raa, b_sqrt=b_sqrt)
        titles['00'] = 'rand axes with add and no norm'
        # mode=add with dist=1 and norm=w
        res['01'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='weight', deepcopy_model=dcopy, a1=dir_one, a2=dir_two, mode='add', raa=raa, b_sqrt=b_sqrt)
        titles['01'] = 'rand axes with add and norm=weight'
        # mode=add with dist=1 and norm=filter, L1
        res['10'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=1, deepcopy_model=dcopy, a1=dir_one, a2=dir_two, mode='add', raa=raa, b_sqrt=b_sqrt)
        titles['10'] = 'rand axes with add and norm=filter L1'
        # mode=add with dist=1 and norm=filter, L2
        res['11'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=2, deepcopy_model=dcopy, a1=dir_one, a2=dir_two, mode='add', raa=raa, b_sqrt=b_sqrt)
        titles['11'] = 'rand axes with add and norm=filter, L2 norm'

        # hessian part
        if axes == 'hessian':
            # mode=add with dist=1 and norm=None, hessian axes
            res['20'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization=None, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['20'] = 'hessian axes with add and no norm'
            # mode=add with dist=1 and norm=w, hessian axes
            res['21'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='weight', deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['21'] = 'hessian axes with add and norm=weight'
            # mode=add with dist=1 and norm=filter, L1, hessian axes
            res['30'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=1, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['30'] = 'hessian axes with add and norm=filter L1'
            # mode=add with dist=1 and norm=filter, L2, hessian axes
            res['31'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=2, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['31'] = 'hessian axes with add and norm=filter, L2 norm'

        # adam part
        if axes == 'adam':
            # mode=add with dist=1 and norm=None, adam axes
            res['20'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization=None, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['20'] = 'adam axes with add and no norm'
            # mode=add with dist=1 and norm=w, adam axes
            res['21'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='weight', deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['21'] = 'adam axes with add and norm=weight'
            # mode=add with dist=1 and norm=filter, L1, adam axes
            res['30'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=1, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['30'] = 'adam axes with add and norm=filter L1'
            # mode=add with dist=1 and norm=filter, L2, adam axes
            res['31'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', order=2, deepcopy_model=dcopy, a1=a1, a2=a2, mode='add', raa=raa, b_sqrt=b_sqrt)
            titles['31'] = 'adam axes with add and norm=filter L2 '
            # see title
            res['40'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization=None, deepcopy_model=dcopy, a1=a1, a2=a2, mode='moment', raa=raa, b_sqrt=True)
            titles['40'] = 'adam axes with adam eq and no norm'
            # see title
            res['41'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter',deepcopy_model=dcopy, a1=a1, a2=a2, mode='moment', raa=raa, b_sqrt=True)
            titles['41'] = 'adam axes with adam eq, filter norm=filter L2'
            ''' # dev version
            # see title
            res['41'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization=None, deepcopy_model=dcopy, a1=a1, a2=a2, mode='moment', raa=raa, b_sqrt=False)
            titles['41'] = 'adam axes with adam eq, no norm and b no sqrt'
            # see title
            res['50'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', deepcopy_model=dcopy, a1=a1, a2=a2, mode='moment', raa=raa, b_sqrt=True)
            titles['50'] = 'adam axes with adam eq, filter norm and sqrt(b)'
            # see title
            res['51'] = src_lla.loss_landscapes.random_plane(model, metric, distance=1, steps=STEPS, normalization='filter', deepcopy_model=dcopy, a1=a1, a2=a2, mode='moment', raa=raa, b_sqrt=False)
            titles['51'] = 'adam axes with adam eq, filter norm and b no sqrt'
            '''
                
        if cap_loss:
            print('clipping loss at ', cap_loss)
            for tag in res:
                res[tag] = res[tag].clip(0,cap_loss)
        
        for h in range(n_v):
            for v in range(n_h):
                ax = axis[h][v]
                tag = str(h)+str(v)
                #ax = plt.axes(projection='3d')
                X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
                Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
                ax.plot_surface(X, Y, res[tag], rstride=1, cstride=1, cmap='viridis', edgecolor='none')
                ax.set_title(titles[tag])

        if to_save:
            figure.savefig(os.path.join(viz_dir, exp_name + '_all_modes' + '.png'))   
        if to_viz:
            plt.show() 
        else:
            plt.close()
        
        return None
        

