# Hessian spectral analysis and visualization functions

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import pickle

from src_lla.hessian import hessian_calc

# TODO: take this from config or allow user input with flag
default_viz_dir = 'viz_results'
default_res_dir = 'analysis_results'


def hessian_criteria(eigenvalues,weights,n):
    
    """
    calculates Hessian criteria re Khn based on Gaussian quadrature
    of spectral decomposition, where n is a power

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition    
    :n - power for eigenvalues*weights product, float; recommended 0<n<1
    returns re and Khn values (float)
    """

    if n < 0:
        warnings.warn('Using negative power for Khn calculation will lead to incrorrect results!')

    re = []
    Khn = []
    
    for i in range(len(eigenvalues)):
        eigs = np.array(eigenvalues[i])
    
        re.append(np.abs(np.min(eigs)/np.max(eigs)))
        
        eig_ws = np.real(np.array(weights[i]))
        eig_pos = np.sum(np.power(eigs[eigs>0]*eig_ws[eigs>0],n))
        eig_neg = np.sum(np.power(np.abs(eigs[eigs<0]*eig_ws[eigs<0]),n))
    
        Khn.append(np.abs(eig_neg/eig_pos))

    re = np.mean(np.array(re), axis=0)
    Khn = np.mean(np.array(Khn), axis=0)

    return re, Khn


def gaussian_conv(x, s2):
    return np.exp(-x**2 / (2.0 * s2)) / np.sqrt(2 * np.pi * s2)


def density_plot(eigenvalues, weights, num_bins=10000, s2=1e-5, ext=0.001):
    """
    evaluates parameters of density plot in histogram form 

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition
    :num_bins - number of eigenvalue bins in histogram
    :s2 - squared sigma used for gaussian convolution
    :ext - horizontal plot offset
    returns density and segments for plotting
    """

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    y_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + ext
    y_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - ext
    segments = np.linspace(y_min, y_max, num=num_bins)
    s2 = s2 * max(1, (y_max - y_min))

    num_runs = eigenvalues.shape[0]
    bin_density = np.zeros((num_runs, num_bins))

    # calculating bin density values
    for i in range(num_runs):
        for j in range(num_bins):
            x = segments[j]
            bin_val = gaussian_conv(x - eigenvalues[i, :], s2)
            bin_density[i, j] = np.sum(bin_val * weights[i, :])
    density = np.mean(bin_density, axis=0)
    density = density / (np.sum(density) * (segments[1] - segments[0])) # normalized
    return density, segments


def esd_plot(eigenvalues, weights,to_save=False,to_viz=True,exp_name='esd_example',viz_dir=default_viz_dir,to_return=False):

    """
    plots and saves esd in histogram form 

    :eigenvalues - eigenvalues of spectral decomposition
    :weights - weights of spectral decomposition
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory for output files
    :exp_name - tag of experiment used in names of output files
    :to_return whether to return density, segments (mainly for debugging)
    returns density, segments or none
    """
    
    density, segments = density_plot(eigenvalues, weights)
    plt.semilogy(segments, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=16, labelpad=10)
    plt.xlabel('Eigenvlaues', fontsize=16, labelpad=10)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    
    if to_save:    
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        plt.savefig(os.path.join(viz_dir, exp_name + '_esd.png'))
        if not to_viz:
            plt.close()
    
    if to_viz:
        plt.show()
    else:
        plt.close()
        
    if to_return: # this option is for debug purposes only
        return density, segments


def eval_save_esd(hessian,n_iter=100,n_v=1,max_v=10,to_save=False,to_viz=True,exp_name='esd_example',
                  viz_dir=default_viz_dir,res_dir=default_res_dir,calc_crit=False,n_kh=0.5):

    """
    calculates and plots hessian esd

    :hessian - hessian class object
    :n_iter - max number of iterations for esd approximation
    :n_v - number of esd evaluation runs
    :max_v - max number of saved orthogonal vectors for esd approximation (increases required memory!!!)
    :to_save - whether to save the results in viz_dir
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory for output files
    :exp_name - tag of experiment used in names of output files
    returns tuple re, Khn or None, None
    """
    
    eigs, weights = hessian.esd_calc(n_iter=n_iter,n_v=n_v,max_v=max_v)
    esd_plot(eigs, weights, to_save=to_save,to_viz=to_viz,viz_dir=viz_dir,exp_name=exp_name)

    if calc_crit:
        re, Khn = hessian_criteria(eigs,weights,n_kh)

        if to_save:
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            with open(os.path.join(res_dir,'hessian_criteria_{}.log'.format(exp_name)), 'a') as log_file:
                    log_file.write('re: {}, Kh{}: {}\n'.format(re,n_kh,Khn))
        
        return re, Khn

    return None, None


def viz_esd(model,metric,eigs=False,top_n=2,eigs_n_iter=100,eigs_tol=1e-3,trace=False,trace_n_iter=100,trace_tol=1e-3,
            esd=True,esd_n_iter=100,n_v=1,max_v=10,to_save=False,
            to_viz=True,exp_name='esd_example',viz_dir=default_viz_dir,res_dir=default_res_dir,calc_crit=False,n_kh=0.5):

    """
    a funtions that collects different operations with hessian: eigs and esd

    :model - neural network torch model object
    :metric - loss evaluator Metric object
    :eigs - wheater to calculated hessian eigenvalues and eigenvectors or not
    :esd - whether to calculated hessian esd or not
    :calc_crit - whether to calculate hessian criteria re and Khn
    :n_kh - power for Khn criterion
    :top_n - number of top eigenvalues to compute
    :eigs_n_iter - number of iterations in eigs calculation
    :tol - tolerance to compare eigenvalues on consecutive iterations
    :esd_n_iter - max number of iterations for esd approximation
    :n_v - number of esd evaluation runs
    :max_v - max number of saved orthogonal vectors for esd approximation (increases required memory!!!)
    :to_save - whether to save the results (into viz_dir for plots and res_dir for criteria)
    :to_viz - whether to show to plots (in notebook)
    :viz_dir - path to directory to save output plots
    :res_dir - path to directory to save output results
    :exp_name - tag of experiment used in names of output files
    returns a list of possible results [eigenvalues,eigenvectors,re,Khn] 
    """

    if calc_crit and not esd:
        raise AttributeError('Hessian criteria calculation is requested but esd calculation is not! Please call viz_esd with esd=True.')

    results = [None,None,None,None,None] # eigenvalues, eigenvectors, trace, re, Khn
    hessian = hessian_calc(model,metric)

    if esd:
        re, Khn = eval_save_esd(hessian,n_iter=esd_n_iter,n_v=n_v,max_v=max_v,to_save=to_save,
                            to_viz=to_viz,viz_dir=viz_dir,res_dir=res_dir,exp_name=exp_name,calc_crit=calc_crit,n_kh=n_kh)

        if calc_crit: # this is redundant since eval_save_esd will return None, None if not calc_crit
            results[3] = re
            results[4] = Khn

    if eigs:
        res = hessian.eigs_calc(top_n=top_n,n_iter=eigs_n_iter,tol=eigs_tol)
        results[0] = res[0]
        results[1] = res[1]
        
    if trace:
        results[2] = hessian.tr_calc(n_iter=trace_n_iter,tol=trace_tol)
        
    hessian.reset()
    
    if to_save:
            
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        
        if results[0] is not None:
            with open(os.path.join(res_dir,'eigenvalues_{}.log'.format(exp_name)), 'a') as log_file:
                log_file.write('{}\n'.format(results[0]))
        if results[1] is not None:
            with open(os.path.join(res_dir,'eigenvectors_{}.pickle'.format(exp_name)), 'wb') as save_file:
                pickle.dump(results[1], save_file, protocol=pickle.HIGHEST_PROTOCOL)
        if results[2] is not None:
            with open(os.path.join(res_dir,'trace_{}.log'.format(exp_name)), 'a') as log_file:
                log_file.write('{}\n'.format(results[2]))
    
    return results

