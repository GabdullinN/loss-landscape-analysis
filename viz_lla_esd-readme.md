# Arguments of functions viz_lla and viz_esd

Input arguments of functions that can be imported from LLA library

**viz_lla arguments**
- model - neural network model object (torch nn.Module); it should be possible to call inference (forward) as pred = model(x)
- metric - object of type Metric for loss evaluation
- device - 'cpu' or 'cuda'
- dist - weight change scale, default 1 (recommended) 
- steps - number of steps for axes for a steps x steps loss landscape plot, default 40 
- num_plots - number of landscapes plotted in one figure for random axes, default 4 
- num_per_plot - number of landscapes in one horizontal line for random axes, default 2 
- axes  - type of axes direction vectors, allowed ['random', 'adam', 'hessian'], default 'random' 
- normalization - type of normalization for direction vectors, allowed [None, 'weight', 'layer', 'model', 'filter'], default None
- order - L1 (1) or L2 (2) normalization for types layer, model, filter, default 2
- cur_name - experiment name (tag) that will be used to name output images, esd plots, logs, default 'default' 
- mode - weight update equation, allowed [add, adameq], default 'add' 
- cap_loss - maximum value at which loss is capped (clipped), default None 
- viz_dev - whether to use 'all_modes' regime, default False
- raa - layer number for 'freeze' regime, default None
- viz_dir - path to directory where loss landscapes and esd plots in png format will be saved, default None (uses default directory './viz_dir') 
- res_dir - path to directory where hessian analysis results will be saved, default None (uses default directory './analysis_results') 
- eval_hessian - whether to calculate Hessian Eigenvalue Spectral Density, default False
- calc_crit - whether to calculate Hessian criteria, default False
- n_kh - power for Hessian criterion Khn calculation, default 0.5 
- optimizer - torch optimizer object (optional), default None
- to_viz - whether to show visualization results with plt.show(), applicable to Jupyter notebooks, default False
- to_save - whether to save the results (plots, spectra, criteria etc), default True

**viz_esd arguments**
- model - neural network model object (torch nn.Module); it should be possible to call inference (forward) as pred = model(x)
- metric - object of type Metric for loss evaluation
- eigs - whether to calculate Hessian eigenvalues and eigenvectors, default False
- esd - whether to calculate Hessian Eigenvalue Spectral Density (esd), default True
- trace - whether to calculate Hessian Trace, default False
- top_n - number of eigenvalues to calculate with eigs=True, default 2
- eigs_n_iter - maximum number of iterations for eigenvalue calculation, default 100
- eigs_tol - eigenvalue estimation tolerance, default 1e-3
- trace_n_iter - maximum number of iterations for trace calculation, default 100
- trace_tol - trace estimation tolerance, default 1e-3
- esd_n_iter - maximum number of iterations for esd calculation, default 100
- n_v - number of esd calculation runs, default 1
- max_v - maximum number of ortagonal vectors saved during esd calculation, default 10 (Important: high max_v might substantially increase memory requirements) 
- n_kh - power for Hessian criterion Khn calculation, default 0.5 
- calc_crit - whether to calculate Hessian criteria, default False
- to_viz - whether to show visualization results with plt.show(), applicable to Jupyter notebooks, default False
- to_save - whether to save the results (plots, spectra, criteria etc), default True
- viz_dir - path to directory where loss landscapes and esd plots in png format will be saved, default None (uses default directory './viz_dir') 
- res_dir - path to directory where hessian analysis results will be saved, default None (uses default directory './analysis_results')
- exp_name - experiment name (tag) that will be used to name output images, esd plots, logs, default 'esd_example'

Please note that eigenvalues, trace, and Hessian criteria are saved as text .log files, whereas eigenvectors are saved as python objects using pickle.


