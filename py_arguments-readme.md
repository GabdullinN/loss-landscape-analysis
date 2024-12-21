# Input arguments for scripts lla_train.py and lla_eval.py

List of arguments for example scripts lla_train.py and lla_eval.py.

**1. Arguments that require specific values:** \
--data - path to data for dataset and dataloader objects, default -1 (takes path from loader.py if there is one) \
--weights - path to model weights, None results in random initialization, default -1 (takes path from loader.py if there is one) \
--seed - seed number for all random values, default None \
-e, --epochs - number of training epochs, ignored for lla_eval, default 1 \
-lr - learning rate, ignored for lla_eval, default 0.01 \
--dist - weight change scale, default 1 (recommended) \
--steps - number of steps for axes for a steps x steps loss landscape plot, default 40 \
-vr, --viz_rate - frequency for output saving in epochs, ignored for lla_eval, default 1 \
-np, --num_plots - number of landscapes plotted in one figure for random axes, default 4 \
-npp, --num_per_plot - number of landscapes in one horizontal line for random axes, default 2 \
-md, --mode - weight update equation, allowed [add, adameq], default add \
-ax, --axes - type of axes direction vectors, allowed [random, adam, hessian], default random \
-nm, --norm - type of normalization for direction vectors, allowed [None, weight, layer, model, filter], default None\
-od, --order - L1 (1) or L2 (2) normalization for types layer, model, filter, default 2\
-lc, --losscap - maximum value at which loss is capped (clipped), default None \
--nkh - power for Hessian criterion Khn calculation, default 0.5 \
-vd, --viz_dir - path to directory where loss landscapes and esd plots in png format will be saved, default ./viz_results \
-ld, --log_dir - path to directory where train logs will be saved, default ./train_logs \
-rd, --res_dir - path to directory where hessian analysis results will be saved, default ./analysis_results \
--name - experiment name (tag) that will be used to name output images, esd plots, logs, default default \
--freeze - layer number (int) used as a start (if positive) or end (if negative) index for layer freezing; for intance, 7 means from 7th layer till the end, -7 means from 0th layer to 6th, where layer order is taken from model.parameters, which corresponds to layer ordering in model.__ init __, default None 

**2. Flags that do not require specific values:** \
--cuda - conduct the analysis on gpu \
--all_modes - plot different predefined combination of visualization parameters, fixes random axes; compatible with --axes adam or --axes hessian. Ignores other visualization settings like num_plots, num_per_plot etc \
-he, --hessian - calculate Hessian Eigenvalue Spectral Density \
-hc, --calc_crit - calculate Hessian criteria 

Example usage of arguments 1 (requires specific value) and 2 (only requires calling): \
python3 lla_train.py --cuda -e 2 --name test
