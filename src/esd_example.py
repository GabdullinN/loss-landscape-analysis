# The main model train script
# You can import your loader from loaders to plot loss landscapes and/or evaluate hessian

# This library is distributed under Apache 2.0 license

# (c) Kryptonite, 2024

import copy
import argparse
import os
import random
import warnings

# import the main Hessian analysis function
from src_lla import viz_esd

#import your loader here
#from src_lla.loaders.src_ResNet import *
from src_lla.loaders.src_LeNet import *
#from src_lla.loaders.src_mlp_mnist import *


def main(args):
    
    if args.cuda:
        print('checking if cuda is available...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == 'cpu':
            print('Warning! --cuda was specified but cuda is not available!')
    else:
        device = 'cpu'
    print("Project running on device: ", device)
    
    exp_name = args.name
    data_path = args.data
    viz_dir = args.viz_dir
    res_dir = args.res_dir
    weight_path = args.weights
    normalization = args.norm
    eval_hessian = True
    seed = args.seed
    calc_crit = args.calc_crit
    n_kh = args.nkh

    if weight_path == 'None':
        weight_path = None

    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    if calc_crit:
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
    
    if seed is not None:
        random.seed(seed)
        #if numpy is not None:
        #    numpy.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    if calc_crit and not eval_hessian:
        warnings.warn('Hessian criteria calculation is requested, but Hessian estimation is False; setting it to True')
        eval_hessian = True  
    
    # init train_loader
    if data_path != '-1':
        train_loader = CustomLoader(shuffle=True,data_path=data_path)
    else: # use default data path specified in loader
        train_loader = CustomLoader(shuffle=True)
        
    # model can take other arguments, but it MUST take device
    # note that weight_path = None is a valid option used for random weight init
    if weight_path != '-1':
        model = ModelInit(device=device,weight_path=weight_path)
    else:
        model = ModelInit(device=device)

    # create metric object
    x_plot, y_plot = iter(train_loader).__next__() # data that the evaluator will use when evaluating loss
    metric = CustomLoss(x_plot, y_plot, device) # loss is evaluated via metric so no need to specify criterion

    # random init esd
    cur_name = 'esd_init_' + exp_name
    res = viz_esd(model,metric,esd=True,eigs=True,trace=True,calc_crit=True,to_save=True,to_viz=False,res_dir=res_dir,viz_dir=viz_dir,exp_name=cur_name)
    
    #pretrained esd
    model.load_state_dict(torch.load('example_weights/lenet_example.pth',map_location=device)) # load example pre-trained weights
    model.to(device)
    cur_name = 'esd_pretrained_' + exp_name
    res = viz_esd(model,metric,esd=True,eigs=True,trace=True,calc_crit=True,to_save=True,to_viz=False,res_dir=res_dir,viz_dir=viz_dir,exp_name=cur_name)
            
    
if __name__ == '__main__':
    
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=False, default='-1', help="path to data, default = -1 (use default from loader)")
    p.add_argument('--weights', type=str, required=False, default='-1', help="path to model weights, default = -1 (use default from loader)")
    p.add_argument('--seed', type=int, required=False, default=None, help="seed value for random values (int), default = None")
    p.add_argument('-e', '--epochs', type=int, required=False, default=1, help="num epochs, default = 1")
    p.add_argument('-lr', type=float, required=False, default=0.01, help="learning rate, default = 0.01")
    p.add_argument('--dist', type=int, required=False, default=1, help="max dist in param space, default = 1")
    p.add_argument('--steps', type=int, required=False, default=40, help="steps for [steps x steps] landscape plot, default = 40")
    p.add_argument('-vr', '--viz_rate', type=int, required=False, default=1, help="save img each vr epochs, default = 1")
    p.add_argument('-np', '--num_plots', type=int, required=False, default=4, help="number of plots for random axes analysis, default = 4")
    p.add_argument('-npp', '--num_per_plot', type=int, required=False, default=2, help="number of figures per horizontal line, default = 2")
    p.add_argument('-md', '--mode', type=str, required=False, default='add', help="mode for weight update calc, allowed [add, adameq], default = add")
    p.add_argument('-ax', '--axes', type=str, required=False, default='random', help="axes to plot landscaped, allowed [random, adam, hessian], default = random")
    p.add_argument('-nm', '--norm', type=str, required=False, default=None, help="weight-dir normalization method, allowed [weight, layer, model, filter], recommended filter, default = None")
    p.add_argument('-od', '--order', type=int, required=False, default=2, help="normalization order for filter,model,layer norm, default = 2 (L2 norm)")
    p.add_argument('-lc', '--losscap', type=float, required=False, default=None, help="loss cap (clip) value, default = None")
    p.add_argument('--nkh', type=float, required=False, default=0.5, help="power for hessian criterion Khn calculation, default = 0.5")
    p.add_argument('-vd', '--viz_dir', type=str, required=False, default='viz_results', help="dir to save landscape images, default = viz_results")
    p.add_argument('-rd', '--res_dir', type=str, required=False, default='analysis_results', help="dir to save hessian criteria, default = analysis_results")
    p.add_argument('-ld', '--log_dir', type=str, required=False, default='train_logs', help="dir to save train logs, default = train_logs")
    p.add_argument('--freeze', type=int, required=False, default=None, 
                   help="freeze and do not update weights in layers, positive int: starting from (7 means no change from layer 7 till the end), negative int: up to (-7: up to layer 7) , default = None")
    p.add_argument('-he', '--hessian',
                   required=False,
                   default=False,
                   action='store_true',
                   help='calculate Hessian eigvalue spectral density, default = False')
    p.add_argument('--all_modes',
                   required=False,
                   default=False,
                   action='store_true',
                   help='use a combination of all modes for viz, this will ignore other viz flags, default = False')
    p.add_argument('-hc', '--calc_crit',
                   required=False,
                   default=False,
                   action='store_true',
                   help='calculate Hessian criteria, default = False')
    p.add_argument('--cuda',
                   required=False,
                   default=False,
                   action='store_true',
                   help='use cuda (gpu), default: no (cpu)')
    p.add_argument('--name',
                   help='experiment name used in output image names',
                   required=False,
                   default='example',
                   type=str)

    
    # Parse and validate script arguments.
    args = p.parse_args()
    
    main(args)
