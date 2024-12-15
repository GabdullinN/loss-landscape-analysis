# Loss Landscape Analysis

Loss Landscape Analysis (LLA) is a PyTorch library that allows to visualize and analyze loss landscapes of neural networks. Its main difference from other libraries is the variety of options that allow to plot loss landscapes along different axes with different weight update rules, use different normalization techniques, etc. This library also makes emphasis on the analysis rather than mere visualization, and incorporates Hessian analysis methods and allows to calculate quantitative analysis metrics.  

Disclaimer: LLA is still in development, which means existing functions might change, and new functions might be added later. Please use the latest version of the library.

<p align="center"><img src="/figure/figure_1.png" width="65%"/></p>

**Requirements**

* Python>=3.8.10
* numpy>=1.22.4
* torch>=1.13.0
* torchvision>=0.14.0
* matplotlib>=3.7.5
* (optional) jupyterlab>=3.2.9

**Run in Docker**

For Docker and Docker Compose users we provide a convenient way of starting with LLA. Required versions start with Docker 27.3.1 and Docker Compose 2.29.7.

Start the container
* git clone this repo
* (optional) change ports in docker-compose.yml
* docker compose up -d {service_name}

The following arguments can be used as {service_name}, which differ by GPU and Jupyter Lab availability
* lla-cpu
* lla-cpu-jup
* lla-gpu
* lla-gpu-jup

There are two ways of working with the project
* Work in container via terminal: docker exec -it lla_container /bin/bash
* Log into Jupyter Lab server: http:{server}:{your_port}, password **land**

**Examples**

We provide a set of examples in examples.sh which illustrate both loss landscape plotting and hessian analysis capabilities of LLA. It will run relevant scripts and save resulting figures in src/example_viz_results and numeric data in src/example_analysis_results. **Note that these folders are cleaned upon example script launch**, so do not use them to store your results. Docker users can run the examples with a single command  

```
docker exec -it lla-container /bin/bash examples-full.sh
```

This example may take some time when running on CPU. You can also run a faster example

```
docker exec -it lla-container /bin/bash examples-fast.sh
```

We provide two scripts lla_train.py and lla_eval.py for training and evaluation of neural networks, respectively. You can run those to test library's capabilities with, for example

```python
# use predefined plot settings with "all_modes", hessian axes, HESD, hessian criteria
python3 lla_eval.py --cuda --seed 42 -lc 10 --weights ./example_weights/lenet_example.pth --all_modes --axes hessian --hessian -hc --name eval_example -vd viz_results -rd analysis_results
# train for 2 epoch, visualize every 2 epochs, 9 plots per figure, random axes
python3 lla_train.py --cuda --seed 42 --axes random --losscap 100 -np 1 -npp 1 --name random_example -vd viz_results -ld train_logs
# train for 4 epochs, visualize every 2 epochs, use adam axes and adam update equation, filter L2 normalization
python3 lla_train.py --cuda --seed 42 -e 4 -vr 2 --axes adam --mode adameq --norm filter --name adam_example -vd viz_results -ld train_logs
```

There is also a separate script showcasing Hessain analysis capabilities without plotting loss landscapes. You can run it with

```python
python3 esd_example.py --cuda --seed 42 --hessian -hc --name hessian_example -vd viz_results -rd analysis_results
```

There are also Jupyter notebooks in src directory with examples for training, evaluation, and how to import LLA into existing project. You can check visualizations made with LLA there.

**LLA functions**

LLA focuses on visualizing 3D plots of loss landscapes and Hessian eigenvalue spectral decomposition (HESD). It allows to control axes type, weight update mode, direction vector normalization mode, weight change scale, freeze certain layers, etc. It also includes "quality of life" options, please refer to py_arguments-readme.md and viz_lla_esd-readme.md for the complete lists. If compared to existing libraries, there are two new options that should be highlighted:  

**axes** allows to choose between random, adam, and hessian axes for loss landscape plotting. Random axes is the most common way of choosing direction vectors present in other libraries, too. Adam axes take Adam optimizer moment axes as direction axes. Hessian axes take two top Hessian eigenvectors as direction axes.

**mode** allows to choose between add and adameq weight update equations. Add is the most common approach present in other libraries, and adameq allows to use weight update equation of Adam optimizer. The latter requires Adam axes. 

In addition to HESD plots we provide options for HESD criteria calculation, please refer to [1] for theoretical details. 

**1. Importing LLA into existing project**

LLA can be imported by copying src/src_lla into another project's directory. For user's convenience all main LLA capabilities can be accessed using functions viz_lla and viz_esd. You can check import_example.ipynb too see their application to the case of LeNet trained on MNIST.

**1.1. Loss landscape visualization with viz_lla**

To use viz_lla simply import it from the library

```python
from src_lla import viz_lla, metrics
```

It will require several objects: 
* model - torch neural network model object (inheriting from nn.Module);
* metric - loss evaluator, see below;
* (optional) optimizer - torch.optim object or similar.

Metric is LLA object that tells the library how the loss is calculated. It requires a batch of data for loss calculation which, for instance, can be taken from torch DataLoader in your project:

```python 
x_plot, y_plot = iter(data_loader).__next__()
```

In many cases metric can be created simply by specifying the loss function:  

```python 
criterion = torch.nn.CrossEntropyLoss() # specify your loss function
metric = metrics.Loss(criterion, x_plot, y_plot, device) # device - 'cuda' or 'cpu'
```

Example calls of viz_lla:

```python
viz_lla(model,metric,device=device,normalization='filter',axes='random',to_save=True,to_viz=True)
viz_lla(model,metric,device=device,axes='hessian',viz_dev=True,eval_hessian=True,to_save=True,to_viz=True)
```

**1.2. Hessian analysis with viz_esd**

Whereas viz_lla allows to access some Hessian analysis capabilities such as HESD plotting, the main Hessian analysis function is viz_esd that can be imported as 

```python
from src_lla import viz_esd
```

This function returns a list of four elements which correspond to eigenvalues, eigenvectors, trace, re, Khn (see [1] for details). Example calls of viz_esd (see viz_lla_esd-readme.md for the complete list of options):

```python
# top 2 eigenvalues and eigenvectors
eigvals, eigvects, _, _, _ = viz_esd(model, metric, esd=False, eigs=True, top_n=2)
# eigenvalues, eigenvectors, trace, HESD and its criteria
eigvals, eigvects, trace, re, Khn = viz_esd(model,metric,esd=True,eigs=True,trace=True,calc_crit=True,to_save=True,to_viz=True)
# only HESD
viz_esd(model, metric, esd=True, eigs=False, to_save=True, to_viz=True)
```

**2. Using loaders for non-standard pipelines**

In case your project has non-standard loss functions, dataloaders, or your model inference is more complex than pred=model(x), we provide an option to use loaders which allow to facilitate visualization for virtually any PyTorch pipeline. It is recommended to make a loaders following loader_template.py and import it into lla_train.py or lla_eval.py as described below.

**2.1. Standard elements in loader.py**

The main purpose of the loader.py is to allow to write a PyTorch pipeline of any complexity in a standardized format so it can be processed by other LLA functions. This is done by defining several standard elements which include

* ModelInit – a function that initializes the model and returns its object. The user should modify ```model = _init_your_model(N_CLASSES)``` line and add everything required for model initialization and weight loading (optional).

* CustomLoader – a function that loads the data and returns the DataLoader object. The user should modify ```train_data = _load_your_data(data_path)``` line and add everything required for data loading and preprocessing (optional). It is also necessary to specify batch_size for torch DataLoader object. It is important that DataLoader must output a tuple ```inputs, labels```. If your dataset (and, hence, dataloader) has different output format, then specify wrap=True and in \_\_next\_\_ method of LoaderWrapper (see below) specify how your dataloader output can be converted into ```inputs, labels``` tuple. 

* LoaderWrapper – this is a wrapper class for dataloader which allows to change its output format. The user should modify ```inputs, labels = output[0]``` line in \_\_next\_\_ so it returns the correct pair ```inputs, labels```.

* CustomLoss – this class is responsible for model inference and loss calculation. Its \_\_init\_\_ requires x (inputs from train_loader), y (labels from train_loader) and device, and optionally user could add any other parameter which might be needed later for \_\_call\_\_. The user should specify how their model calculates logits by modifying ```pred = model_wrapper.forward(inputs.to(self.device))``` and ```pred = model(inputs.to(self.device))``` lines, since both model and model_wrapper might be used by other functions. Then the line ```loss = self.loss_fn(pred, targets.to(self.device))``` should be modified for user's loss functions calculation. Other lines of code should not be modified*.

_*technical note: CustomLoss returns loss as loss.item() when use_wrapper=True, which is needed for landscape plotting. However, specifying use_wrapper=False, return_pred=True allows CustomLoss to calculate loss without detaching it from the computational graph, which allows for loss.backward() so CustomLoss can be used for model inference during training. This is useful when one wants to train the model and plot landscapes using virtually the same code. The user can also use ```batch``` argument to provide data during model training, see lla_train.py._

**2.2 Importing loaders into other scripts**

lla_train.py and lla_eval.py are designed to work with any loader which satisfies the requirements in Section 2.1. Example loaders are provided for mlp, LeNet, and ResNet (which can be found in src/src_lla/loaders). After the loader is imported, all necessary objects can be conveniently initialized as

```python
from src_lla.loaders.{your_loader} import *

# use standard loader elements to initialize all required components
dataloader = CustomLoader()
model = ModelInit(device=device)
metric = CustomLoss(x_plot, y_plot, device)
```

**3. Known peculiarities**

Any LLA function that requires Hessian analysis will throw an error at the initial call. It warns about the possible memory leak, but it can be ignored since the library ensures that no memory leak occurs when viz_lla or viz_esd are used. **However, this is not guaranteed** if the user manually calls only some components of LLA functions, see, for instance, the description of hessian_calc.reset() method in src_lla/hessian. 

**Referencing LLA**

If you use this library in your research, please cite the following paper:

[1] {link to arxiv paper LLA}

**License**

Copyright 2024 Kryptonite

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
