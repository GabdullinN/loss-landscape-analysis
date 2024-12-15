#!/bin/bash

VD=example_viz_results
RD=example_analysis_results

echo "####"
echo "Executing example scripts for LLA library..."
echo "Evaluating LeNet on MNIST"
echo "Using predefined plot settings with all_modes, hessian axes, HESD, hessian criteria"
echo "Figures will be saved to src/viz_results and numeric results will be saved to src/analysis_results, the results will be tagged as full_example"
echo "####"

# checking if previous example results exist
if [ -d "$VD" ]; then
  echo "Clearning old results in $VD."
  rm -r $VD
fi

if [ -d "$RD" ]; then
  echo "Clearning old results in $RD."
  rm -r $RD
fi

# running py scripts
python3 lla_eval.py --cuda --seed 42 -lc 10 --weights ./example_weights/lenet_example.pth --all_modes --axes hessian --hessian -hc --name full_example_figures -vd $VD -rd $RD
echo "####"
echo "Running hessian-only analysis for the same problem, the results will be tagged as hessian_example"
echo "####"
python3 esd_example.py --cuda --seed 42 --hessian -hc --name full_example_hessian -vd $VD -rd $RD
echo "Finished!"
