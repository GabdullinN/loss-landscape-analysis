#!/bin/bash

VD=example_viz_results
RD=example_analysis_results

echo ""
echo "Executing example scripts for LLA library..."
echo "Evaluating LeNet on MNIST"
echo "Plotting loss landscape along Hessian axes, filter normalization"
echo "Figures will be saved to src/$VD and numeric results will be saved to src/$RD, the results will be tagged as fast_example"
echo ""

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
python3 lla_eval.py --cuda --seed 42 -lc 10 --weights ./example_weights/lenet_example.pth --axes hessian --norm filter --name fast_example -vd $VD -rd $RD
echo "Finished!"
echo "Images (loss landscapes and HESDs) are saved in $VD"
echo "Numeric results (eigenvalues, hessian criteria) are saved in $RD"
