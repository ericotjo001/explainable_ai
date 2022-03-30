Future plan: development will continue in another repository: TO BE UPDATED

# Version 2.
We tested BONN and abstract some classes for SQANN. 

Example on how to run BONN: 
```
python main_BONN.py --mode BONN --data donut_example  --elasticsize 48  --show_fig_and_exit 0 --debug_toggles 00000
python main_BONN.py --mode BONN --data big_donut_example --elasticsize 256 --show_fig_and_exit 0 --debug_toggles 00000
```

# Version 1.
## Two Instances of Interpretable Neural Network for Universal Approximations 

The experimental results and iamges in the paper "Two Instances of Interpretable Neural Network for Universal Approximations" can be obtained by running the following commands.

### 1. Triangularly Constructed NN (TNN)
Results are shown in triangular_construction.ipynb

### 2. Semi-Quantized Activation Neural Network (SQANN)
Fig. 3(A) can be obtained using the following:
```
python run_SQANN.py --test_act 1
```

Fig. 3(C) can be obtained from SQANN.ipynb.


The following shows the 2 examples of SQANN constructions. This includes the explicit placement of data in layers and their evaluation results. From SQANN/example.py, turning show_fig_and_exit to 1 will plot figures like fig. 4(A).
```
python run_SQANN.py --mode example1 --show_fig_and_exit 0 --test_data_spread 0.2
python run_SQANN.py --mode example2 --show_fig_and_exit 0 --test_data_spread 0.2
```

Fig. 4(B) can be obtained using the following.
```
python run_SQANN.py --mode example1 --submode collect
python run_SQANN.py --mode example2 --submode collect
```
The results can be found in Checkpoint folder.

Results for table 1 and 2 in appendix can be found in SQANN_boston.ipynb and SQANN_diabetes.ipynb respectively.