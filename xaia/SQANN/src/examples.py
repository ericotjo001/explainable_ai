import numpy as np
from .data import N2LDData, DonutData
from .model import SQANN
from .utils import make_layer_setting, simple_evaluation, standard_evaluation


def run_example1(args):
    print('run_example1()')

    np.random.seed(3) # try different seeds so that we have at least 1 MISSED ACTIVATION
    N = 128
    ld = N2LDData(N, vmin=-1.0, vmax=1., sd=0.1, test_sd=args['test_data_spread'], show_fig_and_exit=args['show_fig_and_exit'])
    X,Y = ld.X, ld.Y
    X_test, Y_test = ld.X_test, ld.Y_test
    
    MAX_LAYER = 24
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }

    net = SQANN(layer_settings, N)
    net.fit_data(X,Y,verbose=20)

    print()
    simple_evaluation(X,Y,net, header_text='Show fitting on training data', verbose=20)

    print('\nevaluate on test dataset.')
    standard_evaluation(X_test, Y_test, net)


def run_example2(args):
    print('run_example2()')

    # np.random.seed(3) # try different seeds so that we have at least 1 MISSED ACTIVATION
    N = 128
    ld = DonutData(N, test_sd=args['test_data_spread'], show_fig_and_exit=args['show_fig_and_exit'])
    X,Y = ld.X, ld.Y
    X_test, Y_test = ld.X_test, ld.Y_test
    
    MAX_LAYER = 24
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }

    net = SQANN(layer_settings, N)
    net.fit_data(X,Y,verbose=20)

    print()
    simple_evaluation(X,Y,net, header_text='Show fitting on training data', verbose=20)

    print('\nevaluate on test dataset.')
    standard_evaluation(X_test, Y_test, net)


