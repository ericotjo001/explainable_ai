import numpy as np
import matplotlib.pyplot as plt

from .model import SQANN , double_selective_activation
from .data import N2LDData, DonutData
from .utils import make_layer_setting, off_ticks, simple_evaluation

def run_tests(args):
    if args['test_act']:
        test_activation(args)
    if args['test_donut_data']:
        test_donut_data(args)
    if args['test_first_layer']:
        test_first_layer(args)
    if args['test_second_layer']:
        test_second_layer(args)
    if args['test_net_allow_miss']:
        test_net_allow_missed_activations(args)

def test_activation(args):
    print('test_activation()')
    font = {'size': 7}
    plt.rc('font', **font)

    x = np.linspace(-2.,2.,12400)
    
    a1s = [1e-3,1e-2,1e-1,1]
    a2s = [1e-2,1e-1, 0.2, 1., 2.]
    n = len(a1s)
    nj = len(a2s)


    plt.figure(figsize=(7,7))
    for j, a2 in enumerate(a2s):
        for i,a1 in enumerate(a1s):
            y = double_selective_activation(x, a1=a1,a2=a2)
            plt.gcf().add_subplot(nj, n, i+1 + j*n)
            plt.gca().plot(x,y,label='a1:%s a2:%s'%(str(a1),str(a2)))
            plt.legend()
            plt.gca().set_ylim([-0.4,1.1])

            off_ticks(i,j,nj)
    plt.tight_layout()
    plt.show()

def test_donut_data(args):
    print('test_donut_data()')
    dd = DonutData(args['N'], show_fig_and_exit=False)    
    dd.show_fig_and_exit()


def test_first_layer(args):
    print('test_first_layer')
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    
    N = 128
    ld = N2LDData(N, vmin=-1., vmax=1., show_fig_and_exit=False)
    X,Y = ld.X, ld.Y

    layer_settings = {
        1: make_layer_setting(1e-2, 0.4, 0.1, 0.99),
    }

    net = SQANN(layer_settings, N)
    net.layer_k_sample_collection(X, Y, layer_k=1)
    print('net.used_indices:')
    print(net.used_indices[1], len(net.used_indices[1]))

    print('double check:')
    for i in net.used_indices[1]:
        y, act = net.activate_layer(X[i,:], layer_k=1)
        print('y:%s y0:%s'%(str(np.round(y,4)),str(np.round(Y[i],4))))
        print('  activations:',act)


def test_second_layer(args):
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)
    
    N = 128
    ld = N2LDData(N, vmin=-1., vmax=1., show_fig_and_exit=False)
    X,Y = ld.X, ld.Y

    layer_settings = {
        1: make_layer_setting(1e-2, 0.4, 0.1, 0.9,),
        2: make_layer_setting(1e-2, 0.4, 0.1, 0.9,),
    }

    net = SQANN(layer_settings, N)

    layer_now = 1
    while True:
        STOP_SIGNAL, COLLISION = net.layer_k_sample_collection(X, Y, layer_k=layer_now)

        print('[%s] net.used_indices: (n nodes:%s, total no of layers:%s)'%(
            str(layer_now),str( len(net.used_indices[layer_now])),str(net.total_n_layer) ))
        print('  ', net.used_indices[layer_now],)
    
        if STOP_SIGNAL=='NO_MORE_DATA': 
            break
        elif STOP_SIGNAL == 'COLLISION':
            collided_layer = COLLISION['collided_layer']

            for layer_j in range(collided_layer+1,layer_now+1):
                net.return_index_from_layer(layer_j)
            kp = COLLISION['perpetrator_index']
            net.push_node_to_layer(this_index= COLLISION['perpetrator_index'], x=X[kp,:], y=Y[kp], layer_k=collided_layer)
            layer_now = collided_layer


            print('<%s> net.used_indices: (n nodes:%s, total no of layers:%s, perpetrator_index:%s)'%(
                str(layer_now),str( len(net.used_indices[layer_now])),str(net.total_n_layer), str(kp) ))
            print('  ',net.used_indices[layer_now],)

        layer_now+=1
        if layer_now>=3: break

    print('double check:')
    for i in net.used_indices[2]:
        y, act = net.forward_to_layer_k(X[i,:], layer_k=2)

        print('y:%s y0:%s'%(str(np.round(y,4)),str(np.round(Y[i],4))))
        print('  activations:',act)


def test_net_allow_missed_activations(args):
    np.random.seed(3) # try different seeds so that we have at least 1 MISSED ACTIVATION
    N = 128
    ld = N2LDData(N, vmin=-1.0, vmax=1., sd=0.1, test_sd=1e-2, show_fig_and_exit=0)
    X,Y = ld.X, ld.Y
    X_test, Y_test = ld.X_test, ld.Y_test
    
    MAX_LAYER = 24
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }

    """ BEST so far
    # 1
    np.random.seed(0)
    N_MISSED_ACTIVATIONS:0, N_large_error:5
    avg error          : 0.03122, avg_frac_error          : 0.07474
    avg exclusive error: 0.03122, avg exclusive frac error: 0.07474
    
    a1s = np.linspace(0.001,1.,51)
    a2s = np.linspace(1.,0.7,101)
    layer_settings = {
        # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
        i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) # BEST NOW
    }
    """

    net = SQANN(layer_settings, N)
    print('Fitting data. With verbose>=100, we get to see layers constructed and reconstructed repeatedly.')
    net.fit_data(X,Y,verbose=100)

    print()
    simple_evaluation(X,Y,net, header_text='Show fitting on training data', verbose=20)
    print()
    simple_evaluation(X_test, Y_test, net, header_text='Show fitting on test data', verbose=100)
