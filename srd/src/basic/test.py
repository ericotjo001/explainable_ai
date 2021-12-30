import numpy as np

import torch
import torch.nn as nn
from .model import selective_activation, threshold_activation, get_torch_tensor_input_from_numpy


def decorprint(x):
    print('\n','='*16 + str(x) +'='*16)

def run_test(args, **kwargs):
    ENV_SHAPE = 3
    ENV_TEMPLATE = [0.,0.,0.,0.,0.5] # food every 5
    
    # python mainfish.py --test_FLD 1 --test_FC 1 --test_FNN 1 --test_PFC 1  --test_cognitive_run 1  --test_env 1

    # ========== Layer/Module design ==========
    if args['test_FLD']:
        test_FoodLocationDetectors(args, ENV_SHAPE=ENV_SHAPE)

    if args['test_FC']:
        test_FullyConnected(args)

    if args['test_FNN']:
        test_FishNeuralNetwork(args,ENV_SHAPE=ENV_SHAPE)

    if args['test_PFC']:
        test_PreFrontalCortex(args,ENV_SHAPE=ENV_SHAPE)

    if args['test_cognitive_run']:
        test_cognitive_run(args,ENV_SHAPE=ENV_SHAPE)

    # =========== ENV ===============
    if args['test_env']:
        test_env(args, ENV_SHAPE=ENV_SHAPE, ENV_TEMPLATE=ENV_TEMPLATE) 

def test_FoodLocationDetectors(args, **kwargs):
    decorprint('test_FoodLocationDetectors()')
    ENV_SHAPE = kwargs['ENV_SHAPE']

    from .model import get_food_location_detectors 
    conv_food = get_food_location_detectors(ENV_SHAPE)    
    print('weight shape', conv_food.weight.data.shape)
    print('bias shape', conv_food.bias.data.shape)

    
    with torch.no_grad():
        print('\nFOOD IS HERE')
        # We expect  to get output [[1, 0]], i.e. the first neuron is activated

        ENV1 = get_torch_tensor_input_from_numpy([0.5,0,0])
        print('input :', ENV1)
        y = conv_food(ENV1)
        y = selective_activation(torch.sum(y, dim=2))
        print('output:',y, y.shape)

        print('\nFOOD IS THERE')
        for env in [[0,0.5,0],[0.,0.,0.5]]:
            # We expect  to get output [[0, 1]], i.e. the second neuron is activated
            print('input :', env)
            env = get_torch_tensor_input_from_numpy(env)
            y = selective_activation(torch.sum(conv_food(env), dim=2))
            print('output: ',y, y.shape)

def test_FullyConnected(args):
    decorprint('test_FullyConnected()')
    from .model import get_fully_connected_layer
    fc = get_fully_connected_layer()

    torch.set_printoptions(precision=2, sci_mode=False)
    with torch.no_grad():
        SAMPLES = {
            'food here and hungry': [1.,0, 0.1],
            'food here not hungry': [1.,0, 0.9],
            'food there and hungry': [0.,1., 0.1],
            'food there not hungry': [0.,1., 0.9],
        }
        for desc,x in SAMPLES.items():
            # x = [[x_fh, x_ft, xF]]
            x = get_torch_tensor_input_from_numpy(x, add_channel=False)
            y = fc(x)
            print(desc)
            print('input :',x)
            print('output:', y)

# sample of situations and the corresponding input vector
SAMPLES = {
    'food here and hungry': [0.5,0,0, 0.1],
    'food here not hungry': [0.5,0,0, 0.9],
    'food there and hungry': [0.,0.5,0, 0.1],
    'food there not hungry': [0.,0.5,0, 0.9],
    'food there 2 and hungry': [0.,0.,0.5, 0.1],
    'food there 2 not hungry': [0.,0.,0.5, 0.9],
}
OTHER_SAMPLES = {
    'no food not hungry': [0,0,0,0.9],
    'no food and hungry': [0,0,0,0.1],
}

def test_FishNeuralNetwork(args, **kwargs):
    decorprint('test_FishNeuralNetwork()')
    ENV_SHAPE = kwargs['ENV_SHAPE']
    from .model import FishNN
    net = FishNN(ENV_SHAPE=ENV_SHAPE)

    import pandas as pd
    np.set_printoptions(precision=2, suppress=True)
    with torch.no_grad():
        row_names = {}
        this_data = []
        for i, (desc, x) in enumerate(SAMPLES.items()):
            x = get_torch_tensor_input_from_numpy(x)
            y, x1 = net(x)

            row_names[i] = desc
            this_data.append(list(x[0][0].numpy()) + list(x1[0].numpy()) + list(y[0].numpy())) 
            print(desc)
            print('input       :', x.numpy()[0,0])
            print('intermediate:', x1.numpy()[0])
            print('output      :', y.numpy()[0], '\n')

        df = pd.DataFrame(this_data,columns=['x1','x2','x3','x4','x1_fh','x1_ft','x1_F','y1','y2'])
        df.index = [y for x,y in row_names.items()]
        print(df)

def test_PreFrontalCortex(args ,**kwargs):
    decorprint('test_PreFrontalCortex()')
    ENV_SHAPE = kwargs['ENV_SHAPE']
    from .model import FishNN
    net = FishNN(ENV_SHAPE=ENV_SHAPE)

    from .model import get_prefrontal_cortex
    pfc_conv, pfc_fc = get_prefrontal_cortex()

    for x,y in OTHER_SAMPLES.items():
        SAMPLES[x] = y

    np.set_printoptions(precision=2, suppress=True)
    with torch.no_grad():
        for i, (desc, x) in enumerate(SAMPLES.items()):
            x = get_torch_tensor_input_from_numpy(x)
            y, x1 = net(x)        

            v0 = torch.cat((x1, net.softmax(y)),dim=1)
            v0 = v0.unsqueeze(1) # for channel=1 of conv1d
            v1_pre_act = net.softmax(pfc_conv(v0),)
            threshold = 0.5 # large threshold to reduce ambiguity
            v1 = threshold_activation(torch.sum(v1_pre_act, dim=2)-threshold)
            v2 = pfc_fc(v1)
            v2 = net.softmax(v2)

            print(desc)
            print('v0:',v0.numpy()[0,0])
            print('v1_pre_act:', v1_pre_act.numpy()[0].reshape(-1))
            print('v1:',v1.numpy()[0])
            print('v2:',v2.numpy()[0]) 

            print()

def test_cognitive_run(args, **kwargs):
    decorprint('test_cognitive_run()')

    ENV_SHAPE = kwargs['ENV_SHAPE']
    from .model import FishNN
    net = FishNN(ENV_SHAPE=ENV_SHAPE)

    np.set_printoptions(precision=3, suppress=True)
    for x,y in OTHER_SAMPLES.items():
        SAMPLES[x] = y

    for i, (desc, x) in enumerate(SAMPLES.items()):
        x = get_torch_tensor_input_from_numpy(x)
        y, x1 = net(x)       
        v = net.self_reward(x1,y)

        print(desc)
        print('y',y.detach().numpy()[0])
        print('v:',v.detach().numpy()[0])
        print()

def test_env(args, **kwargs):
    decorprint('test_cognitive_run()')
    ENV_SHAPE = kwargs['ENV_SHAPE']
    ENV_TEMPLATE = kwargs['ENV_TEMPLATE']

    template_size = len(ENV_TEMPLATE)
    ENV_POINTER = 0
    for i in range(args['n_iter']):
        ENV = np.concatenate((ENV_TEMPLATE[ENV_POINTER:],ENV_TEMPLATE[:ENV_POINTER]))
        ENV = ENV[:ENV_SHAPE]
        print(ENV_POINTER, ENV)
        ENV_POINTER = (ENV_POINTER+1)%template_size