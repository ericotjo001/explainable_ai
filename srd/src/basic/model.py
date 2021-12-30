import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from src.utils import FastPickleClient

def get_torch_tensor_input_from_numpy(x, add_channel=True):
    # unsqueeze(0) needed to make it pytorch tensor batch
    if add_channel:
        # [x] to add 1 dimension for 'channel'
        x = [x]
    return torch.from_numpy(np.array(x)).unsqueeze(0).to(torch.float)

def selective_activation(x, epsilon=1e-4): 
    return epsilon/(epsilon+x**2)

def threshold_activation(x,a=5.):
    return torch.tanh(F.leaky_relu(a*x))


def get_food_location_detectors(ENV_SHAPE,preset=True):
    # OUTPUT_CHANNELS
    # 1. CHANNEL TO DETECT FOOD HERE
    # 2. CHANNEL TO DETECT FOOD THERE (not nearby) 
    OUTPUT_CHANNELS = 2
    
    # kernel size is ENV_SHAPE. 
    # Fish can detect all 3 pixels in front of it
    KERNEL_SIZE = ENV_SHAPE
    conv_food = nn.Conv1d(1,OUTPUT_CHANNELS,KERNEL_SIZE,bias=True)
    
    if preset:
        # FOOD HERE
        conv_food.weight.data[0,:,:] = torch.from_numpy(np.array([1,0,0])) 
        conv_food.bias.data[0] = -0.5
        # FOOD THERE
        conv_food.weight.data[1,:,:] = torch.from_numpy(np.array([0,1,1])) 
        conv_food.bias.data[1] = -0.5
    return conv_food

def get_fully_connected_layer(preset=True):
    # INPUT_CHANNEL : 3 input channels = 2 fld output + 1 FULL variable
    # OUTPUT_CHANNEL: 2 actions are available, EAT or MOVE
    INPUT_CHANNEL, OUTPUT_CHANNEL = 3, 2

    fc = nn.Linear(INPUT_CHANNEL, OUTPUT_CHANNEL,bias=True)
    if preset:
        delta = 1e-3
        fc.weight.data = fc.weight.data * 0 + torch.tensor([[1,delta,-1],[delta,1,1]])
        fc.bias.data = fc.bias.data * 0
    return fc

def get_prefrontal_cortex(preset=True):
    """ We just borrow the name prefrontal cortex
    prefrontal_cortex. According to wiki
        "Executive function relates to abilities to differentiate among conflicting thoughts, 
        determine good and bad, better and best, same and different, future consequences 
        of current activities, working toward a defined goal, prediction of outcomes, 
        expectation based on actions, and social "control" (the ability to suppress urges that, 
        if not suppressed, could lead to socially unacceptable outcomes).""
    """

    # print('get_prefrontal_cortex layers')

    ##################### PART 1 #####################
    # 5 input channels = 3 intermediate channels + 2 output channels
    #   3 intermediate channels: food here, food there, full
    #   2 output channels: eat, move
    # 3 output channels, m1, e1, ex see below.

    pfc_conv = nn.Conv1d(1, 3, 5, bias=False) 
    # e1: eat when hungry and food there
    pfc_conv.weight.data[0,:,:] = torch.from_numpy(np.array([1,0,-1,1,0])) 

    # m1 : move when hungry and food here
    pfc_conv.weight.data[1,:,:] = torch.from_numpy(np.array([0,1,-1,0,1])) 

    # ex: explore to find food
    pfc_conv.weight.data[2,:,:] = torch.from_numpy(np.array([-1,-1,0,0,1])) 

    ##################### PART 2 #####################
    delta = 1e-3
    pfc_fc = nn.Linear(3,2,bias=True)
    pfc_fc.weight.data = 0. + torch.tensor([[1.,1.,1.],[delta,delta,delta]])
    pfc_fc.bias.data = 0. + torch.tensor([0.,0.4])
    return pfc_conv, pfc_fc

class FishNN(nn.Module):
    def __init__(self, ENV_SHAPE,preset_nn=True, preset_meta_nn=True):
        super(FishNN, self).__init__()

        self.fld = get_food_location_detectors(ENV_SHAPE, preset=preset_nn)
        self.fc = get_fully_connected_layer(preset=preset_nn)

        self.pfc_conv, self.pfc_fc = get_prefrontal_cortex(preset=preset_meta_nn)  
        self.threshold=0.5 # large threshold to kill off all decisions made by uniformly low activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        # x is [[x1,x2,x3, FULL]]
        #   expected shape in pytorch, (1,4)
        # x1,x2,x3 are environment variables
        # FULL: if 1, means the fish is full. If 0, means it is hungry

        env = x[:,:,:3]
        x1 = self.fld(env)
        x1 = selective_activation(torch.sum(x1, dim=2))
        x1 = torch.cat((x1,x[:,:,3]), dim=1)
        # now it is [[fh,ft,F]]
        # Neurons description: 
        #   fh: food here
        #   ft: food there
        #   F : FULL

        y = self.fc(x1)
        return y, x1

    def self_reward(self, x1, y):
        v0 = torch.cat((x1, self.softmax(y)),dim=1)
        v0 = v0.unsqueeze(1) # for channel=1 of conv1d
        v1_pre_act = self.softmax(self.pfc_conv(v0),)

        v1 = threshold_activation(v1_pre_act.squeeze(2) -self.threshold)
        v2 = self.pfc_fc(v1)
        v2 = self.softmax(v2)
        return v2

class Fish1D(FastPickleClient):
    def __init__(self, args, **kwargs):
        super(Fish1D, self).__init__()
        ENV_SHAPE = kwargs['ENV_SHAPE']

        self.INTERNAL_STATE = {'energy':1.0}
        self.nn = FishNN(ENV_SHAPE=ENV_SHAPE, preset_nn=True, preset_meta_nn=True)
        self.ACTIONS = ['EAT','MOVE']

    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):        

    def get_input_tensor(self, ENV):
        temp = (ENV, [self.INTERNAL_STATE['energy']])
        x = get_torch_tensor_input_from_numpy(np.concatenate(temp))
        return x

    def make_decision(self, x):
        y, x1 = self.nn(x)
        greedy_decision = torch.argmax(y[0]).data.item()
        return greedy_decision

    def make_self_rewarded_decision(self, x):
        y, x1 = self.nn(x)
        greedy_decision = torch.argmax(y[0]).data.item()

        z = self.nn.self_reward(x1, y)
        return greedy_decision, z

    def update_state(self, action, ENV):
        CURRENT_TILE = ENV[0]

        if self.ACTIONS[action] == 'EAT' and CURRENT_TILE == 0.5:
            self.INTERNAL_STATE['energy'] += 0.5
        else:
            self.INTERNAL_STATE['energy'] -= 0.05
        self.INTERNAL_STATE['energy'] = np.clip(self.INTERNAL_STATE['energy'],None,1.)
        

class Fish1DMapManager(FastPickleClient):
    def __init__(self, args, **kwargs):
        super(Fish1DMapManager, self).__init__()
        self.ACTIONS = ['EAT','MOVE']

        self.ENV_SHAPE = 3
        self.ENV_TEMPLATE = [0.,0.,0.,0.,0.5] # food every 5
        self.template_size = len(self.ENV_TEMPLATE)
        self.ENV_POINTER = 0

    def update_state(self, action, ENV):
        if self.ACTIONS[action] == 'EAT' and ENV[0] == 0.5:
            ENV[0] = 0
        elif self.ACTIONS[action]=='MOVE':
            self.increment_pointer()
            ENV = self.get_env_from_template()
        return ENV

    def get_env_from_template(self):
        ENV_POINTER = self.ENV_POINTER
        ENV_TEMPLATE = self.ENV_TEMPLATE

        ENV = np.concatenate((ENV_TEMPLATE[ENV_POINTER:],ENV_TEMPLATE[:ENV_POINTER]))
        ENV = ENV[:self.ENV_SHAPE]
        return ENV

    def increment_pointer(self):
        self.ENV_POINTER = (self.ENV_POINTER+1)%self.template_size