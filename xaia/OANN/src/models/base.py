import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from .interpolator import OANNInterpolator
from .controller import OANNNodeController
from .utils import double_selective_activation, binarize

class OANN(OANNInterpolator, OANNNodeController):
    def __init__(self, **settings):
        super(OANN, self).__init__(**settings)
        self.r = 0.5

        """
        some inherited properties:
        self.output_mode  # continuous/discrete
        self.interpolation_mode

        Do Implement init new like
        if settings['init_new']:
            self.init_new(**settings)

        """

    def init_new(self, **settings):
        self.layer_nodes = defaultdict(list) 
        self.layer_nodes_alphas = defaultdict(list)
        self.total_n_layer = None # update during training/data fitting

    def activate_layer(self, x, layer_k=1, get_val=True):
        """ 
        x: numpy array, shape (D,)

        Out: 
            act: numpy array, shape (D1,)
            y: numpy array, shape (D2,)
        
        D, D1, D2 arbitrary dimensions, depending on the shape of layers
        """
        lsett = self.get_layer_setting(layer_k)
        # act: activation, at the synapse! we also denote it as v

        act = np.sum((x-self.layer_nodes[layer_k])**2, axis=1)**0.5
        # np.sum(shape (1,D)- shape (N_nodes,D)), axis 1)^2 --> shape (N_nodes,1)
        act = double_selective_activation(act, lsett['a1'],lsett['a2'], r=self.r) 
  
        y = self.layer_nodes_alphas[layer_k][np.argmax(act)] if get_val else None
        return y, act

    def forward(self, x, ALLOW_INTERPOLATION=False,):
        act, y, ACTIVATION_STATUS = x, None, 'MISS'

        if ALLOW_INTERPOLATION:
            self.interpolation_buffer = {} # clear cache, just in case
            
        for layer_k in range(1,1+ self.total_n_layer):
            y, act = self.activate_layer(act, layer_k=layer_k, get_val=True)

            if ALLOW_INTERPOLATION:
                ACTIVATION_STATUS = 'INTERPOLATE'
                self.inlayer_signal_collection(**{'act':act,'layer_k':layer_k})

            if np.any(act > self.get_layer_setting(layer_k)['activation_threshold']):
                ACTIVATION_STATUS = 'HIT'
                break
    
        if ACTIVATION_STATUS == 'HIT':
            if self.output_mode == 'discrete': y = binarize(y, self.n_class)
            return y, act, ACTIVATION_STATUS, layer_k
        elif ACTIVATION_STATUS == 'MISS':
            return None, None, ACTIVATION_STATUS, -1
        elif ACTIVATION_STATUS == 'INTERPOLATE':
            y, act, interp_info = self.perform_interpolation() # y is automatically a binary vector if discrete
            return y, act, ACTIVATION_STATUS, interp_info
        else:
            raise RuntimeError('??')

    def forward_to_layer_k(self,x, layer_k):
        act = x
        y = None
        for k in range(1,1+layer_k): # if layer_k==1, then we go through no loop
            get_val = True if k==layer_k else False
            y, act = self.activate_layer(act, layer_k=k, get_val=get_val)
        return y, act

    #############################################
    # Abstract Functions to implement for Layer Construction
    # This is just a guideline, no need to follow strictly. 
    # See SQANN for implementation.
    #############################################

    def get_layer_setting(self):
        raise NotImplementedError()

    def forward_to_layer_k_for_construction(self):
        raise NotImplementedError()

    def layer_k_sample_collection(self):
        raise NotImplementedError()

    def fit_data(self):
        raise NotImplementedError()


