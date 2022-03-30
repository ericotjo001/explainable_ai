import numpy as np
from collections import defaultdict
from .utils import double_selective_activation

from .base import OANN

class SQANN(OANN):
    def __init__(self, **settings):
        super(SQANN, self).__init__(**settings)
        
        if settings['init_new']:
            self.init_new(**settings)


    def init_new(self, **settings):
        # For the following dictionaries key is index to layer, starting from 1
        if 'layer_settings' in settings:
            self.layer_settings = settings['layer_settings']
        else:
            self.layer_settings = self.make_classic_layer_setting()

        self.layer_nodes = defaultdict(list) 
        self.layer_nodes_alphas = defaultdict(list)
        self.total_n_layer = None # update during training/data fitting

        # only need index references to each data sample
        # N: data_size, no of data. Set this when the data is available.
        self.unused_indices = None # list(range(N)) 
        self.used_indices = defaultdict(list) # key is index to layer, starting from 1 
        
    def make_classic_layer_setting(self):
        print('using classic layer setting, inherited from SQANN...')

        # in SQANN, a1 and a2 are scalars
        # each layer has different a1 and a2
        def make_layer_setting(a1, a2, 
            admission_threshold, activation_threshold,):
            this_layer_setting = {
                'a1':a1, 
                'a2':a2, 
                'admission_threshold':admission_threshold, 
                'activation_threshold': activation_threshold,
            } 
            return this_layer_setting
        MAX_LAYER = 24
        a1s = np.linspace(0.001,1.,51)
        a2s = np.linspace(1.,0.7,101)
        layer_settings = {
            # make_layer_setting(a1, a2,  admission_threshold, activation_threshold, max_n)
            i: make_layer_setting(a1s[i], a2s[i], 0.1, 0.9, ) for i in range(1,1+MAX_LAYER) 
        }
        return layer_settings

    def SQANN_propagation(self, x, ALLOW_INTERPOLATION=False,):
        return self.forward(x, ALLOW_INTERPOLATION=ALLOW_INTERPOLATION)

    #############################################
    # Implementation of Layer Construction functions
    #############################################

    def get_layer_setting(self, layer_k):
        return self.layer_settings[layer_k]

    def forward_to_layer_k_for_construction(self,x, layer_k):
        act, y = x, None
        COLLISION = {'isCollision': False, 'collided_layer': None, 'perpetrator_index':None }
        for k in range(1,1+layer_k):
            get_val = True if k==layer_k else False
            y, act = self.activate_layer(act, layer_k=k, get_val=get_val)
            if np.any(act > self.get_layer_setting(k) ['activation_threshold']):
                COLLISION = {'isCollision': True, 'collided_layer': k, 'perpetrator_index':None }
                break
        return y, act, COLLISION

    def layer_k_sample_collection(self, X, Y, layer_k, verbose=0):
        """
        asssume X is (N,D), Y is (N,Dy), both numpy arrays
        
        output: 
           POST_PROCESSING_INFO =  {
            'STOP_SIGNAL': STOP_SIGNAL, 
            'isCollision': isCollision, 
            'collided_layer': collided_layer, 
            'perpetrator_index':perpetrator_index 
        }

        """

        lsett = self.layer_settings[layer_k] if layer_k in self.layer_settings else self.too_many_layers()
        if len(self.unused_indices)==0:
            self.print('alldataused', verbose=verbose)
            return {'STOP_SIGNAL':'NO_MORE_DATA', 'isCollision': False, 
                'collided_layer': None, 'perpetrator_index':None }

        nodes, node_values, node_added = None, [], 0
        for i in self.unused_indices:
            this_sample = X[i,:]

            _,this_sample, COLLISION = self.forward_to_layer_k_for_construction(this_sample,layer_k=layer_k-1)
            if COLLISION['isCollision']:
                return {'STOP_SIGNAL':'COLLISION', 'isCollision': COLLISION['isCollision'], 
                    'collided_layer': COLLISION['collided_layer'], 'perpetrator_index':i }

            if node_added==0:
                INSERT_NODE = True
            else:
                # check layer k activation
                # np.sum(shape (1,D)- shape (N_nodes,D)), axis 1)^2 --> shape (N_nodes,1)
                # this is simlar to self.activate_layer(), but we are using nodes that are built in-progress
                act = np.sum((this_sample-np.array(nodes))**2, axis=1)**0.5 
                act = double_selective_activation(act,a1=lsett['a1'],a2=lsett['a2'])    
                INSERT_NODE = np.all(act < lsett['admission_threshold'])
            
            if INSERT_NODE:
                nodes = [this_sample] if nodes is None else np.concatenate((nodes,[this_sample]),axis=0)
                node_values.append(Y[i])
                node_added += 1
                self.move_index_to_layer_node(i,layer_k)    

        self.layer_nodes[layer_k] = nodes
        self.layer_nodes_alphas[layer_k] = node_values
        self.total_n_layer = layer_k 

        return {'STOP_SIGNAL':None , 'isCollision': False, 
            'collided_layer': None, 'perpetrator_index':None }

    def fit_data(self, X, Y, verbose=20):
        if self.unused_indices is None:
             self.unused_indices = list(range(len(X)))

        layer_now = 1
        while True:
            POST_PROCESSING_INFO = self.layer_k_sample_collection(X, Y, layer_k=layer_now, verbose=verbose)
            self.print('mainflow', layer_now=layer_now, verbose=verbose)
            
            if POST_PROCESSING_INFO['STOP_SIGNAL']=='NO_MORE_DATA': 
                break
            elif POST_PROCESSING_INFO['STOP_SIGNAL'] == 'COLLISION':
                collided_layer = POST_PROCESSING_INFO['collided_layer']
                for layer_j in range(collided_layer+1,layer_now+1):
                    self.return_index_from_layer(layer_j)
                kp = POST_PROCESSING_INFO['perpetrator_index']
                self.push_node_to_layer(this_index=kp, x=X[kp,:], y=Y[kp], layer_k=collided_layer)
                layer_now = collided_layer

                self.print('collision', layer_now=layer_now, kp=kp, verbose=verbose)
            layer_now+=1
        self.print('finalposition', verbose=verbose)

    def too_many_layers(self):
        THIS_ERROR = "More layers are required than the no. of layer settings specified.\nWe are not handling this for now."
        raise NotImplementedError(THIS_ERROR)

    ####################
    # extra utilities
    ####################

    def print(self, mode, **kwargs):
        if mode=='mainflow':
            self.print_mainflow(**kwargs)
        elif mode=='collision':
            self.print_collision(**kwargs)
        elif mode=='finalposition':
            self.print_finalpos(**kwargs)
        elif mode=='alldataused':
            self.print_alldataused(**kwargs)

    def print_mainflow(self, layer_now, verbose):
        if verbose>=100:
            print('[%s] net.used_indices: (n nodes:%s, total no of layers:%s)'%(
                str(layer_now),str( len(self.used_indices[layer_now])),str(self.total_n_layer) ))
            print('  ', self.used_indices[layer_now],)        

    def print_collision(self, layer_now, kp, verbose):
        if verbose>=100:
            print('<%s> net.used_indices: (n nodes:%s, total no of layers:%s, perpetrator_index:%s)'%(
                str(layer_now),str( len(self.used_indices[layer_now])),str(self.total_n_layer), str(kp) ))
            print('  ',self.used_indices[layer_now],)

    def print_finalpos(self, verbose):
        if verbose>=20:
            print('Final positions of indices in the layers:')
            for layer_now in range(1,1+self.total_n_layer):
                print('  [%s]'%(str(layer_now)), self.used_indices[layer_now],)

    def print_alldataused(self, verbose):
        if verbose>=20:
            print('Exiting layer_k_sample_collection() because all data have been used.')        