import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from .utils import find_two_values_with_max_activations

def supergauss(x,a,n=4):
    x = np.clip(x,-100.,100.)
    return np.exp(-(x/a)**(2*n))

def selective_activation(x,a=0.001):
    return a/(a+x**2)

def double_selective_activation(x,r=0.5, a1=1e-4, a2=1.):
    """
    x = np.linspace(-2.,2.,480)
    y = double_selective_activation(x)
    plt.figure()
    plt.plot(x,y)
    plt.show()
    """
    out = (1-r)*selective_activation(x,a1) + r*supergauss(x,a2)
    return np.clip(out, 0.,1.) 

class SQANN(object):
    # Semi-quantized activation NN 
    def __init__(self, layer_settings, N):
        super(SQANN, self).__init__()
        # N: data_size, no of data

        # For the following dictionaries key is index to layer, starting from 1
        self.layer_settings = layer_settings 
        self.layer_nodes = defaultdict(list) 
        self.layer_nodes_alphas = defaultdict(list)

        # only need references to each data sample
        self.unused_indices = list(range(N))
        self.used_indices = defaultdict(list) # key is index to layer, starting from 1        

        self.total_n_layer = None # update during training/data fitting

        self.interpolation_buffer = {}

    def remove_index_to_layer_node(self, i, layer_k):
        moved_index = self.unused_indices.pop(self.unused_indices.index(i))
        self.used_indices[layer_k].append(moved_index)

    def return_index_from_layer(self, layer_k):
        # for order integrity
        returned_indices = []
        for this_index in self.used_indices[layer_k]:
            returned_indices.append(this_index) 
        self.unused_indices = self.unused_indices + returned_indices
        
        self.used_indices[layer_k] = []
        self.layer_nodes[layer_k] = []
        self.layer_nodes_alphas[layer_k] = []


    def activate_layer(self, x, layer_k=1, get_val=True):
        lsett = self.layer_settings[layer_k]
        # act: activation, at the synapse! we also denote it as v
        act = np.sum((x-self.layer_nodes[layer_k])**2, axis=1)**0.5
        # np.sum(shape (1,D)- shape (N_nodes,D)), axis 1)^2 --> shape (N_nodes,1)
        act = double_selective_activation(act,a1=lsett['a1'],a2=lsett['a2']) 
  
        if get_val:
            y = self.layer_nodes_alphas[layer_k][np.argmax(act)]
        else:
            y = None
        return y, act

    def SQANN_propagation(self, x, ALLOW_INTERPOLATION=False,):
        act = x
        y = None        
        ACTIVATION_STATUS = 'MISS'

        if ALLOW_INTERPOLATION:
            ACTIVATION_STATUS = 'INTERPOLATE'
            self.interpolation_buffer = {} # clear cache, just in case
            
        for layer_k in range(1,1+ self.total_n_layer):
            y, act = self.activate_layer(act, layer_k=layer_k, get_val=True)

            if ALLOW_INTERPOLATION:
                ACTIVATION_STATUS = 'INTERPOLATE'
                # print('act.shape:',act.shape)
                max_lower, max_higher, max_lower_act, max_higher_act, max_lower_index, max_higher_index  = \
                    find_two_values_with_max_activations(act, self.layer_nodes_alphas[layer_k])

                if max_lower_act not in self.interpolation_buffer:
                    self.interpolation_buffer[max_lower_act] = (max_lower, layer_k, max_lower_index)
                if max_higher_act not in self.interpolation_buffer:
                    self.interpolation_buffer[max_higher_act] = (max_higher, layer_k, max_higher_index)

                # Store the TWO values of layer node alphas that are most strongly activated
                # uncomment the print to see what's going on.
                # print('before:')
                # for x,y in self.interpolation_buffer.items():
                #     print('%4s:%4s'%(str(np.round(x,3)),str(np.round(y,3))))
                self.interpolation_buffer = sorted(self.interpolation_buffer.items())[-2:]
                self.interpolation_buffer = {x:y for x,y in self.interpolation_buffer} 
                # print('after:')
                # for x,y in self.interpolation_buffer.items():
                #     print('%4s:%4s'%(str(np.round(x,3)),str(np.round(y,3))))

            if np.any(act > self.layer_settings[layer_k]['activation_threshold']):
                ACTIVATION_STATUS = 'HIT'
                break
    
        if ACTIVATION_STATUS == 'HIT':
            return y, act, ACTIVATION_STATUS, layer_k
        elif ACTIVATION_STATUS == 'MISS':
            return None, None, ACTIVATION_STATUS, -1
        elif ACTIVATION_STATUS == 'INTERPOLATE':
            # only simple linear interpolation of TWO POINTS for now
            # can modify here
            y = 0.
            assert(len(self.interpolation_buffer)==2)

            interp_info = self.interpolation_buffer
            total_act= 0.
            for act, y1 in self.interpolation_buffer.items():
                # y1 is (max_lower, layer_k, max_lower_index)
                y += act * y1[0]
                total_act += act
            y = y/total_act
            act = (total_act)/len(self.interpolation_buffer)
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

    def forward_to_layer_k_for_construction(self,x, layer_k):
        act = x
        y = None
        COLLISION = {'isCollision': False, 'collided_layer': None, 'perpetrator_index':None }
        for k in range(1,1+layer_k):
            get_val = True if k==layer_k else False
            y, act = self.activate_layer(act, layer_k=k, get_val=get_val)
            if np.any(act > self.layer_settings[k]['activation_threshold']):
                COLLISION = {'isCollision': True, 'collided_layer': k, 'perpetrator_index':None }
                break
        
        return y, act, COLLISION

    def push_node_to_layer(self, this_index, x, y, layer_k):      
        lsett = self.layer_settings[layer_k]
        self.remove_index_to_layer_node(this_index, layer_k)
        _ , this_sample = self.forward_to_layer_k(x, layer_k=layer_k-1, )
        
        self.layer_nodes[layer_k] = np.concatenate((self.layer_nodes[layer_k], [this_sample]),axis=0)
        self.layer_nodes_alphas[layer_k].append(y)

    def too_many_layers(self):
        THIS_ERROR = "More layers are required than the no. of layer settings specified.\nWe are not handling this for now."
        raise NotImplementedError(THIS_ERROR)

    def layer_k_sample_collection(self, X, Y, layer_k, verbose=0):
        # asssume X is (N,D), Y is (N,)

        STOP_SIGNAL = None
        lsett = self.layer_settings[layer_k] if layer_k in self.layer_settings else self.too_many_layers()

        if len(self.unused_indices)==0:
            if verbose>=20:
                print('Exiting layer_k_sample_collection() because all data have been used.')
            STOP_SIGNAL = 'NO_MORE_DATA'
            COLLISION = {'isCollision': False, 'collided_layer': None, 'perpetrator_index':None }
            return STOP_SIGNAL, COLLISION

        ######### populate node ###########
        init_index = self.unused_indices[0]
        # print('init_index:',init_index)
        this_sample = X[init_index,:]

        _, this_sample, COLLISION = self.forward_to_layer_k_for_construction(this_sample,layer_k=layer_k-1)        
        if COLLISION['isCollision']:
            STOP_SIGNAL = 'COLLISION'
            COLLISION['perpetrator_index'] = init_index
            return STOP_SIGNAL, COLLISION

        nodes = [this_sample] # neurons
        node_values = [Y[init_index]]
        node_added = 1
        self.remove_index_to_layer_node(init_index,layer_k)

        for i in self.unused_indices:
            this_sample = X[i,:]

            _,this_sample, COLLISION = self.forward_to_layer_k_for_construction(this_sample,layer_k=layer_k-1)
            if COLLISION['isCollision']:
                STOP_SIGNAL = 'COLLISION'
                COLLISION['perpetrator_index'] = i
                return STOP_SIGNAL, COLLISION

            # check layer k activation
            # np.sum(shape (1,D)- shape (N_nodes,D)), axis 1)^2 --> shape (N_nodes,1)
            # this is simlar to self.activate_layer(), but we are using nodes that are built in-progress
            act = np.sum((this_sample-np.array(nodes))**2, axis=1)**0.5 
            act = double_selective_activation(act,a1=lsett['a1'],a2=lsett['a2'])

            if np.all(act < lsett['admission_threshold']):
                ######### populate node ########### 
                nodes = np.concatenate((nodes,[this_sample]),axis=0)
                node_values.append(Y[i])
                node_added += 1
                self.remove_index_to_layer_node(i,layer_k)    


        self.layer_nodes[layer_k] = nodes
        self.layer_nodes_alphas[layer_k] = node_values
        self.total_n_layer = layer_k 


        COLLISION = {'isCollision': False, 'collided_layer': None, 'perpetrator_index':None }
        return STOP_SIGNAL, COLLISION


    def fit_data(self, X, Y, verbose=20):
        if verbose>=20: print()
        layer_now = 1
        while True:
            STOP_SIGNAL, COLLISION = self.layer_k_sample_collection(X, Y, layer_k=layer_now, verbose=verbose)
            if verbose>=100:
                print('[%s] net.used_indices: (n nodes:%s, total no of layers:%s)'%(
                    str(layer_now),str( len(self.used_indices[layer_now])),str(self.total_n_layer) ))
                print('  ', self.used_indices[layer_now],)
            
            if STOP_SIGNAL=='NO_MORE_DATA': 
                break
            elif STOP_SIGNAL == 'COLLISION':
                collided_layer = COLLISION['collided_layer']

                for layer_j in range(collided_layer+1,layer_now+1):
                    self.return_index_from_layer(layer_j)
                kp = COLLISION['perpetrator_index']
                self.push_node_to_layer(this_index=kp, x=X[kp,:], y=Y[kp], layer_k=collided_layer)
                layer_now = collided_layer

                if verbose>=100:
                    print('<%s> net.used_indices: (n nodes:%s, total no of layers:%s, perpetrator_index:%s)'%(
                        str(layer_now),str( len(self.used_indices[layer_now])),str(self.total_n_layer), str(kp) ))
                    print('  ',self.used_indices[layer_now],)

            layer_now+=1

        if verbose>=20:
            print('Final positions of indices in the layers:')
            for layer_now in range(1,1+self.total_n_layer):
                print('  [%s]'%(str(layer_now)), self.used_indices[layer_now],)