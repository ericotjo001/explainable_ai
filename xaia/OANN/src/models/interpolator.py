import copy
import numpy as np

from .utils import binarize

class OANNInterpolator(object):
    def __init__(self, **settings):
        super(OANNInterpolator, self).__init__()
        self.output_mode = settings['output_mode'] if 'output_mode' in settings \
            else 'continuous' # or discrete
        self.interpolation_mode = settings['interp_mode'] if 'interp_mode' in settings \
            else 'two_max'
        self.interpolation_buffer = {}

        if self.output_mode == 'discrete':
            assert('n_class' in settings)
            self.n_class = settings['n_class']
        else:
            self.n_class = None

    def inlayer_signal_collection(self, **kwargs):
        if self.interpolation_mode == 'two_max':
            self.inlayer_signal_collection_two_max(**kwargs)
        elif self.interpolation_mode == 'layerwise_top_accumulation':
            self.inlayer_signal_collection_layerwise_top_max(**kwargs)
        else:
            raise NotImplementedError()

    def perform_interpolation(self, **kwargs):
        if self.interpolation_mode == 'two_max':
            return self.interpolate_two_max(**kwargs)
        elif self.interpolation_mode == 'layerwise_top_accumulation':
            return self.interpolate_layerwise_top_accumulation(**kwargs)
        else:
            raise NotImplementedError() 

    ###############################
    # FIRST MODE: two max
    ###############################
    def inlayer_signal_collection_two_max(self, act=None, layer_k=None):
        assert(not (act is None or layer_k is None))
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


    def interpolate_two_max(self, ):
        # only simple linear interpolation of TWO POINTS 
        y = 0.
        assert(len(self.interpolation_buffer)==2)

        interp_info = self.interpolation_buffer

        total_act= 0.
        for act, y1 in self.interpolation_buffer.items():
            # y1 is (max_lower, layer_k, max_lower_index)

            if self.output_mode=='continuous':
                y += act * y1[0] 
            elif self.output_mode =='discrete':
                y += act * binarize(y1[0], self.n_class)
            else:
                raise NotImplementedError('unknown output_mode')

            total_act += act
        y = y/total_act
        act = (total_act)/len(self.interpolation_buffer)

        return y, act, interp_info

    ###############################
    # SECOND MODE: layerwise_top_max
    ###############################
    def inlayer_signal_collection_layerwise_top_max(self, act=None, layer_k=None):
        assert(not (act is None or layer_k is None))
        actmax = np.max(act)
        idx = np.argmax(act)
        self.interpolation_buffer[layer_k] = {'idx':idx,'actmax':actmax, 
            'alpha': self.layer_nodes_alphas[layer_k][idx] ,'act': act}

        best_candidate = {'layer_k':layer_k, 'idx': idx, 'actmax':actmax, 
            'alpha': self.layer_nodes_alphas[layer_k][idx],'act': act }
        if 'best' not in self.interpolation_buffer:
            self.interpolation_buffer['best'] = best_candidate
        else:
            if actmax > self.interpolation_buffer['best']['actmax']:
                self.interpolation_buffer['best'] = best_candidate

    def interpolate_layerwise_top_accumulation(self, ):
        # only simple linear interpolation of TWO POINTS 
        y = np.zeros(shape=(self.n_class,)) # will be binarized
        y_counter = np.zeros(shape=(self.n_class,))
        interp_info = self.interpolation_buffer

        if self.output_mode=='continuous':
            raise NotImplementedError()
        elif self.output_mode =='discrete':
            for layer_k, info in self.interpolation_buffer.items():
                if layer_k=='best': continue
                actmax = self.interpolation_buffer[layer_k]['actmax']
                alpha = self.interpolation_buffer[layer_k]['alpha']

                if actmax>self.activation_threshold:
                    # this means we get a perfect hit
                    y = binarize(alpha, self.n_class)
                    break
                y = y + actmax * binarize(alpha, self.n_class)
            act = None
        else:
            raise NotImplementedError('unknown output_mode')

        return y, act, interp_info


def find_two_values_with_max_activations(act, y):
    """
    assume act is between 0 and 1
    
    Example:
    N = 5
    act = np.random.uniform(0,1,size=(N,))
    y = np.array(range(N))
    print('act:',act)
    print('y  :',y)
    max_lower, max_higher = find_two_values_with_max_activations(act,y)
    print('max_lower :',max_lower )
    print('max_higher:',max_higher)
    """
    x = copy.deepcopy(act)
    max_higher_index = np.argmax(x)
    max_higher = y[max_higher_index]
    max_higher_act = x[max_higher_index]

    x[max_higher_index] = -np.inf
    max_lower_index = np.argmax(x)
    max_lower = y[max_lower_index]
    max_lower_act = x[max_lower_index]
    return max_lower, max_higher, max_lower_act, max_higher_act, max_lower_index, max_higher_index