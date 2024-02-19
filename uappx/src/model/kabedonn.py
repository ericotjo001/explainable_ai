"""
k-width and bifold embedded data ordered nn
"""
import numpy as np
from src.utils import double_selective_activation

from .nodes import LayerNodes, a1, a2, r
from .indexer import DataIndexer
from .trainer import Trainer
from .interpolator import Interpolator
from .information import LayerInformation

def get_admission_th(L):
    return 0.5

class KABEDONN(LayerNodes, Trainer, Interpolator, LayerInformation):
    def __init__(self, **settings):
        super(KABEDONN, self).__init__()

        
        if settings['init_new']:
            self.activation_threshold = 0.95 if 'activation_threshold' not in settings else settings['activation_threshold']
            self.admission_threshold = get_admission_th if 'admission_threshold' not in settings else settings['admission_threshold']
            self.init_interpolator(settings['interpolator_settings'])
            self.layer_hierarchy = None # to be updated only using self.get_latest_layer_hierarchy()

            self.init_new(**settings)            
            self.ix = DataIndexer (settings['DATA_DIR'], 
                settings['folder_to_class_mapping'], 
                settings['kwidth'], 
                settings['data_fetcher'],
                init_new=settings['init_new'])
        else:
            raise NotImplementedError('For now, loading can be done directly using joblib.load()')
        
    def init_new(self, **settings):
        self.layers = {} # {layer : LayerNodes()}

    def forward(self,x,):
        act, act_pre = None, x
        for layer_ in range(1,1+ len(self.layers)):
            receptors = self.layers[layer_].assemble_receptors() 
            act = self.normalized_stimulation(act_pre, receptors)

            if np.any(act>=self.admission_threshold(layer_)):
                act_idx = np.argmax(act) # activated node index
                activated_node = self.layers[layer_].node_list[act_idx]
                
                y_pred, NODE_INFO = activated_node.forward(act_pre)
                OUTPUT_INFO = {
                    'output_mode': 'activation',  
                    'act':act,
                    'layer': layer_,
                    'act_idx': act_idx,
                    'activated_node':activated_node, 
                    'NODE_INFO': NODE_INFO,}
                return y_pred, OUTPUT_INFO

            interp_buffer = {'layer': layer_,'act': act,'act_pre': act_pre,}
            self.interpolator_signal_collection(interp_buffer)
            self.interpolator_processing()

            act_pre = act
            
        y_pred, OUTPUT_INFO = self.interpolator_output(x)
        return y_pred, OUTPUT_INFO

    def integrate_nodes_layer(self, L, nodes_layer):
        self.layers[L] = nodes_layer # this is a LayerNodes() object, see nodes.py

    def activate_layer_l(self, x, L, filter_mode=False):
        act = x
        for layer_ in range(1,1+L):
            # print('layer_:',layer_)
            receptors = self.layers[layer_].assemble_receptors()  
            act = self.normalized_stimulation(act, receptors)

            if filter_mode:
                FILTER_INFO = {'STATUS': None, }
                if np.any(act>=self.admission_threshold(layer_)):
                    FILTER_INFO['STATUS'] =  'ACTIVATED'
                    FILTER_INFO['LAYER'] = layer_
                    return act, FILTER_INFO

        if filter_mode:
            FILTER_INFO = {'STATUS': None}
            return act, FILTER_INFO         

        return act

    def normalized_stimulation(self, act, receptors):
        norm = np.linalg.norm(act)+ 1e-7
        act = np.sum((act/norm-receptors/norm)**2, axis=1)**0.5 
        act = double_selective_activation(act,a1=a1,a2=a2, r=r)  
        return act

    def get_latest_layer_hierarchy(self): 
        self.layer_hierarchy = self.get_layer_hierarchy() # LayerInformation method

