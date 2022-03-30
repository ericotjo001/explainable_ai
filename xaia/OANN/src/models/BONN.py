import os, joblib
import queue
import numpy as np
from collections import defaultdict
import joblib

from .base import OANN
from OANN.src.utils import get_file_extension
from .utils import double_selective_activation

ALLOWED_DATA_FORMAT = ['npy']

class BONN(OANN):

    STATES = ['output_mode', 'interpolation_mode', 'n_class',
        'layer_nodes', 'layer_nodes_alphas', 'total_n_layer',
        'init_values','layer_settings', 'activation_threshold', 'admission_threshold',
        ]
    IX_STATES = [
        'DATA_DIR', 'DATA_INDEXER_DIR', 'folder_to_class_mapping', 'class_to_folder_mapping','elasticsize',
        'data_layer_status', 'data_index_status', 'data_pointer', 'data_sizes'
    ]
    
    def __init__(self, **settings):
        super(BONN, self).__init__(**settings)
        self.r = 0.1
        if settings['init_new']:
            self.init_new(**settings)
        else:
            self.load_states(**settings)

    def init_new(self, **settings):
        self.output_mode = settings['output_mode']

        self.layer_nodes = defaultdict(list) 
        self.layer_nodes_alphas = defaultdict(list)

        self.total_n_layer = 0 # update during training/data fitting
        self.ix = DataIndexer(settings['DATA_DIR'], 
            settings['folder_to_class_mapping'], 
            elasticsize=settings['elasticsize'], 
            init_new=settings['init_new'],
            data_fetcher=settings['data_fetcher'])

        self.init_values = {
            'a1':0.001,
            'a2':0.1,
        }
        self.layer_settings = {}
        self.activation_threshold = 1- 1e-7 # 1.0 - 1e-7 
        self.admission_threshold = 0.9 # 1. - 1e-2

    def load_states(self,**settings):
        print('loading states...')
        states = joblib.load(settings['MODEL_DIR'])
        ix_states = joblib.load(settings['MODEL_DIR']+'.indexer.states')
        for state, data in states.items():
            setattr(self, state, data)

        self.ix = DataIndexer(None, None, data_fetcher='dummy')
        for state, data in ix_states.items():
            setattr(self.ix, state, data)

    def save_state(self, MODEL_DIR):
        states = {state: getattr(self,state) for state in self.STATES}
        ix_states = {state: getattr(self.ix,state) for state in self.IX_STATES}
        joblib.dump(states, MODEL_DIR)
        joblib.dump(ix_states, MODEL_DIR+'.indexer.states')


    ###############################################
    # Data fitting
    ###############################################

    def fit_data(self, max_iter=None, verbose=100):
        iter_ = 0 
        elasticset = []
        current_layer = 1
        def update_net(iter_ , elasticset, current_layer, nodes, node_values, 
            a1vec, a2vec, verbose):
            self.get_layer_setting(current_layer) # to initiate layer, if not yet existent

            if verbose>=100: print('updating net...')
            self.ix.integrate_data_indices(elasticset, layer_k=current_layer)
            self.integrate_data_nodes(current_layer, nodes, node_values, a1vec, a2vec,)

        all_decomissioned_nodes = []
        all_unresolvable_nodes = []
        while True:
            print(f'\n[pre iter {iter_}] current_layer:{current_layer}')
            elasticset, INFO = self.ix.prepare_elasticset_(elasticset)
            
            elasticset, decomissioned, nodes, node_values, a1vec, a2vec, POST_PROCESSING_INFO \
                = self.test_elastic_set_activation(elasticset, current_layer)            

            if len(elasticset)>=self.ix.elasticsize:
                update_net(iter_ , elasticset,current_layer, nodes, node_values,  a1vec, a2vec, verbose)
                current_layer+=1 # point to next layer

                # reset
                elasticset = []

            self.ix.update_decommisioned_data(decomissioned)
            all_decomissioned_nodes.extend(decomissioned)
            all_unresolvable_nodes.extend(POST_PROCESSING_INFO['UNRESOLVABLES'])

            if INFO['ALL_DATA_USED_UP']:
                if len(elasticset)>0:
                    update_net(iter_ , elasticset, current_layer, nodes, node_values, a1vec, a2vec, verbose)
                break

            self.print_fitting_info('layerstatus', *[iter_, current_layer, elasticset, decomissioned], verbose=verbose )
            self.print_fitting_info('post_iter', *[], verbose=verbose)
            iter_ += 1
            if max_iter is not None:
                if iter_ >= max_iter: break

        eofit = "========== End of fitting ============"
        print(f"\n{eofit}\nUNRESOLVABLES{all_unresolvable_nodes}")
        print('all_decomissioned_nodes:', all_decomissioned_nodes)
        self.print_fitting_info('layerstatus', *[iter_, current_layer, [], []], verbose=verbose )
        self.print_fitting_info('post_iter', *[], verbose=verbose)
        total_nodes = 0
        for k, alphas in self.layer_nodes_alphas.items():
            print(f'layer:{k} has {len(alphas)} nodes')
            total_nodes += len(alphas)
        print(f'self.total_n_layer:{self.total_n_layer}. total_nodes:{total_nodes}')

        return { # EXTRA INFO
            'decomissioned': all_decomissioned_nodes, 
            'unresolvable': all_unresolvable_nodes,
        }

 
    def print_fitting_info(self,mode, *args, verbose=100):
        if mode=='layerstatus' and verbose>=100:
            self.print_fitting_info_layerstatus(*args)
        if mode=='post_iter' and verbose>=100:
            self.print_fitting_info_post_iter()
        if mode=='layer_settings' and verbose>=100:
            self.print_fitting_info_layersettings()
    def print_fitting_info_layerstatus(self, i, current_layer, elasticset, decomissioned):
        print(f'[post iter {i}], working on current_layer:{current_layer}\nelasticset:{elasticset}\ndecomissioned:{decomissioned}', )

    def print_fitting_info_post_iter(self):
        print('layer status:')
        self.ix.print_(mode='datalayerstatus')
        print('datapointer:')
        self.ix.print_(mode='datapointer')

    def print_fitting_info_layersettings(self):
        print('layer settings:')
        for k,lsett in self.layer_settings.items():
            print(f'layer {k}\n  {lsett}')

    ###############################################
    # Data fitting components
    ###############################################

    def get_layer_setting(self, layer_k):
        if layer_k not in self.layer_settings: 
            self.layer_settings[layer_k] = {
                'a1': None, # shape: like activation (act) in that layer,
                'a2': None, # shape: like a1,
                }

        this_layer_setting = self.layer_settings[layer_k]

        # value to pass to base class
        this_layer_setting['activation_threshold'] = self.activation_threshold

        return this_layer_setting

    def forward_to_layer_k_for_construction(self,x, layer_k, y0):
        act, y = x, None
        INFO = {'STATUS':None}
        for k in range(1,1+layer_k):
            y, act1 = self.activate_layer(act, layer_k=k, get_val=True)

            if k<layer_k: # at the last layer, finetune_fresh_layer() do the job]                
                IS_DISTINCT = self.is_distinct_and_intermediate_but_unrecognized(y,y0)
                if IS_DISTINCT and np.any(act1>self.activation_threshold):
                    INFO['STATUS'] = 'UNRESOLVABLE'
                    break
            act = act1
        return y, act, INFO

    def compute_layer_activation(self, this_sample, nodes,current_layer, a1vec, a2vec):
        act = np.sum((this_sample-np.array(nodes))**2, axis=1)**0.5 
        act = double_selective_activation(act,a1vec,a2vec) 
        return act

    def test_elastic_set_activation(self, elasticset, current_layer):
        new_elastic_set, decomissioned = [], []
        elasticset_to_index = {idx:i for i, idx in enumerate(elasticset)}
        POST_PROCESSING_INFO = {'UNRESOLVABLES':[]}

        #################################
        # main process here
        x_batch,y0_batch = self.ix.fetch_data_by_elastic_set(elasticset)

        nodes, node_values, node_added = None, [], 0
        a1vec, a2vec = [], []
        for i,(x,y0) in enumerate(zip(x_batch, y0_batch)):
            _, this_sample, INFO = self.forward_to_layer_k_for_construction(x, current_layer-1, y0)

            if INFO['STATUS']=='UNRESOLVABLE':               
                INSERT_NODE = False
                POST_PROCESSING_INFO['UNRESOLVABLES'].append(elasticset[i])
            elif node_added==0:
                INSERT_NODE = True
            else:
                # check layer k activation
                # np.sum(shape (1,D)- shape (N_nodes,D)), axis 1)^2 --> shape (N_nodes,1)
                # this is simlar to self.activate_layer(), but we are using nodes that are built in-progress
                act = self.compute_layer_activation(this_sample, nodes, current_layer, a1vec, a2vec)
                INSERT_NODE = np.all(act < self.admission_threshold)

                # COLLISION MAY OCCUR. IF COLLISIONS OCCUR BETWEEN DIFFERENT ALPHA VALUES, 
                # IT MEANS THE DISTINGUISHING POWER OF IS TOO LOW.
                # THEN, WE "TRAIN" THE PARAMETER SO THAT IT NO LONGER TREATS IT AS A COLLISION
                if not INSERT_NODE:
                    IS_DISTINCT = self.is_distinct_but_unrecognized( y0, x_batch, 
                        y0_batch,new_elastic_set[np.argmax(act)], elasticset, elasticset_to_index)
                    activated_elastic_index = elasticset_to_index[new_elastic_set[np.argmax(act)]]                                     

                    if IS_DISTINCT:
                        while not INSERT_NODE:
                            a1vec, a2vec = self.finetune_fresh_layer(this_sample, nodes, current_layer, act, a1vec, a2vec)
                            act = self.compute_layer_activation(this_sample, nodes, current_layer, a1vec, a2vec)
                            INSERT_NODE = np.all(act < self.admission_threshold)

            if INSERT_NODE:
                nodes = [this_sample] if nodes is None else np.concatenate((nodes,[this_sample]),axis=0)
                node_values.append(y0)
                new_elastic_set.append(elasticset[i])                
                a1vec.extend([self.init_values['a1']])
                a2vec.extend([self.init_values['a2']])
                node_added += 1                
            else:
                decomissioned.append(elasticset[i])
                
                #######################################
                # double check, help debugging
                #######################################
                act = self.compute_layer_activation(this_sample, nodes, current_layer, a1vec, a2vec)
                pred_idx = np.argmax(act)
                if node_values[pred_idx] != y0:
                    raise RuntimeError('DANGEROUS HERE!')


        return new_elastic_set, decomissioned, \
            nodes, node_values, a1vec, a2vec, \
            POST_PROCESSING_INFO

    def integrate_data_nodes(self, layer_k, nodes, node_values, a1vec, a2vec):
        self.layer_nodes[layer_k] = nodes
        self.layer_nodes_alphas[layer_k] = node_values
        self.total_n_layer = layer_k 

        self.layer_settings[layer_k]['a1'] = np.array(a1vec) 
        self.layer_settings[layer_k]['a2'] = np.array(a2vec)


    ###############################################
    # fine tuning
    ###############################################

    def is_distinct_and_intermediate_but_unrecognized(self, y, y0):
        if int(y) != int(y0):
            return True
        return False

    def is_distinct_but_unrecognized(self, y0, x_batch, y0_batch,
        elastic_index, elasticset, elasticset_to_index):
        activated_elastic_index = elasticset_to_index[elastic_index] 

        # check if activated_elastic_index activates other nodes instead here
        if int(y0) == int(y0_batch[activated_elastic_index]):
            return False
        return True

    def finetune_intermediate(self, k, act, act1):
        raise RuntimeError('Not finetuneable')

    def finetune_fresh_layer_(self, layer_k, act,a1vec, a2vec):
        PERPETRATOR_INDICES = act > self.admission_threshold
        for i,is_perpetrator in enumerate(PERPETRATOR_INDICES):
            if is_perpetrator:
                # generally, decrease a1 and a2 
                a1vec[i] *= 0.9 
                a2vec[i] *= 0.9 
        return a1vec, a2vec

    def finetune_fresh_layer(self, this_sample, nodes, current_layer, act, a1vec, a2vec):
        while True:
            a1vec, a2vec = self.finetune_fresh_layer_(current_layer, act, a1vec, a2vec)
            act = self.compute_layer_activation(this_sample, nodes, current_layer,a1vec, a2vec,)
            
            INSERT_NODE = np.all(act < self.admission_threshold)
            if INSERT_NODE: break
        return a1vec, a2vec

    ###############################################
    # lookup
    ###############################################

    def get_node_to_data_lookup_dict(self):
        return

class DataIndexer():
    def __init__(self, DATA_DIR, folder_to_class_mapping, 
                elasticsize=10,
                data_fetcher=None,
                init_new=False,):
        super(DataIndexer, self).__init__()        
        """
        DATA_DIR: str, points to a folder containing folders of images arranged based on class e.g.
        + class0
          - x0.npy
          - x1.npy
          - ...
        + class1
        + specialfolder

        !! VERY IMPORTANT !!
        Index of data in each folders are in lexical order.
        For example, if your folder contains 1.npy, 2.npy, ..., 99.npy
        the order in which the data is referred too are:
          1.npy, 10.npy, 11.npy ...., 2.npy,...
        """
        self.data_fetcher = data_fetcher
        assert(self.data_fetcher is not None)
        if init_new:
            self.do_init_new(DATA_DIR, folder_to_class_mapping, elasticsize)


    def do_init_new(self, DATA_DIR, folder_to_class_mapping, elasticsize): 
        self.DATA_DIR = DATA_DIR
        self.DATA_INDEXER_DIR = DATA_DIR + '.ix'
        self.folder_to_class_mapping = folder_to_class_mapping
        self.class_to_folder_mapping = self.get_class_to_folder_mapping(folder_to_class_mapping)
        self.elasticsize = elasticsize

        self.indices_buffer = queue.Queue() 
        self.DATA_INDEXER = self.load_data_indexer() # load it on the spot

        """
        ########################
        Init the indices here
        ########################
        The following are arranged according to data folder
        self.data_layer_status:
            { class/label (int) : list of int: -1 ==> unused. -2 ==> decommissioned. >= 1 ==> layer it belongs to now }
            Yes, for layer status, index 0 is not used, just for consistency 
        self.data_index_status: 
            { class/label (int) : list of int: -1 ==> unused/decommisioned.  >=0 ==> the node index it belongs to now} 
        self.data_pointer:
            { class/label (int) : int: -1 ==> unused, -2==> all used, >=0 ==> current position}    
        self.data_sizes     
            { class/label (int) : int (size of this class)}    
        """
        self.data_layer_status = {} 
        self.data_index_status = {}  
        self.data_pointer = {} 
        self.data_sizes = {}

        self.initiate_indices(DATA_DIR, catmap=folder_to_class_mapping)


    def initiate_indices(self, DATA_DIR, catmap, data_format='npy'):
        print('DataIndexer. initiate_indices...')

        class_folders = os.listdir(DATA_DIR)
        for classname in class_folders:
            list_of_data = [x for x in os.listdir(os.path.join(DATA_DIR, classname)) \
                if get_file_extension(x) in ALLOWED_DATA_FORMAT]
            n_data = len(list_of_data)
            self.data_layer_status[catmap[classname]] = np.zeros(shape=(n_data,), dtype=int) -1
            self.data_index_status[catmap[classname]] = np.zeros(shape=(n_data,), dtype=int) -1
            self.data_pointer[catmap[classname]] = -1
            self.data_sizes[catmap[classname]] = n_data 

    #########################################
    # Fetch data!
    #########################################

    def fetch_data_by_elastic_set(self, elasticset):
        x_batch, y_batch = [], []
        for classlabel,current_index in elasticset:
            classname = str(self.class_to_folder_mapping[classlabel])
            dataname = self.DATA_INDEXER[classname][current_index]
            DATA_DIR = os.path.join(self.DATA_DIR, classname, str( dataname))    

            x = self.data_fetcher(DATA_DIR)
            x_batch.append(x)
            y_batch.append(classlabel)
        return x_batch, y_batch

    #########################################
    # Data Management
    #########################################

    def prepare_elasticset_(self, elasticset):
        """
        elasticset: a set of indices. It will be continually readjusted until a BONN layer is formed.
        INFO = {'ALL_DATA_USED_UP':ALL_DATA_USED_UP}
        """
        assert(len(elasticset) <= self.elasticsize)
        ALL_DATA_USED_UP = False
        while self.indices_buffer.qsize() < self.elasticsize:
            ALL_DATA_USED_UP = self.check_if_all_data_used_up()
            self.populate_buffer()
            if ALL_DATA_USED_UP: break

        n_pop = self.indices_buffer.qsize() if ALL_DATA_USED_UP else self.elasticsize - len(elasticset)
        for i in range(n_pop):
            deq = self.indices_buffer.get()
            elasticset.append(deq)

        if ALL_DATA_USED_UP and self.indices_buffer.qsize()==0: 
            return elasticset, {'ALL_DATA_USED_UP':ALL_DATA_USED_UP} 
        return elasticset, {'ALL_DATA_USED_UP':False}

    def populate_buffer(self):
        # traverse through each class ONCE
        for classlabel, current_index in self.data_pointer.items():
            if current_index == -2: continue

            if current_index+1 >= self.data_sizes[classlabel]:
                self.data_pointer[classlabel] = -2                
                continue
            # now we know there is an available data in this class, let's put it into the buffer
            current_index = current_index + 1 
            self.indices_buffer.put((classlabel,current_index))

            # update the pointer
            self.data_pointer[classlabel] = current_index

    def check_if_all_data_used_up(self):
        ALL_DATA_USED_UP = True
        for classlabel, current_index in self.data_pointer.items():
            if current_index+1 >= self.data_sizes[classlabel]:
                self.data_pointer[classlabel] = -2                
                continue
            if current_index != -2: return False
        return ALL_DATA_USED_UP

    """
    Let's update data by indices
    both "decomissioned" and "elasticset" are list of tuples (classlabel, thisindex)
    """
    def update_decommisioned_data(self, decomissioned):
        for i, (classlabel, current_index) in enumerate(decomissioned):
            self.data_layer_status[classlabel][current_index] = -2
            self.data_index_status[classlabel][current_index] = -2

    def integrate_data_indices(self, elasticset, layer_k):
        for i, (classlabel, current_index) in enumerate(elasticset):
            self.data_layer_status[classlabel][current_index] = layer_k
            self.data_index_status[classlabel][current_index] = i

    ###################
    # utils
    ###################
    def load_data_indexer(self):
        if not os.path.exists(self.DATA_INDEXER_DIR):
            print(f'creating new data indexer for data in {self.DATA_INDEXER_DIR}')
            data_indexer = {}
            for classname in os.listdir(self.DATA_DIR):
                data_indexer[classname] = os.listdir(os.path.join(self.DATA_DIR, classname))                
            joblib.dump(data_indexer, self.DATA_INDEXER_DIR)
        else:
            print(f'loading data indexer from {self.DATA_INDEXER_DIR}')
            data_indexer = joblib.load(self.DATA_INDEXER_DIR)
        return data_indexer

    def get_class_to_folder_mapping(self, folder_to_class_mapping):
        class_to_folder_mapping = ['dummy' for i in range(len(folder_to_class_mapping))]
        for foldername, classlabel in folder_to_class_mapping.items():
            class_to_folder_mapping[classlabel] = foldername
        return class_to_folder_mapping

    def print_(self, **kwargs):
        if kwargs['mode']=='datalayerstatus':
            self.print_datalayerstatus(**kwargs)
        elif kwargs['mode']=='datasizes':
            self.print_datasizes(**kwargs)
        elif kwargs['mode']=='datapointer':
            self.print_dataspointer(**kwargs)

    def print_datalayerstatus(self, **kwargs):
        print('data indexer data layer status...')        
        for i,x in self.data_layer_status.items():
            print(f'  {i}:{x}')

    def print_datasizes(self, **kwargs):
        print('data indexer datasizes...')
        for i,n in self.data_sizes.items():
            print(f'  {i}:{n}')

    def print_dataspointer(self, **kwargs):
        print('data indexer datapointer...')
        for i,p in self.data_pointer.items():
            print(f'  {i}:{p}')
