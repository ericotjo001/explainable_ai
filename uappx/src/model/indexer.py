import os, joblib
import numpy as np
import queue
import random 
class kFetcher(object):
    def __init__(self, kwidth):
        super(kFetcher, self).__init__()
        self.kwidth = kwidth

        # variables/properties to be implemented downstream
        self.DATA_INDEXER = None 
        self.DATA_DIR = None
        self.class_to_folder_mapping = None
        self.data_fetcher = None

    def mould_elasticset_by_queue(self, elasticset, q):
        # q is a queue.Queue()
        kFetcherInfo = {'ALL_USED_UP': False}
        while len(elasticset) < self.kwidth:
            if q.empty():
                kFetcherInfo['ALL_USED_UP'] = True
                break
            elasticset.append(q.get())

        return elasticset, q, kFetcherInfo

    def fetch_data_by_elastic_set(self, elasticset, as_numpy=False):
        x_batch, y_batch = [], []
        for classlabel,current_index in elasticset:
            classname = str(classlabel)
            dataname = self.DATA_INDEXER[int(classname)][current_index]

            DATA_DIR = os.path.join(self.DATA_DIR, self.class_to_folder_mapping[classlabel] , str( dataname))    

            x = self.data_fetcher(DATA_DIR)
            x_batch.append(x)
            y_batch.append(classlabel)
        if as_numpy:
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
        return x_batch, y_batch

class DataIndexer(kFetcher):
    def __init__(self, DATA_DIR, folder_to_class_mapping, kwidth, data_fetcher, init_new=False):
        super(DataIndexer, self).__init__(kwidth)
        
        """
        DATA_DIR: str, points to a folder containing folders of images arranged based on class e.g.
        + class0
          - x0.npy
          - x1.npy
          - ...
        + class1
        + specialfolder

        """        

        if init_new:
            self.do_init_new(DATA_DIR, folder_to_class_mapping, kwidth, data_fetcher)
        else:
            raise NotImplementedError('not implemented')


    def do_init_new(self, DATA_DIR, folder_to_class_mapping, kwidth, data_fetcher):
        self.DATA_DIR = DATA_DIR
        self.DATA_INDEXER_DIR = DATA_DIR + '.ix'
        self.folder_to_class_mapping = folder_to_class_mapping
        self.class_to_folder_mapping = self.get_class_to_folder_mapping(folder_to_class_mapping)

        self.data_fetcher = data_fetcher

        """
        data_to_net: 
            dictionary, indexed by data's index (class,filename), class is int.

        net_to_data:
            dictionary, indexed by position in the neural network
        """ 
        self.data_to_net = {}
        self.net_to_data = {}
        self.n_data_by_class = self.get_n_data_by_class()

        """
        self.DATA_INDEXER is like
        {
            0: ['0.npy', '1.npy', '11.npy',...],
            1: [...], 
            ...
            class: [dataname1, dataname2,...],
            ...
        }
        to facilitate easy data reference
        """
        self.DATA_INDEXER = self.load_data_indexer() # load it on the spot


    def get_data_status_in_net(self,):
        # given the index of a data sample, find which layer, which node etc it belongs to in the network
        raise NotImplementedError()

    def get_data_index_from_net(self):
        # given the layer and node etc of a network, find which sampel data it is.
        raise NotImplementedError()

    def get_n_data_by_class(self):
        n_data_by_class = {}
        for c in self.class_to_folder_mapping:
            folderdir = os.path.join(self.DATA_DIR, str(c))
            n_data = len(os.listdir(folderdir))   
            n_data_by_class[c] = n_data         
        return n_data_by_class        

    def create_queue_one_runthrough(self, config=None):
        if config is None:
            # run through all data once
            q = self.standard_queue_()
        elif config['mode']=='firstn':
            q = self.first_n_queue(config)
        elif config['mode']=='scrambledfirstn':
            q = self.first_n_queue(config)
            scrambled = []
            while q.qsize()>0: scrambled.append(q.get())
            random.shuffle(scrambled)
            q = queue.Queue()
            for x in scrambled: q.put(x)
        else:
            raise NotImplementedError()
        return q

    def standard_queue_(self):
        q = queue.Queue()
        n_class = len(self.class_to_folder_mapping)
        current_idx_by_class = { i:-1 for i,c in enumerate(self.class_to_folder_mapping) }
        total_data = sum([n for _,n in self.n_data_by_class.items()])

        counted = 0
        cptr = 0 # current class pointer
        while True:
            next_idx = current_idx_by_class[cptr]+1
            if next_idx < self.n_data_by_class[self.class_to_folder_mapping[cptr]]:
                q.put((cptr,next_idx))

                current_idx_by_class[cptr]+=1
                counted+=1

                if counted>= total_data:
                    break
            cptr = (cptr+1)%n_class
        return q

    def first_n_queue(self, config):
        q = queue.Queue()
        n_data_queried = {}
        n_class, total_data = 0, 0
        for c,n in zip(config['classes'], config['firstn']):
            # c is str of integer
            n_class+=1
            class_name = self.class_to_folder_mapping[int(c)]
            n_data_queried[c] = np.min([int(self.n_data_by_class[class_name]),int(n)])
            total_data+= n_data_queried[c]

        counted = 0 
        current_idx_by_class = {c:-1 for i,c in enumerate(config['classes'])}
        cptr_list = config['classes']
        cptr = 0 # current class pointer
        while True:
            next_idx = current_idx_by_class[config['classes'][cptr]]+1
            n_data = n_data_queried [cptr_list[cptr]]
            if next_idx < int(n_data):
                q.put((int(config['classes'][cptr]),next_idx))
                current_idx_by_class[ config['classes'][cptr] ]+=1
                counted+=1
                if counted>= total_data:
                    break                
            cptr = (cptr+1)%n_class
        return q


    ###################
    # utils
    ###################

    def get_class_to_folder_mapping(self, folder_to_class_mapping):
        class_to_folder_mapping = ['dummy' for i in range(len(folder_to_class_mapping))]
        for foldername, classlabel in folder_to_class_mapping.items():
            class_to_folder_mapping[classlabel] = foldername
        return class_to_folder_mapping

    def load_data_indexer(self):
        if not os.path.exists(self.DATA_INDEXER_DIR):
            print(f'creating new data indexer for data in {self.DATA_INDEXER_DIR}')
            data_indexer = {}
            for classname in sorted(os.listdir(self.DATA_DIR)):
                data_indexer[ self.folder_to_class_mapping [classname] ] = sorted(os.listdir(os.path.join(self.DATA_DIR, classname)) )              
            joblib.dump(data_indexer, self.DATA_INDEXER_DIR)
        else:
            print(f'loading data indexer from {self.DATA_INDEXER_DIR}')
            data_indexer = joblib.load(self.DATA_INDEXER_DIR)
        return data_indexer

    #### THIS IS THE GUY THE TELLS US WHERE IS THE RELEVANT FILE ####
    def get_folder_and_filename_by_index(self, y0,idx):
        classfolder = self.class_to_folder_mapping[y0]
        filename = self.DATA_INDEXER[y0][idx]
        return classfolder,filename
