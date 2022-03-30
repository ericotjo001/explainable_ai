import numpy as np

class OANNNodeController(object):
    def __init__(self, ):
        super(OANNNodeController, self).__init__()

    def init_new(self, **settings):
        """
        Override this function whenever necessary
        """
        self.layer_nodes = defaultdict(list) 
        self.layer_nodes_alphas = defaultdict(list)

        # N: data_size, no of data. Set this when the data is available.
        self.unused_indices = None # list(range(N)) 
        self.used_indices = defaultdict(list) # key is index to layer, starting from 1 

    def move_index_to_layer_node(self, i, layer_k):
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


    def push_node_to_layer(self, this_index, x, y, layer_k):      
        self.move_index_to_layer_node(this_index, layer_k)
        _ , this_sample = self.forward_to_layer_k(x, layer_k=layer_k-1, )
        
        self.layer_nodes[layer_k] = np.concatenate((self.layer_nodes[layer_k], [this_sample]),axis=0)
        self.layer_nodes_alphas[layer_k].append(y)