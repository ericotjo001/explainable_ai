import numpy as np
# a1, a2, r = 0.01, 0.5, 0.5 
a1, a2, r = 0.1, 0.5, 0.2
from src.utils import double_selective_activation

class Node(object):
    """
    nrvec: normalized node relative vector, 
    """
    def __init__(self):
        super(Node, self).__init__()

        self.main_node = None # act_pre
        self.main_key =  None # (y0,idx) = (class, data_index)

        # representative nodes that is not main, 
        # dictionary {(y0,idx): nrvec} where nrvec = (act_pre - main_node)/np.linalg.norm(nrvec)
        # idx same as before (class, data_index)
        self.sub_nodes_ = {}
        self.sub_nodes_buffer = None

        # nodes that are already well represented by main_node, list [(y0,idx)]
        # or the like_nodes: nodes like the main nodes
        self.wr_nodes_ = []

    def new_node(self, y0, idx, act_pre):
        self.main_node = act_pre
        self.main_key = (y0,idx)

    def new_subnode(self, nrvec, y0, idx, activation_threshold=0.9):
        self.sub_nodes_[(y0,idx)] = {
            'nrvec':nrvec,
            'activation_threshold': activation_threshold,
        }
        self.sub_nodes_buffer = None # reset

    def forward(self, act_pre):
        NODE_INFO = {'subactivation':False}
        y_node,idx = self.main_key
        
        nrvec = act_pre - self.main_node
        norm = np.linalg.norm(nrvec)
        if norm==0. or len(self.sub_nodes_)==0:
            return y_node, NODE_INFO
        nrvec = nrvec/norm

        receptors, temp_idx = self.assemble_sub_receptors()
        side_act = np.matmul(receptors, nrvec.T)

        activation_threshold = np.array([subnode['activation_threshold'] for _,subnode in self.sub_nodes_.items()])
        if np.any(side_act>=activation_threshold):
            receptor_idx = np.argmax(side_act - activation_threshold )
            y_pred = temp_idx[receptor_idx][0]
            NODE_INFO = {'subactivation':True, 
                'subnode_idx':temp_idx[receptor_idx],  # (y0,idx)
                'side_act':side_act}
        else:
            y_pred = y_node

        return y_pred, NODE_INFO

    def forward_get_all_signals(self,act_pre):
        """
        Special function defined mainly for evaluations.
        """
        NODE_INFO = {'subactivation':False}
        y_node,idx = self.main_key
        
        nrvec = act_pre - self.main_node
        norm = np.linalg.norm(nrvec)
        if norm==0. or len(self.sub_nodes_)==0:
            return y_node, NODE_INFO
        nrvec = nrvec/norm

        receptors, temp_idx = self.assemble_sub_receptors()
        side_act = np.matmul(receptors, nrvec.T)

        activation_threshold = np.array([subnode['activation_threshold'] for _,subnode in self.sub_nodes_.items()])
        if np.any(side_act>=activation_threshold):
            receptor_idx = np.argmax(side_act - activation_threshold )
            y_pred = temp_idx[receptor_idx][0]
            NODE_INFO = {'subactivation':True, 
                'subnode_idx':temp_idx[receptor_idx],  # (y0,idx)
                'side_act':side_act}
            return side_act, NODE_INFO
        return None, NODE_INFO

    def assemble_sub_receptors(self):
        if self.sub_nodes_buffer is None:
            receptors = [] # shape: (n, act_pre dim)
            temp_idx = []
            for (y0,idx), node_nrvec_ in self.sub_nodes_.items():
                receptors.append(node_nrvec_['nrvec']) 
                temp_idx.append((y0,idx))
            receptors = np.array(receptors)
            self.sub_nodes_buffer = {
                'receptors': receptors,
                'temp_idx' : temp_idx,
            }
        receptors = self.sub_nodes_buffer['receptors']
        temp_idx = self.sub_nodes_buffer['temp_idx']
        return receptors, temp_idx

class LayerNodes(object):
    def __init__(self, ):
        super(LayerNodes, self).__init__()

        self.node_list = [] # list of Node()
        self.receptors_buffer = None
        
    def count_nodes(self):
        return len(self.node_list)

    def insert_new_node(self, y0, idx, act_pre):
        self.receptors_buffer = None # reset
        n = Node()
        n.new_node(y0, idx, act_pre)
        self.node_list.append(n)

    def assemble_receptors(self,):
        if self.receptors_buffer is None:
            nodes = []
            for n in self.node_list:
                if len(nodes)==0:
                    nodes = [n.main_node]
                else:
                    nodes = np.concatenate((nodes,[n.main_node]),axis=0)
            nodes = np.array(nodes)
            self.receptors_buffer = nodes
        
        nodes = self.receptors_buffer
        return nodes

