import numpy as np
HIERARCHY_FORMAT = """FORMAT:
layer k:
  (class, data_index): [(class, data_index),...]"""

def print_dict(d):
    for x,y in d.items():
        print('%s:%s'%(str(x),str(y)))


class InterpolatorInformation(object):
    def __init__(self, ):
        super(InterpolatorInformation, self).__init__()

    def get_status_by_interpolation(self, y_pred, OUTPUT_INFO, print_status=True):
        ###############################
        # for first type of interpolator, predict_by_max_activation()
        # see .interpolator.py
        ###############################        
        if OUTPUT_INFO['NODE_INFO']['subactivation']:
            y0,idx= OUTPUT_INFO['NODE_INFO']['subnode_idx']

            y0_parent, idx_parent = OUTPUT_INFO['activated_node'].main_key
            parent_folder_name = self.ix.class_to_folder_mapping[y0_parent]
            parent_data_name = self.ix.DATA_INDEXER[y0_parent][idx_parent]
            parent_node_loc = (parent_folder_name, parent_data_name)

        else:
            y0,idx = OUTPUT_INFO['activated_node'].main_key
            parent_node_loc = 'same as data_loc'

        folder_name = self.ix.class_to_folder_mapping[y0]
        data_name = self.ix.DATA_INDEXER[y0][idx]
        data_loc = (folder_name, data_name)


        status = {
            'activation': 'interpolation',
            'y_pred': y_pred,
            'layer': OUTPUT_INFO['layer'],
            'node_idx': OUTPUT_INFO['act_idx'],   
            'data_idx': (y0,idx),
            'data_loc': data_loc,     
            'parent_node_loc': parent_node_loc,           
        }
        if print_status:
            print_dict(status)
        return status        

class LayerInformation(InterpolatorInformation):
    """
    An abstract class of KABEDONN for printing the network's layers information
    """
    def __init__(self, ):
        super(LayerInformation, self).__init__()

    def get_layer_hierarchy(self):
        layer_hierarchy = {}
        for l, layernode in self.layers.items():
            layer_hierarchy[l] = {}
            for node in layernode.node_list:
                # main node's sample data index is here :)
                y0,idx = node.main_key 

                # the node's sub nodes (sub_nodes_) that are not main
                sub_nodes_idx = [ (y0,i) for y0,i in node.sub_nodes_]
                like_nodes_idx = node.wr_nodes_

                layer_hierarchy[l][(y0,idx)] = (sub_nodes_idx, like_nodes_idx)
        return layer_hierarchy
        
    def print_layer_hierarchy(self):
        print('print_layer_hierarchy!')
        print(HIERARCHY_FORMAT)
        layer_hierarchy = self.get_layer_hierarchy()
        total_layer = 0
        total_subnodes = 0
        for l, layers in layer_hierarchy.items():
            print('layer:',l)
            for main_node, (sub_nodes_,like_nodes_idx) in layers.items():
                print('  %s : %s'%(str(main_node), str(sub_nodes_)))
                print('     like nodes: %s'%(str(like_nodes_idx)))
                print('     sub nodes : %s'%(str(sub_nodes_)))

                total_subnodes += len(sub_nodes_)
            total_layer+=1
        print('total_layer   :', total_layer)
        print('total_subnodes:', total_subnodes)

    def forward_get_all_signals(self,x):
        """
        Special function defined mainly for evaluations.
        """
        main_signals = {}
        act, act_pre = None, x
        side_signals = None
        if self.layer_hierarchy is None:
            self.layer_hierarchy = self.get_layer_hierarchy()

        for layer_ in range(1,1+ len(self.layers)):
            receptors = self.layers[layer_].assemble_receptors() 
            act = self.normalized_stimulation(act_pre, receptors)
            main_signals[layer_] = {
                'indices': [(y0,idx) for y0,idx in self.layer_hierarchy[layer_]] , 
                'act':act,
            }

            if side_signals is None:
                if np.any(act>=self.admission_threshold(layer_)):
                    act_idx = np.argmax(act) # activated node index
                    activated_node = self.layers[layer_].node_list[act_idx]
                    side_signals, NODE_INFO = activated_node.forward_get_all_signals(act_pre)
                    side_signals = {
                        'indices': [(y0,idx) for y0,idx in activated_node.sub_nodes_],
                        'subactivation': NODE_INFO['subactivation'],
                        'act': side_signals,
                    }

            act_pre = act
        return main_signals, side_signals


    def get_data_status_in_net(self, x_batch, y0_batch, indices, verbose=100):      
        np.set_printoptions(precision=3)
        # print(x_batch.shape, y0_batch)
        for i,(x,y0) in enumerate(zip(x_batch, y0_batch)):
            if verbose>=100:
                index_banner = '=== Index:%s ==='%(str(indices[i]))
                print()
                print(index_banner)
                print('  x:',x, )
                print('  y0:',y0 )
                print('='*len(index_banner))
            y_pred, OUTPUT_INFO = self.forward(x)

            print_status = True if verbose>=100 else False
            if OUTPUT_INFO['output_mode'] == 'activation':
                if  OUTPUT_INFO['NODE_INFO']['subactivation']:
                    INFO = self.get_status_subactivation(y_pred, OUTPUT_INFO, print_status=print_status)
                else:
                    info = self.get_status_standard_activation(y_pred, OUTPUT_INFO, print_status=print_status)
            elif OUTPUT_INFO['output_mode'] == 'interpolation':
                info = self.get_status_by_interpolation(y_pred, OUTPUT_INFO, print_status=print_status)
            else:
                raise NotImplementedError()

            # do something with info if you like

    def get_status_standard_activation(self, y_pred, OUTPUT_INFO, print_status=True):
        y0,idx = OUTPUT_INFO['activated_node'].main_key

        folder_name = self.ix.class_to_folder_mapping[y0]
        data_name = self.ix.DATA_INDEXER[y0][idx]
        # or alternatively use get_folder_and_filename_by_index() function from self.ix.DATA_INDEXER
        data_loc = (folder_name, data_name)

        status = {
            'activation': 'standard',
            'y_pred': y_pred,
            'layer': OUTPUT_INFO['layer'],
            'node_idx': OUTPUT_INFO['act_idx'],
            'data_idx': (y0,idx),
            'data_loc': data_loc,
            'parent_node_loc': 'same as data_loc', 
        }
        if print_status:
            print_dict(status)
        return status

    def get_status_subactivation(self, y_pred, OUTPUT_INFO, print_status=True):
        y0,idx= OUTPUT_INFO['NODE_INFO']['subnode_idx']

        folder_name = self.ix.class_to_folder_mapping[y0]
        data_name = self.ix.DATA_INDEXER[y0][idx]
        # or alternatively use get_folder_and_filename_by_index() function from self.ix.DATA_INDEXER
        data_loc = (folder_name, data_name)

        # during subactivation, we may still be interested with the main node that is activated.
        y0_parent,idx_parent = OUTPUT_INFO['activated_node'].main_key
        folder_name_parent = self.ix.class_to_folder_mapping[y0_parent]
        data_name_parent = self.ix.DATA_INDEXER[y0_parent][idx_parent]
        parent_node_loc = (folder_name_parent, data_name_parent)

        status = {
            'activation': 'subactivation',
            'y_pred': y_pred,
            'layer': OUTPUT_INFO['layer'],
            'node_idx': OUTPUT_INFO['act_idx'],   
            'data_idx': (y0,idx),
            'data_loc': data_loc,     
            'parent_node_loc':parent_node_loc,
        }
        if print_status:
            print_dict(status)
        return status



    def get_data_index_from_net(self, NODES_OF_INTEREST): 
        for layer_,idx in NODES_OF_INTEREST:
            layer_,idx = int(layer_), int(idx)
            print('========= layer:%s, index:%s ========='%(str(layer_),str(idx)))

            try:
                layer = self.layers[layer_]
            except:
                print('layer not found!')
                continue

            try:
                node = layer.node_list[idx]
            except:
                print('node not found!')
                continue
            print('main_key (main node):',node.main_key)
            print('sub_nodes_ (distinct nodes):',[x for x in node.sub_nodes_])
            print('wr_nodes_  (similar nodes) :', node.wr_nodes_)
            print()
