import numpy as np
from .nodes import LayerNodes, a1, a2, r
import queue

class StatusPrinter(object):
    # A VERY ABSTRACT CLASS
    def __init__(self, ):
        super(StatusPrinter, self).__init__()
        
    def print_final_(self):
        count, n_embedded = 0, 0
        for layer, nodes_layer  in self.layers.items():
            # nodes_layer is a LayerNodes()
            print('LAYER: ',layer)
            print('%-5s %-3s %-5s %-5s'%(str('y0'), str('idx'), 
                str('n(wr)'), str('n(subnode)')))
            for node in nodes_layer.node_list:
                y0, idx = node.main_key
                print('%-5s %-3s %-5s %-5s'%(str(y0), str(idx), 
                    str(len(node.wr_nodes_)), str(len(node.sub_nodes_))
                    ))
                count +=1 
                count += len(node.sub_nodes_)
                count += len(node.wr_nodes_ )

                n_embedded += 1
                n_embedded += len(node.sub_nodes_)
        print('total data used:', count)
        print('total embedded data:', n_embedded)

from .metatrainer import MetaTrainer
class Trainer(StatusPrinter, MetaTrainer):
    def __init__(self, ):
        super(Trainer, self).__init__()

        """
        Abstracted from a KABEDONN
        Do implement every abstracted detail at KABEDONN
        
        Some examples: 
        self.layers = {}
        self.ix = None # a DataIndexer
        etc
        """

    def integrate_nodes_layer(self, L, nodes_layer):
        raise NotImplementedError('Please implement this downstream.')

    def forward(self, x):
        raise NotImplementedError('Please implement this downstream.')
        
    def fit_data(self, config=None, verbose=100):
        if verbose>=10:
            print('fitting data...')

        if config is None:
            config = {
                'print_final_info':True,
                'balance_test': True,
                'qconfig': None,
            }

        qtemp = self.ix.create_queue_one_runthrough(config=config['qconfig'])
        q = queue.Queue()
        qlist = []
        while qtemp.qsize()>0:
            qitem = qtemp.get()
            qlist.append(qitem)
            q.put(qitem)

        elasticset, altset = [], []
        current_layer = 1
        counter = 0
        while q.qsize()>0 or len(elasticset)>0:
            elasticset, q, kFetcherInfo = self.ix.mould_elasticset_by_queue(elasticset, q)
            elasticset, setaside, nodes_layer = self.filter_by_layer_activation (current_layer, elasticset)
            altset.extend(setaside)

            if config['balance_test']:
                self.check_altset_balance(altset) # meta trainer

            counter+=1
            if verbose>=100:
                if (counter)%100==0:
                    update_text= 'qsize:[%s] esize:%s asize:%s'%(
                        str(q.qsize()),
                        str(len(elasticset)),
                        str(len(altset)))
                    print('%-48s'%(str(update_text)),end='\r')

            if len(elasticset)>= self.ix.kwidth:
                self.integrate_nodes_layer(current_layer, nodes_layer)
                recycledset = self.bifold_activation(current_layer, altset)
                if verbose>=100:
                    print('\nLayer %s construction completed! qsize:[%s]'%(str(current_layer),str(q.qsize())))
                current_layer+=1 

                # reset
                elasticset, altset = [], []
                for rec in recycledset: q.put(rec)

            if kFetcherInfo['ALL_USED_UP']:
                if verbose>=100:
                    print('\nALL USED UP!')
                if len(elasticset)>0:
                    self.integrate_nodes_layer(current_layer, nodes_layer)
                    recycledset = self.bifold_activation(current_layer, altset)
                    elasticset = []

        if config['print_final_info']:
            self.print_final_()
        if verbose>=100:
            print('training/data fitting done!')
        return qlist

    def filter_by_layer_activation(self, L, elasticset):
        # L: (int) layer
        # elasticset: list of (y0,idx) as usual
        new_elasticset, setaside = [],[]
        x_batch, y0_batch = self.ix.fetch_data_by_elastic_set(elasticset)
        
        nodes_layer = LayerNodes()
        for i, (x,y0_actual) in enumerate(zip(x_batch, y0_batch)):
            # we are constructing layer L, so activate all the way to L-1 first
            act_pre, FILTER_INFO = self.activate_layer_l(x, L-1, filter_mode=True) 

            if FILTER_INFO['STATUS'] == 'ACTIVATED':
                INSERT_NODE = False
            elif nodes_layer.count_nodes()==0:
                INSERT_NODE = True
            else:
                receptors = nodes_layer.assemble_receptors()        
                act = self.normalized_stimulation(act_pre, receptors)

                INSERT_NODE = np.all(act < self.admission_threshold(L))

            if INSERT_NODE:
                y0,idx = elasticset[i] 
                assert(y0==y0_actual)
                nodes_layer.insert_new_node(y0,idx, act_pre)
                new_elasticset.append(elasticset[i])
            else:
                setaside.append(elasticset[i])

        return new_elasticset, setaside, nodes_layer

    def bifold_activation(self, L, altset):
        # L: (int) layer
        # altset: list of (y0,idx) like elasticset. 

        recycledset = []

        x_batch, y0_batch = self.ix.fetch_data_by_elastic_set(altset)
        for i, (x,y0) in enumerate(zip(x_batch, y0_batch)):
            act, FILTER_INFO = self. activate_layer_l(x, L, filter_mode=True)

            if FILTER_INFO['STATUS'] != 'ACTIVATED':
                raise Exception('watch out for this for now. Not likely though.')
                recycledset.append(altset[i])
                continue

            activated_layer = FILTER_INFO['LAYER']
            act_idx = np.argmax(act) # activated node index
            activated_node = self.layers[activated_layer].node_list[act_idx]

            ############# core bifold mechanism ###################
            y_node, idx = activated_node.main_key
            main_node = activated_node.main_node

            # concept: y_node should be equal to y0. But in fringe cases (esp at boundaries)
            #   they may not be equal. In that case, insert y0 as a near "boundary" nodes.
            INSERT_SUBNODE = (int(y_node)!=int(y0))
            if INSERT_SUBNODE:
                act_pre, _ = self.activate_layer_l(x, activated_layer-1, filter_mode=True)    
                nrvec = act_pre - main_node
                nrvec = nrvec/np.linalg.norm(nrvec)

                y0,idx = altset[i]
                self.layers[activated_layer].node_list[act_idx].new_subnode(nrvec, y0, idx)
            else:
                self.layers[activated_layer].node_list[act_idx].wr_nodes_.append(altset[i])

            ############################################

        return recycledset

    def subforward(self,x, L):
        # like forward(), but without many unnecessary info
        act, act_pre = None, x
        for layer_ in range(1,1+ L):
            receptors = self.layers[layer_].assemble_receptors() 
            act = self.normalized_stimulation(act_pre, receptors)

            act_idx = np.argmax(act) # activated node index
            activated_node = self.layers[layer_].node_list[act_idx]
            
            y_pred, _ = activated_node.forward(act_pre)
            act_pre = act
        return act_pre

    def evaluate_and_finetune_on_train_data(self, qlist=None, verbose=100):
        if verbose>=100:
            print('evaluate_on_train_data...')
            print('  By construct, we should get near perfect accuracy.')
            
        if qlist is None:
            q = self.ix.create_queue_one_runthrough()
        else:
            q = queue.Queue()
            for qitem in qlist:
                q.put(qitem)

        elasticset = []
        n_total = q.qsize()
        total = 0
        while q.qsize()>0 or len(elasticset)>0:
            elasticset, q, kFetcherInfo = self.ix.mould_elasticset_by_queue(elasticset, q)
            x_batch, y0_batch = self.ix.fetch_data_by_elastic_set(elasticset)

            for i, (x,y0) in enumerate(zip(x_batch, y0_batch)):
                y, OUTPUT_INFO = self.forward(x)

                if int(y)!=int(y0):
                    while True:
                        self.finetune(i, elasticset, x, OUTPUT_INFO)
                        y, OUTPUT_INFO = self.forward(x)
                        if int(y)==int(y0):
                            break

                total+=1    
                if verbose>=100:
                    if total%100==0:
                        update_text = 'finetune qsize: [%s]'%(str(q.qsize()))
                        print('%-64ss'%(str(update_text)), end='\r')

            elasticset = []

    def reevaluate_train_data_status(self, qlist=None, verbose=100):
        if qlist is None:
            q = self.ix.create_queue_one_runthrough()
        else:
            q = queue.Queue()
            for qitem in qlist:
                q.put(qitem)

        BEYOND_PRECISION = False

        elasticset = []
        n_total = q.qsize()
        total, correct = 0, 0
        while q.qsize()>0 or len(elasticset)>0:
            elasticset, q, kFetcherInfo = self.ix.mould_elasticset_by_queue(elasticset, q)
            x_batch, y0_batch = self.ix.fetch_data_by_elastic_set(elasticset)

            for i, (x,y0) in enumerate(zip(x_batch, y0_batch)):
                y, OUTPUT_INFO = self.forward(x)

                if int(y)==int(y0):
                    correct+=1
                else:
                    NODE_INFO = OUTPUT_INFO['NODE_INFO']
                    act_idx = OUTPUT_INFO['act_idx']
                    layer = OUTPUT_INFO['layer']
                    activated_node = OUTPUT_INFO['activated_node']

                    y0p, idxp = elasticset[i]
                    if (y0p, idxp) in activated_node.wr_nodes_:
                        thelist = self.layers[layer].node_list[act_idx].wr_nodes_ 
                        self.layers[layer].node_list[act_idx].wr_nodes_ .pop( thelist.index((y0p, idxp)))
                    elif (y0p, idxp) in activated_node.sub_nodes_:
                        BEYOND_PRECISION = True                      
                        # self.layers[layer].node_list[act_idx].sub_nodes_.pop((y0p, idxp)) # no go
                    else:
                        raise Exception('unknown error..')

                total+=1    
                if verbose>=100:
                    if total%100==0:
                        update_text = 'eval qsize: [%s]'%(str(q.qsize()))
                        print('%-64ss'%(str(update_text)), end='\r')

            elasticset = []

        from .eval import EvaluationResult   
        evalargs = {
            'mode': 'simple_accuracy',
            'correct': correct, 'total': total, 
            'header_text': 'training accuracy: (should be near perfect by construct)',
            
        }         
        EVAL_RESULT = EvaluationResult(**evalargs)
        if BEYOND_PRECISION:
            print("WARNING!!!!!!!! Error beyond current setting's precision detected~+! Try larger kwdith! ")
        return EVAL_RESULT


    def finetune(self, i, elasticset, x, OUTPUT_INFO):
        NODE_INFO = OUTPUT_INFO['NODE_INFO']
        if OUTPUT_INFO['output_mode']=='interpolation':
            raise NotImplementedError('construction error!')
        activated_layer = OUTPUT_INFO['layer'] 
        act_idx = OUTPUT_INFO['act_idx']

        if NODE_INFO['subactivation']:
            # subnode(s) is/are activated but prediction is wrong
            subnode_idx = NODE_INFO['subnode_idx']
            at = self.layers[activated_layer].node_list[act_idx].sub_nodes_[subnode_idx]['activation_threshold']
            at = np.min([1.-1e-7, np.max(NODE_INFO['side_act'])])

            self.layers[activated_layer].node_list[act_idx].sub_nodes_[subnode_idx]['activation_threshold'] = at*(1+1e-5)
        else:  
            # subnode is NOT activated, but main node prediction is wrong
            node = self.layers[activated_layer].node_list[act_idx]

            act_pre = self.subforward(x, activated_layer-1)
            nrvec = act_pre - node.main_node
            norm = np.linalg.norm(nrvec)
            nrvec = nrvec/norm
            y0, idx = elasticset[i]
            self.layers[activated_layer].node_list[act_idx].new_subnode(nrvec, y0, idx, activation_threshold= 1.-1e-5)
