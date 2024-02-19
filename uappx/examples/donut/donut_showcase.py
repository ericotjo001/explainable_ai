import numpy as np
import matplotlib.pyplot as plt
from src.model.indexer import DataIndexer
import joblib

def load_training_data_for_showcase(net):
    q = net.ix.create_queue_one_runthrough()
    all_training_set = []
    while q.qsize()>0:
        y0,idx = q.get()
        all_training_set.append((y0,idx))
    x_train_batch, y_train_batch = net.ix.fetch_data_by_elastic_set(all_training_set, as_numpy=True)
    return x_train_batch, y_train_batch


def load_data_assimilated_into_net(net):
    print('load_data_assimilated_into_net...')

    layer_hierarchy = net.get_layer_hierarchy()
    anodes = []
    for l, layers in layer_hierarchy.items():
        for main_node, (sub_nodes_,like_nodes_) in layers.items():
            x_mainnodes, y_mainnodes = net.ix.fetch_data_by_elastic_set([main_node], as_numpy=True)
            x_subnodes, y_subnodes = net.ix.fetch_data_by_elastic_set(sub_nodes_, as_numpy=True)
            x_likenodes,y_likenodes = net.ix.fetch_data_by_elastic_set(like_nodes_, as_numpy=True)
            anodes.append({
                'main':(x_mainnodes, y_mainnodes),
                'sub' :(x_subnodes, y_subnodes),
                'like':(x_likenodes,y_likenodes,),
                'main_annot': main_node,
                'sub_annot': sub_nodes_,
                }) 
    return anodes


def load_test_data( net, DIRS, DATA_POINTS_OF_INTEREST):
    """
    Load test data samples specified in DATA_POINTS_OF_INTEREST
      DATA_POINTS_OF_INTEREST = zip([y0_1,y0_2,...],[i_1,i_2,...])
    as x_test_batch, y0_test_batch

    x_wrong_batch, y0_wrong_batch are subset of x_test_batch, y0_test_batch
      that are wrongly predicted by net 
    """
    kwidth = 10 # dummy
    TEST_EVAL_RESULT = joblib.load(DIRS['TEST_RESULT_DIR']) 
    elasticset = []
    mark_wrong_predictions = []
    for y0,idx in DATA_POINTS_OF_INTEREST:
        y0,idx = int(y0),int(idx)
        elasticset.append((y0,idx))
        if (y0,idx) in TEST_EVAL_RESULT.indices_wrong_data:
            mark_wrong_predictions.append((int(y0),int(idx)))

    test_ix = DataIndexer(DIRS['TEST_DATA_DIR'], 
        net.ix.folder_to_class_mapping, kwidth, 
        net.ix.data_fetcher, 
        init_new=True)
    x_test_batch, y0_test_batch = test_ix.fetch_data_by_elastic_set(elasticset, as_numpy=True)
    x_wrong_batch, y0_wrong_batch = test_ix.fetch_data_by_elastic_set(mark_wrong_predictions, as_numpy=True)

    ##########################################
    # For interpretability: find the most activated nodes! 
    # Hence, find the MOST
    ##########################################
    x_act_nodes, activation_hierarcy = get_activations(net, x_test_batch, elasticset)

    # for this_index, nodes in activation_hierarcy.items():
    #     print('%s <-> [%s] : %s'%(str(this_index),str(nodes['main_node']),str(nodes['like_nodes'])))

    return {
        'test'  :{'x': x_test_batch, 'y': y0_test_batch},
        'wrong' :{'x': x_wrong_batch,'y': y0_wrong_batch}, 
        'anodes':{'x': x_act_nodes}, # activated nodes
        'indices_wrong_data': TEST_EVAL_RESULT.indices_wrong_data,
        'activation_hierarcy': activation_hierarcy,
    }

def get_activations(net, x_test_batch, elasticset):
    activation_hierarcy = {}
    activatedset = []
    for this_index,x in zip(elasticset, x_test_batch):
        y, OUTPUT_INFO = net.forward(x)
        node = OUTPUT_INFO['activated_node']
        activation_hierarcy[this_index] = {
            'main_node' :node.main_key,
            'like_nodes' :node.wr_nodes_,
            }
        activatedset.append(node.main_key)

    x_act_nodes, _ = net.ix.fetch_data_by_elastic_set(activatedset, as_numpy=True)
    return x_act_nodes, activation_hierarcy


######### plots ######### 

def plot_scatter(x,y,cmap,alpha, marker='o', colorbar=True, 
    annotations=None, annotate=False, annot_color='m'):
    vmin, vmax = np.min(y), np.max(y)+1
    plt.scatter(x[:,0], x[:,1], c=y, cmap=cmap,marker=marker, alpha=alpha, vmin=vmin, vmax=vmax)
    
    if annotate and annotations is not None:
        assert(len(x)==len(annotations))
        for coords, annot in zip(x,annotations):
            # Note: if multidim, only use the first 2 dim for plotting
            plt.annotate(str(annot),coords[:2], alpha=0.5,c=annot_color)
    
    if colorbar:
        plt.colorbar(ticks=list(set(y)))

def plot_marker(x,s,marker,edgecolor, alpha=1.):
    plt.scatter(x[:,0], x[:,1], s,
        marker=marker, facecolor='none', edgecolor=edgecolor, alpha=alpha) 

def plot_line_to_mainnodes(mainnode,subnodes, alpha=0.5, c='m', linewidth=0.5, linestyle='-'):
    for subnode in subnodes:
        line = np.array([mainnode, subnode]).T
        plt.plot(line[0,:], line[1,:], linewidth=linewidth, c=c, alpha=alpha,linestyle=linestyle)

def plot_nodes_assimilated_to_net(anodes, alpha=0.5, annotate=False, annot_main_only=False):
    # anodes: check out load_data_assimilated_into_net() above

    if annotate:
        n_labels=0
        for dat in anodes:
            x_subnodes, y_subnodes = dat['sub']
            n_labels += len(x_subnodes)
        circle = [(0.5*np.sin(theta),0.5*np.cos(theta)) for theta in np.linspace(-np.pi/2,np.pi/2,num=n_labels, ) ]
        counter = 0

    for dat in anodes:
        x_mainnodes, y_mainnodes = dat['main']
        x_likenodes, y_likenodes = dat['like']
        x_subnodes, y_subnodes = dat['sub']

        plt.scatter( x_mainnodes [:,0], x_mainnodes [:,1], 128, marker='^', facecolor='none', edgecolor='m', alpha=alpha)
        if annotate or annot_main_only:
            # Note: if multidim, only use the first 2 dim for plotting
            plt.gca().annotate(str(dat['main_annot']), x_mainnodes[0,:2],c='b', alpha=alpha,)
        
        if len(x_likenodes)>0:
            # plt.scatter( x_likenodes [:,0], x_likenodes [:,1], 128, marker='^', facecolor='none', edgecolor='m', linestyle='--', alpha=alpha)
            plot_line_to_mainnodes(mainnode=x_mainnodes[0],subnodes=x_likenodes, alpha=alpha)            
        if len(x_subnodes)>0:
            # plt.scatter( x_subnodes [:,0], x_subnodes [:,1], 128, marker='v', facecolor='none', edgecolor='m', linestyle='--', alpha=alpha)
            if annotate:
                for annot, coords in zip(dat['sub_annot'],x_subnodes):
                    # Note: if multidim, only use the first 2 dim for plotting
                    plt.gca().annotate(annot ,coords[:2] ,c='r', alpha=alpha, fontsize=8,
                        xytext=circle[counter],
                        arrowprops=dict(arrowstyle="->",connectionstyle="arc3",color='r',alpha=alpha))            
                    counter+=1
            plot_line_to_mainnodes(mainnode=x_mainnodes[0],subnodes=x_subnodes, linestyle='--',
                alpha=alpha,c='r',)


############ report ###################

ASUMMARY = """========= Summary ===========
Activation summary of your selected data samples
FORMAT: 
[b] %-10s ==> %-10s
  list of like nodes = [node1,node2,...]
where b=T/F (boolean, True or False)
"""%(str('data'),str('main_node'))

def write_timestamped_report(REPORT_DIR, TESTDAT):
    print('REPORT SAVED TO:',REPORT_DIR)
    with open(REPORT_DIR, 'w') as f:
        f.write(ASUMMARY)
        for this_index, nodes in TESTDAT ['activation_hierarcy'].items():
            IS_CORRECT = 'F' if this_index in TESTDAT['indices_wrong_data'] else 'T'
            f.write('[%s] %-10s ==> %s :\n  %s\n'%(str(IS_CORRECT), str(this_index),str(nodes['main_node']),
                str(nodes['like_nodes'])))
        f.write('\nwrong indices for you to check out:\n')
        f.write(str(TESTDAT['indices_wrong_data']))