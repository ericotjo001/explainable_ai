import os, joblib
from src.model.indexer import DataIndexer
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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
    # x_wrong_batch, y0_wrong_batch = test_ix.fetch_data_by_elastic_set(mark_wrong_predictions, as_numpy=True)

    TESTDAT = {
        'test': {'x': x_test_batch, 'y': y0_test_batch},
        # 'wrong':{'x': x_wrong_batch, 'y': y0_wrong_batch},
        'elasticset': elasticset,
        'mark_wrong_predictions': mark_wrong_predictions,
    }
    return TESTDAT

def fetch_image_by_class_and_index(classlabel, current_index, net):
    classname = str(net.ix.class_to_folder_mapping[classlabel])
    dataname = net.ix.DATA_INDEXER[classlabel][current_index]

    DATA_DIR = os.path.join(net.ix.DATA_DIR, classname, str( dataname))  
    pil_img = Image.open(DATA_DIR)
    img = np.asarray(pil_img)/255.
    return img

def pick_relevant_indices(activated_node, OUTPUT_INFO, n_pick=5):
    NODE_INFO = OUTPUT_INFO['NODE_INFO']
    if NODE_INFO['subactivation']:
        similar_samples = [NODE_INFO['subnode_idx']]
    else:
        n_similar = len(activated_node.wr_nodes_)
        if n_similar>n_pick:
            similar_samples = np.random.choice(range(n_similar), n_pick, replace=False)
            similar_samples = [activated_node.wr_nodes_[i] for i in similar_samples]
        else:
            similar_samples = activated_node.wr_nodes_
    INDICES = {
        'influential': activated_node.main_key, # y0,idx
        'similar': similar_samples,
    } 
    return INDICES

def get_similar_images(INDICES, net):
    similar_imgs = []
    for y0_sim, idx_sim in INDICES['similar']:
        img_sim = fetch_image_by_class_and_index(y0_sim, idx_sim, net)
        similar_imgs.append(img_sim)
    if len(similar_imgs)>0:
        similar_imgs = np.concatenate(similar_imgs, axis=1)
    return similar_imgs

def plot_relevant_examples(TESTDAT, net, DIRS):
    print('plot_relevant_examples')

    def offticks():
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    plt.figure()
    n_rows = len(TESTDAT['elasticset'])
    counter = 0
    for (y0,idx), x, y in zip( TESTDAT['elasticset'] , TESTDAT['test']['x'],TESTDAT['test']['y']):
        img = fetch_image_by_class_and_index(y0,idx, net)

        y, OUTPUT_INFO = net.forward(x)
        NODE_INFO = OUTPUT_INFO['NODE_INFO']
        activated_node = OUTPUT_INFO['activated_node']

        INDICES = pick_relevant_indices(activated_node, OUTPUT_INFO, n_pick=5)
        y0_inf,idx_inf = INDICES['influential']
        img_inf = fetch_image_by_class_and_index(y0_inf, idx_inf,net)
        similar_imgs = get_similar_images(INDICES, net)

        plt.gcf().add_subplot(n_rows,3,3*counter+1)
        plt.gca().imshow(img, cmap='gray')
        offticks()
        if (y0,idx) in TESTDAT['mark_wrong_predictions']:
            plt.gca().set_ylabel('(x)')
        plt.gcf().add_subplot(n_rows,3,3*counter+2)
        plt.gca().imshow(img_inf,cmap='gray')
        plt.gca().set_ylabel('sub:%s\nGT:%s'%(str(int(NODE_INFO['subactivation'])),
            str(y0_inf)))
        offticks()
        plt.gcf().add_subplot(n_rows,3,3*counter+3)
        plt.gca().imshow(similar_imgs,cmap='gray')
        offticks()
        counter+=1
    plt.show()