import os
import numpy as np
from pipeline.objgen.random_simple_gen_implemented import TenClassesPyIO
from skimage.transform import resize

DEBUG_MODE = 0
if DEBUG_MODE:# toggle freely
    DEBUG_IMSHOW_LOADER = 0
    DEBUG_IMSHOW_LOADER_XAI = 0
else: # do not edit
    DEBUG_IMSHOW_LOADER = 0 # bool
    DEBUG_IMSHOW_LOADER_XAI = 0 # bool

DEFAULT_DATA_CONFIG_DATA = {
    'data_cache_name': 'small_training_data_10c',
    'training_data':{  
        'number_of_data_shards':4,
        'number_of_data_per_shard':48,},    
    'validation_data_cache_name':'small_val_data_10c',
    'val_data':{
        'number_of_data_shards':2,
        'number_of_data_per_shard':16,},
    'test_data_cache_name':'small_test_data_10c',
    'test_data':{
        'number_of_data_chunks':2,
        'number_of_data_per_chunk':16,}, 
}

VERBOSE_THRESHOLD = 100
def generate_ten_classes_training_val_test_data(config_data=None, realtime_update=False):
    # generate training, validation and test data.
    print('generate_ten_classes_training_val_test_data()')
    if config_data is None:
        config_data = DEFAULT_DATA_CONFIG_DATA
    CACHE_FOLDER_DIR = 'checkpoint/cache'
    # training data
    create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['data_cache_name'], 
        n_shard=config_data['training_data']['number_of_data_shards'] , 
        n_per_shard= config_data['training_data']['number_of_data_per_shard'],realtime_update=realtime_update)
    # validation data
    create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['validation_data_cache_name'], 
        n_shard=config_data['val_data']['number_of_data_shards'] , 
        n_per_shard= config_data['val_data']['number_of_data_per_shard'],realtime_update=realtime_update)           
    # test data
    create_or_load_data_shards(CACHE_FOLDER_DIR, config_data['test_data_cache_name'], 
        n_shard=config_data['test_data']['number_of_data_chunks'] , 
        n_per_shard= config_data['test_data']['number_of_data_per_chunk'],
        realtime_update=realtime_update, include_xai_variables=True)           

def create_or_load_data_shards(CACHE_FOLDER_DIR, DATA_NAME, n_shard, n_per_shard, 
    realtime_update=False, include_xai_variables=False):
    denomination = 'shard'
    if include_xai_variables: denomination = 'chunk'         
    for i in range(1,1+n_shard):
        shard_dir = os.path.join(CACHE_FOLDER_DIR, '%s.%s.%s'%(str(DATA_NAME),str(i),str(denomination)))
        if not os.path.exists(shard_dir):
            print('creating %s %s number %s'%(str(denomination), str(DATA_NAME),str(i)))
            if not include_xai_variables:
                save_one_shard(shard_dir, n_per_shard,realtime_update=realtime_update)
            else:
                save_one_chunk(shard_dir, n_per_shard,realtime_update=realtime_update)
        else:
            print('%s %s number %s already exists'%(str(denomination), str(DATA_NAME),str(i)))
            # there is a misnomer in the method name. Data is not actually loaded here.

def load_dataset_from_a_shard(k, CACHE_FOLDER_DIR, DATA_NAME, include_xai_variables=False,
    reshape_size=None):
    # reshape_size : (C, H, W) tuple
    denomination = 'shard'
    if include_xai_variables: denomination = 'chunk' 
    this_dataset = TenClassesPyIO()        
    shard_dir = os.path.join(CACHE_FOLDER_DIR, '%s.%s.%s'%(str(DATA_NAME),str(k),str(denomination)))    
    this_dataset = this_dataset.load_pickled_data(shard_dir, tv=(1,0,VERBOSE_THRESHOLD))
    this_dataset.x = np.array(this_dataset.x) # original shape (N, C, H, W)

    if reshape_size is not None:
        s, N = reshape_size, len(this_dataset.x)
        temp_x = []
        
        if include_xai_variables:
            size_HW = reshape_size[1:]
            temp_h = []

        for i in range(N):
            temp = this_dataset.x[i].transpose(1,2,0)
            temp = resize(temp, (s[1],s[2],s[0]))
            temp = temp.transpose(2,0,1)
            temp_x.append(temp)

            if include_xai_variables:   
                h_resize = resize(this_dataset.h[i],size_HW)
                temp_h.append(h_resize)
                if DEBUG_IMSHOW_LOADER_XAI: do_DEBUG_IMSHOW_LOADER_XAI(this_dataset, i, h_resize, DEBUG_IMSHOW_LOADER)
 
        this_dataset.x = np.array(temp_x)
        if include_xai_variables:
            this_dataset.h = np.array(temp_h)

    if DEBUG_IMSHOW_LOADER: do_DEBUG_IMSHOW_LOADER(this_dataset, DEBUG_IMSHOW_LOADER)
    return this_dataset



def save_one_shard(shard_dir, n_per_shard, realtime_update=False):
    this_dataset = TenClassesPyIO()
    this_dataset.setup_training_0001(general_meta_setting=None, explanation_setting=None, 
        data_size=n_per_shard, realtime_update=realtime_update)
    this_dataset.pickle_data(this_dataset, shard_dir, tv=(1,0,VERBOSE_THRESHOLD))

def save_one_chunk(shard_dir, n_per_shard, realtime_update=False):
    this_dataset = TenClassesPyIO()
    this_dataset.setup_xai_evaluation_0001(general_meta_setting=None, explanation_setting=None, 
        data_size=n_per_shard, realtime_update=realtime_update)
    this_dataset.pickle_data(this_dataset, shard_dir, tv=(1,0,VERBOSE_THRESHOLD))

def do_DEBUG_IMSHOW_LOADER(this_dataset, DEBUG_IMSHOW_LOADER):
    """
    Use this to observe the image x used for evaluation.
    """
    print('DEBUG_IMSHOW_LOADER. this_dataset.x.shape:', this_dataset.x.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(this_dataset.x[0].transpose(1,2,0))
    plt.show()
    raise Exception('terminating for DEBUG_IMSHOW_LOADER (!)')

def do_DEBUG_IMSHOW_LOADER_XAI(this_dataset, i, h_resize, DEBUG_IMSHOW_LOADER):
    """
    Use this to observe that the resizing of heatmaps are done properly
    """
    print('DEBUG_IMSHOW_LOADER_XAI.\nthis_dataset.h[i].shape:%s\nh_resize.shape:%s'%(str(this_dataset.h[i].shape),str(h_resize.shape)))
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(this_dataset.h[i], vmin=-1,vmax=1, cmap='bwr')
    ax2 = fig.add_subplot(122)
    ax2.imshow(h_resize, vmin=-1,vmax=1, cmap='bwr')
    plt.show()                   
    raise Exception('terminating for DEBUG_IMSHOW_LOADER_XAI (!)')