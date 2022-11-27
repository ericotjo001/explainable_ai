from .utils import *
# .utils include joblib, manage_dir

def prepare_data(dargs):
    print('prepare_data...')

    DIRS = manage_dir(dargs)
    DATA_MODES = {
        'train': 'SHARD_FOLDER_DIR',
        'val': 'SHARD_VAL_FOLDER_DIR',
        'test': 'SHARD_TEST_FOLDER_DIR',
    }
    SHARD_FOLDER_DIR = DIRS[DATA_MODES[dargs['data_mode']]]

    if len(os.listdir(SHARD_FOLDER_DIR))==0:
        for i in range(dargs['n_shards']):
            SHARD_NAME = f'data-{str(dargs["n_classes"])}_{i}.shard'
            SHARD_DIR = os.path.join(SHARD_FOLDER_DIR, str(SHARD_NAME), )

            print(f'preparing data to {SHARD_DIR}')
            save_one_chunk(SHARD_DIR, dargs['n_classes'], dargs['n_per_shard'], realtime_update=False)
    else:
        print(f"Data ALREADY exists at {SHARD_FOLDER_DIR}")


    for i in range(dargs['n_shards']):
        SHARD_NAME = f'data-{str(dargs["n_classes"])}_{i}.shard'
        SHARD_DIR = os.path.join(SHARD_FOLDER_DIR, str(SHARD_NAME), )
        display_some_shard_samples(SHARD_DIR, SHARD_FOLDER_DIR, name=f'samples-shard-{i}')

def save_one_chunk(SHARD_DIR, n_classes, n_per_shard, realtime_update=False):
    if n_classes==10:
        from .objgen.random_simple_gen_implemented import TenClassesPyIO
        dataset = TenClassesPyIO()
    elif n_classes==3:
        from .objgen.random_simple_gen_implemented2 import ThreeClassesPyIO
        dataset = ThreeClassesPyIO()        
    else:
        raise NotImplementedError()

    dataset.setup_xai_evaluation_0001(general_meta_setting=None, explanation_setting=None, 
        data_size=n_per_shard, realtime_update=realtime_update)
    joblib.dump(dataset, SHARD_DIR)


def display_some_shard_samples(SHARD_DIR, SHARD_FOLDER_DIR, name):
    SAMPLES_DIR = os.path.join(SHARD_FOLDER_DIR, 'samples_display')
    if not os.path.exists(SAMPLES_DIR):
        os.makedirs(SAMPLES_DIR, exist_ok=True)
    
    print('\nsource:',SHARD_DIR)
    dataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
    nshow = np.min([4, dataset.__len__()])

    plt.figure(figsize=(8,4))
    for i in range(nshow):
        x, y0 = dataset.__getitem__(i)
        h = dataset.h[i] # heatmap
        print(f'x.shape: {x.shape} | y0:{y0} | h.shape: {h.shape} | v["type"]: {dataset.v[i]["type"]} ' )
        if i==0: print('  v:', dataset.v[i].keys())

        plt.gcf().add_subplot(2,nshow, i+1)
        plt.gca().imshow(x.transpose(1,2,0), vmin=0,vmax=1)
        plt.gca().set_title(f'y0:{y0}')
        plt.gcf().add_subplot(2,nshow, i+1+nshow)
        plt.gca().imshow(h, vmin=-1.,vmax=1, cmap='bwr')
    plt.tight_layout()
    plt.savefig(os.path.join(SAMPLES_DIR,name+'.png'))

def load_dataset_from_a_shard(SHARD_DIR, reshape_size=None):
    # reshape_size : (C, H, W) tuple

    this_dataset = joblib.load(SHARD_DIR)
    this_dataset.x = np.array(this_dataset.x) # original shape (N, C, H, W)

    if reshape_size is not None:
        s, N = reshape_size, len(this_dataset.x)
        temp_x = []
        
        size_HW = reshape_size[1:]
        temp_h = []

        for i in range(N):
            temp = this_dataset.x[i].transpose(1,2,0)
            temp = resize(temp, (s[1],s[2],s[0]))
            temp = temp.transpose(2,0,1)
            temp_x.append(temp)

            h_resize = resize(this_dataset.h[i],size_HW)
            temp_h.append(h_resize)
 
        this_dataset.x = np.array(temp_x)
        this_dataset.h = np.array(temp_h)

    return this_dataset        