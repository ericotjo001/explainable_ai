from ..utils import FastPickleClient
from .utils import create_folder_if_not_exists
import os
from .maps import MapManager

def manage_data_dir(args):
    DIRS = {}
    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    CKPT_DIR = os.path.join(args['ROOT_DIR'],'checkpoint')
    create_folder_if_not_exists(CKPT_DIR)
    DATA_FOLDER_DIR = os.path.join(args['ROOT_DIR'],'data')
    create_folder_if_not_exists(DATA_FOLDER_DIR)

    DATA_DIR = os.path.join(DATA_FOLDER_DIR,args['map_data_name'])

    DIRS['CKPT_DIR'] = CKPT_DIR
    DIRS['DATA_DIR'] = DATA_DIR
    args['DIRS'] = DIRS
    return args

class DataStorage(FastPickleClient):
    def __init__(self, ):
        super(DataStorage, self).__init__()
        self.data = []

    def store_data(self, this_map, pos):
        self.data.append((this_map, pos))

def store_map_data(args):
    print('store_map_data()')

    args = manage_data_dir(args)
    H,W = args['map_size']
    mm = MapManager(args, ACTIONS=None)
    dat = DataStorage()

    for i in range(args['n_maps']):
        pos, _ = mm.get_random_pos(H, W)     
        this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=args['lava_fraction'], map_size=args['map_size'],
            target_pos=None, target_exclude=pos)

        if args['peek_map']:
            peek_map(this_map, pos)

        dat.store_data(this_map, pos, )

        update_text = 'n data: %-4s '%(str(len(dat.data)),)
        print('%-64s'%(update_text),end='\r')

    print()
    dat.pickle_data(dat, args['DIRS']['DATA_DIR'])

def peek_map(this_map, pos):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.gcf().add_subplot(121)
    plt.gca().imshow(this_map.transpose(1,2,0))
    plt.gca().set_xlabel('this map')

    idx,idy= pos
    print('(x,y):',idx,idy)
    plt.gca().scatter([idx],[idy], marker='x',c='b')

    plt.show()
    if do_exit:
        exit()        
