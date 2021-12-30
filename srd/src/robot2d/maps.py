import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging

class MapManager(object):
    def __init__(self, args, ACTIONS):
        super(MapManager, self).__init__()
        self.args = args
       
        self.TILES_RAW = {
            'target':np.array((255,215,0)),
            'dirt': np.array((139,69,19)), 
            'grass':np.array((0,128,0)),
            'lava': np.array((255,0,0)),
        }

        self.TILES = {x:y/255. for x,y in self.TILES_RAW.items()}        
        self.target_pix = self.TILES['target']

        self.ACTIONS = ACTIONS

        # GLOBAL INFORMATION:
        # intiiate it using this object's methods
        # these variables should NOT be visible to the units
        # They are actually irrelevant if the unit can see the whole map.
        self.pos = None # idx,idy
        self.this_map = None # C,H,W

        # buffer
        self.buffer = defaultdict(list)
        self.counter = 0

        # others
        self.recognition_threshold = 1e-4

    def get_random_map(self, grass_fraction=0.3, lava_fraction=0, map_size=(10,12),
        target_pos=None, target_exclude=None):
        # target_exclude = (idx,idy), specify the position which we don't want the target to appear on
        # If we set target_exclude to be the unit's starting position, 
        # we prevent target appearing on the unit's starting position

        H,W = map_size

        this_map = np.zeros(shape=(3,)+tuple(map_size)) + self.TILES['dirt'].reshape(3,1,1)
        grass = self.TILES['grass'].reshape(3,1,1)

        if lava_fraction==0:
            grass_pos = np.random.uniform(0,1,size=map_size)<grass_fraction
            this_map = this_map * (1- grass_pos) + grass_pos * grass
        else:
            lava = self.TILES['lava'].reshape(3,1,1)
            random_pos = np.random.uniform(0,1,size=map_size)<(grass_fraction+lava_fraction)

            grass_or_lava = np.random.uniform(0,grass_fraction+lava_fraction,size=map_size)<grass_fraction 
            grass_pos = random_pos * grass_or_lava 
            lava_pos = random_pos * (1-grass_or_lava)
            this_map = this_map * (1-random_pos) + grass_pos*grass +lava_pos*lava

        if target_pos is None:
            while True:
                idx = np.random.randint(0,W)
                idy = np.random.randint(0,H)
                if target_exclude is None:
                    break
                else:
                    idx_ex, idy_ex = target_exclude
                    if not (idx==idx_ex and idy==idy_ex):
                        # make sure the robot starts at a dirt tile
                        this_map[:,idy_ex,idx_ex] = self.TILES['dirt']
                        break
        else:
            idx,idy = target_pos
        this_map[:,idy,idx] = self.TILES['target'] # remember it's C,H,W
        return this_map

    def get_random_pos(self, H, W):
        pos = np.random.randint(0,W),np.random.randint(0,H) # idx, idy
        relative_pos = self.get_relative_pos(H,W,pos)
        return pos, relative_pos

    def get_relative_pos(self, H,W,pos):
        relative_pos =  np.zeros(shape=(H,W))
        idx,idy = pos
        relative_pos[idy,idx] = 1.
        return relative_pos

    def register_state(self, args, pos, this_map):
        H,W = self.args['map_size']
        self.pos = pos 
        self.this_map = this_map



    ################## for making gifs ##################
    def reset_gif_save(self):
        self.buffer = defaultdict(list)
        self.counter = 0

    def animation_maker_tools(self, debug_gif_maker=None, mode='collect',**kwargs):
        if debug_gif_maker is None: return

        if mode=='collect':
            # temp = kwargs['Robot2NN'].x_attn.clone().detach().cpu().numpy()[0].transpose(1,2,0)
            self.buffer['this_map'].append(kwargs['this_map'])
            # idy, idx = kwargs['Robot2NN'].get_mental_position_indices()        
            # idy, idx = idy[0], idx[0]
            self.buffer['pos'].append(kwargs['pos'])
            self.buffer['title'].append(kwargs['title'])

        elif mode=='save':  
            self._save_gif(debug_gif_maker['save_dir'], )
            self._save_static_trail(debug_gif_maker['static_img_save_dir'], )
            


    def _save_gif(self, save_dir):
        logging.disable(30) # just to disable the buggy matplotlib.animation warning

        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        im = plt.imshow(self.buffer['this_map'][0], origin='upper')

        H,W,_ = self.buffer['this_map'][0].shape
        
        ax = plt.gca()

        idx,idy = self.buffer['pos'][0]
        imxy = plt.scatter([idx],[idy] ,marker='x',c='b')

        plt.gca().set_xlim([-1,W])
        plt.gca().set_ylim([-1,H])
        plt.gca().invert_yaxis()
        def update(i):
            label = 'timestep {0}'.format(i)
            im.set_data(self.buffer['this_map'][i])

            plt.title('%s : %s'%(str(i),str(self.buffer['title'][i])))

            idx,idy = self.buffer['pos'][i]
            imxy.set_offsets((idx,idy))
            return im

        anim = FuncAnimation(fig, update, frames=range(len(self.buffer['this_map'])), interval=200)
        anim.save(save_dir, dpi=80,) 
        
        logging.disable(0) # make sure that other warnings still apply
        plt.close()

    def _save_static_trail(self, save_dir):
        pos_array = np.array(self.buffer['pos']).T
        plt.figure()
        plt.gca().imshow(self.buffer['this_map'][0])
        plt.gca().plot(pos_array[0], pos_array[1],c='b')
        for i, (idx,idy) in enumerate(zip(pos_array[0],pos_array[1])):
            plt.gca().annotate('%s'%(str(i)), (idx,idy))
        plt.savefig(save_dir)
        plt.close()