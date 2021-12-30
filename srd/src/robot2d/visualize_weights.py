import os
import numpy as np
from .model import Robot2NN
import matplotlib.pyplot as plt
from .utils import create_folder_if_not_exists

import torch

def manage_robot_dir(args):
    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    
    CHECKPOINT_DIR = os.path.join(args['ROOT_DIR'],'checkpoint')
    create_folder_if_not_exists(CHECKPOINT_DIR)
    if args['CHECKPOINT_LAYER'] is not None:
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,args['CHECKPOINT_LAYER'])
        create_folder_if_not_exists(CHECKPOINT_DIR)

    DIRS = {
        'ROOT_DIR': args['ROOT_DIR'],
        'CHECKPOINT_DIR': CHECKPOINT_DIR,
    }
    args['DIRS'] = DIRS
    return args

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_weights(args):
    print('compare_weights()')

    args = manage_robot_dir(args)
    net = Robot2NN(args)
    if args['PROJECT_NAME'] is not None:
        MODEL_DIR = os.path.join(args['DIRS']['CHECKPOINT_DIR'], args['PROJECT_NAME'], 'robot2D.model')
        checkpoint = torch.load(MODEL_DIR)
        net.load_state_dict(checkpoint['net'])  
        print('loading from %s'%(str(MODEL_DIR)))
    else:
        print('loading default model...')

    n_params = count_parameters(net)
    print('n_params:', n_params)

    def plot_graph(vmin=-1., vmax=1., cmap='bwr', cmap_mode='single'):
        plt.figure(figsize=(10,7))
        running_max, running_min = -np.inf, np.inf
        for i,(mod_name,mod) in enumerate(net.tile_modules.items()):
            print(mod_name)
            for j, deconv in enumerate(mod.kernel):
                plt.gcf().add_subplot(4,5, j+1+5*i)
                dweight = deconv.weight.data.clone().detach().numpy()[0,0]
                # print(dweight.shape)
                # print(j+1 + 5*i)

                this_max, this_min = np.max(dweight), np.min(dweight)
                if this_max>running_max:
                    running_max = this_max
                if this_min<running_min:
                    running_min = this_min

                ax = plt.gca().imshow(dweight, cmap=cmap, vmax=vmax, vmin=vmin)
                offticks()

                if cmap_mode=='all':
                    cbar = plt.gcf().colorbar(ax)
                    cbar.ax.tick_params(labelsize=7) 
                if j==0:
                    plt.gca().set_ylabel(mod_name)
                if i==0:
                    plt.gca().set_title('deconv %s'%(str(j+1)))

        if cmap_mode=='single':
            cbar = plt.gcf().colorbar(ax)
            cbar.ax.tick_params(labelsize=7) 
        return running_max, running_min

    print('Plotting with weights...')
    running_max, running_min = plot_graph()
    print('Plotting with weights with adjusted intensity range...')
    _,_ = plot_graph(vmin=running_min - 0.1*np.abs(running_min), vmax=running_max+ 0.1*np.abs(running_max), cmap='hot', cmap_mode='all')
    plt.show()

def offticks():
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])