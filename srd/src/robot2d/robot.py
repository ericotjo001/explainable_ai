import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from .utils import create_folder_if_not_exists
from .maps import MapManager
from .model import Robot2NN

from .data import DataStorage

device = None # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def manage_robot_dir(args):
    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()
    
    CHECKPOINT_DIR = os.path.join(args['ROOT_DIR'],'checkpoint')
    create_folder_if_not_exists(CHECKPOINT_DIR)
    if args['CHECKPOINT_LAYER'] is not None:
        CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR,args['CHECKPOINT_LAYER'])
        create_folder_if_not_exists(CHECKPOINT_DIR)


    DATA_FOLDER_DIR = os.path.join(args['ROOT_DIR'],'data')
    create_folder_if_not_exists(DATA_FOLDER_DIR)
    
    DIRS = {
        'ROOT_DIR': args['ROOT_DIR'],
        'CHECKPOINT_DIR': CHECKPOINT_DIR,
    }

    if args['PROJECT_NAME'] is not None:
        PROJECT_DIR = os.path.join(CHECKPOINT_DIR,args['PROJECT_NAME'])
        create_folder_if_not_exists(PROJECT_DIR)
        MODEL_DIR = os.path.join(PROJECT_DIR, 'robot2D.model')
        DIRS['PROJECT_DIR'] = PROJECT_DIR
        DIRS['MODEL_DIR'] = MODEL_DIR

    if args['train_data_name'] is not None:
        TRAIN_DATA_DIR = os.path.join(DATA_FOLDER_DIR,args['train_data_name'])
        DIRS['TRAIN_DATA_DIR'] = TRAIN_DATA_DIR
    if args['map_data_name'] is not None:
        DATA_DIR = os.path.join(DATA_FOLDER_DIR,args['map_data_name'])
        DIRS['DATA_DIR'] = DATA_DIR

    args['DIRS'] = DIRS
    return args


#####################################
# To run and save robot exploration images
#####################################
def run_robot(args):
    print('run_robot()')

    from .test import run_tests
    run_tests(args)

    args = manage_robot_dir(args)
    DIRS =  args['DIRS']
    H,W = args['map_size']

    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)    
    with torch.no_grad():
        for i in range(args['n_maps']):
            # initiate data
            pos, relative_pos = mm.get_random_pos(H, W)  
            idx,idy = pos
            this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=args['lava_fraction'] , map_size=args['map_size'],
                target_pos=None, target_exclude=pos) 
            net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
            
            # run network
            plans, current_best_index = net.make_plans()
            plan, v_plan, positions, tiles, reached = plans[current_best_index]

            # store gif
            new_map_text = 'new map: %s'%(str(i+1)) 
            save_dir = os.path.join(DIRS['PROJECT_DIR'], 'robot_%s.gif'%(str(i+1)))
            static_img_save_dir = os.path.join(DIRS['PROJECT_DIR'], 'robot_%s.static.jpg'%(str(i+1)))
            debug_gif_maker = {'save_dir': save_dir,'static_img_save_dir':static_img_save_dir}

            for action, this_pos, tile in zip(plan, positions, tiles):
                mm.animation_maker_tools(debug_gif_maker=debug_gif_maker, mode='collect',
                    this_map=this_map.transpose(1,2,0), pos=this_pos, 
                    title='%s %-12s [%s]'%(str(action), str(tile), str(reached)),)

            print('Done, saving for map %s at iter %s'%(str(i+1), str(len(plan)),))
            mm.animation_maker_tools(debug_gif_maker=debug_gif_maker,mode='save')      
            mm.reset_gif_save()

def run_robot_srd(args):
    print('run_robot_srd()')

    args = manage_robot_dir(args)
    DIRS =  args['DIRS']
    H,W = args['map_size']

    net = Robot2NN(args)
    print('\nLoading srd traind model!\n')
    checkpoint = torch.load(DIRS['MODEL_DIR'])
    net.load_state_dict(checkpoint['net'])        

    mm = MapManager(args, net.ACTIONS)    
    with torch.no_grad():
        for i in range(args['n_maps']):
            # initiate data
            pos, relative_pos = mm.get_random_pos(H, W)  
            idx,idy = pos
            this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=args['lava_fraction'] , map_size=args['map_size'],
                target_pos=None, target_exclude=pos) 
            net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
            
            # run network
            plans, current_best_index = net.make_plans()
            plan, v_plan, positions, tiles, reached = plans[current_best_index]

            # store gif
            new_map_text = 'new map: %s'%(str(i+1)) 
            save_dir = os.path.join(DIRS['PROJECT_DIR'], 'robot_%s.gif'%(str(i+1)))
            static_img_save_dir = os.path.join(DIRS['PROJECT_DIR'], 'robot_%s.static.jpg'%(str(i+1)))
            debug_gif_maker = {'save_dir': save_dir,'static_img_save_dir':static_img_save_dir}

            for action, this_pos, tile in zip(plan, positions, tiles):
                mm.animation_maker_tools(debug_gif_maker=debug_gif_maker, mode='collect',
                    this_map=this_map.transpose(1,2,0), pos=this_pos, 
                    title='%s %-12s [%s]'%(str(action), str(tile), str(reached)),)

            print('Done, saving for map %s at iter %s'%(str(i+1), str(len(plan)),))
            mm.animation_maker_tools(debug_gif_maker=debug_gif_maker,mode='save')      
            mm.reset_gif_save()


#####################################
# For experimental data collection
#####################################

from ..utils import FastPickleClient 
class ResultsData(FastPickleClient):
    def __init__(self, ):
        super(ResultsData, self).__init__()
        self.tile_names = ['grass','dirt','unrecognized']
        self.results = {tile_name:[] for tile_name in self.tile_names}
        self.n_correct = 0
        self.n_total = 0

    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):

    def add_result(self, tiles, reached):
        if reached:
            self.n_correct += 1
        for tile_name in self.tile_names:
            n_tile = sum(np.array(tiles)==tile_name) 
            self.results[tile_name].append(n_tile)

    def save_histogram(self, IMG_DIR, iter_limit=48, n_max=None):
        plt.figure(figsize=(8,4))

        ax = plt.gcf().add_subplot(121)
        vn,vx,_ = plt.gca().hist( self.results['grass'], bins=range(iter_limit+1), rwidth=0.5 )
        plt.gca().set_xlabel('grass. max:%s'%(str(np.max(vn))))
        this_max = [vn]

        ax2 = plt.gcf().add_subplot(122)
        vn,vx,_ = plt.gca().hist( self.results['dirt'], bins=range(iter_limit+1), rwidth=0.5 )
        this_max.append(vn)

        n_max = int(np.max(this_max) *1.1)
        ax.set_ylim((0,n_max))
        ax2.set_ylim((0,n_max))
        plt.gca().set_xlabel('dirt. max:%s'%(str(np.max(vn))))

        plt.savefig(IMG_DIR)
        plt.close()
        

def run_robot_eval(args, run_srd=False, ResultsData=ResultsData):
    print('run_robot_eval()')

    H,W = args['map_size']
    args = manage_robot_dir(args)
    DIRS =  args['DIRS']
    RESULT_DIR = os.path.join(DIRS['PROJECT_DIR'], 'result.data')
    IMG_DIR = os.path.join(DIRS['PROJECT_DIR'], 'hist.%s.jpg'%(str(args['PROJECT_NAME'])))

    net = Robot2NN(args).to(device=device)
    if run_srd:
        # load srd trained model
        print('loading model...')
        checkpoint = torch.load(DIRS['MODEL_DIR'])
        net.load_state_dict(checkpoint['net'])

    dat = DataStorage()
    dat = dat.load_pickled_data(args['DIRS']['DATA_DIR'], tv=(0,0,None), 
        text='loading eval data...')
    n_data = len(dat.data)

    rd = ResultsData()
    mm = MapManager(args, net.ACTIONS)    
    with torch.no_grad():
        for i,(this_map, pos) in enumerate(dat.data):
            relative_pos = mm.get_relative_pos(H,W, pos)
            net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
            
            # run network
            plans, current_best_index = net.make_plans()
            plan, v_plan, positions, tiles, reached = plans[current_best_index]

            rd.add_result(tiles, reached)
            rd.n_total += 1

    # for x,y in rd.results.items():
    #     print('%s:%s'%(str(x),str(y)))
    print('n_correct/total: %s/%s'%(str(rd.n_correct),str(n_data)))
    rd.pickle_data(rd, RESULT_DIR, tv=(0,0,None), text='Saving result dictionary...')
    rd.save_histogram(IMG_DIR, iter_limit=args['iter_limit'], n_max=12)


def run_robot_srd_training(args, ResultsData=ResultsData):
    print('run_robot_srd_training()')

    if args['eval_after_train']:
        assert(args['map_data_name'] is not None)

    H,W = args['map_size']
    args = manage_robot_dir(args)
    DIRS =  args['DIRS']

    net = Robot2NN(args)
    net.to(device=device)
    
    dat = DataStorage()
    dat = dat.load_pickled_data(args['DIRS']['TRAIN_DATA_DIR'], tv=(0,0,None), text='loading train data...')
    n_data = len(dat.data)
    import random
    random.shuffle(dat.data)

    optimizer = optim.SGD(net.parameters(), lr=0.0001, )

    mm = MapManager(args, net.ACTIONS)    
    for i,(this_map, pos) in enumerate(dat.data):
        net.zero_grad()
        update_text = '%s/%s'%(str(i+1),str(n_data))
        if (i+1)%12==0 or (i+1)==n_data:
            print('%-64s'%(str(update_text)),end='\r')

        relative_pos = mm.get_relative_pos(H,W, pos)
        net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
        
        # run network
        plans, current_best_index = net.make_plans()
        # plan, v_plan, positions, tiles, reached = plans[current_best_index]

        loss = 0.
        for i,this_plan in plans.items():
            plan, v_plan, positions, tiles, reached = this_plan 
            loss = loss + (1.-net.tanh(v_plan))**2
        loss = loss/len(plans)
        loss.backward()
        optimizer.step()
    print('\nDone...')

    checkpoint = {'net': net.state_dict(),'optimizer': optimizer.state_dict(),}
    torch.save(checkpoint, DIRS['MODEL_DIR'])
    print('done saving to %s'%(str(DIRS['MODEL_DIR'])))

    if args['eval_after_train']:
        run_robot_eval(args,run_srd=True, ResultsData=ResultsData)

def aggregate_result(args):
    print('aggregate_result()')

    args['CHECKPOINT_LAYER'] = args['PROJECT_NAME']  
    PROJECT_HEADER = args['PROJECT_NAME'] 
    args = manage_robot_dir(args)
    IMG_DIR = os.path.join(args['DIRS']['CHECKPOINT_DIR'], 'AGG_HIST.jpeg')
    TXT_DIR = os.path.join(args['DIRS']['CHECKPOINT_DIR'], 'AGG_RESULT.txt')

    def get_results(i, args, srd=False):
        args['PROJECT_NAME'] = PROJECT_HEADER + str(1000+i)[1:]
        if srd:
            args['PROJECT_NAME'] = args['PROJECT_NAME'] + '.srd'
        args = manage_robot_dir(args)
        DIRS = args['DIRS']

        rd = ResultsData()
        RESULT_DIR = os.path.join(DIRS['PROJECT_DIR'], 'result.data')
        rd = rd.load_pickled_data(RESULT_DIR, tv=(0,0,100), text='loading results...')

        results = rd.results
        n_correct = rd.n_correct
        n_total = rd.n_total
        return results, n_correct, n_total

    iter_limit = args['iter_limit']
    def arrange_and_display_results(tile_name,
        results, n_correct, n_total,
        results_srd, n_correct_srd, n_total_srd):
        this_max = []
        
        h = np.array(results[tile_name])
        vn,vx,_ = plt.gca().hist( h, bins=range(iter_limit+1),  alpha=0.5, 
            histtype='step', cumulative=-1,label='standard')
        this_max.append(vn)

        h = np.array(results_srd[tile_name])
        vn,vx,_ = plt.gca().hist( h , bins=range(iter_limit+1), 
            rwidth=0.5, color='r', alpha=0.5,
            histtype='step', cumulative=-1, label='srd')
        this_max.append(vn)
        
        n_max = int(np.max(this_max) *1.1)
        return n_max

    n_tile_type = 2
    fig_height = 3
    if args['include_lava']: 
        n_tile_type+=1
        fig_height = 5
        
    txt = open(TXT_DIR,'w')
    update_text = '%10s %-7s %-7s %-7s\n'%(str(''),str('correct'),str('fraction'),str('n_total'))
    txt.write(update_text)
    print(update_text, end='')
    plt.figure(figsize=(int(2*args['n_expt']+1), int(fig_height)))
    for i in range(1,1+args['n_expt']):
        results, n_correct, n_total = get_results(i, args, srd=False)
        results_srd, n_correct_srd, n_total_srd = get_results(i, args, srd=True)

        update_text = '%-10s %7s %7s %7s\n'%('expt'+str(i),str(n_correct),str(np.round(n_correct/n_total,3)),str(n_total))
        update_text2 = '%-10s %7s %7s %7s\n'%('expt'+str(i)+'_SRD',str(n_correct_srd),str(np.round(n_correct_srd/n_total_srd,3)),str(n_total))
        print(update_text,end='')
        print(update_text2, end='')
        txt.write('%s%s'%(update_text,update_text2))

        this_max = []
        ##################################
        # RESULT FOR GRASS TILE
        ##################################
        ax = plt.gcf().add_subplot(n_tile_type,args['n_expt'], i)
        n_max = arrange_and_display_results('grass',
            results, n_correct, n_total, results_srd, n_correct_srd, n_total_srd)
        this_max.append(n_max)
        if i==1: 
            plt.gca().set_ylabel('grass')        
            plt.legend()
        else:
            plt.gca().set_yticks([])
        plt.gca().set_xticks([])

        ##################################
        # RESULT FOR DIRT TILE
        ##################################
        ax2 = plt.gcf().add_subplot(n_tile_type,args['n_expt'], i + args['n_expt'])
        n_max = arrange_and_display_results('dirt',
            results, n_correct, n_total, results_srd, n_correct_srd, n_total_srd)
        this_max.append(n_max)
        if i==1: 
            plt.gca().set_ylabel('dirt')        
        else:
            plt.gca().set_yticks([])

        ######## MAX VAL SETTING
        n_max = np.max(this_max)
        ax.set_ylim((0,n_max))
        ax2.set_ylim((0,n_max))

        ##################################
        # RESULT FOR LAVA TILE
        ##################################
        if args['include_lava']:
            plt.gca().set_xticks([])

            ax3 = plt.gcf().add_subplot(n_tile_type,args['n_expt'], i + 2*args['n_expt'])
            n_max = arrange_and_display_results('unrecognized',
                results, n_correct, n_total, results_srd, n_correct_srd, n_total_srd)
            this_max.append(n_max)
            if i==1: 
                plt.gca().set_ylabel('lava')
            else:
                plt.gca().set_yticks([])

        plt.gca().set_xlabel('expt %s'%(str(i)))        

    plt.tight_layout()

    txt.close()
    plt.savefig(IMG_DIR)