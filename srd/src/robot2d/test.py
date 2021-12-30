import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from .utils import create_folder_if_not_exists
from .maps import MapManager
from .model import Robot2NN

def run_tests(args):
    if args['test_map']:
        run_test_map(args)
    if args['test_act']:
        run_test_activations(args)
    if args['test_act2']:
        run_test_activations2(args)
    if args['test_local_act']:
        run_test_local_action(args)
    if args['test_plan']:
        run_test_plan(args)
    if args['test_srd']:
        test_self_reward(args)

def off_ticks():
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

def run_test_map(args):
    print('run_test_map')    
    
    N = 12
    H,W = args['map_size']
    net = Robot2NN(args)

    mm = MapManager(args, net.ACTIONS)
    plt.figure(figsize=(10,6))
    for i in range(N):
        pos,_ = mm.get_random_pos(H, W)        
        this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=0, 
            map_size=args['map_size'],target_pos=None, target_exclude=pos)

        plt.gcf().add_subplot(3,4,1+i)
        plt.gca().imshow(this_map.transpose(1,2,0))
        plt.gca().scatter(pos[0],pos[1],marker='x',c='b')
        # plt.gca().scatter(attn_pos[0],attn_pos[1], s=64, edgecolor='orange', facecolor='none')
        if i>0: off_ticks()

    plt.tight_layout()
    plt.show()


def run_test_activations(args, ):
    print('run_test_activations()')
    # np.random.seed(0)

    N = 6
    H,W = args['map_size']
    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)

    def display_w_activations(lava_fraction=0.):
        plt.figure(figsize=(6,7))
        for j in range(N):
            pos, relative_pos = mm.get_random_pos(H, W)  
            idys, idxs = np.where(relative_pos==1)
            this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=lava_fraction , map_size=args['map_size'],
                target_pos=None, target_exclude=pos) 
            net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)
            
            plt.gcf().add_subplot(N, 5, 1 + j*5)
            plt.gca().imshow(this_map.transpose(1,2,0))
            plt.gca().scatter(pos[0],pos[1], c='b', marker='x')
            if j==0: 
                plt.gca().set_title('map')
            else:
                off_ticks()

            for i,x_name in zip([2,3,4,5],['target','dirt','grass', 'unknown']):
                plt.gcf().add_subplot(N, 5,i + j*5)
                aba = getattr(net, 'w_%s'%(str(x_name)))
                plt.gca().imshow(aba, cmap='gray', vmin=0.,vmax=1.)
                off_ticks()
                if j==0: 
                    plt.gca().set_title(x_name)
        plt.tight_layout()

    display_w_activations(lava_fraction=0.)
    display_w_activations(lava_fraction=0.1)
    plt.show()

def run_test_activations2(args, ):
    print('run_test_activations2()')

    N = 6
    n_row = 5
    H,W = args['map_size']
    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)    

    def test_act(lava_fraction):
        plt.figure(figsize=(8,7))
        for j in range(N):
            pos, relative_pos = mm.get_random_pos(H, W)  
            idx,idy = pos
            this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=lava_fraction , map_size=args['map_size'],
                target_pos=None, target_exclude=pos) 
            net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)
            v = net.compute_v1()
            v_sigma= net.compute_v_sigma(v)
            for i, x_name in enumerate(['target','self','grass','dirt']):
                plt.gcf().add_subplot(N,n_row+1, 1+i+(1+n_row)*j)
                this_v = v[x_name].clone().detach().numpy()[0,0]
                plt.gca().imshow(this_v, vmin=-1., vmax=1., cmap='bwr')

                if j==0:
                    plt.gca().set_title(x_name)

                if i>0 or j>0:
                    off_ticks()

            plt.gcf().add_subplot(N,n_row+1, 2+i+(1+n_row)*j)
            plt.gca().imshow(this_map.transpose(1,2,0) )
            plt.gca().scatter(idx,idy, c='b', marker='x')

            plt.gcf().add_subplot(N,n_row+1, 3+i+(1+n_row)*j)
            plt.gca().imshow(v_sigma.clone().detach().numpy()[0,0], cmap='bwr')

        plt.tight_layout()

    test_act(lava_fraction=0)
    test_act(lava_fraction=0.05)
    plt.show()

def run_test_local_action(args):
    print('run_test_local_action()')
    np.random.seed(0)

    H,W = args['map_size']
    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)    


    N = 10
    for i in range(N):
        pos, relative_pos = mm.get_random_pos(H, W)  
        idys, idxs = np.where(relative_pos==1)
        idx,idy = idxs[0], idys[0]
        
        this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=0. , map_size=args['map_size'],
            target_pos=None, target_exclude=pos) 
        net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)
        v = net.compute_v1()
        v_sigma= net.compute_v_sigma(v)

        choices = net.get_top_two_choices(v_sigma, idy, idx)
        print(choices)

def run_test_plan(args):
    print('run_test_plan()')
    H,W = args['map_size']
    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)    

    def debug_map_trail_temp(net, idx, idy, positions):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.gcf().add_subplot(121)
        plt.gca().imshow(net.x_attn.transpose(1,2,0))
        plt.gca().set_xlabel('x_attention')
    
        
        print('(x,y):',idx,idy)
        plt.gca().scatter([idx],[idy], marker='x',c='r')
        positions = np.array(positions)
        plt.gca().scatter(positions[:,0], positions[:,1],s=3, marker='.',c='b')
        plt.show()
        exit()

    N = 10
    n_reached = 0
    for i in range(N):
        pos, relative_pos = mm.get_random_pos(H, W)  
        idx,idy = pos

        this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=0. , map_size=args['map_size'],
            target_pos=None, target_exclude=pos) 
        net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
        plan, v_plan, positions, tiles, reached = net.make_a_plan()
        print(plan)
        print('v_plan:',v_plan)
        # print(reached)
        if reached:
            n_reached+=1

        # for debugging
        # net.peek_self_attention(marker_coords='auto')
        # debug_map_trail_temp(net, idx, idy, positions)

    print('n_reached:%s/%s'%(str(n_reached),str(N)))

def test_self_reward(args):
    print('test_self_reward()')
    H,W = args['map_size']
    net = Robot2NN(args)
    mm = MapManager(args, net.ACTIONS)    

    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.5,0.999))

    for i in range(args['n_maps']):
        update_text = '%s/%s'%(str(i+1),str(args['n_maps']))
        print('%-64s'%(str(update_text)),end='\r')
        pos, relative_pos = mm.get_random_pos(H, W)  
        idx,idy = pos
        this_map = mm.get_random_map(grass_fraction=0.3, lava_fraction=0. , map_size=args['map_size'],
            target_pos=None, target_exclude=pos) 
        net.register_current_mental_representation(attn_map=this_map, w_self=relative_pos)    
        
        plans, current_best_index = net.make_plans()
        
        # pretend it's lika batch optimization
        loss = 0.
        for i,this_plan in plans.items():
            plan, v_plan, positions, tiles, reached = this_plan 
            loss = loss + (v_plan-net.tanh(v_plan))**2
        loss = loss/len(plans)
        loss.backward()
        optimizer.step()

    print('\nDone...')
