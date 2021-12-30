import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


def create_folder_if_not_exists(this_dir):
    if not os.path.exists(this_dir):
        os.mkdir(this_dir)

def manage_dir(args):
    if args['ROOT_DIR'] is None:
        args['ROOT_DIR'] = os.getcwd()

    args['CKPT_DIR'] = os.path.join(args['ROOT_DIR'], 'checkpoint')
    create_folder_if_not_exists(args['CKPT_DIR'])
    args['PROJECT_DIR'] = os.path.join(args['CKPT_DIR'], args['PROJECT_NAME'])
    create_folder_if_not_exists(args['PROJECT_DIR'])
    return args

def run_fish(args):
    print('run_fish()')

    args= manage_dir(args)

    # ========== Specify Environment ==========
    # See main text for details.
    # In short: 
    # 1. environment is a vector x = [x1,x2,x3]
    # 2. food is xk=0.5. No food is xk=0.01
    from .test import run_test
    run_test(args,)

    from .data_collector import FishUnitDataCollector
    dc = FishUnitDataCollector(args)

    from .model import Fish1D, Fish1DMapManager
    mm = Fish1DMapManager(args)
    fish = Fish1D(args, ENV_SHAPE=mm.ENV_SHAPE)

    ENV = mm.get_env_from_template()
    for i in range(args['n_iter']):
        x = fish.get_input_tensor(ENV=ENV)

        with torch.no_grad():
            greedy_decision = fish.make_decision(x) # returns y, x1, greedy_decision

        fish.update_state(action=greedy_decision, ENV=ENV)
        ENV = mm.update_state(action=greedy_decision, ENV=ENV)
        dc.get_unit_data(i,fish, ENV)

        if fish.INTERNAL_STATE['energy']<=0:
            print('fish is dead.')
            break

    save_dir = os.path.join(args['PROJECT_DIR'],'robotfish.png')
    dc.display_data(save_dir=save_dir)

def run_fish_srd(args):
    print('run_fish_srd()')
    """
    See run_fish(), similar. No tests here.
    """
    assert(args['n_iter']>=256)
    args= manage_dir(args)

    from .data_collector import FishUnitDataCollector
    dc = FishUnitDataCollector(args)

    from .model import Fish1D, Fish1DMapManager
    mm = Fish1DMapManager(args)
    fish = Fish1D(args, ENV_SHAPE=mm.ENV_SHAPE)

    MEMORY_SIZE = 8
    optimizer = optim.Adam(fish.nn.parameters(), lr=0.002, betas=(0.5,0.999))
    criterion = nn.CrossEntropyLoss()

    ENV = mm.get_env_from_template()
    z = None
    fish.nn.zero_grad()

    torch.set_printoptions(precision=2, sci_mode=False)
    for i in range(args['n_iter']):
        if (i+1)%128==0 or (i+1)==args['n_iter']:
            text = '%s/%s'%(str(i+1),str(args['n_iter']))
            print('%-64s'%(str(text)), end='\r')
        x = fish.get_input_tensor(ENV=ENV)

        greedy_decision, z2 = fish.make_self_rewarded_decision(x)
        z = z + z2 if z is not None else z2
        if (i+1)%MEMORY_SIZE==0:
            loss = criterion(z, torch.argmax(z,dim=1))
            loss.backward()     
            optimizer.step()   

            # reset
            z = None    
            fish.nn.zero_grad()

        fish.update_state(action=greedy_decision, ENV=ENV)
        ENV = mm.update_state(action=greedy_decision, ENV=ENV)
        
        dc.get_unit_data(i,fish, ENV)

        if fish.INTERNAL_STATE['energy']<=0:
            print('fish is dead.')
            break

    print('\nrun over')
    save_dir = os.path.join(args['PROJECT_DIR'],'robotfishsrd.png')
    dc.display_srd_data(args, save_dir=save_dir)