import os, joblib, json
import numpy as np
from OANN.src.data import DonutDataX, DonutDataY


import OANN.src.model as model
from OANN.src.utils import parse_bool_from_string, strbool_description, readjust_bools

# Define here how data is loaded directly from directory:
def donut_numpy_loader(data_dir):
    x = np.load(data_dir)
    # include any processing here if needed
    return x

def get_data_dir(dataname='discretedonut'):
    DATA_DIR = os.path.join('data', 'BONN', dataname)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR+'.test', exist_ok=True)
    return DATA_DIR


def get_ckpt_dir(dataname='discretedonut'):
    CKPT_DIR = os.path.join('Checkpoint', 'BONN', dataname)
    os.makedirs(CKPT_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(CKPT_DIR, 'model.pth')
    return CKPT_DIR, MODEL_DIR

def prep_data(dargs, label_mode):
    if dargs['data'] == 'donut_example':
        datax = DonutDataX(100, test_sd=dargs['test_data_spread'], label_mode=label_mode,
            show_fig_and_exit=dargs['show_fig_and_exit'])
        folder_to_class_mapping = {f'class{i}':i for i in [0,1,2]}
        DATA_DIR = get_data_dir()
        CKPT_DIR, MODEL_DIR = get_ckpt_dir()
    elif dargs['data'] == 'big_donut_example':
        datax = DonutDataY(1000, test_sd=0.01, label_mode=label_mode,
            show_fig_and_exit=dargs['show_fig_and_exit'])
        folder_to_class_mapping = {f'class{i}':i for i in [0,1,2,3,4]}
        DATA_DIR = get_data_dir('bigdiscretedonut')
        CKPT_DIR, MODEL_DIR = get_ckpt_dir('bigdiscretedonut')
    # X,Y = datax.X, datax.Y
    # X_test, Y_test = datax.X_test, datax.Y_test

    TEST_DATA_DIR = DATA_DIR+".test"
    print(f'DATA_DIR:{DATA_DIR}\nTEST_DATA_DIR:{TEST_DATA_DIR}')
    if len(os.listdir(DATA_DIR))==0:
        datax.save_data_by_class(DATA_DIR)
    if len(os.listdir(DATA_DIR+'.test'))==0:
        datax.save_testdata_by_class(TEST_DATA_DIR)
    RESULT_DIR = os.path.join(CKPT_DIR,'model.results.json')

    DIRS = {'CKPT_DIR': CKPT_DIR,'MODEL_DIR':MODEL_DIR,
        'DATA_DIR':DATA_DIR, 'TEST_DATA_DIR': TEST_DATA_DIR,
        'RESULT_DIR': RESULT_DIR,
        }
    return datax, folder_to_class_mapping, DIRS

def train_donut_example(dargs, label_mode='discrete'):
    print('donut_example...')
    TOGGLES = [parse_bool_from_string(x) for x in dargs['debug_toggles']]

    if label_mode!='discrete':
        raise NotImplementedError()

    ############# DATA #############
    datax, folder_to_class_mapping, DIRS = prep_data(dargs, label_mode)

    ########## SETUP MODEL #########
    from OANN.src.model import BONN
    settings = {'init_new':True,
        'folder_to_class_mapping': folder_to_class_mapping,
        'DATA_DIR': DIRS['DATA_DIR'],
        'n_class':len(folder_to_class_mapping),
        'output_mode': 'discrete',
        'elasticsize':dargs['elasticsize'], # for donut data, this is about 3 samples per class in a layer
        'data_fetcher': donut_numpy_loader,
        'interp_mode': 'layerwise_top_accumulation',
    }
    net = BONN(**settings)
    net.ix.print_(**{'mode':'datasizes'})
    redirect_for_testing(dargs, TOGGLES, net=net) # for debugging!    
    net.fit_data(max_iter=None, verbose=dargs['verbose'])

    print(f"\nSaving model to {DIRS['MODEL_DIR']}")
    net.save_state(DIRS['MODEL_DIR'])
    print()

    # ########### TEST MODEL ON THE TRAINING DATA ############
    # from OANN.src.eval import evaluate_net
    # evaluate_net(net, folder_to_class_mapping, DIRS['DATA_DIR'], dargs,verbose=dargs['verbose'])


def eval_donut_example(dargs, label_mode='discrete'):
    print('eval donut_example...')

    if label_mode!='discrete':
        raise NotImplementedError()

    ############# DATA #############
    _, _ , DIRS =prep_data(dargs, label_mode)

    ############# LOAD MODEL ###########
    from OANN.src.model import BONN
    settings = {'init_new':False,'MODEL_DIR': DIRS['MODEL_DIR'],}
    net = BONN(**settings)
    net.ix.print_(**{'mode':'datasizes'})
    net.ix.print_(**{'mode':'datalayerstatus'})

    ########### TEST MODEL ON THE TRAINING DATA ############
    print(f'\n{48*"="}\nPrinting incorrectly predicted training samples...')
    from OANN.src.eval import evaluate_net
    results_train = evaluate_net(net, net.ix.folder_to_class_mapping, DIRS['DATA_DIR'], dargs,verbose=20)
    print(48*"=")

    ########### TEST MODEL ON EVAL DATA ############
    print(f'\n{48*"="}\nEvaluating on test samples...')
    results_test = evaluate_net(net, net.ix.folder_to_class_mapping, DIRS['TEST_DATA_DIR'], dargs,verbose=100)

    results = {
        'train': results_train,
        'test': results_test,
    }
    with open(DIRS['RESULT_DIR'], 'w') as json_file:
        json.dump(results, json_file, indent=4, sort_keys=True)

INSTRUCTIONS = \
""" h: help
q: quit
x: incorrect indices. 
"""
def collect_random_samples(n, data_sizes):
    n_class = len(data_sizes)
    random_samples = []
    for i in range(n):
        random_class = np.random.randint(0,n_class)
        n_indices = data_sizes[random_class]
        random_index = np.random.randint(0,n_indices)
        random_samples.append((random_class,random_index))
    return random_samples

def random_inspect_donut_example(dargs, label_mode='discrete'):
    print('random_inspect_donut_example...')
    if label_mode!='discrete':
        raise NotImplementedError()

    ############# DATA #############
    _, _ , DIRS =prep_data(dargs, label_mode)

    from OANN.src.model import BONN
    settings = {'init_new':False,'MODEL_DIR': DIRS['MODEL_DIR'],}
    net = BONN(**settings)

    with open(DIRS['RESULT_DIR']) as f:
        results = json.load(f)
    incorrect_indices_test = [tuple(map(int, x[1:-1].split(', '))) for x in results['test']['incorrect_indices']]

    while True:
        userkey = input("Enter command (enter h for help):")

        if userkey=='h':
            print(INSTRUCTIONS)
            continue
        elif userkey=='q':
            break
        
        if userkey=='x':
            print('Finding wrongly predicted indices.')
            nmax = input('Enter integers n samples to display:')
        else:
            nmax = userkey
            
        if int(nmax) + 0 == int(nmax) :
            nmax = int(nmax)
            print(nmax, )
        else: 
            print('try again.')
            continue

        if userkey=='x':
            pass
        else:
            random_samples = collect_random_samples(nmax, net.ix.data_sizes)
            print(random_samples)
            raise Exception('gg')

    print('exiting...')




def redirect_for_testing(dargs, TOGGLES, **kwargs):
    # ANYTHING HERE MIGHT BE REMOVED ANYTIME
    # SOLELY FOR DEBUGGING

    EXIT= False
    import OANN.src.tests as tex
    
    if TOGGLES[0]==1:
        tex.redirect_for_testing_indexmanagement(dargs, TOGGLES, **kwargs); EXIT = True
    if TOGGLES[1]==1:
        tex.redirect_for_testing_fetch_data_by_elasticset(dargs, TOGGLES, **kwargs); EXIT = True
    if TOGGLES[2]==1:        
        tex.redirect_for_testing_data_fitting(dargs, TOGGLES, **kwargs); EXIT = True
    if TOGGLES[3]==1:    
        kwargs['ALLOW_INTERPOLATION'] = False    
        tex.redirect_for_prediction(dargs, TOGGLES, **kwargs); EXIT = True
    if TOGGLES[4]==1:    
        kwargs['ALLOW_INTERPOLATION'] = True    
        tex.redirect_for_prediction(dargs, TOGGLES, **kwargs); EXIT = True
    if EXIT:
        exit()




