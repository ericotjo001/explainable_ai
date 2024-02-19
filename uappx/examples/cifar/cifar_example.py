import torch
import os, joblib
import numpy as np
from PIL import Image
from src.utils import parse_bool_from_string, strbool_description, readjust_bools
from .cifar_prep import prep_data_and_dirs, get_admission_th, prep_deep_neural_network_and_data_loader
from .dnn import device


def run_cifar_example(args, dargs, parser):
    print('cifar_example...')

    parser.add_argument('--submode', default='train', type=str, help=None)
    parser.add_argument('--kwidth', default=16, type=int, help=None)

    BOOLS = { # if any
        'DNN_TRAINING':0,
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)


    if dargs['submode'] == 'train_dnn':
        train_(dargs, parser, BOOLS, convNN=True)
    elif dargs['submode'] == 'ces':
        # cse: compare with euclidean dist, selected samples
        ces_(dargs, parser, BOOLS)
    elif dargs['submode'] == 'hyper':
        hyper_(dargs, parser, BOOLS)
    elif dargs['submode'] == 'train':
        train_(dargs, parser, BOOLS,)
    elif dargs['submode'] == 'eval_train':
        eval_train_(dargs, parser, BOOLS,)
    elif dargs['submode'] == 'eval':
        eval_(dargs, parser, BOOLS)
    elif dargs['submode'] == 'result':
        result_(dargs)
    elif dargs['submode'] == 'showcase':
        showcase_(parser, BOOLS)
    else:
        raise NotImplementedError('invalid submode?')

def ces_(dargs, parser, BOOLS):
    from .cifar_showcase_select import ces
    ces(dargs, parser, BOOLS)

def hyper_(dargs, parser, BOOLS):
    from .cifar_hyper import cifar_hyper_
    cifar_hyper_(dargs, parser, BOOLS)

def train_(dargs, parser, BOOLS, convNN=False):
    print('train_ (training on all)')

    modelname = 'cifar_k%s.pth'%(str(dargs['kwidth']))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)
    dnn, cifar_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)
    if convNN: exit() # just training convolutional Neural Network, not building kabedonn
    
    from src.model.kabedonn import KABEDONN
    settings = {'init_new':True,
        'folder_to_class_mapping': folder_to_class_mapping,
        'DATA_DIR': DIRS['DATA_DIR'],
        'n_class':len(folder_to_class_mapping),
        'kwidth':dargs['kwidth'], # for donut data, this is about 3 samples per class in a layer
        'data_fetcher': cifar_img_loader,
        'interpolator_settings': None,
        'activation_threshold': 0.999,
        'admission_threshold':get_admission_th,
    }
    net = KABEDONN(**settings)    

    import time
    start = time.time()
    net.fit_data(config={
        'print_final_info':True,
        'balance_test': False,        
        'qconfig': None,
        })
    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] '%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)),))

    start = time.time()
    net.evaluate_and_finetune_on_train_data()
    end = time.time()
    elapsed = end - start
    print(' Finetune time taken %s[s] = %s [min] '%(str(round(elapsed,1)), 
        str(round(elapsed/60.,1)),))

    print('saving to %s...'%(str(DIRS['MODEL_DIR'])))
    net.ix.data_fetcher = None # so that it is now pickleable
    joblib.dump(net, DIRS['MODEL_DIR'])


def eval_train_(dargs, parser, BOOLS):
    print('eval_train_')
    modelname = 'cifar_k%s.pth'%(str(dargs['kwidth']))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)
    dnn, mnist_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':False,
        'MODEL_DIR': DIRS['MODEL_DIR'],
        'interpolator_settings': None,
    }

    print(DIRS['MODEL_DIR'])
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)
    net.ix.data_fetcher = mnist_img_loader

    TRAINING_EVAL_RESULT = net.reevaluate_train_data_status()
    TRAINING_EVAL_RESULT.print_accuracy()

    joblib.dump(TRAINING_EVAL_RESULT, DIRS['TRAIN_RESULT_DIR'])
    print('saving reevaluated model...')
    net.ix.data_fetcher = None # so that it is now pickleable
    joblib.dump(net, DIRS['MODEL_DIR'])


def eval_(dargs, parser, BOOLS):
    print('eval_')
    modelname = 'cifar_k%s.pth'%(str(dargs['kwidth']))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)
    dnn, cifar_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':False,
        'MODEL_DIR': DIRS['MODEL_DIR'],
        'interpolator_settings': None,
    }
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)
    net.ix.data_fetcher = cifar_img_loader
    net.print_final_()
    print()

    from src.model.eval import evaluate_on_test_data
    print('Starting external evaluation...')
    eval_settings = {
        'DIRS':DIRS,
        'folder_to_class_mapping': folder_to_class_mapping,
        'data_fetcher': cifar_img_loader,
    }
    TEST_EVAL_RESULT = evaluate_on_test_data(net, **eval_settings)
    print('indices of wrongly predicted samples:',TEST_EVAL_RESULT.indices_wrong_data)

    print('\nSummary:')
    TEST_EVAL_RESULT.print_accuracy()
    joblib.dump(TEST_EVAL_RESULT, DIRS['TEST_RESULT_DIR'])

def result_(dargs):
    print('result_')
    modelname = 'cifar_k%s.pth'%(str(dargs['kwidth']))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':False,
        'MODEL_DIR': DIRS['MODEL_DIR'],
        'interpolator_settings': None,
    }
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)

    TRAINING_EVAL_RESULT = joblib.load(DIRS['TRAIN_RESULT_DIR'])
    TEST_EVAL_RESULT = joblib.load(DIRS['TEST_RESULT_DIR'])

    # can be too much printed, so we suppress them
    # net.print_final_() 
    # net.print_layer_hierarchy() 

    print('='*32)
    TRAINING_EVAL_RESULT.print_accuracy()
    TEST_EVAL_RESULT.print_accuracy()

def showcase_(parser,BOOLS, settings=None):
    print('showcase_')
    parser.add_argument('--classes', nargs='+', default=[0]) 
    parser.add_argument('--idx', nargs='+', default=[0]) 
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)    

    from datetime import datetime
    now = datetime.now()
    dated_showcase = now.strftime("%Y-%m-%d_%H-%M-%S")        

    ######### net ########
    modelname = 'cifar_k%s.pth'%(str(dargs['kwidth']))
    folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs, modelname=modelname)
    dnn, cifar_img_loader = prep_deep_neural_network_and_data_loader(dargs, parser, BOOLS, DIRS, device=device)
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)
    net.ix.data_fetcher = cifar_img_loader
    net.print_final_()

    ######### load test data that we want to showcase ########
    from .cifar_showcase import load_test_data
    DATA_POINTS_OF_INTEREST = zip(dargs['classes'],dargs['idx'])
    TESTDAT = load_test_data( net, DIRS, DATA_POINTS_OF_INTEREST)    

    ########## plot ##########
    from .cifar_showcase import plot_relevant_examples
    plot_relevant_examples(TESTDAT, net, DIRS)