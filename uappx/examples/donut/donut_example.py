import os, joblib
import numpy as np
from src.utils import parse_bool_from_string, strbool_description, readjust_bools

def first_donut_example(args, dargs,parser, TOGGLES):
    print('first_donut_example...')

    parser.add_argument('--submode', default='train', type=str, help=None)
    parser.add_argument('--kwidth', default=16, type=int, help=None)
    parser.add_argument('--redir_id', default=0, type=int, help=None)
    BOOLS = { # if any
        # 'init_new': 1,
        'show_fig_and_exit': 1, 
    }
    for bkey,b in BOOLS.items():
        parser.add_argument('--%s'%(str(bkey)), default=str(b), type=str, help=strbool_description)

    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)

    if dargs['submode'] == 'train':
        train_(dargs, extra_settings=None)
    elif dargs['submode'] == 'eval_train':
        eval_train_(dargs)
    elif dargs['submode'] == 'eval':
        eval_(dargs)
    elif dargs['submode'] == 'find':
        find_(parser, BOOLS)
    elif dargs['submode'] == 'showcase':
        showcase_(parser, BOOLS)
    else:
        raise NotImplementedError('invalid submode?')


def train_(dargs, extra_settings=None):
    print('train_')

    datax, folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':True,
        'folder_to_class_mapping': folder_to_class_mapping,
        'DATA_DIR': DIRS['DATA_DIR'],
        'n_class':len(folder_to_class_mapping),
        'kwidth':dargs['kwidth'], # for donut data, this is about 3 samples per class in a layer
        'data_fetcher': donut_numpy_loader,
        'interpolator_settings': None,
    }
    if extra_settings is not None:
        for x,y in extra_settings.items():
            settings[x] = y
            
    net = KABEDONN(**settings)

    from .donut_tests import redirect_for_testing
    redirect_for_testing(dargs, datax, folder_to_class_mapping, DIRS, net)

    net.fit_data()
    net.evaluate_and_finetune_on_train_data()

    print('saving to %s...'%(str(DIRS['MODEL_DIR'])))
    joblib.dump(net, DIRS['MODEL_DIR'])

def eval_train_(dargs):
    datax, folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':False,
        'MODEL_DIR': DIRS['MODEL_DIR'],
        'interpolator_settings': None,
    }
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)

    TRAINING_EVAL_RESULT = net.reevaluate_train_data_status()
    TRAINING_EVAL_RESULT.print_accuracy()

def eval_(dargs):
    print('eval_')
    datax, folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs)

    from src.model.kabedonn import KABEDONN
    settings = {'init_new':False,
        'MODEL_DIR': DIRS['MODEL_DIR'],
        'interpolator_settings': None,
    }
    net = joblib.load(DIRS['MODEL_DIR']) # KABEDONN(**settings)
    net.print_final_()
    print()
    

    from src.model.eval import evaluate_on_test_data
    print('Starting external evaluation...')
    eval_settings = {
        'DIRS':DIRS,
        'folder_to_class_mapping': folder_to_class_mapping,
        'data_fetcher': donut_numpy_loader,
    }
    TEST_EVAL_RESULT = evaluate_on_test_data(net, **eval_settings)
    print('indices of wrongly predicted samples:',TEST_EVAL_RESULT.indices_wrong_data)

    print('\nSummary:')
    TEST_EVAL_RESULT.print_accuracy()
    joblib.dump(TEST_EVAL_RESULT, DIRS['TEST_RESULT_DIR'])


def showcase_(parser,BOOLS, settings=None):
    print('showcase_')
    parser.add_argument('--classes', nargs='+', default=[0]) 
    parser.add_argument('--idx', nargs='+', default=[0]) 
    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)
    dargs['show_fig_and_exit'] = False; args.show_fig_and_exit = False

    from datetime import datetime
    now = datetime.now()
    dated_showcase = now.strftime("%Y-%m-%d_%H-%M-%S")    

    if settings is None:
        settings = {
            'subplot1':{
                'annotate':True,
            },
            'subplot2':{
                'background_alpha':0.1,
                'annotate':True,
            }
        }
    #### prepare data ####
    _, folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs)
    net = joblib.load(DIRS['MODEL_DIR'])

    # ======= showcase layer hierarchy! ===========
    print('\n')
    net.print_layer_hierarchy()

    ####### load training data ###########
    from .donut_showcase import load_training_data_for_showcase
    x_train_batch, y_train_batch = load_training_data_for_showcase(net)

    print('\n')
    from .donut_showcase import load_data_assimilated_into_net
    # x_mainnodes, y_mainnodes, x_subnodes, y_subnodes = load_data_assimilated_into_net(net)
    anodes = load_data_assimilated_into_net(net)

    # ######## load test data that we want to showcase #########
    from .donut_showcase import load_test_data
    DATA_POINTS_OF_INTEREST = zip(dargs['classes'],dargs['idx'])
    TESTDAT = load_test_data( net, DIRS, DATA_POINTS_OF_INTEREST)

    ####### plot ###########
    import matplotlib.pyplot as plt
    from .donut_showcase import plot_nodes_assimilated_to_net, plot_line_to_mainnodes, \
        plot_scatter, plot_marker, write_timestamped_report

    plt.figure(figsize=(12,5))
    plt.gcf().add_subplot(121)
    s1 = settings['subplot1']
    plot_scatter(x_train_batch,y_train_batch,cmap='gray',alpha=0.5, annotate=s1['annotate'])
    plot_nodes_assimilated_to_net(anodes, alpha=0.5, annotate=s1['annotate'])

    plt.gcf().add_subplot(122)
    s2 = settings['subplot2']
    # background
    plot_scatter(x_train_batch,y_train_batch,
        cmap='gray',alpha=s2['background_alpha'], colorbar=False, annotate=s2['annotate'])
    plot_nodes_assimilated_to_net(anodes, alpha=s2['background_alpha'], 
        annotate=s2['annotate'],annot_main_only=True)

    #########################
    # for selected data
    #########################
    DATA_POINTS_OF_INTEREST = zip(dargs['classes'],dargs['idx'])
    plot_marker(TESTDAT['anodes']['x'],
        s=32,marker='^',edgecolor='m', alpha=1)  # small upright triangle: activated node
    if len(TESTDAT['wrong']['x'])>0:
        plot_marker(TESTDAT['wrong']['x'],s=128,marker='p',edgecolor='r')
    for x, mainnode in zip(TESTDAT['test']['x'], TESTDAT['anodes']['x']): # for wrong prediction
        plot_line_to_mainnodes(mainnode,[x], alpha=0.5, c='r', linestyle='--', linewidth=1 )
    # THE data
    plot_scatter(TESTDAT['test']['x'],TESTDAT['test']['y'], 
        annotations=['(%s,%s)'%(str(y0),str(idx)) for y0,idx in DATA_POINTS_OF_INTEREST], 
        annotate=True, cmap='jet',marker='x',alpha=0.9, annot_color='m') # main test data

    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['SHOWCASE_FOLDER_DIR'],dated_showcase+'.png' ))

    REPORT_DIR = os.path.join(DIRS['SHOWCASE_FOLDER_DIR'],dated_showcase+'.txt')
    write_timestamped_report(REPORT_DIR, TESTDAT)

def find_(parser, BOOLS,):
    parser.add_argument('--classes', nargs='+', default=[0]) 
    parser.add_argument('--layers', nargs='+', default=[1]) 
    parser.add_argument('--idx', nargs='+', default=[0]) 
    parser.add_argument('--find_what', type=str, default='activated_nodes') 
    parser.add_argument('--input', type=str, default='training_data') 


    args, unknown = parser.parse_known_args()
    dargs = vars(args)  # is a dictionary
    args, dargs = readjust_bools(args, dargs, BOOLS)
    dargs['show_fig_and_exit'] = False; args.show_fig_and_exit = False

    _, folder_to_class_mapping, DIRS = prep_data_and_dirs(dargs)
    net = joblib.load(DIRS['MODEL_DIR'])
    net.print_final_()
    net.print_layer_hierarchy()

    
    if dargs['find_what'] == 'activated_nodes':
        # we demonstrate this using training data samples

        print('\n','='*24,'OBSERVE DATA STATUS HERE', '='*24)
        if dargs['input'] == 'training_data':
            elasticset = []
            for y0,idx in zip(dargs['classes'],dargs['idx']):
                elasticset.append((int(y0),int(idx)))
            x_batch, y0_batch = net.ix.fetch_data_by_elastic_set(elasticset, as_numpy=True)
        elif dargs['input'] == 'freeinput':
            if dargs['data'] == 'donut':
                from .freeinput import x_batch_donut
                x_batch = x_batch_donut
            elif dargs['data'] == 'bigdonut':
                from .freeinput import x_batch_bigdonut
                x_batch = x_batch_bigdonut
            else:
                raise NotImplementedError()

            y0_batch = [-1 for _ in range(len(x_batch))]
            elasticset = ['free input!' for _ in range(len(x_batch))]
        else:
            raise NotImplementedError()

        net.get_data_status_in_net(x_batch, y0_batch, indices=elasticset)
    elif dargs['find_what'] == 'node_info':
        NODES_OF_INTEREST = zip(dargs['layers'], dargs['idx'])
        net.get_data_index_from_net(NODES_OF_INTEREST)
    else:
        raise NotImplementedError()


######### Utils #############

def donut_numpy_loader(data_dir):
    x = np.load(data_dir, )
    # include any processing here if needed
    return x

def get_data_dir(dataname='discretedonut', folderdir='data'):
    DATA_DIR = os.path.join(folderdir, dataname)
    TEST_DATA_DIR = DATA_DIR+".test"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    return DATA_DIR, TEST_DATA_DIR

def get_ckpt_dir(dataname='discretedonut', modeldir='model.pth', ckptdir='checkpoint'):
    CKPT_DIR = os.path.join(ckptdir, dataname)
    os.makedirs(CKPT_DIR, exist_ok=True)
    MODEL_DIR = os.path.join(CKPT_DIR, modeldir)
    return CKPT_DIR, MODEL_DIR

def prep_data_and_dirs(dargs):
    print('prep_data_and_dirs...')
    if dargs['data'] == 'donut' and dargs['type']=='classification':
        ckptname='donutclassifier'
        from .data import DonutDataX
        datax = DonutDataX(120, test_sd=0.1, label_mode='discrete',
            show_fig_and_exit=dargs['show_fig_and_exit'])
        folder_to_class_mapping = {f'class{i}':i for i in [0,1,2]}

        DATA_DIR, TEST_DATA_DIR = get_data_dir(dataname='donutclasses', folderdir='data')
        CKPT_DIR, MODEL_DIR = get_ckpt_dir(dataname=ckptname, modeldir='model.pth', ckptdir='checkpoint')
    elif dargs['data'] == 'bigdonut' and dargs['type']=='classification':
        ckptname='bigdonutclassifier'
        from .data import DonutDataY       
        datax = DonutDataY(1000, test_sd=0.1, label_mode='discrete',
            show_fig_and_exit=dargs['show_fig_and_exit'])
        folder_to_class_mapping = {f'class{i}':i for i in [0,1,2,3,4]}

        DATA_DIR, TEST_DATA_DIR = get_data_dir(dataname='bigdonutclasses', folderdir='data')
        CKPT_DIR, MODEL_DIR = get_ckpt_dir(dataname=ckptname, modeldir='model.pth', ckptdir='checkpoint') 
    else:
        raise NotImplementedError('check --data and --type arguments?')
    # X,Y = datax.X, datax.Y
    # X_test, Y_test = datax.X_test, datax.Y_test

    
    print(f'  DATA_DIR:{DATA_DIR}\n  TEST_DATA_DIR:{TEST_DATA_DIR}')
    if len(os.listdir(DATA_DIR))==0:
        datax.save_data_by_class(DATA_DIR)
    if len(os.listdir(DATA_DIR+'.test'))==0:
        datax.save_testdata_by_class(TEST_DATA_DIR)
    TEST_RESULT_DIR = os.path.join(CKPT_DIR, ckptname+ '.result')  
    SHOWCASE_FOLDER_DIR = os.path.join(CKPT_DIR,'showcase')
    os.makedirs(SHOWCASE_FOLDER_DIR, exist_ok=True)

    DIRS = {
        'CKPT_DIR': CKPT_DIR,
        'MODEL_DIR':MODEL_DIR,
        'DATA_DIR':DATA_DIR, 
        'TEST_DATA_DIR': TEST_DATA_DIR,
        'TEST_RESULT_DIR': TEST_RESULT_DIR,
        'SHOWCASE_FOLDER_DIR': SHOWCASE_FOLDER_DIR,
        }
    return datax, folder_to_class_mapping, DIRS