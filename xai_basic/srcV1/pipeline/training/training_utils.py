from pipeline.training.shared_dependencies import *

def do_save(evaluator, net, MODEL_DIR, INFO_DIR):
    # print('Saving model at (lifetime) iter:%s'%(str(evaluator.iter),))
    torch.save(net.state_dict(), MODEL_DIR)  
    evaluator.pickle_data(evaluator, INFO_DIR, tv=(1,0,VERBOSE_THRESHOLD))

def prepare_branch_dirs(MODEL_DIR, config_data):
    BRANCH_FOLDER_DIR = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s'%(str(config_data['branch_name_label']))
    if not os.path.exists(BRANCH_FOLDER_DIR):
        os.mkdir(BRANCH_FOLDER_DIR)
    BRANCH_MODEL_DIR = BRANCH_FOLDER_DIR + '/%s.%s.model'%(str(config_data['model_name']),str(config_data['branch_name_label'])) 
    BRANCH_INFO_DIR = BRANCH_FOLDER_DIR + '/%s.%s.info'%(str(config_data['model_name']),str(config_data['branch_name_label']))
    return BRANCH_MODEL_DIR, BRANCH_INFO_DIR

def prepare_save_dirs(config_data):
    SAVE_NAME = config_data['model_name']
    MODEL_FOLDER = os.path.join('checkpoint',SAVE_NAME) 
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)
    MODEL_DIR = os.path.join(MODEL_FOLDER, SAVE_NAME + '.model')
    INFO_DIR = os.path.join(MODEL_FOLDER, SAVE_NAME + '.info')

    CACHE_FOLDER_DIR = os.path.join('checkpoint', 'cache') 
    return MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR

def optimizer_setup(setting, net, mode='adam'):
    if mode=='adam':
        lr = setting['lr']
        weight_decay = setting['weight_decay']
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay=weight_decay)
    else:
        raise RuntimeError('Invalid mode for optimizer.')
    return optimizer