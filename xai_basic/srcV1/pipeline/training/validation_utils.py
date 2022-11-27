from pipeline.training.shared_dependencies import *
from pipeline.eval.evaluation_utils import EvalMidTraining

DEBUG_MODE = 0
if DEBUG_MODE: #toggle the following freely
    DEBUG_TRY_S2_EARLY_STOPPING = 0
else: # do not edit the following
    DEBUG_TRY_S2_EARLY_STOPPING = 0 # bool

class RegularEvaluator(EvalMidTraining):
    def __init__(self):
        super(RegularEvaluator, self).__init__()

        self.best_acc = 0. # on validation dataset shard by shard
        self.total_validation_acc = 0.
        self.last_updated_iter = 0
        self.early_stopping = False
        self.early_stopping_counter = 0
        self.early_stopping_limit = 24
        self.early_stopping_msg = None
        self.refresh_fraction = 0.4 # recommend below 0.5
        self.target_acc= 0.96

        self.reshape_size = None # tuple (C,H,W)

        # inherited
        # self.iter = 0
        self.DEBUG_ITER = 0

    # inherited from FastPickleClient
    # def pickle_data(self, save_data, save_dir, tv=(0,0,None)):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None)):

    # inherited from EvalMidTraining
    # def setup(self, avg_loss_every_n_iter=24):
        # the following will be initiated by setup()
        # self.avg_loss_every_n_iter = avg_loss_every_n_iter 
        # self.running_loss = 0.
        # self.loss_iter, self.losses = [],[] # x,y -axes to plot loss
    # def compute_running_average_loss(self, loss):

    # Used for training validation (not testing)

    def do_DEBUG_TRY_S2_EARLY_STOPPING(self):
        self.DEBUG_ITER += 1
        if self.DEBUG_ITER>=3: 
            self.early_stopping_counter = self.early_stopping_limit+1
            print('\nTRIGGER S2 EARLY STOPPING BY DEBUG MODE.\n')

    def evaluate_and_save(self, net, BRANCH_MODEL_DIR, CACHE_FOLDER_DIR, config_data, this_device=None):
        k = np.random.randint(config_data['val_data']['number_of_data_shards'])
        val_dataset = self.preloaded_val_dataset[k]
        valloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
        acc = self.evaluate_single_shard(valloader, net)

        if not os.path.exists(BRANCH_MODEL_DIR):
            torch.save(net.state_dict(), BRANCH_MODEL_DIR) 

        if DEBUG_TRY_S2_EARLY_STOPPING: self.do_DEBUG_TRY_S2_EARLY_STOPPING()

        if acc > self.best_acc:
            self.best_acc = acc
            torch.save(net.state_dict(), BRANCH_MODEL_DIR) 
            self.last_updated_iter = self.iter 
            self.early_stopping_counter = int(self.refresh_fraction*self.early_stopping_counter)
            
            ###########################################
            # BEST MODEL ACHIEVED TARGET ACCURACY
            ###########################################
            if acc >= self.target_acc:
                self.early_stopping_msg = '\n[S1] target acc %s achieved.\n  Backup is saved as .optim.'%(str(self.target_acc))
                print(self.early_stopping_msg)
                self.early_stopping = True
                self.evaluate_all_shards(net, config_data)
                torch.save(net.state_dict(), BRANCH_MODEL_DIR + '.optim') 
        else:   
            self.early_stopping_counter+=1
            if self.early_stopping_counter>= self.early_stopping_limit:
                self.early_stopping_msg = '\n[S2] Early stopping limit reached without achieving target.\n'
                print(self.early_stopping_msg)
                self.early_stopping = True
        # print('\n - evaluation end - ')

    # Used for training validation (not testing)
    def evaluate_all_shards(self, net, config_data):
        net.eval()
        n_correct = 0
        total_n = 0
        for k in range(config_data['val_data']['number_of_data_shards']):
            val_dataset = self.preloaded_val_dataset[k]
            valloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=1)
            for i, data in enumerate(valloader, 0):
                x, y0 = data
                x = x.to(torch.float).to(this_device)
                y0 = y0.to(this_device)
                y = net(x)

                y_pred = torch.argmax(y)
                pred_is_correct = (y_pred.item() == y0.item())
                total_n += 1 
                if pred_is_correct:
                    n_correct += 1
        acc = n_correct/total_n
        self.total_validation_acc = acc
        print('')
        print('total validation acc: %s/%s = %s'%(str(n_correct),str(total_n),str(acc)))

    # Used for training validation (not testing)
    def evaluate_single_shard(self, valloader, net):
        net.eval()
        n_correct = 0
        for i, data in enumerate(valloader, 0):
            x, y0 = data
            x = x.to(torch.float).to(this_device)
            y0 = y0.to(this_device)
            y = net(x)

            y_pred = torch.argmax(y)
            pred_is_correct = (y_pred.item() == y0.item())
            if pred_is_correct:
                n_correct += 1
        acc = n_correct/len(valloader)
        return acc


import pipeline.data.prepare_10classes_data as data10
def preloading_validation_datasets(CACHE_FOLDER_DIR, config_data, reshape_size=None):
    print('  preloading validation dataset...')
    preloaded_val_dataset = []
    val_shards = range(1,1+config_data['val_data']['number_of_data_shards'])
    for k in val_shards:
        val_dataset = data10.load_dataset_from_a_shard(k, CACHE_FOLDER_DIR, config_data['validation_data_cache_name'],
            reshape_size=reshape_size)
        preloaded_val_dataset.append(val_dataset)
    return preloaded_val_dataset


def branch_validation_info(config_data,):
    from pipeline.training.training_utils import prepare_save_dirs
    MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR = prepare_save_dirs(config_data)

    BRANCH_FOLDER_DIR = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s'%(str(config_data['branch_name_label']))
    
    BRANCH_INFO_DIR = BRANCH_FOLDER_DIR + '/%s.%s.info'%(str(config_data['model_name']), str(config_data['branch_name_label']))
    BRANCH_INFO_TXT = BRANCH_FOLDER_DIR + '/%s.%s.txt'%( str(config_data['model_name']),str(config_data['branch_name_label']))

    evaluator = RegularEvaluator()
    evaluator = evaluator.load_pickled_data(pickled_dir=BRANCH_INFO_DIR, tv=(1,0,None))
    
    ATTRIBUTE_OF_INTEREST = ['total_validation_acc', 'target_acc', 'best_acc', 'early_stopping','early_stopping_msg']

    f1 = open(BRANCH_INFO_TXT,"w")
    this_msg = 'performance summary:\n'
    print(this_msg, end='')
    f1.write(this_msg)
    for att in ATTRIBUTE_OF_INTEREST:
        this_val = getattr(evaluator,att) 
        this_msg = '  %-24s: %s\n'%(str(att), str(this_val))
        print(this_msg, end='')
        f1.write(this_msg)
        if att == 'best_acc':
            f1.write('  >> (best acc when validated on a single shard only. Consider instead total_validation_acc.)\n')
    f1.close()