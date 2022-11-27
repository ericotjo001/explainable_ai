from pipeline.training.vgg_ten_classes_root import *
from pipeline.training.validation_utils import RegularEvaluator

def initiate_or_load_model(MODEL_DIR, INFO_DIR, config_data, print_spacing='  ', verbose=100):
    net = AdjVGG()
    
    evaluator = RegularEvaluator()
    if verbose>=100:
        print('%sinitiate_or_load_model:\n  %s%s'%(str(print_spacing),str(MODEL_DIR),str(print_spacing)))
    if os.path.exists(MODEL_DIR) and os.path.exists(INFO_DIR):
        net.load_state_dict(torch.load(MODEL_DIR))  
        evaluator = evaluator.load_pickled_data(pickled_dir=INFO_DIR, tv=(1,verbose,VERBOSE_THRESHOLD))
    else:
        evaluator.setup(avg_loss_every_n_iter=config_data['avg_loss_every_n_iter'])

    net.to(this_device)
    return net, evaluator

from pipeline.training.validation_utils import preloading_validation_datasets
from pipeline.training.training_utils import do_save, prepare_save_dirs, optimizer_setup, prepare_branch_dirs