from pipeline.training.resnet34_ten_classes import *
from pipeline.eval.evaluation_xai_implementation import evaluation_ten_classes

def eval_resnet34_ten_classes_xai(config_data=None, singleton_scope=False, FIND_OPTIM_BRANCH_MODEL=False, realtime_update=False, ALLOW_ADHOC_NOPTIM=False):
    # print('eval_resnet34_ten_classes_xai()')
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    evaluation_ten_classes(initiate_or_load_model, config_data, singleton_scope=singleton_scope, reshape_size=None,
        FIND_OPTIM_BRANCH_MODEL=FIND_OPTIM_BRANCH_MODEL, realtime_update=realtime_update,
        ALLOW_ADHOC_NOPTIM=ALLOW_ADHOC_NOPTIM)

def eval_resnet34_ten_classes_plot_loss(config_data=None, **kwargs):
    print('eval_resnet34_ten_classes_plot_loss()')
    if config_data is None: config_data = DEFAULT_CONFIG_DATA
    
    from pipeline.eval.evaluation_utils import eval_plot_loss
    eval_plot_loss(initiate_or_load_model, config_data, **kwargs)    

def eval_resnet34_ten_classes_branch_validation_info(config_data=None):
    print('eval_resnet34_ten_classes_branch_validation_info()')
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    from pipeline.training.validation_utils import branch_validation_info
    branch_validation_info(config_data,)
 