from pipeline.training.alexnet_depth_scenario import * 
from pipeline.workflow_config import DATA_SIZE_ALEXNET
from pipeline.eval.evaluation_xai_implementation import evaluation_ten_classes

def eval_alexnet_depth_scenario_xai(config_data=None, singleton_scope=False, FIND_OPTIM_BRANCH_MODEL=False, realtime_update=False, ALLOW_ADHOC_NOPTIM=False):
    print('eval_alexnet_depth_scenario_xai()')
    if config_data is None: config_data = DEFAULT_CONFIG_DATA
    
    evaluation_ten_classes(initiate_or_load_model, config_data, singleton_scope=singleton_scope, reshape_size=DATA_SIZE_ALEXNET,
        FIND_OPTIM_BRANCH_MODEL=FIND_OPTIM_BRANCH_MODEL, realtime_update=realtime_update,
        ALLOW_ADHOC_NOPTIM=ALLOW_ADHOC_NOPTIM)

def eval_alexnet_depth_scenario_branch_validation_info(config_data=None):
    print('eval_alexnet_ten_classes_branch_validation_info()')
    if config_data is None: config_data = DEFAULT_CONFIG_DATA

    from pipeline.training.validation_utils import branch_validation_info
    branch_validation_info(config_data,)