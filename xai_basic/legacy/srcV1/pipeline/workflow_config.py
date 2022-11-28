DEFAULT_CONFIG_DATA = {
    'model_name': 'SOME_MODEL_NAME', # 'resnet34adj_0001',
    'branch_name_label': 1,

    # training
    'training_scheme': 'continuous', # 1. continuous 2. regular_evaluation
    'training_data_scheme': 'shard_by_shard', # 1. shard_by_shard 2. renew (NOT IMPLEMENTED)
    'n_epoch' : 2,
    'batch_size' : 4,
    'learning': {
        'adam':{'lr':1e-3, 'weight_decay':1e-5,},
    },

    # validation
    'avg_loss_every_n_iter': 4,
    'eval_every_n_iter':4, # Evaluation every n iterations, for part 2 i.e. training in regular evaluation mode
    'validation_data_scheme': 'shard_by_shard', # 1. shard_by_shard 2. renew (NOT IMPLEMENTED)

    # eval, xai, test
    'xai_mode':'Saliency',
    'shap_n_baseline': 4, # for Shap based method (setting large value may cause memory error)
    'gallery_resize': (224,224,3), # to save memory, make it smaller than the original size    
}

DATA_SIZE_ALEXNET = (3,224,224) # specify in (C,H,W)
DATA_SIZE_VGG = (3,224,224)