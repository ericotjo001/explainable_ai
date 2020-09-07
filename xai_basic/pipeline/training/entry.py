from utils.descriptions.training_description import TRAINING_DESCRIPTION
from utils.descriptions.main_description import NULL_DESCRIPTION

def select_training_mode(console_modes):
    print('selecting training mode...')

    if console_modes['mode2'] is None:
        print(TRAINING_DESCRIPTION)

    elif console_modes['mode2'] == 'resnet34_ten_classes':
        from pipeline.training.resnet34_ten_classes import training_resnet34_ten_classes
        training_resnet34_ten_classes()
    elif console_modes['mode2'] == 'resnet34_ten_classes_branch':
        from pipeline.training.resnet34_ten_classes import training_resnet34_ten_classes
        from pipeline.training.resnet34_ten_classes_root import DEFAULT_CONFIG_DATA
        config_data = DEFAULT_CONFIG_DATA
        config_data['training_scheme'] = 'regular_evaluation'
        training_resnet34_ten_classes(config_data=config_data)

    elif console_modes['mode2'] == 'alexnet_ten_classes': 
        from pipeline.training.alexnet_ten_classes import training_alexnet_ten_classes
        training_alexnet_ten_classes()
    elif console_modes['mode2'] == 'alexnet_ten_classes_branch': 
        from pipeline.training.alexnet_ten_classes import training_alexnet_ten_classes
        from pipeline.training.alexnet_ten_classes_root import DEFAULT_CONFIG_DATA
        config_data = DEFAULT_CONFIG_DATA
        config_data['training_scheme'] = 'regular_evaluation'
        training_alexnet_ten_classes(config_data=config_data)   

    elif console_modes['mode2'] == 'vgg_ten_classes': 
        from pipeline.training.vgg_ten_classes import training_vgg_ten_classes
        training_vgg_ten_classes()
    elif console_modes['mode2'] == 'vgg_ten_classes_branch': 
        from pipeline.training.vgg_ten_classes import training_vgg_ten_classes
        from pipeline.training.vgg_ten_classes_root import DEFAULT_CONFIG_DATA
        config_data = DEFAULT_CONFIG_DATA
        config_data['training_scheme'] = 'regular_evaluation'
        training_vgg_ten_classes(config_data=config_data)   
        
    elif console_modes['mode2'] == 'alex_depth_scenario':
        from pipeline.training.alexnet_depth_scenario import training_alexnet_ten_classes_depth_scenario
        training_alexnet_ten_classes_depth_scenario()
        

    else:
        print(NULL_DESCRIPTION+'\n'+TRAINING_DESCRIPTION)
