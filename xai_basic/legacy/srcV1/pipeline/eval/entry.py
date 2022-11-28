from utils.descriptions.eval_description import EVAL_DESCRIPTION
from utils.descriptions.main_description import NULL_DESCRIPTION

def select_evaluation_mode(console_modes):
    print('selecting evaluation mode...')

    submode = None
    if 'mode3' in console_modes:
        submode = console_modes['mode3']  

    if console_modes['mode2'] is None:
        print(EVAL_DESCRIPTION)
    elif console_modes['mode2'] == 'resnet34_ten_classes':
        import pipeline.eval.resnet34_ten_classes_eval as ev
        import pipeline.eval.resnet34_ten_classes_eval_unpack_xai_data as evis

        if submode is None:
            ev.eval_resnet34_ten_classes_plot_loss(plot_mode='savefig')
        elif submode == 'branch_validation_info':
            ev.eval_resnet34_ten_classes_branch_validation_info(config_data=None)
        elif submode == 'xai':
            ev.eval_resnet34_ten_classes_xai(realtime_update=True)
            ev.eval_resnet34_ten_classes_xai(realtime_update=True, 
                FIND_OPTIM_BRANCH_MODEL=True, ALLOW_ADHOC_NOPTIM=True)
        elif submode == 'xai_singleton_scope': #just for observing single data point
            ev.eval_resnet34_ten_classes_xai(singleton_scope=True)
        elif submode == 'unpack_and_pointwise_process':
            evis.unpack_and_pointwise_process_xai_data(config_data=None)
            evis.unpack_and_pointwise_process_xai_data(config_data=None, BRANCH_DATA=True)
        elif submode == 'view_gallery':
            evis.view_gallery(config_data=None)
            evis.view_gallery(config_data=None, BRANCH_DATA=True)
        elif submode == 'roc':
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=False)
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=True)
        else:
            print(NULL_DESCRIPTION+'\n'+EVAL_DESCRIPTION)

    elif console_modes['mode2'] == 'alexnet_ten_classes':
        import pipeline.eval.alexnet_ten_classes_eval as ev
        import pipeline.eval.alexnet_ten_classes_eval_unpack_xai_data as evis

        if submode is None:
            ev.eval_alexnet_ten_classes_plot_loss(plot_mode='savefig')
        elif submode == 'branch_validation_info':
            ev.eval_alexnet_ten_classes_branch_validation_info(config_data=None)
        elif submode == 'xai':
            ev.eval_alexnet_ten_classes_xai(realtime_update=True)
            ev.eval_alexnet_ten_classes_xai(realtime_update=True, 
                FIND_OPTIM_BRANCH_MODEL=True, ALLOW_ADHOC_NOPTIM=True)
        elif submode == 'xai_singleton_scope': #just for observing single data point
            ev.eval_alexnet_ten_classes_xai(singleton_scope=True)
        elif submode == 'unpack_and_pointwise_process':
            evis.unpack_and_pointwise_process_xai_data(config_data=None)
            evis.unpack_and_pointwise_process_xai_data(config_data=None, BRANCH_DATA=True)
        elif submode == 'view_gallery':
            evis.view_gallery(config_data=None)
            evis.view_gallery(config_data=None, BRANCH_DATA=True)
        elif submode == 'roc':
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=False)
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=True)
        else:
            print(NULL_DESCRIPTION+'\n'+EVAL_DESCRIPTION)

    elif console_modes['mode2'] == 'vgg_ten_classes':
        import pipeline.eval.vgg_ten_classes_eval as ev
        import pipeline.eval.vgg_ten_classes_eval_unpack_xai_data as evis

        if submode is None:
            ev.eval_vgg_ten_classes_plot_loss(plot_mode='savefig')
        elif submode == 'branch_validation_info':
            ev.eval_vgg_ten_classes_branch_validation_info(config_data=None)
        elif submode == 'xai':
            ev.eval_vgg_ten_classes_xai(realtime_update=True)
            ev.eval_vgg_ten_classes_xai(realtime_update=True, 
                FIND_OPTIM_BRANCH_MODEL=True, ALLOW_ADHOC_NOPTIM=True)
        elif submode == 'xai_singleton_scope': #just for observing single data point
            ev.eval_vgg_ten_classes_xai(singleton_scope=True)
        elif submode == 'unpack_and_pointwise_process':
            evis.unpack_and_pointwise_process_xai_data(config_data=None)
            evis.unpack_and_pointwise_process_xai_data(config_data=None, BRANCH_DATA=True)
        elif submode == 'view_gallery':
            evis.view_gallery(config_data=None)
            evis.view_gallery(config_data=None, BRANCH_DATA=True)
        elif submode == 'roc':
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=False)
            evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=None, BRANCH_DATA=True)
        else:
            print(NULL_DESCRIPTION+'\n'+EVAL_DESCRIPTION)
            
    else:
        print(NULL_DESCRIPTION+'\n'+EVAL_DESCRIPTION)