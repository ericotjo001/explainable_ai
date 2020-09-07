import time, os, sys, shutil
import pipeline.training.alexnet_ten_classes as tr
import pipeline.eval.alexnet_ten_classes_eval as ev
import pipeline.eval.alexnet_ten_classes_eval_unpack_xai_data as evis
import pipeline.workflow_utils as wfut

def workflow2():
    print('workflow2()')

    DEBUG_MODE = 1
    FULL_DATA_MODE = 0
    MACHINE_OPTION = None # 'NSCC'
    TOGGLE = {'DATA':1,
        'PART1':1, 'PART1.2':1,
        'PART2':1, 'PART2.2':1,
        'PART3':1, 'PART3.2':1, 'PART3.3':1,
    }

    # Testing with alexnet
    XAI_MODES = [ 
        'Saliency', 
        # 'IntegratedGradients', 
        'InputXGradient', 
        # 'DeepLift', 
        # 'GuidedBackprop', 
        # 'GuidedGradCam',
        # 'Deconvolution', 

        # 'GradientShap',
        # 'DeepLiftShap',
        ]

    BRANCHES_PER_MODEL = [1,2] # each number will become a new optimized model [PART 2]
    config_data = tr.DEFAULT_CONFIG_DATA
    realtime_update = True # Bool [PART 1] 
    N_CONTINUOUS_TRANING_EPOCH = 16
    N_REGULAR_EVALUATION_EPOCH = 16 # [PART 2]  it is ok to set as many as possible. There is early stopping mechanism
    ALLOW_NOPTIM = 1 # allow evaluation of NON-OPTIMALLY TUNED MODEL [PART 2.2]
    config_data['model_name'] = 'workflow2_0001'
    config_data['branch_name_label'] = None # Configure on the spot to prevent error


    if MACHINE_OPTION=='NSCC':
        # for job sent to NSCC, Singapore.
        os.chdir('MeowSupport/meim2venv/xai_basic')
        realtime_update = False # prevent unexpected printing behaviour
        from utils.utils import check_default_aux_folders
        check_default_aux_folders()
        
    elif 'mylocalmachine':
        pass # set directory to ~/xai_basic, not necessary if you run main.py from the correct dir.

    if FULL_DATA_MODE:
        config_data['data_cache_name'] = 'training_data_10c_6400'
        config_data['validation_data_cache_name'] = 'val_data_10c_1600'
        config_data['test_data_cache_name'] = 'test_data_10c_1600'
        config_data['training_data'] = {'number_of_data_shards':32, 'number_of_data_per_shard':200,}
        config_data['val_data'] = {'number_of_data_shards':8, 'number_of_data_per_shard':200,}
        config_data['test_data'] = {'number_of_data_chunks':8, 'number_of_data_per_chunk':200,}
    else:
        print('FULL_DATA_MODE: OFF') # use default, small_training_data_10c etc, see pipeline.prepare_10classes_data.py


    if DEBUG_MODE:
        N_CONTINUOUS_TRANING_EPOCH = 1
        N_REGULAR_EVALUATION_EPOCH = 1
        config_data['training_data']['number_of_data_shards'] = 2
        config_data['val_data']['number_of_data_shards'] = 2
        config_data['test_data']['number_of_data_chunks'] = 2

    MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR = tr.prepare_save_dirs(config_data)
    if TOGGLE['DATA']:
        from pipeline.workflow_utils import get_data
        get_data(config_data, CACHE_FOLDER_DIR, realtime_update); print('EXITED PART DATA\n')

    """ PART 1. Fine-tune. 
    Run through several epochs of training to adjust the parameters towards the new dataset. """
    if TOGGLE['PART1']:
        start = time.time()
        config_data['training_scheme'] = 'continuous'
        config_data['n_epoch'] = N_CONTINUOUS_TRANING_EPOCH
        tr.training_alexnet_ten_classes(config_data=config_data, realtime_update=realtime_update)
        end = time.time()
        elapsed = end - start
        print('\nPART 1.\n  time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), 
            str(round(elapsed/60.,1)), str(round(elapsed/3600.,1)))) 
        # output [model_name].model, [model_name]s.info

    if TOGGLE['PART1.2']:   
        print('\nPART1.2')     
        ev.eval_alexnet_ten_classes_plot_loss(config_data=config_data, plot_mode='savefig')

    """ PART 2. Save only when evaluation shows improvement."""
    if TOGGLE['PART2']:
        print('\nPART2')
        config_data['training_scheme'] = 'regular_evaluation'
        config_data['n_epoch'] = N_REGULAR_EVALUATION_EPOCH # it's ok to set it high, early stopping mechanism is implemented
        for branch_number in BRANCHES_PER_MODEL:
            config_data['branch_name_label'] = branch_number
            print('\nPART2 branch %s sub-part (A)'%(str(branch_number)))
            tr.training_alexnet_ten_classes(config_data=config_data, realtime_update=realtime_update)
            print('\nPART2 branch %s sub-part (B)'%(str(branch_number)))
            ev.eval_alexnet_ten_classes_branch_validation_info(config_data=config_data)
            print('\nPART2 branch %s DONE\n\n'%(str(branch_number)))
        config_data['branch_name_label'] = None # for safety
        # output [model_name].[branch no.].model, [model_name].[branch no.].info

    """ PART 2.2. Get optimized model.  The idea is to fine tune a model till the desired 
    target accuracy is achieved for evaluation. Branch models will be created here. """
    if TOGGLE['PART2.2']:
        print('\nPART2.2')

        for branch_number in BRANCHES_PER_MODEL:
            config_data['branch_name_label'] = branch_number
            BRANCH_FOLDER = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s'%(str(branch_number))
            r_eval_model_dir =  os.path.join(BRANCH_FOLDER,'%s.%s.model'%(str(config_data['model_name']),str(config_data['branch_name_label'])) )
            
            if os.path.exists(r_eval_model_dir + '.optim'):
                print('  Optimized model exists in \n%s'%(str(r_eval_model_dir + '.optim')))
            else:
                if ALLOW_NOPTIM:
                    noptim_model_dir = r_eval_model_dir + '.noptim'
                    print('Creating %s'%(str(noptim_model_dir)))
                    shutil.copyfile(r_eval_model_dir, noptim_model_dir)
                else:
                    print('DO SOME OPTIMIZATION to build .optim model. __NOT_IMPLEMENTED__')
        config_data['branch_name_label'] = None # for safety
    
    """ PART 3 """
    if TOGGLE['PART3']:
        print('\nPART3')
        msg = 'No model for XAI evaluation exists. User part2 or part2.2 to build .optim or .noptim models'

        for branch_number in BRANCHES_PER_MODEL:
            config_data['branch_name_label'] = branch_number
            xai_model_dir = wfut.check_model_for_xai_exist(branch_number, MODEL_DIR, config_data)
            print('Start PART 3 xai processing for a model branch...\n  %s'%(str(xai_model_dir)))

            start = time.time()
            for xai_mode in XAI_MODES: 
                config_data['xai_mode'] = xai_mode
                ev.eval_alexnet_ten_classes_xai(config_data=config_data, FIND_OPTIM_BRANCH_MODEL=True, realtime_update=realtime_update)
                # output is .xai data
            end = time.time()
            elapsed = end - start
            print('End PART 3 xai processing for a model branch...\n  time taken %s[s] = %s [min] = %s [hr]\n'%(
                str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1)) ))

    if TOGGLE['PART3.2']:
        print('\nPART3.2')

        for branch_number in BRANCHES_PER_MODEL:
            config_data['branch_name_label'] = branch_number
            start = time.time()
            for xai_mode in XAI_MODES: 
                config_data['xai_mode'] = xai_mode
                evis.unpack_and_pointwise_process_xai_data(config_data=config_data, BRANCH_DATA=True, do_print=False)
                # output [model_name].[BRANCH NO]_[xai_method].csv
                evis.view_gallery(config_data=config_data, BRANCH_DATA=True)
                # output folder gallery_[model_name].[BRANCH NO]_[xai_method]
            end = time.time()
            elapsed = end - start
            print('END PART 3.2 Branch %s \n  time taken %s[s] = %s [min] = %s [hr]\n'%(str(branch_number),
                str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))  ))

    if TOGGLE['PART3.3']:
        print('\nPART3.3')
        for branch_number in BRANCHES_PER_MODEL:
            config_data['branch_name_label'] = branch_number
            start = time.time()
            for xai_mode in XAI_MODES: 
                config_data['xai_mode'] = xai_mode
                evis.unpack_and_pointwise_process_xai_data_for_roc(config_data=config_data, BRANCH_DATA=True)
            end = time.time()
            elapsed = end - start
            print('END PART 3.3 Branch %s \n  time taken %s[s] = %s [min] = %s [hr]\n'%(str(branch_number),
                str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))  ))

"""
The equivalent sequence of command is:

python main.py --mode data --mode2 ten_classes
python main.py --mode training --mode2 alexnet_ten_classes
python main.py --mode evaluation --mode2 alexnet_ten_classes
python main.py --mode training --mode2 alexnet_ten_classes_branch
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 branch_validation_info
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 xai
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 unpack_and_pointwise_process
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 view_gallery
python main.py --mode evaluation --mode2 alexnet_ten_classes --mode3 roc

"""