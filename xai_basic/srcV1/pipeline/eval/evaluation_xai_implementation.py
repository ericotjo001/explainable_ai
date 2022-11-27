from pipeline.training.shared_dependencies import *
from pipeline.eval.evaluation_xai_utils import EvaluationPackage,\
    singleton_scope_oberservation, compute_single_data_attribution
import pipeline.data.prepare_10classes_data as data10
from skimage.transform import resize
from captum.attr import Saliency, IntegratedGradients,InputXGradient, DeepLift, \
    GuidedBackprop, GuidedGradCam, Deconvolution, GradientShap, DeepLiftShap


def evaluation_ten_classes(initiate_or_load_model, config_data, singleton_scope=False, reshape_size=None,
    FIND_OPTIM_BRANCH_MODEL=False, realtime_update=False, ALLOW_ADHOC_NOPTIM=False):
    from pipeline.training.training_utils import prepare_save_dirs
    xai_mode = config_data['xai_mode']
    MODEL_DIR, INFO_DIR, CACHE_FOLDER_DIR = prepare_save_dirs(config_data)

    ############################
    VERBOSE = 0
    ############################
    
    if not FIND_OPTIM_BRANCH_MODEL: 
        print('Using the following the model from (only) continuous training for xai evaluation [%s]'%(str(xai_mode)))
        net, evaluator = initiate_or_load_model(MODEL_DIR, INFO_DIR, config_data, verbose=VERBOSE)
    else: 
        BRANCH_FOLDER_DIR = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s'%(str(config_data['branch_name_label']))    
        BRANCH_MODEL_DIR = os.path.join(BRANCH_FOLDER_DIR, '%s.%s.model'%(str(config_data['model_name']),str(config_data['branch_name_label'])))
        # BRANCH_MODEL_DIR = MODEL_DIR[:MODEL_DIR.find('.model')] + '.%s.model'%(str(config_data['branch_name_label'])) 

        if ALLOW_ADHOC_NOPTIM: # this is intended only for debug runs
            print('<< [EXY1] ALLOWING ADHOC NOPTIM >>') 
            import shutil
            shutil.copyfile(BRANCH_MODEL_DIR, BRANCH_MODEL_DIR + '.noptim')

        if os.path.exists(BRANCH_MODEL_DIR+'.optim'):
            BRANCH_MODEL_DIR = BRANCH_MODEL_DIR+'.optim'
            print('  Using the OPTIMIZED branch model for [%s] xai evaluation: %s'%(str(xai_mode),str(BRANCH_MODEL_DIR)))
        elif os.path.exists(BRANCH_MODEL_DIR+'.noptim'):
            BRANCH_MODEL_DIR = BRANCH_MODEL_DIR+'.noptim'
            print('  Using the partially optimized branch model for [%s] xai evaluation: %s'%(str(xai_mode),str(BRANCH_MODEL_DIR)))
        else:
            raise RuntimeError('Attempting to find .optim or .noptim model, but not found.')       
        if VERBOSE>=250:
            print('  """You may see a warning by pytorch for ReLu backward hook. It has been fixed externally, so you can ignore it."""')
        net, evaluator = initiate_or_load_model(BRANCH_MODEL_DIR, INFO_DIR, config_data, verbose=VERBOSE)

    
    if xai_mode == 'Saliency': attrmodel = Saliency(net)
    elif xai_mode == 'IntegratedGradients': attrmodel = IntegratedGradients(net)
    elif xai_mode == 'InputXGradient': attrmodel = InputXGradient(net)
    elif xai_mode == 'DeepLift': attrmodel = DeepLift(net)
    elif xai_mode == 'GuidedBackprop': attrmodel = GuidedBackprop(net)
    elif xai_mode == 'GuidedGradCam': attrmodel = GuidedGradCam(net, net.select_first_layer()) # first layer
    elif xai_mode == 'Deconvolution': attrmodel = Deconvolution(net)
    elif xai_mode == 'GradientShap': attrmodel = GradientShap(net)
    elif xai_mode == 'DeepLiftShap': attrmodel = DeepLiftShap(net)
    else: raise RuntimeError('No valid attribution selected.')

    if singleton_scope: # just to observe a single datapoint, mostly for debugging
        singleton_scope_oberservation(net, attrmodel, config_data, CACHE_FOLDER_DIR)
    else:
        aggregate_evaluation(net, attrmodel, config_data, CACHE_FOLDER_DIR, reshape_size=reshape_size, realtime_update=realtime_update, EVALUATE_BRANCH=FIND_OPTIM_BRANCH_MODEL)

from .eval_metrics import FiveBandXAIMetric
def setup_xai_using_xai_setting(config_data, this_xai_setting, EVALUATE_BRANCH, file_ext='xai'):
    xai = FiveBandXAIMetric()
    xai_mode = config_data['xai_mode']
    model_name = config_data['model_name']
    if not EVALUATE_BRANCH:
        XAI_FOLDER = os.path.join('checkpoint', model_name ,'XAI_results')
        if not os.path.exists(XAI_FOLDER):
            os.mkdir(XAI_FOLDER)
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s_%s.%s'%(str(model_name), str(xai_mode), str(file_ext)))
    else:
        XAI_FOLDER = os.path.join('checkpoint', model_name , '%s.%s'%(str(model_name),str(config_data['branch_name_label'])), 'XAI_results')
        if not os.path.exists(XAI_FOLDER):
            os.mkdir(XAI_FOLDER)
        XAI_SAVEDIR = os.path.join(XAI_FOLDER, '%s.%s_%s.%s'%(str(model_name), str(config_data['branch_name_label']), str(xai_mode), str(file_ext)))   
    XAI_INFO_DIR = XAI_SAVEDIR + '.txt'
    xai.soft_fiveband_info(INFO_DIR=XAI_INFO_DIR, setting=this_xai_setting)
    return xai, XAI_SAVEDIR, this_xai_setting

def setup_xai_using_setting_0001(config_data, EVALUATE_BRANCH):
    from pipeline.eval.xai_settings import XAI_SETTING_0001 
    return setup_xai_using_xai_setting(config_data, XAI_SETTING_0001, EVALUATE_BRANCH, file_ext='xai')

def setup_xai_using_setting_0002(config_data, EVALUATE_BRANCH):
    from pipeline.eval.xai_settings import XAI_SETTING_0002
    return setup_xai_using_xai_setting(config_data, XAI_SETTING_0002, EVALUATE_BRANCH, file_ext='clamp.xai')

def aggregate_evaluation(net, attrmodel, config_data, CACHE_FOLDER_DIR, reshape_size=None, 
    realtime_update=True, EVALUATE_BRANCH=False): 

    VERBOSE = 0

    evalpack = EvaluationPackage()
    evalpack2 = EvaluationPackage()
    net.eval()
    xai, XAI_SAVEDIR, xai_setting0001 = setup_xai_using_setting_0001(config_data, EVALUATE_BRANCH)
    xai2, XAI_SAVEDIR2, xai_setting0002 = setup_xai_using_setting_0002(config_data, EVALUATE_BRANCH)
 
    total_n, n_correct = 0, 0
    n_shard = config_data['test_data']['number_of_data_chunks']
    test_shards = range(1,1+n_shard)
    for k in test_shards:
        test_dataset = data10.load_dataset_from_a_shard(k, CACHE_FOLDER_DIR, config_data['test_data_cache_name'], 
            include_xai_variables=True, reshape_size=reshape_size)
        # print(np.array(test_dataset.x).shape, np.array(test_dataset.h).shape)
        for j in range(test_dataset.data_size):
            identifier = '%s_chunk%s_queue%s'%(str(config_data['test_data_cache_name']),str(k),str(j))

            ################################################
            # perform XAI evaluation pytorch API
            ################################################
            x, y, y_pred, y0, attr, h0 = compute_single_data_attribution(test_dataset, j, net, attrmodel, this_device, config_data)            
            pred_is_correct = int(y_pred)==int(y0)

            ################################################
            # perform XAI evaluation via fiveband metrics 
            # First on the dafault xai_setting is used. Then, the process is 
            # repeated for the xai_setting that includes clamping process.
            ################################################
            labels = {'y':y, 'y_pred': y_pred, 'y0':y0, 'pred_is_correct': pred_is_correct,}
            compute_scores_and_gallery_items(identifier, x, attr, h0, labels, 
                xai_setting=xai_setting0001, MetricObject=xai, evalpack=evalpack, config_data=config_data)
            compute_scores_and_gallery_items(identifier, x, attr, h0, labels, 
                xai_setting=xai_setting0002, MetricObject=xai, evalpack=evalpack2, config_data=config_data)

            ################################################
            # perform accuracy testing on evaluation dataset as well
            ################################################
            total_n+=1
            if pred_is_correct: n_correct+=1
            update_text = 'shard: %2s/%2s, j:%4s/%4s, acc:%s/%s'%(str(k),str(n_shard),
                str(j+1),str(test_dataset.data_size),str(n_correct),str(total_n))
            if realtime_update: print('    %-96s'%(str(update_text)), end='\r')

    acc = n_correct/total_n
    print('\n    pred acc. %s/%s=%s [XAIPred Marker]'%(str(n_correct), str(total_n), str(round(acc,2))))
    evalpack.pickle_data(evalpack, XAI_SAVEDIR, tv=(1,VERBOSE,100), text=None)
    evalpack.pickle_data(evalpack2, XAI_SAVEDIR2, tv=(1,VERBOSE,100), text=None)
    # XAI_SAVEDIR example: ../XAI_results/workflow1_0001.1_DeepLift.csv

def compute_scores_and_gallery_items(identifier, x, attr, h0, labels, 
    xai_setting, MetricObject, evalpack, config_data):
    s = config_data['gallery_resize']
    y0, y_pred, pred_is_correct = labels['y0'], labels['y_pred'], labels['pred_is_correct']

    metrics_collection = MetricObject.soft_fiveband_score(attr, h0, setting=xai_setting)
    data_point_package = { 'labels': labels, 'metrics_collection': metrics_collection}           
    x, attr, h0 = gallery_resize(s, x, attr, h0) # to save gallery memory
    evalpack.add_xai_data_by_xai_method(identifier, data_point_package, xai_method_name=config_data['xai_mode'])
    gallery_item = {'identifier': identifier, 'metrics_collection': metrics_collection,
        'x':x,'y0':y0,'attr':attr,'h0':h0,'y_pred':y_pred,}
    evalpack.add_xai_output_to_gallery_by_groundtruth(y0, pred_is_correct, gallery_item)    

def gallery_resize(s, x, attr, h0):
    if s is not None: 
        x = x.transpose(1,2,0)
        x = resize(x, s).transpose(2,0,1)
        attr = attr.transpose(1,2,0)
        attr = resize(attr, s).transpose(2,0,1)
        h0 = resize(h0,(s[0],s[1]))
    #     print('x.shape:',x.shape,) # (C,H,W)
    #     print('attr.shape:',attr.shape) # (C,H,W)
    #     print('h0.shape',h0.shape) # (H,W)
    # raise Exception('DEBUGGING')
    return x, attr, h0