from .utils import *
from .model import mabSPA
from .data import load_dataset_from_a_shard
from .objgen.random_simple_gen_implemented2 import ThreeClassesPyIOwithHeatmap 
from .objgen.random_simple_gen_implemented import TenClassesPyIOwithHeatmap
from .xai import compute_single_data_attribution,compute_attribution_score, XAI_WITH_STANDARD_ATTR_SETTINGS, XAI_WITH_BASELINE_ATTR_SETTINGS

DEBUG_TOGGLES = {
    'xai_eval_iter' : 0,
    'xai_methods' : 0,
}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_mabcam(dargs):
    import warnings
    warnings.filterwarnings('ignore','.*Setting backward hooks on.*')
    warnings.filterwarnings('ignore','.*Setting forward, backward hooks and attributes.*')

    start = time.time()

    DIRS = manage_dir(dargs)
    if dargs['n_classes'] == 3:
        net = mabSPA().to(device=device)
    elif dargs['n_classes'] == 10:
        net = mabSPA(fc_output_c=10).to(device=device)
    else:
        raise NotImplementedError()    
    checkpoint = torch.load(DIRS['MODEL_DIR'])

    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()   

    metrics_aggregate = pd.DataFrame({
        'id':[], 'xai_method':[], 'thresholds':[], 'pixel_acc':[], 'recall':[], 'precision':[], 'FPR': [],})
    metrics_aggregate.to_csv(DIRS['METRIC_AGGREGATE_DIR'], index=False)

    shardlist = os.listdir(DIRS['SHARD_TEST_FOLDER_DIR'])
    for ns,SHARD_NAME in enumerate(shardlist):
        if not SHARD_NAME[-6:] == '.shard': continue

        SHARD_DIR = os.path.join(DIRS['SHARD_TEST_FOLDER_DIR'], str(SHARD_NAME), )
        print('processing SHARD_DIR %s'%(str(SHARD_DIR)))
        testdataset = load_dataset_from_a_shard(SHARD_DIR, reshape_size=None)
        if dargs['n_classes'] == 3:
            testdataset = ThreeClassesPyIOwithHeatmap(testdataset.x,testdataset.y,testdataset.h)
        elif dargs['n_classes'] == 10:
            testdataset = TenClassesPyIOwithHeatmap(testdataset.x,testdataset.y, testdataset.h)
        else:
            raise NotImplementedError()
        testloader = DataLoader(testdataset,batch_size=1) 
        
        with torch.no_grad():
            label = SHARD_NAME
            for j,xai_method in enumerate([
                'Saliency',
                'IntegratedGradients', 
                'InputXGradient', 
                'DeepLift', 
                'GuidedBackprop', 
                'GuidedGradCam', 
                'Deconvolution',
                'GradientShap',
                'DeepLiftShap',
                'mab',
            ]):  
                print('xai_method:',xai_method)
                update_metric_aggregate_(label ,testloader, net, xai_method=xai_method, METRIC_AGGREGATE_DIR=DIRS['METRIC_AGGREGATE_DIR'])
                if DEBUG_TOGGLES['xai_methods']:
                    if j>=2: break

    end = time.time()
    elapsed = end - start
    print(' time taken %s[s] = %s [min] = %s [hr]'%(str(round(elapsed,1)), str(round(elapsed/60.,1)), str(round(elapsed/3600.,1))))
    return

def update_metric_aggregate_(label, testloader, net, xai_method, METRIC_AGGREGATE_DIR):
    metrics_aggregate = pd.read_csv(METRIC_AGGREGATE_DIR, float_precision='high', index_col=False)

    for i,(x,y0, h0) in enumerate(testloader):
        x = x.to(torch.float).to(device=device)
        y0 = y0.to(torch.long).to(device=device)
        h0 = h0[0].clone().cpu().numpy()
        h0 = (h0==1)*0.4 + (h0==2)*0.9

        five_band_setting = None
        if xai_method == 'mab':
            net.forward_mode = 'label+heatmap' # IMPT!
            y, h = net(x)
            config_data = {'h':h, 'xai_mode': xai_method}
            y_pred = torch.argmax(y,dim=1)            
            y_pred = y_pred[0].detach().cpu().numpy()
            five_band_setting = 'mab'
            attr = compute_single_data_attribution(x, y_pred, None, config_data)

        elif xai_method in XAI_WITH_STANDARD_ATTR_SETTINGS:
            config_data = {'xai_mode': xai_method}
            net.forward_mode = None
            
            y = net(x)
            y_pred = torch.argmax(y,dim=1)            
            attrmodel = select_attr_model(xai_method, net)
            attr = compute_single_data_attribution(x, y_pred, attrmodel, config_data=config_data)

        elif xai_method in XAI_WITH_BASELINE_ATTR_SETTINGS:
            config_data = {'shap_n_baseline':4, 'xai_mode': xai_method} 
            net.forward_mode = None

            y = net(x)
            y_pred = torch.argmax(y,dim=1)            
            attrmodel = select_attr_model(xai_method, net)
            attr = compute_single_data_attribution(x, y_pred, attrmodel, config_data, device=device)
        else:
            raise NotImplementedError()

        #####################################################################
        # Result collection!
        # note: this collection (list) applies for a single sample
        #####################################################################
        metrics_collection = compute_attribution_score(y_pred, int(y0[0].item()), attr, h0, five_band_setting=five_band_setting, display_only=False)
        for metrics in metrics_collection:
            # print(metrics) # {'pixel_acc': 0.80511474609375, 'recall': 0.9114715267654219, 'precision': 0.7332151268760893, 'FPR': 0.28688921140934887, 'thresholds': array([-0.7, -0.3,  0.3,  0.7]), 'thresholds0': array([-0.7, -0.3,  0.3,  0.7])}

            mc = { mname : [metric_value] for mname, metric_value in metrics.items()}
            mc.update({'id': ['%s_%s'%(str(label),str(i))] ,'xai_method': [xai_method]})            
            metrics_aggregate = pd.concat([metrics_aggregate, pd.DataFrame(mc)])

        if DEBUG_TOGGLES['xai_eval_iter']:
            if i>=3: break

    metrics_aggregate.to_csv(METRIC_AGGREGATE_DIR, index=False)


def select_attr_model(xai_mode, net):
    net.forward_mode = None
    if xai_mode == 'Saliency': 
        from captum.attr import Saliency
        attrmodel = Saliency(net)
    elif xai_mode == 'IntegratedGradients': 
        from captum.attr import IntegratedGradients
        attrmodel = IntegratedGradients(net)
    elif xai_mode == 'InputXGradient':
        from captum.attr import InputXGradient
        attrmodel = InputXGradient(net)
    elif xai_mode == 'DeepLift': 
        from captum.attr import DeepLift
        attrmodel = DeepLift(net)
    elif xai_mode == 'GuidedBackprop':
        from captum.attr import GuidedBackprop
        attrmodel = GuidedBackprop(net)
    elif xai_mode == 'GuidedGradCam':
        from captum.attr import GuidedGradCam
        attrmodel = GuidedGradCam(net, net.select_first_layer()) # first layer
    elif xai_mode == 'Deconvolution':
        from captum.attr import Deconvolution
        attrmodel = Deconvolution(net)
    elif xai_mode == 'GradientShap':
        from captum.attr import GradientShap
        attrmodel = GradientShap(net)
    elif xai_mode == 'DeepLiftShap':
        from captum.attr import DeepLiftShap
        attrmodel = DeepLiftShap(net)
    else: 
        raise RuntimeError('No valid attribution selected.')
    return attrmodel
