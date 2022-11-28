from pipeline.training.shared_dependencies import *
from .evaluation_utils import  pytorch_singleton_to_batch_reshape
from utils.utils import FastPickleClient

XAI_WITH_STANDARD_ATTR_SETTINGS = ['Saliency',
    'IntegratedGradients', # still under testing, memory issue
    'InputXGradient', 
    'DeepLift',
    'GuidedBackprop', 
    'GuidedGradCam', 
    'Deconvolution',
]

XAI_WITH_BASELINE_ATTR_SETTINGS = ['GradientShap',
    'DeepLiftShap']

class EvaluationPackage(FastPickleClient):
    def __init__(self):
        super(EvaluationPackage, self).__init__()
        self.GALLERY = {}
        self.n_per_class = 7 # max number of samples to store in gallery per class
        self.evaluation_result = {'test_accuracy':0.}
        
    # def pickle_data(self, save_data, save_dir, tv=(0,0,None), text=None):
    # def load_pickled_data(self, pickled_dir, tv=(0,0,None), text=None):

    def add_xai_data_by_xai_method(self, identifier, data_point_package, xai_method_name):
        if not hasattr(self, xai_method_name): setattr(self, xai_method_name,{})        
        getattr(self, xai_method_name)[identifier] = data_point_package

    def add_xai_output_to_gallery_by_groundtruth(self, y0, pred_is_correct, gallery_item):
        gallery_label = '%s_%s'%(str(y0),str(bool(pred_is_correct)))
        if gallery_label not in self.GALLERY: self.GALLERY[gallery_label] = []
        if len(self.GALLERY[gallery_label]) < self.n_per_class:
            self.GALLERY[gallery_label].append(gallery_item)

def singleton_scope_oberservation(net, attrmodel, config_data, CACHE_FOLDER_DIR):
    print('singleton_scope')
    raise Exception('This function has not been updated along the versions upgrade.')
    from .eval_metrics import FiveBandXAIMetric
    xai = FiveBandXAIMetric()

    test_shards = range(1,1+config_data['test_data']['number_of_data_chunks'])
    k = random.choice(test_shards)
    test_dataset = data10.load_dataset_from_a_shard(k, CACHE_FOLDER_DIR, 
        config_data['test_data_cache_name'], include_xai_variables=True)
    j = random.choice(range(test_dataset.data_size))
    
    x, y, y_pred, y0, attr, h0 = compute_single_data_attribution(test_dataset, j, net, attrmodel, 
        this_device, config_data)
    # print('attr.shape, h0.shape', attr.shape, h0.shape) # (3, 512, 512) (512, 512)

    # # metrics = xai.fiveband_score(attr, h0, setting=None) # same as metrics_collection[0]
    metrics_collection = xai.soft_fiveband_score(attr, h0, setting=None)
    for x in metrics_collection:
        for mname, metric in x.items():
            if isinstance(metric, float): metric = round(metric, 3)
            print('  %-12s %s'%(str(mname), str(metric )))
        print()

def compute_single_data_attribution(test_dataset, j, net, attrmodel, this_device, config_data):
    xai_mode = config_data['xai_mode']
    # assume batch size is 1 for evaluation
    # if resized, both x and h in test_dataset should have been resized

    x = pytorch_singleton_to_batch_reshape(test_dataset.x[j],this_device=this_device) 
    h = test_dataset.h[j] # (512,512) without resize
    y0 = test_dataset.y[j]
    # v = test_dataset.v[j]
    y = net(x)
    y_pred = torch.argmax(y)
    # print(x.shape, y0, y_pred.item(), h.shape)

    x1 = x.clone().detach().to(device=this_device)
    x1.requires_grad = True
    y = y.clone().cpu().detach().numpy()
    y_pred = int(y_pred.clone().cpu().detach().numpy())

    if xai_mode in XAI_WITH_STANDARD_ATTR_SETTINGS:
        attr = attrmodel.attribute(x1[0:0+1], target=y_pred).squeeze(0).cpu().detach().numpy() # (3,512,512)
    elif xai_mode in XAI_WITH_BASELINE_ATTR_SETTINGS:
        baseline_dist = torch.randn((config_data['shap_n_baseline'],)+x1[0].shape).to(device=this_device) * 0.001
        attr = attrmodel.attribute(x1[0:0+1], target=y_pred, baselines=baseline_dist).squeeze(0).cpu().detach().numpy() # (3,512,512)        
    else:
        raise RuntimeError('Invalide xai_mode.')
    return test_dataset.x[j], y, y_pred, y0, attr, h


"""
>>legacy<<

from captum.attr import Saliency, IntegratedGradients ,GradientShap, Occlusion, DeepLiftShap, InputXGradient
from captum.attr import DeepLift, GuidedBackprop, GuidedGradCam, Deconvolution, FeatureAblation 

def quick_setup(attrmodel, x,y, option=None):
    x.requires_grad=True
    with torch.cuda.device(0):
        net.zero_grad()
        for i in range(len(x)): # iterate thru batch number (otherwise too memory consuming)
            if option == 'GradientShap':
                baseline_dist = torch.randn(x.shape) * 0.001
                attr = attrmodel.attribute(x[i:i+1], target = y0[i], baselines=baseline_dist)
            elif option == 'DeepLifShap':
                baseline_dist = torch.randn(x.shape) * 0.001
                attr = attrmodel.attribute(x[i:i+1], target = y0[i], baselines=baseline_dist)
            elif option == 'Occlusion':
                attr = attrmodel.attribute(x[i:i+1], target = y0[i], sliding_window_shapes=(3,40, 40),strides = (3, 20,20))
            elif option == 'FeatureAblation':
                feature_mask = torch.tensor(0.1*np.ones((3,64,64))).to(torch.int64)
                attr = attrmodel.attribute(x[i:i+1], target = y0[i], feature_mask=feature_mask)
            else:
                attr = attrmodel.attribute(x[i:i+1], target = y0[i])
            
            this_min, this_max = torch.max(attr).item(), torch.min(attr).item()
            heatmap_magnitude = np.max((np.abs(this_min), np.abs(this_max)))
            print('attr.shape:%s, max:%s, min:%s'%(str(attr.shape), str(this_max),str(this_min)))
            this_title = 'hmag:%s'%(str(np.round(heatmap_magnitude,5)))
            create_comparison(x[i],heatmap=attr[0], vmax=heatmap_magnitude,vmin=-heatmap_magnitude,
                              this_title=this_title) # grads still in (1,C,H,W)
"""