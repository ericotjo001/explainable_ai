from .utils import *

XAI_WITH_STANDARD_ATTR_SETTINGS = [
    'Saliency',
    'IntegratedGradients', # still under testing, memory issue
    'InputXGradient', 
    'DeepLift',
    'GuidedBackprop', 
    'GuidedGradCam', 
    'Deconvolution',
]

XAI_WITH_BASELINE_ATTR_SETTINGS = ['GradientShap',
    'DeepLiftShap']

def compute_single_data_attribution(x, y_pred, attrmodel, config_data, device=None):
    # x is a tensor with batch size=1 (1,C,H,W)
    # y_pred is long, the predicted integer label 
    # output: attr, attribution heatmap
    xai_mode = config_data['xai_mode']

    if xai_mode in XAI_WITH_STANDARD_ATTR_SETTINGS:
        x.requires_grad = True
        attr = attrmodel.attribute(x, target=y_pred).squeeze(0).cpu().detach().numpy() # (3,512,512)
    elif xai_mode in XAI_WITH_BASELINE_ATTR_SETTINGS:
        x.requires_grad = True
        baseline_dist = torch.randn((config_data['shap_n_baseline'],)+x[0].shape).to(device=device) * 0.001
        attr = attrmodel.attribute(x[0:0+1], target=y_pred, baselines=baseline_dist).squeeze(0).cpu().detach().numpy() # (3,512,512)        
    elif xai_mode == 'mab':
        h = config_data['h'] # since it's already computed during forward propagation, (1,3,H_,W_) 
        b,c,h_,w_ = x.shape
        h1 = torch.argmax(h[:,:,:h_,:w_], dim=1)
        h1 = h1.clone().detach().cpu().numpy()
        attr = (h1==1)*0.4 + (h1==2)*0.9              
    else:
        raise RuntimeError('Invalid xai_mode.')
    return attr

def get_xai_fiveband_setting(five_band_setting=None):
    from .xai_eval import XAI_SETTING_FIVE_BAND

    setting = {x:y for x,y in XAI_SETTING_FIVE_BAND.items()}
    if five_band_setting == 'mab':
        setting.update({'n_soft_steps':1})
    return setting

def compute_attribution_score(y_pred, y0, attr, h0, five_band_setting=None, display_only=False):
    # x is a tensor with batch size=1 (1,C,H,W)
    # y_pred is long, the predicted integer label 
    # attr is computed using compute_single_data_attribution, still in raw form
    #   attr will be processed inside the soft_fiveband_score via process like sum_pixels_over_channels
    #   normalization will occur there as well

    from .xai_eval import FiveBandXAIMetric
    
    xai_setting = get_xai_fiveband_setting(five_band_setting=five_band_setting)
    fbm = FiveBandXAIMetric()
    metrics_collection = fbm.soft_fiveband_score(attr, h0, setting=xai_setting)

    if display_only:
        def round_decimal(x):
            return np.round(x,3)        
        print('metrics_collection:\n %-32s %-10s %-10s %-10s %-10s '%('thresholds' , 'pixel_acc' , 'recall' , 'precision' , 'FPR'))
        for x in metrics_collection:
            print('%-32s %-10s %-10s %-10s %-10s'%(str(x["thresholds"]),
                str(round_decimal(x["pixel_acc"])),
                str(round_decimal(x["recall"])),
                str(round_decimal(x["precision"])),
                str(round_decimal(x["FPR"])))
                )        
        return None

    return metrics_collection

