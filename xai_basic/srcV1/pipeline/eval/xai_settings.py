import numpy as np
XAI_SETTING_0001 = {
    'xai_channel_transformation': 'sum_pixels_over_channels',
    'sum_pixels_over_channels': {
        'normalize': 'absmax_after_sum'
    },
    'five_band_stratification':{
        'thresholds': np.array([-0.5, -0.3, 0.3, 0.5]),
        'targets': np.array([-2,-1,0,1,2]),                     
    },
    'five_band_stratification_for_groundtruth':{
        'thresholds': np.array([-0.7, -0.3, 0.3, 0.7]),
        'targets': np.array([-2,-1,0,1,2]),  
    },
    'soft_delta':0.005,
    'n_soft_steps':56,
}

XAI_SETTING_0002 = {
    'xai_channel_transformation': 'clamp_then_sum_pixels_over_channels',
    'sum_pixels_over_channels': {
        'normalize': 'absmax_after_sum'
    },
    'clamp_setting':{
        'min_max': [-0.1,0.1],
    },
    'five_band_stratification':{
        'thresholds': np.array([-0.9, -0.5, 0.5, 0.9]),
        'targets': np.array([-2,-1,0,1,2]),                     
    },
    'five_band_stratification_for_groundtruth':{
        'thresholds': np.array([-0.7, -0.3, 0.3, 0.7]),
        'targets': np.array([-2,-1,0,1,2]),  
    },
    'soft_delta':0.01,
    'n_soft_steps':41,
}
