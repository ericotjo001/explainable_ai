import numpy as np
import copy

class XAITranformation3c1c(object):
    """3c1c: 3 channel XAI output and 1 channel groundtruth
    For example, Saliency output can be (3,512,512) but the 
    groundtruth is (512,512)."""
    def __init__(self, ):
        super(XAITranformation3c1c, self).__init__()

    def check_is_proper_subset(self, subset , target):
        is_proper_subset = True
        return is_proper_subset

    def sum_pixels_over_channels(self, h, normalize='absmax_before_sum'):
        # h is (C, H, W)
        if np.all(h==0): return np.sum(h,axis=0)

        if normalize=='absmax_before_sum':
            # will ensure zero stays as zero
            h = h/np.max(np.abs(h))
            h = np.sum(h,axis=0)
            h = h/np.max(np.abs(h))
            return h
        elif normalize=='absmax_after_sum':
            # will ensure zero stays as zero
            h = np.sum(h,axis=0)
            h = h/np.max(np.abs(h))
            return h
        else:
            raise RuntimeError('Invalid mode.')

    def five_band_stratification(self, x, setting=None):
        # make sure x is normalized to [-1,1]
        if setting is None:
            setting = {
                'thresholds': [-0.7, -0.3, 0.3, 0.7],
                'targets': [-2,-1,0,1,2], 
            }

        y = setting['targets']
        yth = setting['thresholds']
        b1 = (x<=yth[0])
        b2 = (x>yth[0])*(x<=yth[1])
        b3 = (x>yth[1])*(x<=yth[2])
        b4 = (x>yth[2])*(x<=yth[3])
        b5 = (x>yth[3]) 
        assert(np.all(b1+b2+b3+b4+b5==1))

        x1 = x*0 + b1*y[0] + b2*y[1] + b3*y[2] + b4*y[3] + b5*y[4]
        return x1

class FiveBandXAIMetric(XAITranformation3c1c):
    def __init__(self):
        super(FiveBandXAIMetric, self).__init__()
        self.default_setting = {
            'xai_channel_transformation': 'sum_pixels_over_channels',
            'sum_pixels_over_channels': {
                'normalize': 'absmax_after_sum'
            },
            'clamp_setting':{
                'min_max': [-0.1,0.1],
            },
            'five_band_stratification':{
                'thresholds': np.array([-0.7, -0.3, 0.3, 0.7]),
                'targets': np.array([-2,-1,0,1,2]),                     
            },
            'five_band_stratification_for_groundtruth':{
                'thresholds': np.array([-0.7, -0.3, 0.3, 0.7]),
                'targets': np.array([-2,-1,0,1,2]),  
            },
            'soft_delta':0.02,
            'n_soft_steps':11,
        }
        self.ep = 1e-6

    def fiveband_score(self, h, h0, setting=None, verbose=0):
        # h: (C, H, W)
        # h0: (H, W)
        if setting is None: setting = self.default_setting

        if setting['xai_channel_transformation'] == 'sum_pixels_over_channels':
            tsetting = setting['sum_pixels_over_channels']
            h = self.sum_pixels_over_channels(h, normalize=tsetting['normalize'])
        elif setting['xai_channel_transformation'] == 'clamp_then_sum_pixels_over_channels':
            tsetting = setting['sum_pixels_over_channels']
            csetting = setting['clamp_setting']
            h = np.clip(h, csetting['min_max'][0],csetting['min_max'][1],)
            h = self.sum_pixels_over_channels(h, normalize=tsetting['normalize'])
        else:
            raise RuntimeError('Invalid mode.')

        if verbose>=250:
            print('='*64)
            print('fiveband_score()')
            print('h after normalization\n',h)

        fsetting = setting['five_band_stratification']
        h = self.five_band_stratification(h, setting=fsetting)
        f0setting = setting['five_band_stratification_for_groundtruth']
        h0 = self.five_band_stratification(h0, setting=f0setting)

        if verbose>=250:
            print('\nAfter strat\n h.shape:',h.shape)
            print(h)
            print('h0.shape:',h0.shape)
            print(h0)
            print()

        assert(self.check_is_proper_subset(np.unique(h),fsetting['targets']))
        assert(self.check_is_proper_subset(np.unique(h0),fsetting['targets']))
        assert(h.shape==h0.shape)

        binary_equality = (h==h0).astype(float)
        # print('binary_equality:',binary_equality)
        pixel_acc = np.sum(binary_equality.reshape(-1))/len(binary_equality.reshape(-1)) 
        TP = ((h!=0) * (h0!=0) * binary_equality).astype(float) # a little modification, because it's not strictly binary
        FP = ((h!=0) * (h0==0)).astype(float) 
        FP2 = ((h!=0) * (h0!=0) * (1-binary_equality)).astype(float)
        TN = ((h==0) * (h0==0)).astype(float)
        FN = ((h==0) * (h0!=0)).astype(float)

        if verbose>=250:
            print('binary_equality\n', binary_equality)
            print('TP\n',TP)
            print('FP\n',FP)
            print('FN\n',FN)
            print('='*64)

        TP = np.sum(TP.reshape(-1))
        FP = np.sum(FP.reshape(-1)) + np.sum(FP2.reshape(-1))
        TN = np.sum(TN.reshape(-1))
        FN = np.sum(FN.reshape(-1))
        # print('TP:%s, FP:%s, FN:%s'%(str(TP),str(FP),str(FN)))

        recall = TP/(TP+FN + self.ep)
        precision = TP/(TP+FP + self.ep)
        FPR = FP/(FP+TN+ self.ep)
        metrics = {
            'pixel_acc':pixel_acc,
            'recall': recall,
            'precision': precision,
            'FPR':FPR,
            # included in a later process
            # 'thresholds': None 
            # 'thresholds0': None # for groundtruth
        }
        return metrics

    def soft_fiveband_score(self, h, h0, setting=None):
        # h: (C, H, W)
        # h0: (H, W)

        if setting is None: setting = self.default_setting
        
        g = setting['five_band_stratification']['thresholds']
        d = setting['soft_delta']
        assert(isinstance(g, np.ndarray))
        running_setting = copy.deepcopy(setting)

        metrics_collection = []
        for i in range(setting['n_soft_steps']):
            if i>0: g = g + (-1)**(g>0.).astype(float) *d
            running_setting['five_band_stratification']['thresholds'] = g
            # print(setting['five_band_stratification']['thresholds'], 
            #   running_setting['five_band_stratification']['thresholds'])
            metrics = self.fiveband_score(h,h0,setting=running_setting)
            metrics['thresholds'] = g
            metrics['thresholds0'] = running_setting['five_band_stratification_for_groundtruth']['thresholds']
            metrics_collection.append(metrics)
        return metrics_collection

    def soft_fiveband_info(self, INFO_DIR, setting=None):
        f1 = open(INFO_DIR,"w")
        f1.write('FiveBandXAIMetric class has been deployed.\n')
        if setting is None: 
            setting = self.default_setting
            f1.write('Default setting is used.\n')
        else:
            f1.write('User-defined setting is used.\n')

        for xkey, xitem in setting.items():
            msg = '  %s : %s\n'%(str(xkey), str(xitem))
            f1.write(msg)

        f1.write('\nThis means the following [index] thresholds are used:\n')
        g = setting['five_band_stratification']['thresholds']
        d = setting['soft_delta']
        for i in range(setting['n_soft_steps']):
            if i>0: g = g + (-1)**(g>0.).astype(float) *d
            f1.write('[%s] %s\n'%(str(i),str(np.round(g,5))))
        f1.close()