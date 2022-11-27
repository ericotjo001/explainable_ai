import numpy as np
from .cells import CCell
from .rcells import RCell
from .ccells import CCellM, CCellP
from .tcells import CCellT,CCellTs

unique_heatmap_error_msg = 'Heatmap features contain values that are not allowed'
def unique_entries_check(heatmap, unique_set_allowed):
    unique_entries = set(list(heatmap.reshape(-1)))
    for a in unique_entries:
        if a not in unique_set_allowed:
            print('allowed:', unique_set_allowed)
            print('but contains:', a)
            raise RuntimeError(unique_heatmap_error_msg)   

class CCellX(CCell):
    """
    Note that explanation is implemented IN CONTEXT, i.e.
    in different classification problems, explanations will be different.
    Hence, this is just an example.
    """
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(CCellX, self).__init__()

        if fast_init:
            self.change_array_shape(array_shape)            
            self.setup_explanation_attributes(explanation_setting)
            self.setup_basic_ball(color_setting, shape_setting)

    def make_explanation(self):
        # print('make_explanation()...')
        explanation = None
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        # print('  eborder max:%s, min:%s'%(str(np.max(explanation_border)),str(np.min(explanation_border))))
        explanation_border = (explanation_border>self.discriminative_feature_threshold).astype(np.float) 
        explanation_body =  1/3* np.sum(self.parts['inner_ball']**2,axis=2)**0.5
        # print('  ebody max:%s, min:%s'%(str(np.max(explanation_body)),str(np.min(explanation_body))))
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float) 
        # print('  explanation_border.shape:%s'%(str(explanation_border.shape)))
        # print('  explanation_body.shape  :%s'%(str(explanation_body.shape)))

        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap
        heatmap = explanation_border * self.discriminative_feature_value + explanation_body* self.localization_value 
        
        unique_set_allowed = {0., self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap

class CCellMX(CCellM):
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(CCellMX, self).__init__()

        if fast_init:
            self.change_array_shape(array_shape)
            self.setup_explanation_attributes(explanation_setting)
            self.setup_ccell(color_setting, shape_setting) 
            
    def make_explanation(self):
        # print('make_explanation()...')
        explanation = None
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        explanation_border = (explanation_border>self.localization_threshold).astype(np.float)  
        explanation_body =  1/3* np.sum(self.parts['inner_ball']**2,axis=2)**0.5
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float) 
        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap 
        explanation = explanation_border + explanation_body

        skeletal_feature = 1/3* np.sum(self.parts['bar']**2,axis=2)**0.5
        skeletal_feature = (skeletal_feature>self.discriminative_feature_threshold).astype(np.float) 
        heatmap = explanation * (1- skeletal_feature ) * self.localization_value + skeletal_feature * self.discriminative_feature_value

        unique_set_allowed = {0.,self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap

class CCellPX(CCellP):
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(CCellPX, self).__init__()     

        if fast_init:
            self.change_array_shape(array_shape)
            self.setup_explanation_attributes(explanation_setting)
            self.setup_ccell(color_setting, shape_setting) 

    def make_explanation(self):
        # print('make_explanation()...')
        explanation = None
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        explanation_border = (explanation_border>self.localization_threshold).astype(np.float)  
        explanation_body =  1/3* np.sum(self.parts['inner_ball']**2,axis=2)**0.5
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float)
        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap 
        explanation = explanation_border + explanation_body

        skeletal_feature = 1/3* np.sum(self.parts['bar']**2,axis=2)**0.5
        skeletal_feature = (skeletal_feature>self.discriminative_feature_threshold).astype(np.float) 
        heatmap = explanation * (1- skeletal_feature ) * self.localization_value + skeletal_feature * self.discriminative_feature_value

        unique_set_allowed = {0.,self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap      

class RCellX(RCell):
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(RCellX, self).__init__()

        if fast_init:
            self.change_array_shape(array_shape)
            self.setup_explanation_attributes(explanation_setting)
            self.setup_rectangle(color_setting, shape_setting)   

    def make_explanation(self):
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        explanation_border = (explanation_border>self.discriminative_feature_threshold ).astype(np.float)
        explanation_body =  1/3* np.sum(self.parts['inner_rect']**2,axis=2)**0.5
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float)  
        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap 
        heatmap = explanation_border * self.discriminative_feature_value + explanation_body * self.localization_value

        unique_set_allowed = {0.,self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap

class CCellTX(CCellT):
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(CCellTX, self).__init__()

        if fast_init:
            self.change_array_shape(array_shape)
            self.setup_explanation_attributes(explanation_setting)
            self.setup_tcell(color_setting, shape_setting)

    def make_explanation(self):
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        explanation_border = (explanation_border>self.localization_threshold).astype(np.float) 
        explanation_body =  1/3* np.sum(self.parts['inner_ball']**2,axis=2)**0.5
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float) 
        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap 
        explanation = explanation_border + explanation_body

        tail_feature = 1/3* np.sum(self.parts['tail']**2,axis=2)**0.5
        tail_feature = (tail_feature>self.discriminative_feature_threshold).astype(np.float) 
        heatmap = explanation * (1- tail_feature ) * self.localization_value + tail_feature * self.discriminative_feature_value

        unique_set_allowed = {0.,self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap      


class CCellTsX(CCellTs):
    def __init__(self, fast_init=False, array_shape=None,
        color_setting=None, shape_setting=None, explanation_setting=None):
        super(CCellTsX, self).__init__()

        if fast_init:
            self.change_array_shape(array_shape)
            self.setup_explanation_attributes(explanation_setting)
            self.setup_tcell(color_setting, shape_setting)

    def make_explanation(self):
        explanation_border = 1/3* np.sum(self.parts['border']**2,axis=2)**0.5
        explanation_border = (explanation_border>self.localization_threshold).astype(np.float)  
        explanation_body =  1/3* np.sum(self.parts['inner_ball']**2,axis=2)**0.5
        explanation_body = (explanation_body>self.localization_threshold).astype(np.float)
        explanation_body = explanation_body * (1 - explanation_border) # prevent overlap 
        explanation = explanation_border + explanation_body

        tail_feature = 1/3* np.sum(self.parts['tail']**2,axis=2)**0.5
        tail_feature = (tail_feature>self.discriminative_feature_threshold).astype(np.float) 
        heatmap = explanation * (1- tail_feature ) * self.localization_value + tail_feature * self.discriminative_feature_value

        unique_set_allowed = {0.,self.explanation_setting['localization_value'],self.explanation_setting['discriminative_feature_value']}
        unique_entries_check(heatmap, unique_set_allowed)
        return heatmap      
