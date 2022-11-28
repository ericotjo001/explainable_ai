import matplotlib.pyplot as plt
import numpy as np


class ObjetOutline2D():
    def __init__(self):
        super(ObjetOutline2D, self).__init__()
        self.s = (512,512)
        self.c = (int(self.s[0]/2),int(self.s[1]/2))

    def change_array_shape(self,s):
        self.s = s
        self.c = (int(self.s[0]/2),int(self.s[1]/2))

    def get_lattice_coord(self):
        s = self.s
        c = self.c
        xg = np.meshgrid(np.linspace(0,s[0]-1,s[0]).astype(int))
        yg = np.meshgrid(np.linspace(0,s[1]-1,s[1]).astype(int))    
        mx,my = np.meshgrid(xg,yg)  
        x, y = mx - c[0], my - c[1]
        return x, y

    def stack_color(self, array, rgb, channel_CHW=False):
        # array: single channel, for example (512,512)
        array = np.stack((array*rgb[0], array*rgb[1], array*rgb[2]))
        if channel_CHW:
            array = array.transpose((1,2,0))
        return array

    def make_explanation(self):
        """ make sure explanation is generated using properties of the object
        _IMPLEMENT_IT_IN_CONTEXT_
        Tips:
        1. During implementation of objects derived from this class, initiate self.parts.
          make_explanation() will then manipulate items stored in self.parts.        
        2. always normalize to [-1,1.] where 0. means no contribution, 1.0 means fully discriminative features,
          -1.0 means negative contribution. To soften the values, set discriminative features as >=0.9, localization
          to be >=0.1. This should be flexible.

        Possibly useful attributes to initiate using setup_explanation_attributes():
        self.localization_threshold = 0.1
        self.localization_value = 0.1
        self.discriminative_feature_threshold = 0.9
        self.discriminative_feature_value = 0.9
        """
        
        explanation = None
        
        return explanation

    def setup_explanation_attributes(self, explanation_setting):
        # _IMPLEMENT_IT_IN_CONTEXT_ according to make_explanation()
        # the following is a basic, but universal implementation
        for attr_name, value in explanation_setting.items():
            setattr(self,attr_name, value)
        self.explanation_setting = explanation_setting
        