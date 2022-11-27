from .baseclass import ObjetOutline2D
import numpy as np
from skimage.transform import resize, rotate
from scipy.ndimage import shift

class CCell(ObjetOutline2D):
    def __init__(self, ):
        super(CCell, self).__init__()

    def setup_basic_ball(self, color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {
                'border': np.array([0.8,0.8,0.8]),
                'inner': np.array([0.2,0.2,0.8])
            }
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {
                'radius': 100,
                'vertical_stretch':1.0, # to roughly make ellipse
                'thickness': 20, # border thickness
                'd_noise':0.1,
                'noise_roughness': 20 # 1 is very fine, pixel level noise
            }           
        else:
            self.shape_setting = shape_setting

    def make_basic_ball(self, centerpos=(0,0), rotate_angle=None):
        r = self.shape_setting['radius']
        d_noise = self.shape_setting['d_noise']
        thickness = self.shape_setting['thickness']
        noise_roughness = self.shape_setting['noise_roughness']
        vertical_stretch = self.shape_setting['vertical_stretch']
        border_color = self.color_setting['border']
        inner_color = self.color_setting['inner']        

        return self.build_basic_ball_body(r, thickness, border_color, inner_color, vertical_stretch=vertical_stretch,
            d_noise=d_noise, noise_roughness=noise_roughness, rotate_angle=rotate_angle, centerpos=centerpos)        

    def build_basic_ball_body(self, r, thickness, border_color, inner_color, vertical_stretch=1.0,
        d_noise=None, noise_roughness=None, centerpos=(0,0), rotate_angle=None, 
        do_make_explanation=True):
        """
        r (radius of ball), thickness (of border) are floats in the scale of array pixels (1 means 1 pixel of array)
        d is the array containing distance from origin (defined as the center of the array)

        also in the same order of magnitude as r:
        + centerpos is (x0,y0) : float 
        + d_noise : float. Set None to disable. 
        + noise_roughness : float. Set equals to 1. to generate random normal number per pixel. If set to, for example 2,
            noise is generated every 2x2 pixels
        """

        x, y = self.get_lattice_coord()
        x0,y0 = centerpos

        # d = ((x-x0) **2 + ((y-y0)/vertical_stretch)**2)**0.5
        d = ((x) **2 + ((y)/vertical_stretch)**2)**0.5

        if d_noise is not None:
            n = resize(np.random.normal(0,d_noise, size=[int(dn/noise_roughness) for dn in d.shape]), d.shape)
            d = d + n
        else:
            n = None

        border = (d<=r) * (d>=r-thickness) 
        border = self.stack_color(border, border_color, channel_CHW=True)
        inner_ball = (d<r-thickness)
        inner_ball = self.stack_color(inner_ball, inner_color, channel_CHW=True)
        ball = border + inner_ball

        ############### AFFINE TRANSFORMATION ###############
        # rotate
        if rotate_angle is not None:
            ball = rotate(ball, rotate_angle)
            inner_ball = rotate(inner_ball, rotate_angle)
            border = rotate(border , rotate_angle)            
            
        # translate
        ball = shift(ball, centerpos+(0,))
        inner_ball = shift(inner_ball, centerpos+(0,))
        border = shift(border, centerpos+(0,))

        ball = np.clip(ball, a_min=0., a_max=1.)
        border = np.clip(border, a_min=0., a_max=1.)
        inner_ball = np.clip(inner_ball, a_min=0., a_max=1.)

        ########### EXPLANATION ZONE ###########
        self.parts = {
            'xyd' :(x,y,d),
            'centerpos':(x0,y0),
            'noise':n,
            'border': border,
            'inner_ball': inner_ball,
            'ball': ball # it is okay to save items in conservative way, just for convenience now. No worry about memory efficiency
        }
        if do_make_explanation:
            heatmap = self.make_explanation()
        else:
            heatmap = None
        return ball, heatmap

class RCell(ObjetOutline2D):
    def __init__(self):
        super(RCell, self).__init__()
        

    def setup_rectangle(self,color_setting, shape_setting):
        if color_setting is None:
            self.color_setting = {}
        else:
            self.color_setting = color_setting

        if shape_setting is None:
            self.shape_setting = {}
        else:
            self.shape_setting = shape_setting

